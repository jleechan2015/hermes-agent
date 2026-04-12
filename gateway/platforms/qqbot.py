"""
QQBot platform adapter using the official QQ Bot Open Platform Python SDK (qq-botpy).

Supports four message scenes:
  - guild_at      : @bot in a guild channel  (post_message)
  - direct_message: private DM              (post_dms)
  - c2c           : C2C (single-user app)   (message.reply)
  - group_at      : @bot in a group         (message.reply)

Configuration in config.yaml:
    platforms:
      qqbot:
        enabled: true
        extra:
          appid: "your_appid"   # or QQBOT_APPID env var
        token: "your_secret"    # or QQBOT_SECRET env var (BotSecret, NOT BotToken)

Environment variables:
    QQBOT_APPID   - BotAppID
    QQBOT_SECRET  - BotSecret (used with botpy ≥ 2.x start(appid, secret))
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

try:
    import botpy
    from botpy.message import C2CMessage, DirectMessage, GroupMessage, Message
    QQBOTPY_AVAILABLE = True
except ImportError:
    QQBOTPY_AVAILABLE = False
    botpy = None  # type: ignore[assignment]
    Message = None  # type: ignore[assignment]
    DirectMessage = None  # type: ignore[assignment]
    C2CMessage = None  # type: ignore[assignment]
    GroupMessage = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_bytes,
)

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 2000
DEDUP_WINDOW_SECONDS = 300
HISTORY_MAXLEN = 500
RECONNECT_BACKOFF = [2, 5, 10, 30, 60]

# C2C / group messages require a monotonically increasing msg_seq per session
_C2C_MSG_SEQ_MAX = 1_000_000


def check_qqbot_requirements() -> bool:
    """Check if QQBot dependencies are available and configured."""
    if not QQBOTPY_AVAILABLE:
        logger.warning("[QQBot] qq-botpy not installed. Run: uv add qq-botpy")
        return False
    appid = os.getenv("QQBOT_APPID")
    secret = os.getenv("QQBOT_SECRET")
    if not appid or not secret:
        logger.warning("[QQBot] QQBOT_APPID / QQBOT_SECRET env vars not set.")
        return False
    return True


# ---------------------------------------------------------------------------
# Internal hub models
# ---------------------------------------------------------------------------

@dataclass
class _PendingEvent:
    """Normalised inbound event stored in the hub."""
    event_id: str
    scene: str        # guild_at | direct_message | c2c | group_at
    source: str       # "{platform}:{chat_id}"
    content: str
    sender_id: str
    created_at: str
    attachments: List[Dict] = field(default_factory=list)
    handled: bool = False

    @classmethod
    def from_message_event(cls, event: MessageEvent, scene: str) -> "_PendingEvent":
        src = event.source
        source_str = f"{src.platform}:{src.chat_id}" if src else ""
        attachments = [
            {"url": u, "type": t}
            for u, t in zip(event.media_urls, event.media_types)
        ]
        return cls(
            event_id=event.message_id or uuid.uuid4().hex,
            scene=scene,
            source=source_str,
            content=event.text,
            sender_id=str(src.user_id) if src and src.user_id else "",
            created_at=datetime.now(timezone.utc).isoformat(),
            attachments=attachments,
        )


@dataclass
class _ReplyResult:
    event_id: str
    ok: bool
    message: str


# Responder is the callable stored alongside each pending event.
# It accepts the same **kwargs as botpy message.reply() / post_message().
Responder = Callable[..., Awaitable[None]]


# ---------------------------------------------------------------------------
# MessageHub
# ---------------------------------------------------------------------------

class _MessageHub:
    """
    Coroutine-safe message bus embedded in QQBotAdapter.

    Mirrors the interface of qq-bot-mcp's MessageHub so the same MCP
    tool layer can drive either implementation.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        # event_id -> (_PendingEvent, Responder)
        self._pending: Dict[str, Tuple[_PendingEvent, Responder]] = {}
        self._history: deque[_PendingEvent] = deque(maxlen=HISTORY_MAXLEN)
        self._total_received: int = 0

    async def publish(
        self,
        event: MessageEvent,
        scene: str,
        responder: Responder,
    ) -> _PendingEvent:
        async with self._condition:
            ev = _PendingEvent.from_message_event(event, scene)
            self._pending[ev.event_id] = (ev, responder)
            self._history.appendleft(ev)
            self._total_received += 1
            self._condition.notify_all()
            return ev

    async def pending_count(self) -> int:
        async with self._lock:
            return len(self._pending)

    async def total_received(self) -> int:
        async with self._lock:
            return self._total_received

    async def list_pending(self, limit: int = 20) -> List[Dict]:
        async with self._lock:
            items = [asdict(item[0]) for item in self._pending.values()]
        return items[: max(1, min(limit, 100))]

    async def list_history(self, limit: int = 20) -> List[Dict]:
        async with self._lock:
            items = [asdict(ev) for ev in list(self._history)[: max(1, min(limit, 100))]]
        return items

    async def wait_event(self, timeout_seconds: int = 15) -> Dict:
        timeout = max(1, min(timeout_seconds, 120))
        async with self._condition:
            if self._pending:
                first = next(iter(self._pending.values()))[0]
                return {"ok": True, "event": asdict(first)}
            try:
                await asyncio.wait_for(self._condition.wait(), timeout=timeout)
            except (asyncio.TimeoutError, TimeoutError):
                return {"ok": False, "message": "timeout"}
            if not self._pending:
                return {"ok": False, "message": "timeout"}
            first = next(iter(self._pending.values()))[0]
            return {"ok": True, "event": asdict(first)}

    async def reply(self, event_id: str, **kwargs: Any) -> _ReplyResult:
        """
        Dispatch a reply via the stored responder.

        Supported kwargs (passed directly to botpy):
            content, msg_type, image, markdown, ark, embed, media, keyboard
        """
        async with self._lock:
            target = self._pending.pop(event_id, None)
        if not target:
            return _ReplyResult(event_id=event_id, ok=False, message="event_id 不存在或已处理")

        ev, responder = target
        try:
            await responder(**kwargs)
            ev.handled = True
            return _ReplyResult(event_id=event_id, ok=True, message="回复成功")
        except Exception as exc:
            logger.error("[QQBot] hub reply failed for %s: %s", event_id, exc)
            async with self._lock:
                self._pending[event_id] = (ev, responder)
            return _ReplyResult(event_id=event_id, ok=False, message=f"回复失败: {exc}")

    async def discard(self, event_id: str) -> _ReplyResult:
        async with self._lock:
            target = self._pending.pop(event_id, None)
        if not target:
            return _ReplyResult(event_id=event_id, ok=False, message="event_id 不存在或已处理")
        ev, _ = target
        ev.handled = True
        return _ReplyResult(event_id=event_id, ok=True, message="已丢弃")


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class QQBotAdapter(BasePlatformAdapter):
    """
    QQBot adapter using the official qq-botpy SDK.

    Supports guild @-messages, direct messages, C2C, and group @-messages.
    Inbound events are dispatched to the gateway handler AND published to
    the embedded MessageHub so external tools can poll / reply via the hub.
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig) -> None:
        super().__init__(config, Platform.QQBOT)

        extra = config.extra or {}
        self._appid: str = str(extra.get("appid") or os.getenv("QQBOT_APPID", "")).strip()
        # botpy ≥ 2.x uses "secret" (BotSecret), not the old token
        self._secret: str = str(
            config.token
            or extra.get("secret")
            or os.getenv("QQBOT_SECRET", "")
            or os.getenv("QQBOT_TOKEN", "")  # legacy fallback
        ).strip()

        self._bot_client: Optional[Any] = None
        self._connect_task: Optional[asyncio.Task] = None

        # Dedup: msg_id -> timestamp
        self._seen_messages: Dict[str, float] = {}
        # DM guild_ids (private session IDs) for routing post_dms
        self._dm_guild_ids: set = set()
        # Per-session C2C/group msg_seq counter
        self._msg_seq: int = 1
        # chat_id -> (scene, responder) for group_at / c2c scenes that require
        # message.reply() instead of post_message / post_dms
        self._responder_cache: Dict[str, Tuple[str, Responder]] = {}

        self._intents = (
            botpy.Intents(
                public_guild_messages=True,
                direct_message=True,
                public_messages=True,   # C2C + group
                guild_messages=True,
            )
            if QQBOTPY_AVAILABLE
            else None
        )

        # Embedded message hub
        self.hub = _MessageHub()

    # -- Connection lifecycle ------------------------------------------------

    async def connect(self) -> bool:
        if not QQBOTPY_AVAILABLE:
            msg = "qq-botpy not installed. Run: uv add qq-botpy"
            self._set_fatal_error("missing_dependency", msg, retryable=True)
            logger.error("[%s] %s", self.name, msg)
            return False
        if not self._appid or not self._secret:
            msg = "QQBOT_APPID / QQBOT_SECRET not set"
            self._set_fatal_error("missing_credentials", msg, retryable=False)
            logger.error("[%s] %s", self.name, msg)
            return False

        try:
            self._bot_client = _QQBotClient(
                intents=self._intents,
                adapter=self,
            )
            self._connect_task = asyncio.create_task(self._run_client())
            self._mark_connected()
            logger.info("[%s] WebSocket task started (appid=%s)", self.name, self._appid)
            return True
        except Exception as exc:
            msg = f"Connection failed: {exc}"
            self._set_fatal_error("connect_error", msg, retryable=True)
            logger.error("[%s] %s", self.name, msg)
            return False

    async def _run_client(self) -> None:
        try:
            if self._bot_client:
                async with self._bot_client:
                    await self._bot_client.start(
                        appid=self._appid,
                        secret=self._secret,
                    )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("[%s] Client error: %s", self.name, exc)
            self._set_fatal_error("connection_failed", str(exc), retryable=True)
            await self._notify_fatal_error()

    async def disconnect(self) -> None:
        self._running = False
        if self._connect_task:
            self._connect_task.cancel()
            try:
                await self._connect_task
            except asyncio.CancelledError:
                pass
        self._bot_client = None
        self._mark_disconnected()
        logger.info("[%s] Disconnected", self.name)

    # -- Send (gateway-initiated outbound) -----------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a plain-text message to a guild channel, DM, group, or C2C."""
        if not self._bot_client or not hasattr(self._bot_client, "api"):
            return SendResult(success=False, error="Bot not connected")
        try:
            if len(content) > MAX_MESSAGE_LENGTH:
                content = content[: MAX_MESSAGE_LENGTH - 3] + "..."

            # group_at / c2c must reply via message.reply(), not post_message
            cached = self._responder_cache.get(chat_id)
            if cached:
                scene, responder = cached
                kwargs: Dict[str, Any] = {"content": content}
                if scene in ("group_at", "c2c"):
                    kwargs["msg_seq"] = self._next_msg_seq()
                await responder(**kwargs)
                return SendResult(success=True)

            is_dm = chat_id in self._dm_guild_ids
            if is_dm:
                result = await self._bot_client.api.post_dms(
                    guild_id=chat_id,
                    content=content,
                    msg_id=reply_to or None,
                )
            else:
                result = await self._bot_client.api.post_message(
                    channel_id=chat_id,
                    content=content,
                    msg_id=reply_to or None,
                )
            msg_id = result.get("id") if isinstance(result, dict) else str(result)
            return SendResult(success=True, message_id=msg_id, raw_response=result)
        except Exception as exc:
            logger.error("[%s] send failed: %s", self.name, exc)
            retryable = any(p in str(exc).lower() for p in ("timeout", "network", "connection"))
            return SendResult(success=False, error=str(exc), retryable=retryable)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image to a guild channel, DM, group, or C2C (URL-based)."""
        if not self._bot_client or not hasattr(self._bot_client, "api"):
            return SendResult(success=False, error="Bot not connected")
        try:
            # group_at / c2c: use cached responder
            cached = self._responder_cache.get(chat_id)
            if cached:
                scene, responder = cached
                kwargs: Dict[str, Any] = {"image": image_url}
                if scene in ("group_at", "c2c"):
                    kwargs["msg_seq"] = self._next_msg_seq()
                await responder(**kwargs)
                if caption:
                    await self.send(chat_id, caption)
                return SendResult(success=True)

            is_dm = chat_id in self._dm_guild_ids
            if is_dm:
                result = await self._bot_client.api.post_dms(
                    guild_id=chat_id,
                    image=image_url,
                    msg_id=reply_to or None,
                )
            else:
                result = await self._bot_client.api.post_message(
                    channel_id=chat_id,
                    image=image_url,
                    msg_id=reply_to or None,
                )
            if caption:
                await self.send(chat_id, caption, reply_to=reply_to)
            msg_id = result.get("id") if isinstance(result, dict) else None
            return SendResult(success=True, message_id=msg_id, raw_response=result)
        except Exception as exc:
            logger.error("[%s] send_image failed: %s", self.name, exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        try:
            if self._bot_client and hasattr(self._bot_client, "api"):
                info = await self._bot_client.api.get_channel(channel_id=chat_id)
                if info:
                    return {
                        "name": info.get("name", chat_id),
                        "type": "group" if info.get("guild_id") else "direct",
                        "chat_id": chat_id,
                    }
        except Exception as exc:
            logger.debug("[%s] get_chat_info failed: %s", self.name, exc)
        return {"name": chat_id, "type": "unknown", "chat_id": chat_id}

    # -- Hub public interface ------------------------------------------------

    async def hub_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "pending": await self.hub.pending_count(),
            "total_received": await self.hub.total_received(),
        }

    async def hub_list_pending(self, limit: int = 20) -> List[Dict]:
        return await self.hub.list_pending(limit)

    async def hub_wait_event(self, timeout_seconds: int = 15) -> Dict:
        return await self.hub.wait_event(timeout_seconds)

    async def hub_list_history(self, limit: int = 20) -> List[Dict]:
        return await self.hub.list_history(limit)

    async def hub_reply_event(
        self,
        event_id: str,
        content: Optional[str] = None,
        msg_type: Optional[int] = None,
        image: Optional[str] = None,
        markdown: Optional[Dict] = None,
        ark: Optional[Dict] = None,
        embed: Optional[Dict] = None,
        media: Optional[Dict] = None,
        keyboard: Optional[Dict] = None,
    ) -> _ReplyResult:
        """
        Reply to a pending hub event.

        All parameters optional — at least one required.
        Passed directly to botpy (message.reply / post_message / post_dms).
        """
        kwargs: Dict[str, Any] = {}
        if content is not None:
            kwargs["content"] = content
        if msg_type is not None:
            kwargs["msg_type"] = msg_type
        if image is not None:
            kwargs["image"] = image
        if markdown is not None:
            kwargs["markdown"] = markdown
        else:
            kwargs["markdown"] = {"content":content}
        if ark is not None:
            kwargs["ark"] = ark
        if embed is not None:
            kwargs["embed"] = embed
        if media is not None:
            kwargs["media"] = media
        if keyboard is not None:
            kwargs["keyboard"] = keyboard
        if not kwargs:
            return _ReplyResult(event_id=event_id, ok=False, message="至少需要提供一个回复参数")
        return await self.hub.reply(event_id=event_id, **kwargs)

    async def hub_discard_event(self, event_id: str) -> _ReplyResult:
        return await self.hub.discard(event_id)

    # -- Internal helpers ----------------------------------------------------

    def _next_msg_seq(self) -> int:
        self._msg_seq += 1
        if self._msg_seq > _C2C_MSG_SEQ_MAX:
            self._msg_seq = 1
        return self._msg_seq

    def _is_duplicate(self, msg_id: str) -> bool:
        now = time.time()
        cutoff = now - DEDUP_WINDOW_SECONDS
        self._seen_messages = {k: v for k, v in self._seen_messages.items() if v > cutoff}
        if msg_id in self._seen_messages:
            return True
        self._seen_messages[msg_id] = now
        return False

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip @bot mention tags from message content."""
        return re.sub(r"<@!?\S+?>", "", text or "").strip()

    @staticmethod
    def _extract_attachments(message: Any) -> List[Dict]:
        raw = getattr(message, "attachments", None) or []
        result = []
        for att in raw:
            result.append({
                "content_type": getattr(att, "content_type", "") or "",
                "filename": getattr(att, "filename", "") or "",
                "url": getattr(att, "url", "") or "",
                "width": getattr(att, "width", None),
                "height": getattr(att, "height", None),
                "size": getattr(att, "size", None),
            })
        return result

    def _get_sender_id(self, message: Any) -> str:
        for attr in ("author", "src_guild_id", "group_openid", "openid"):
            value = getattr(message, attr, None)
            if not value:
                continue
            for sub in ("id", "member_openid", "user_openid"):
                v = getattr(value, sub, None)
                if v:
                    return str(v)
            if isinstance(value, str):
                return value
        return "unknown"

    def _make_responder(self, message: Any, scene: str) -> Responder:
        """Build a responder callable that wraps botpy's reply API."""
        adapter = self

        async def _responder(**kwargs: Any) -> None:
            # C2C and group require msg_seq
            if scene in ("c2c", "group_at"):
                kwargs.setdefault("msg_seq", adapter._next_msg_seq())
            await message.reply(**kwargs)

        return _responder

    async def _handle_inbound(
        self,
        message: Any,
        scene: str,
        chat_id: str,
        chat_type: str,
        user_id: str,
        user_name: str,
    ) -> None:
        """Common inbound processing for all four scenes."""
        raw_id = getattr(message, "id", None) or str(uuid.uuid4())
        if self._is_duplicate(raw_id):
            return

        content = self._clean_text(getattr(message, "content", "") or "")
        attachments = self._extract_attachments(message)

        if not content and not attachments:
            return

        msg_type = MessageType.TEXT
        if attachments and any(
            a.get("content_type", "").startswith("image/") for a in attachments
        ):
            msg_type = MessageType.PHOTO

        source = self.build_source(
            chat_id=chat_id,
            chat_type=chat_type,
            user_id=user_id or None,
            user_name=user_name or None,
        )

        event = MessageEvent(
            text=content,
            message_type=msg_type,
            source=source,
            raw_message=message,
            message_id=raw_id,
            media_urls=[a["url"] for a in attachments if a.get("url")],
            media_types=[a["content_type"] for a in attachments if a.get("url")],
            timestamp=datetime.now(tz=timezone.utc),
        )

        responder = self._make_responder(message, scene)

        # Cache responder for group_at / c2c so send() can route correctly
        if scene in ("group_at", "c2c"):
            self._responder_cache[chat_id] = (scene, responder)

        # Publish to hub (for MCP / external tool access)
        await self.hub.publish(event, scene, responder)

        # Also dispatch through the standard gateway handler
        await self.handle_message(event)

    # -- QQ event handlers (called by _QQBotClient) --------------------------

    async def on_at_message_create(self, message: Any) -> None:
        """Guild channel @bot message."""
        guild_id = getattr(message, "guild_id", "") or ""
        channel_id = getattr(message, "channel_id", "") or ""
        author = getattr(message, "author", None)
        user_id = self._get_sender_id(message)
        user_name = getattr(author, "username", None) or getattr(author, "name", None) or ""
        await self._handle_inbound(
            message=message,
            scene="guild_at",
            chat_id=channel_id,
            chat_type="group",
            user_id=user_id,
            user_name=user_name,
        )

    async def on_direct_message_create(self, message: Any) -> None:
        """Private DM (direct message)."""
        # guild_id on a DM is the private session ID required by post_dms
        dm_guild_id = getattr(message, "guild_id", "") or ""
        author = getattr(message, "author", None)
        user_id = self._get_sender_id(message)
        user_name = getattr(author, "username", None) or getattr(author, "name", None) or ""

        if dm_guild_id:
            self._dm_guild_ids.add(dm_guild_id)

        await self._handle_inbound(
            message=message,
            scene="direct_message",
            chat_id=dm_guild_id or user_id,
            chat_type="dm",
            user_id=user_id,
            user_name=user_name,
        )

    async def on_c2c_message_create(self, message: Any) -> None:
        """C2C (single-user app) message."""
        user_id = self._get_sender_id(message)
        await self._handle_inbound(
            message=message,
            scene="c2c",
            chat_id=user_id,
            chat_type="dm",
            user_id=user_id,
            user_name="",
        )

    async def on_group_at_message_create(self, message: Any) -> None:
        """Group @bot message."""
        group_id = getattr(message, "group_openid", None) or getattr(message, "group_id", "") or ""
        user_id = self._get_sender_id(message)
        await self._handle_inbound(
            message=message,
            scene="group_at",
            chat_id=group_id or user_id,
            chat_type="group",
            user_id=user_id,
            user_name="",
        )


# ---------------------------------------------------------------------------
# botpy Client subclass
# ---------------------------------------------------------------------------

class _QQBotClient(botpy.Client):
    """Thin botpy.Client subclass that forwards events to QQBotAdapter."""

    def __init__(self, intents: Any, adapter: QQBotAdapter, **kwargs: Any) -> None:
        self._adapter = adapter
        super().__init__(intents=intents, **kwargs)
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

    async def on_ready(self) -> None:
        self._adapter._running = True
        self._adapter._mark_connected()
        logger.info("[QQBot] Bot ready")

    async def on_at_message_create(self, message: Message) -> None:
        await self._adapter.on_at_message_create(message)

    async def on_direct_message_create(self, message: DirectMessage) -> None:
        await self._adapter.on_direct_message_create(message)

    async def on_c2c_message_create(self, message: C2CMessage) -> None:
        await self._adapter.on_c2c_message_create(message)

    async def on_group_at_message_create(self, message: GroupMessage) -> None:
        await self._adapter.on_group_at_message_create(message)
