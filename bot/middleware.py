"""Security middleware guaranteeing fail-closed private chat, allowlist, rate limiting, and input size boundaries."""

import functools
import time
from typing import Callable, Coroutine, List, Optional, Set, Tuple
from telegram import Update
from telegram.ext import ApplicationHandlerStop, ContextTypes, TypeHandler
from core.logger import get_logger
from core.security import enforce_user_message_length

logger = get_logger("bot_middleware")


class TokenBucketRateLimiter:
    """Thread/coroutine safe in-memory token bucket rate limiter per user ID."""

    def __init__(self, requests_per_minute: int = 10):
        self.capacity = float(requests_per_minute)
        self.refill_rate = self.capacity / 60.0  # tokens per second
        self._buckets: dict[int, Tuple[float, float]] = {}  # user_id -> (tokens, last_timestamp)

    def allow_request(self, user_id: int) -> bool:
        """Check if request from user_id is permitted under the rate limit."""
        now = time.monotonic()
        if user_id not in self._buckets:
            # First request: consume 1 token from full capacity
            self._buckets[user_id] = (self.capacity - 1.0, now)
            return True

        tokens, last_time = self._buckets[user_id]
        elapsed = now - last_time
        tokens = min(self.capacity, tokens + elapsed * self.refill_rate)

        if tokens >= 1.0:
            self._buckets[user_id] = (tokens - 1.0, now)
            return True
        else:
            self._buckets[user_id] = (tokens, now)
            return False

    def reset(self, user_id: Optional[int] = None) -> None:
        """Reset rate limiter state (for tests or admin override)."""
        if user_id is not None:
            self._buckets.pop(user_id, None)
        else:
            self._buckets.clear()


class SecurityGate:
    """Automated security gate enforcing strict pre-execution isolation."""

    def __init__(
        self,
        allowed_user_ids: List[int],
        rate_limit_per_minute: int = 10,
        max_message_chars: int = 4000,
    ):
        self.allowed_user_ids: Set[int] = set(allowed_user_ids)
        self.rate_limiter = TokenBucketRateLimiter(requests_per_minute=rate_limit_per_minute)
        self.max_message_chars = max_message_chars

    async def evaluate_security(self, update: Update) -> Tuple[bool, Optional[str]]:
        """Strictly evaluate security rules in order:
        1. Private-chat check
        2. User allowlist check
        3. Rate limiter check
        4. Input-size validation
        Returns (is_allowed, failure_reason).
        """
        # 1. Private-chat check
        chat = update.effective_chat
        if not chat or chat.type != "private":
            logger.warning("Rejected non-private chat update.")
            return False, "non_private_chat"

        # 2. User allowlist check
        user = update.effective_user
        if not user or user.id not in self.allowed_user_ids:
            logger.warning("Rejected unauthorized user ID.")
            return False, "unauthorized_user"

        # 3. Rate limiter check
        if not self.rate_limiter.allow_request(user.id):
            logger.warning("Rate limit exceeded for user.")
            return False, "rate_limit_exceeded"

        # 4. Input-size validation
        message = update.effective_message
        if message and message.text:
            if len(message.text) > self.max_message_chars:
                logger.warning("Message length exceeds maximum allowed characters.")
                return False, "input_too_long"

        return True, None

    async def middleware_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Application-level TypeHandler callback registered at group -1.
        
        Guarantees that if ANY check fails, the update never reaches any downstream handler.
        """
        is_allowed, reason = await self.evaluate_security(update)
        if not is_allowed:
            if reason == "rate_limit_exceeded":
                if update.effective_message:
                    await update.effective_message.reply_text(
                        "⏳ *Rate limit reached* (maximum 10 requests per minute). Please wait a moment before trying again.",
                        parse_mode="Markdown",
                    )
            elif reason == "input_too_long":
                if update.effective_message:
                    await update.effective_message.reply_text(
                        f"⚠️ *Message too long*: Incoming text exceeds the limit of {self.max_message_chars} characters. "
                        "Please send a shorter query or excerpt.",
                        parse_mode="Markdown",
                    )
            elif reason == "unauthorized_user":
                if update.effective_message:
                    # Minimal generic rejection without exposing internal information
                    await update.effective_message.reply_text("⛔ Unauthorized access.")
            
            # Stop all further handlers across all groups
            raise ApplicationHandlerStop()

    def guard(self, handler_func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
        """Decorator wrapping individual handlers to guarantee pre-execution authorization
        even if invoked outside the PTB group dispatcher.
        """
        @functools.wraps(handler_func)
        async def guarded_wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            is_allowed, reason = await self.evaluate_security(update)
            if not is_allowed:
                if reason == "rate_limit_exceeded":
                    if update.effective_message:
                        await update.effective_message.reply_text(
                            "⏳ *Rate limit reached* (maximum 10 requests per minute). Please wait a moment before trying again.",
                            parse_mode="Markdown",
                        )
                elif reason == "input_too_long":
                    if update.effective_message:
                        await update.effective_message.reply_text(
                            f"⚠️ *Message too long*: Incoming text exceeds the limit of {self.max_message_chars} characters.",
                            parse_mode="Markdown",
                        )
                elif reason == "unauthorized_user":
                    if update.effective_message:
                        await update.effective_message.reply_text("⛔ Unauthorized access.")
                # Halt execution immediately; do NOT invoke handler_func
                return
            return await handler_func(update, context, *args, **kwargs)

        return guarded_wrapper
