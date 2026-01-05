from dataclasses import dataclass, field
from typing import Optional
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProcessingCheckpoint:
    """Tracks processing progress for resume capability."""

    processed_files: dict = field(default_factory=dict)
    current_file: Optional[str] = None
    current_page: int = 0
    total_processed: int = 0
    last_updated: str = ""

    def to_dict(self) -> dict:
        return {
            "processed_files": self.processed_files,
            "current_file": self.current_file,
            "current_page": self.current_page,
            "total_processed": self.total_processed,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingCheckpoint":
        return cls(
            processed_files=data.get("processed_files", {}),
            current_file=data.get("current_file"),
            current_page=data.get("current_page", 0),
            total_processed=data.get("total_processed", 0),
            last_updated=data.get("last_updated", ""),
        )


@dataclass
class APIKeyManager:
    """Manages multiple API keys with automatic rotation on failure."""

    keys: list = field(default_factory=list)
    current_index: int = 0
    failed_keys: set = field(default_factory=set)

    REQUESTS_PER_MINUTE: int = 5
    REQUESTS_PER_DAY: int = 20
    MIN_REQUEST_INTERVAL: float = 13.0

    key_usage: dict = field(default_factory=dict)

    def _get_today(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _init_key_usage(self, key_index: int) -> None:
        today = self._get_today()
        if key_index not in self.key_usage:
            self.key_usage[key_index] = {
                "daily_count": 0,
                "last_request_time": 0.0,
                "date": today,
            }
        elif self.key_usage[key_index]["date"] != today:
            self.key_usage[key_index]["daily_count"] = 0
            self.key_usage[key_index]["date"] = today
            if key_index in self.failed_keys:
                self.failed_keys.discard(key_index)
                logger.info(f"API key {key_index + 1} reset for new day")

    def load_keys(self, keys_file: str) -> bool:
        try:
            with open(keys_file, "r", encoding="utf-8") as f:
                self.keys = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
            if not self.keys:
                logger.error(f"No API keys found in {keys_file}")
                return False
            logger.info(f"Loaded {len(self.keys)} API key(s)")
            for i in range(len(self.keys)):
                self._init_key_usage(i)
            return True
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
            return False

    def get_current_key(self) -> Optional[str]:
        for i in range(len(self.keys)):
            self._init_key_usage(i)

        available_keys = [
            k for i, k in enumerate(self.keys) if i not in self.failed_keys
        ]
        if not available_keys:
            return None
        
        while self.current_index in self.failed_keys:
            self.current_index = (self.current_index + 1) % len(self.keys)
            if all(i in self.failed_keys for i in range(len(self.keys))):
                return None
        return self.keys[self.current_index]

    def wait_for_rate_limit(self) -> None:
        self._init_key_usage(self.current_index)
        usage = self.key_usage[self.current_index]

        if usage["last_request_time"] > 0:
            elapsed = time.time() - usage["last_request_time"]
            if elapsed < self.MIN_REQUEST_INTERVAL:
                wait_time = self.MIN_REQUEST_INTERVAL - elapsed
                logger.info(f"⏳ Rate limit: waiting {wait_time:.1f}s before next request...")
                time.sleep(wait_time)

    def record_request(self) -> bool:
        self._init_key_usage(self.current_index)
        usage = self.key_usage[self.current_index]

        usage["daily_count"] += 1
        usage["last_request_time"] = time.time()

        remaining = self.REQUESTS_PER_DAY - usage["daily_count"]
        logger.info(
            f"📊 API key {self.current_index + 1}: {usage['daily_count']}/{self.REQUESTS_PER_DAY} requests today ({remaining} remaining)"
        )

        if usage["daily_count"] >= self.REQUESTS_PER_DAY:
            logger.warning(f"API key {self.current_index + 1} reached daily limit")
            return False
        return True

    def mark_current_failed(self, reason: str = "error") -> bool:
        self.failed_keys.add(self.current_index)
        logger.warning(
            f"API key {self.current_index + 1} marked as failed ({reason})"
        )

        if len(self.failed_keys) >= len(self.keys):
            logger.error("All API keys have been exhausted!")
            return False

        self.current_index = (self.current_index + 1) % len(self.keys)
        while self.current_index in self.failed_keys:
            self.current_index = (self.current_index + 1) % len(self.keys)

        logger.info(f"🔄 Rotated to API key {self.current_index + 1}")
        return True

    def has_available_keys(self) -> bool:
        for i in range(len(self.keys)):
            self._init_key_usage(i)
        return len(self.failed_keys) < len(self.keys)

    def get_status_summary(self) -> str:
        lines = ["API Keys Status:"]
        for i in range(len(self.keys)):
            self._init_key_usage(i)
            usage = self.key_usage[i]
            status = (
                "❌ FAILED"
                if i in self.failed_keys
                else ("✅ CURRENT" if i == self.current_index else "⏸️ AVAILABLE")
            )
            lines.append(
                f"  Key {i + 1}: {usage['daily_count']}/{self.REQUESTS_PER_DAY} requests - {status}"
            )
        return "\n".join(lines)
