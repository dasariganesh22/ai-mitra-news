from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo
import dateutil.parser
from core.models import ConflictInfo, NewsFactSheet, ReliabilityState, SourceFact, ToolMetadata


def parse_to_timezone(val: Optional[str], tz: ZoneInfo) -> Optional[datetime]:
    """Safely parse ISO, RFC, or arbitrary date string into a timezone-aware datetime."""
    if not val or not isinstance(val, str) or not val.strip():
        return None
    try:
        dt = dateutil.parser.parse(val.strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(tz)
    except Exception:
        return None


def get_effective_article_time(item: NewsFactSheet, tz: ZoneInfo) -> Optional[datetime]:
    """Determine the effective article timestamp according to strict precedence:
    1. updated_at when a trustworthy update timestamp is available.
    2. Otherwise published_at when a trustworthy publication timestamp is available.
    3. If neither is available, treat the item as undated (None).
    Do NOT use retrieved_at or event_date for digest eligibility.
    """
    if item.updated_at:
        dt = parse_to_timezone(item.updated_at, tz)
        if dt is not None:
            return dt
    if item.published_at:
        dt = parse_to_timezone(item.published_at, tz)
        if dt is not None:
            return dt
    return None


class DatabaseManager:
    """Manages SQLite database connections with strict parameter scoping."""

    def __init__(self, db_path: str = "data/ai_mitra.db"):
        self.db_path = Path(db_path)
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Get connection with Row factory enabled."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Initialize database schema if not already present."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # News items table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS news_items (
                    event_id TEXT PRIMARY KEY,
                    headline TEXT NOT NULL,
                    primary_source_name TEXT NOT NULL,
                    primary_source_url TEXT NOT NULL,
                    supporting_sources_json TEXT NOT NULL,
                    reliability TEXT NOT NULL,
                    is_official_source INTEGER NOT NULL,
                    source_reported_facts_json TEXT NOT NULL,
                    ai_summary TEXT NOT NULL,
                    ai_interpretation TEXT,
                    conflicts_json TEXT NOT NULL,
                    tool_metadata_json TEXT,
                    category TEXT NOT NULL,
                    published_at TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # User conversations table strictly scoped by user_id
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Create index for fast user-scoped conversation lookups
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_bookmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    event_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (event_id) REFERENCES news_items(event_id)
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_conversations_user_id ON user_conversations(user_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_news_items_category ON news_items(category)"
            )

            # Migration: Safely add timestamp columns to news_items if missing (preserves existing DB)
            cursor.execute("PRAGMA table_info(news_items)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            if "updated_at" not in existing_columns:
                cursor.execute("ALTER TABLE news_items ADD COLUMN updated_at TEXT")
            if "retrieved_at" not in existing_columns:
                cursor.execute("ALTER TABLE news_items ADD COLUMN retrieved_at TEXT")
            if "event_date" not in existing_columns:
                cursor.execute("ALTER TABLE news_items ADD COLUMN event_date TEXT")

            conn.commit()

    # --------------------------------------------------------------------------
    # News Persistence
    # --------------------------------------------------------------------------

    def save_news_item(self, item: NewsFactSheet) -> None:
        """Persist or update a news item using parameterized queries."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO news_items (
                    event_id, headline, primary_source_name, primary_source_url,
                    supporting_sources_json, reliability, is_official_source,
                    source_reported_facts_json, ai_summary, ai_interpretation,
                    conflicts_json, tool_metadata_json, category, published_at,
                    updated_at, retrieved_at, event_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.event_id,
                    item.headline,
                    item.primary_source_name,
                    item.primary_source_url,
                    json.dumps(item.supporting_sources),
                    item.reliability.value,
                    1 if item.is_official_source else 0,
                    json.dumps([f.model_dump() for f in item.source_reported_facts]),
                    item.ai_summary,
                    item.ai_interpretation,
                    json.dumps([c.model_dump() for c in item.conflicts]),
                    json.dumps(item.tool_metadata.model_dump()) if item.tool_metadata else None,
                    item.category,
                    item.published_at,
                    item.updated_at,
                    item.retrieved_at,
                    item.event_date,
                ),
            )
            conn.commit()

    def get_recent_news(self, limit: int = 10, category: Optional[str] = None) -> List[NewsFactSheet]:
        """Fetch recent news items ordered by insertion."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if category:
                cursor.execute(
                    "SELECT * FROM news_items WHERE category = ? ORDER BY created_at DESC LIMIT ?",
                    (category, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM news_items ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            rows = cursor.fetchall()
            return [self._row_to_news_fact_sheet(row) for row in rows]

    def get_news_by_event_id(self, event_id: str) -> Optional[NewsFactSheet]:
        """Fetch a specific news event by its ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM news_items WHERE event_id = ?", (event_id,))
            row = cursor.fetchone()
            return self._row_to_news_fact_sheet(row) if row else None

    def get_news_for_digest(
        self,
        cutoff_time: datetime,
        lookback_hours: int = 24,
        limit: int = 6,
        category: Optional[str] = None,
        timezone_str: str = "Asia/Kolkata",
    ) -> List[NewsFactSheet]:
        """Fetch news items strictly filtered and prioritized by recent lookback window.

        Precedence & Window Rules:
        1. Effective timestamp = updated_at (if valid), else published_at (if valid).
           retrieved_at and event_date are NOT used for eligibility.
        2. Future exclusion: Any item where effective_time > cutoff_time is excluded.
        3. Priority 1: Items within [cutoff_time - lookback_hours, cutoff_time] (e.g. 24h),
           ordered by effective_time DESC.
        4. Priority 2: Window expansion to 48h and 72h if insufficient items in 24h window.
        5. Priority 3: Undated items (effective_time is None) can fill remaining slots
           if limit is not reached, without being falsely presented as today's news.
        6. Does not force a fixed number if genuinely relevant stories are fewer than limit.
        """
        tz = ZoneInfo(timezone_str)
        if cutoff_time.tzinfo is None:
            cutoff_dt = cutoff_time.replace(tzinfo=tz)
        else:
            cutoff_dt = cutoff_time.astimezone(tz)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            if category:
                cursor.execute(
                    "SELECT * FROM news_items WHERE category = ? ORDER BY created_at DESC LIMIT 100",
                    (category,),
                )
            else:
                cursor.execute(
                    "SELECT * FROM news_items ORDER BY created_at DESC LIMIT 100"
                )
            rows = cursor.fetchall()
            candidates = [self._row_to_news_fact_sheet(row) for row in rows]

        window_primary: List[Tuple[datetime, NewsFactSheet]] = []
        window_48h: List[Tuple[datetime, NewsFactSheet]] = []
        window_72h: List[Tuple[datetime, NewsFactSheet]] = []
        undated_items: List[NewsFactSheet] = []

        for item in candidates:
            eff_dt = get_effective_article_time(item, tz)

            if eff_dt is not None:
                # Rule 3: Future exclusion
                if eff_dt > cutoff_dt:
                    continue

                age_seconds = (cutoff_dt - eff_dt).total_seconds()
                age_hours = age_seconds / 3600.0

                if 0 <= age_hours <= lookback_hours:
                    window_primary.append((eff_dt, item))
                elif lookback_hours < age_hours <= 48:
                    window_48h.append((eff_dt, item))
                elif 48 < age_hours <= 72:
                    window_72h.append((eff_dt, item))
            else:
                undated_items.append(item)

        # Sort dated windows by effective time descending (freshest first)
        window_primary.sort(key=lambda x: x[0], reverse=True)
        window_48h.sort(key=lambda x: x[0], reverse=True)
        window_72h.sort(key=lambda x: x[0], reverse=True)

        selected: List[NewsFactSheet] = []

        # 1. Primary window (e.g. 24h)
        for _, item in window_primary:
            if len(selected) < limit:
                selected.append(item)

        # 2. Window expansion: 48h
        if len(selected) < limit:
            for _, item in window_48h:
                if len(selected) < limit:
                    selected.append(item)

        # 3. Window expansion: 72h
        if len(selected) < limit:
            for _, item in window_72h:
                if len(selected) < limit:
                    selected.append(item)

        # 4. Undated items (only if still needed and available)
        if len(selected) < limit:
            for item in undated_items:
                if len(selected) < limit:
                    selected.append(item)

        return selected

    # --------------------------------------------------------------------------
    # User Scoping & Boundaries (Change 5)
    # --------------------------------------------------------------------------

    def save_chat_message(self, user_id: int, role: str, content: str) -> None:
        """Store conversation record strictly scoped to the given authorized user_id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO user_conversations (user_id, role, content) VALUES (?, ?, ?)",
                (user_id, role, content),
            )
            conn.commit()

    def get_user_conversation_history(self, user_id: int, limit: int = 20) -> List[dict]:
        """Retrieve conversation history strictly scoped to user_id using parameterized SQL.

        Guarantees that user A's data can never be returned to user B.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, user_id, role, content, created_at
                FROM user_conversations
                WHERE user_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (user_id, limit),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def clear_user_conversation_history(self, user_id: int) -> int:
        """Delete conversation history strictly for the designated user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_conversations WHERE user_id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount

    # --------------------------------------------------------------------------
    # Helper Deserialization
    # --------------------------------------------------------------------------

    @staticmethod
    def _row_to_news_fact_sheet(row: sqlite3.Row) -> NewsFactSheet:
        tool_data = json.loads(row["tool_metadata_json"]) if row["tool_metadata_json"] else None
        keys = row.keys() if hasattr(row, "keys") else []
        return NewsFactSheet(
            event_id=row["event_id"],
            headline=row["headline"],
            primary_source_name=row["primary_source_name"],
            primary_source_url=row["primary_source_url"],
            supporting_sources=json.loads(row["supporting_sources_json"]),
            reliability=ReliabilityState(row["reliability"]),
            is_official_source=bool(row["is_official_source"]),
            source_reported_facts=[
                SourceFact(**f) for f in json.loads(row["source_reported_facts_json"])
            ],
            ai_summary=row["ai_summary"],
            ai_interpretation=row["ai_interpretation"],
            conflicts=[ConflictInfo(**c) for c in json.loads(row["conflicts_json"])],
            tool_metadata=ToolMetadata(**tool_data) if tool_data else None,
            category=row["category"],
            published_at=row["published_at"],
            updated_at=row["updated_at"] if "updated_at" in keys else None,
            retrieved_at=row["retrieved_at"] if "retrieved_at" in keys else None,
            event_date=row["event_date"] if "event_date" in keys else None,
        )

