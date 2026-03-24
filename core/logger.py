"""会話ログの自動保存."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from config import BASE_DIR

logger = logging.getLogger(__name__)


class ConversationLogger:
    """会話履歴をMarkdown形式で自動保存するクラス."""

    def __init__(
        self,
        log_dir: Path | None = None,
        llm_name: str = "",
        mode: str = "casual",
        character: str = "ずんだもん",
    ) -> None:
        self._log_dir: Path = log_dir or (BASE_DIR / "logs")
        self._llm_name: str = llm_name
        self._mode: str = mode
        self._character: str = character
        self._entries: list[dict[str, str]] = []
        self._session_start: datetime = datetime.now()

    @property
    def entry_count(self) -> int:
        """記録済みエントリ数を返す."""
        return len(self._entries)

    def set_llm_name(self, name: str) -> None:
        """使用中のLLM名を更新."""
        self._llm_name = name

    def set_mode(self, mode: str) -> None:
        """会話モードを更新."""
        self._mode = mode

    def set_character(self, character: str) -> None:
        """キャラクター名を更新."""
        self._character = character

    def add_entry(self, role: str, content: str) -> None:
        """会話エントリを追加.

        Args:
            role: "user" または "assistant"
            content: メッセージ内容
        """
        self._entries.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "role": role,
            "content": content,
        })

    def save(self) -> Path | None:
        """現在の会話をMarkdownファイルに保存.

        Returns:
            保存先のパス。エントリが空の場合は None。
        """
        if not self._entries:
            return None

        self._log_dir.mkdir(parents=True, exist_ok=True)

        filename = self._session_start.strftime("%Y-%m-%d_%H%M%S") + ".md"
        filepath = self._log_dir / filename

        content = self._format_markdown()

        filepath.write_text(content, encoding="utf-8")
        logger.info("会話ログを保存: %s (%d件)", filepath, len(self._entries))
        return filepath

    def save_and_reset(self) -> Path | None:
        """保存してからエントリをリセットし、新しいセッションを開始.

        Returns:
            保存先のパス。エントリが空の場合は None。
        """
        result = self.save()
        self._entries.clear()
        self._session_start = datetime.now()
        return result

    def _format_markdown(self) -> str:
        """会話履歴をMarkdown形式にフォーマット."""
        date_str = self._session_start.strftime("%Y-%m-%d %H:%M:%S")
        lines: list[str] = [
            f"# Voice Chat Log - {date_str}",
            f"## Meta: mode={self._mode}, llm={self._llm_name}, "
            f"character={self._character}",
            "",
        ]

        for entry in self._entries:
            role_label = "User" if entry["role"] == "user" else "AI"
            lines.append(
                f"[{entry['timestamp']}] **{role_label}**: {entry['content']}"
            )

        lines.append("")  # 末尾改行
        return "\n".join(lines)
