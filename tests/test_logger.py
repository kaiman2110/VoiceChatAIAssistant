"""core/logger.py のユニットテスト."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from core.logger import ConversationLogger


class TestConversationLogger:
    """ConversationLogger クラスのテスト."""

    @pytest.mark.unit
    def test_add_entry(self, tmp_path: Path) -> None:
        """会話エントリを追加できる."""
        logger = ConversationLogger(log_dir=tmp_path)
        logger.add_entry("user", "こんにちは")
        logger.add_entry("assistant", "こんにちはなのだ！")

        assert logger.entry_count == 2

    @pytest.mark.unit
    def test_save_creates_file(self, tmp_path: Path) -> None:
        """保存するとMarkdownファイルが作成される."""
        logger = ConversationLogger(log_dir=tmp_path)
        logger.add_entry("user", "テスト")
        logger.add_entry("assistant", "テストなのだ")

        result = logger.save()

        assert result is not None
        assert result.exists()
        assert result.suffix == ".md"

    @pytest.mark.unit
    def test_save_empty_returns_none(self, tmp_path: Path) -> None:
        """エントリが空の場合は None を返す."""
        logger = ConversationLogger(log_dir=tmp_path)
        result = logger.save()

        assert result is None

    @pytest.mark.unit
    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """ログディレクトリが存在しない場合は自動作成する."""
        log_dir = tmp_path / "nested" / "logs"
        logger = ConversationLogger(log_dir=log_dir)
        logger.add_entry("user", "テスト")

        result = logger.save()

        assert result is not None
        assert log_dir.exists()

    @pytest.mark.unit
    def test_filename_format(self, tmp_path: Path) -> None:
        """ファイル名が YYYY-MM-DD_HHmmss.md 形式である."""
        mock_now = datetime(2026, 3, 25, 14, 30, 45)
        with patch("core.logger.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.strftime = datetime.strftime
            logger = ConversationLogger(log_dir=tmp_path)

        logger.add_entry("user", "テスト")
        result = logger.save()

        assert result is not None
        assert result.name == "2026-03-25_143045.md"

    @pytest.mark.unit
    def test_markdown_format_header(self, tmp_path: Path) -> None:
        """Markdown にヘッダーとメタデータが含まれる."""
        logger = ConversationLogger(
            log_dir=tmp_path,
            llm_name="gemini-2.5-flash",
            mode="casual",
            character="ずんだもん",
        )
        logger.add_entry("user", "こんにちは")

        result = logger.save()
        assert result is not None
        content = result.read_text(encoding="utf-8")

        assert "# Voice Chat Log" in content
        assert "mode=casual" in content
        assert "llm=gemini-2.5-flash" in content
        assert "character=ずんだもん" in content

    @pytest.mark.unit
    def test_markdown_format_entries(self, tmp_path: Path) -> None:
        """Markdown に発言がタイムスタンプ付きで記録される."""
        logger = ConversationLogger(log_dir=tmp_path)
        logger.add_entry("user", "やぁ")
        logger.add_entry("assistant", "やぁなのだ")

        result = logger.save()
        assert result is not None
        content = result.read_text(encoding="utf-8")

        assert "**User**: やぁ" in content
        assert "**AI**: やぁなのだ" in content

    @pytest.mark.unit
    def test_save_and_reset(self, tmp_path: Path) -> None:
        """save_and_reset で保存後にエントリがクリアされる."""
        logger = ConversationLogger(log_dir=tmp_path)
        logger.add_entry("user", "テスト")

        result = logger.save_and_reset()

        assert result is not None
        assert logger.entry_count == 0

    @pytest.mark.unit
    def test_set_llm_name(self, tmp_path: Path) -> None:
        """LLM名を動的に更新できる."""
        logger = ConversationLogger(log_dir=tmp_path, llm_name="gemini")
        logger.set_llm_name("ollama/gemma3")
        logger.add_entry("user", "テスト")

        result = logger.save()
        assert result is not None
        content = result.read_text(encoding="utf-8")
        assert "llm=ollama/gemma3" in content

    @pytest.mark.unit
    def test_set_mode(self, tmp_path: Path) -> None:
        """モードを動的に更新できる."""
        logger = ConversationLogger(log_dir=tmp_path)
        logger.set_mode("code_review")
        logger.add_entry("user", "テスト")

        result = logger.save()
        assert result is not None
        content = result.read_text(encoding="utf-8")
        assert "mode=code_review" in content

    @pytest.mark.unit
    def test_multiple_saves_same_file(self, tmp_path: Path) -> None:
        """同一セッションの複数回 save は同じファイルに上書きする."""
        logger = ConversationLogger(log_dir=tmp_path)
        logger.add_entry("user", "1回目")
        result1 = logger.save()

        logger.add_entry("assistant", "応答")
        result2 = logger.save()

        assert result1 == result2
        content = result2.read_text(encoding="utf-8")  # type: ignore[union-attr]
        assert "1回目" in content
        assert "応答" in content
