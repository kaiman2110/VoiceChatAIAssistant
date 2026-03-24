"""テスト共通フィクスチャ."""

import pytest

from config import Settings


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """テスト用の設定。ダミーAPIキーを注入."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    return Settings()
