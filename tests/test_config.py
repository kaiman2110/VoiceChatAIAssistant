"""config.py のユニットテスト."""

import pytest

from config import Settings


class TestSettings:
    """Settings クラスのテスト."""

    @pytest.mark.unit
    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """デフォルト値が正しく設定される."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("VOICEVOX_HOST", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
        monkeypatch.delenv("WAKEWORD_MODEL", raising=False)

        s = Settings(_env_file=None)

        assert s.gemini_api_key == ""
        assert s.voicevox_host == "http://localhost:50021"
        assert s.ollama_host == "http://localhost:11434"
        assert s.whisper_model == "large-v3"
        assert s.wakeword_model is None
        assert s.max_chat_history == 10
        assert s.default_speaker_id == 3

    @pytest.mark.unit
    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """環境変数でデフォルト値をオーバーライドできる."""
        monkeypatch.setenv("GEMINI_API_KEY", "my-secret-key")
        monkeypatch.setenv("VOICEVOX_HOST", "http://remote:50021")
        monkeypatch.setenv("WHISPER_MODEL", "medium")
        monkeypatch.setenv("MAX_CHAT_HISTORY", "20")
        monkeypatch.setenv("DEFAULT_SPEAKER_ID", "2")

        s = Settings()

        assert s.gemini_api_key == "my-secret-key"
        assert s.voicevox_host == "http://remote:50021"
        assert s.whisper_model == "medium"
        assert s.max_chat_history == 20
        assert s.default_speaker_id == 2

    @pytest.mark.unit
    def test_wakeword_model_optional(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """wakeword_model は None を許容する."""
        monkeypatch.delenv("WAKEWORD_MODEL", raising=False)
        s = Settings()
        assert s.wakeword_model is None

    @pytest.mark.unit
    def test_wakeword_model_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """wakeword_model を環境変数で設定できる."""
        monkeypatch.setenv("WAKEWORD_MODEL", "models/wakeword/custom.onnx")
        s = Settings()
        assert s.wakeword_model == "models/wakeword/custom.onnx"

    @pytest.mark.unit
    def test_settings_fixture(self, settings: Settings) -> None:
        """conftest の settings フィクスチャが機能する."""
        assert settings.gemini_api_key == "test-api-key"
