"""アプリケーション設定管理."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR: Path = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """環境変数から読み込むアプリケーション設定."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Gemini API
    gemini_api_key: str = Field(default="", description="Google AI Studio APIキー")

    # VOICEVOX
    voicevox_host: str = Field(
        default="http://localhost:50021",
        description="VOICEVOX Engineのホスト",
    )

    # Ollama
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollamaのホスト",
    )

    # Whisper
    whisper_model: str = Field(
        default="large-v3",
        description="Whisperモデルサイズ",
    )

    # Wake Word
    wakeword_model: str | None = Field(
        default=None,
        description="カスタムWake Wordモデルのパス",
    )

    # VAD (Voice Activity Detection)
    vad_threshold: float = Field(
        default=0.5,
        description="VAD音声判定の閾値 (0.0-1.0)",
    )
    vad_silence_duration_ms: int = Field(
        default=1000,
        description="発話終了と判定する無音継続時間 (ミリ秒)",
    )

    # Audio
    audio_sample_rate: int = Field(
        default=16000,
        description="音声サンプリングレート (Hz)",
    )

    # アプリケーション設定
    max_chat_history: int = Field(
        default=10,
        description="保持する会話履歴の最大往復数",
    )
    default_speaker_id: int = Field(
        default=3,
        description="VOICEVOXデフォルトキャラ (3=ずんだもん)",
    )
