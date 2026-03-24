"""STT + VAD の手動動作確認スクリプト.

使い方:
    python scripts/test_stt.py

マイクに話しかけると、VADで発話区間を検出し、
faster-whisper で文字起こし結果を表示します。
Ctrl+C で終了。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import Settings
from core.audio import AudioRecorder, VADDetector
from core.stt import WhisperSTT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """STT + VAD 動作確認メイン."""
    settings = Settings()

    # STT モデルロード
    logger.info("Whisper モデルをロード中 (%s)...", settings.whisper_model)
    stt = WhisperSTT(model_size=settings.whisper_model)
    stt.load_model()

    # VAD モデルロード
    logger.info("VAD モデルをロード中...")
    vad = VADDetector(
        threshold=settings.vad_threshold,
        silence_duration_ms=settings.vad_silence_duration_ms,
        sample_rate=settings.audio_sample_rate,
    )
    vad.load_model()

    def on_speech_end(speech_data: np.ndarray) -> None:
        """発話終了時のコールバック: STT で文字起こし."""
        duration_sec: float = len(speech_data) / settings.audio_sample_rate
        logger.info("発話を受信: %.2f秒", duration_sec)
        text: str = stt.transcribe(speech_data, sample_rate=settings.audio_sample_rate)
        logger.info("認識結果: 「%s」", text)

    recorder = AudioRecorder(
        vad,
        sample_rate=settings.audio_sample_rate,
        chunk_size=512,
    )

    logger.info("マイク録音を開始します。話しかけてください (Ctrl+C で終了)")
    recorder.start(on_speech_end=on_speech_end)

    try:
        import time

        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("終了します...")
    finally:
        recorder.stop()


if __name__ == "__main__":
    main()
