"""VAD + マイク入力の手動動作確認スクリプト.

使い方:
    python scripts/test_vad.py

マイクに話しかけると、発話開始/終了がコンソールに表示されます。
Ctrl+C で終了。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.audio import AudioRecorder, VADDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def on_speech_end(speech_data: np.ndarray) -> None:
    """発話終了時のコールバック."""
    duration_sec: float = len(speech_data) / 16000
    logger.info("発話を受信: %.2f秒 (%d サンプル)", duration_sec, len(speech_data))


def main() -> None:
    """VAD 動作確認メイン."""
    logger.info("VAD モデルをロード中...")
    vad = VADDetector(threshold=0.5, silence_duration_ms=1000, sample_rate=16000)
    vad.load_model()
    logger.info("VAD モデルのロード完了")

    recorder = AudioRecorder(vad, sample_rate=16000, chunk_size=512)

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
