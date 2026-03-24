"""音声認識（faster-whisper による STT）."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class WhisperSTT:
    """faster-whisper を利用した音声認識クライアント."""

    def __init__(
        self,
        model_size: str = "large-v3",
        language: str = "ja",
    ) -> None:
        self._model_size: str = model_size
        self._language: str = language
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._compute_type: str = "float16" if self._device == "cuda" else "int8"
        self._model: WhisperModel | None = None

    def load_model(self) -> None:
        """Whisper モデルをロードする."""
        logger.info(
            "Whisper モデルをロード中 (model=%s, device=%s, compute=%s)",
            self._model_size,
            self._device,
            self._compute_type,
        )
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        logger.info("Whisper モデルのロード完了")

    def is_available(self) -> bool:
        """Whisper モデルが利用可能か確認."""
        try:
            if self._model is None:
                self.load_model()
            return self._model is not None
        except Exception:
            return False

    @property
    def device(self) -> str:
        """使用中のデバイス (cuda/cpu)."""
        return self._device

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """音声データをテキストに変換する.

        Args:
            audio: 音声データ (float32, -1.0〜1.0)
            sample_rate: サンプリングレート (Hz)

        Returns:
            認識されたテキスト
        """
        if self._model is None:
            raise RuntimeError("Whisperモデルが未ロード。load_model() を先に呼んでください")

        if len(audio) == 0:
            return ""

        segments, _info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=5,
            vad_filter=False,
        )

        text: str = "".join(segment.text for segment in segments)
        logger.info("音声認識結果: %s", text)
        return text
