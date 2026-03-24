"""Wake Word 検知（openWakeWord）."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """openWakeWord による Wake Word 検知.

    80ms フレーム（1280サンプル @16kHz）単位で予測し、
    スコアが閾値を超えた場合に検知とみなす。
    """

    def __init__(
        self,
        wakeword_model: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        self._wakeword_model: str | None = wakeword_model
        self._threshold: float = threshold
        self._model: Any = None

    def load_model(self) -> None:
        """openWakeWord モデルをロード."""
        from openwakeword.model import Model

        if self._wakeword_model:
            self._model = Model(wakeword_models=[self._wakeword_model])
            logger.info(
                "Wake Word カスタムモデルをロード: %s", self._wakeword_model,
            )
        else:
            self._model = Model()
            logger.info("Wake Word プリトレインモデルをロード")

    def is_available(self) -> bool:
        """モデルが利用可能か確認."""
        try:
            if self._model is None:
                self.load_model()
            return self._model is not None
        except Exception:
            logger.warning("Wake Word モデルのロードに失敗")
            return False

    def detect(self, audio_frame: np.ndarray) -> bool:
        """音声フレームを受け取り Wake Word が検知されたか判定.

        Args:
            audio_frame: 16kHz, int16 の音声データ（80ms = 1280サンプル推奨）

        Returns:
            いずれかのモデルがスコア閾値を超えた場合 True
        """
        if self._model is None:
            raise RuntimeError("モデル未ロード。先に load_model() を呼んでください")

        predictions: dict[str, float] = self._model.predict(audio_frame)

        for model_name, score in predictions.items():
            if score >= self._threshold:
                logger.info(
                    "Wake Word 検知: %s (score=%.3f)", model_name, score,
                )
                return True
        return False

    def reset(self) -> None:
        """内部バッファをリセット."""
        if self._model is not None:
            self._model.reset()
