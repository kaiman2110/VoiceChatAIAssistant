"""core/wakeword.py のユニットテスト."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.wakeword import WakeWordDetector


class TestWakeWordDetector:
    """WakeWordDetector クラスのテスト."""

    @pytest.mark.unit
    def test_init_defaults(self) -> None:
        """デフォルトパラメータで初期化できる."""
        detector = WakeWordDetector()
        assert detector._wakeword_model is None
        assert detector._threshold == 0.5
        assert detector._model is None

    @pytest.mark.unit
    def test_init_custom_params(self) -> None:
        """カスタムパラメータで初期化できる."""
        detector = WakeWordDetector(
            wakeword_model="custom.onnx",
            threshold=0.7,
        )
        assert detector._wakeword_model == "custom.onnx"
        assert detector._threshold == 0.7

    @pytest.mark.unit
    @patch("core.wakeword.WakeWordDetector.load_model")
    def test_is_available_success(self, mock_load: MagicMock) -> None:
        """モデルロード成功時は True を返す."""
        detector = WakeWordDetector()

        def _set_model() -> None:
            detector._model = MagicMock()

        mock_load.side_effect = _set_model

        assert detector.is_available() is True
        mock_load.assert_called_once()

    @pytest.mark.unit
    @patch("core.wakeword.WakeWordDetector.load_model")
    def test_is_available_failure(self, mock_load: MagicMock) -> None:
        """モデルロード失敗時は False を返す."""
        mock_load.side_effect = Exception("ロード失敗")

        detector = WakeWordDetector()
        assert detector.is_available() is False

    @pytest.mark.unit
    def test_detect_above_threshold(self) -> None:
        """スコアが閾値以上で True を返す."""
        detector = WakeWordDetector(threshold=0.5)
        detector._model = MagicMock()
        detector._model.predict.return_value = {"hey_mycroft": 0.85}

        audio = np.zeros(1280, dtype=np.int16)
        assert detector.detect(audio) is True

    @pytest.mark.unit
    def test_detect_below_threshold(self) -> None:
        """スコアが閾値未満で False を返す."""
        detector = WakeWordDetector(threshold=0.5)
        detector._model = MagicMock()
        detector._model.predict.return_value = {"hey_mycroft": 0.2}

        audio = np.zeros(1280, dtype=np.int16)
        assert detector.detect(audio) is False

    @pytest.mark.unit
    def test_detect_multiple_models(self) -> None:
        """複数モデルのうち1つでも閾値超えで True."""
        detector = WakeWordDetector(threshold=0.5)
        detector._model = MagicMock()
        detector._model.predict.return_value = {
            "alexa": 0.1,
            "hey_mycroft": 0.8,
        }

        audio = np.zeros(1280, dtype=np.int16)
        assert detector.detect(audio) is True

    @pytest.mark.unit
    def test_detect_no_model_raises(self) -> None:
        """モデル未ロードで RuntimeError."""
        detector = WakeWordDetector()
        audio = np.zeros(1280, dtype=np.int16)

        with pytest.raises(RuntimeError, match="モデル未ロード"):
            detector.detect(audio)

    @pytest.mark.unit
    def test_reset(self) -> None:
        """reset でモデルの内部バッファがリセットされる."""
        detector = WakeWordDetector()
        detector._model = MagicMock()

        detector.reset()
        detector._model.reset.assert_called_once()

    @pytest.mark.unit
    def test_reset_no_model(self) -> None:
        """モデル未ロード時の reset は何もしない."""
        detector = WakeWordDetector()
        detector.reset()  # エラーなし

    @pytest.mark.unit
    @patch("core.wakeword.WakeWordDetector.load_model")
    def test_is_available_skips_if_loaded(self, mock_load: MagicMock) -> None:
        """既にロード済みなら再ロードしない."""
        detector = WakeWordDetector()
        detector._model = MagicMock()

        assert detector.is_available() is True
        mock_load.assert_not_called()
