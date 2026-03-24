"""core/stt.py のユニットテスト."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.stt import WhisperSTT


# ---------------------------------------------------------------------------
# WhisperSTT テスト
# ---------------------------------------------------------------------------


class TestWhisperSTT:
    """WhisperSTT クラスのテスト."""

    @pytest.mark.unit
    def test_init_defaults(self) -> None:
        """デフォルトパラメータで初期化できる."""
        with patch("core.stt.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            stt = WhisperSTT()
            assert stt._model_size == "large-v3"
            assert stt._language == "ja"
            assert stt.device == "cpu"
            assert stt._compute_type == "int8"

    @pytest.mark.unit
    def test_init_custom_params(self) -> None:
        """カスタムパラメータで初期化できる."""
        with patch("core.stt.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            stt = WhisperSTT(model_size="medium", language="en")
            assert stt._model_size == "medium"
            assert stt._language == "en"

    @pytest.mark.unit
    def test_init_gpu_device(self) -> None:
        """CUDA 利用可能時は GPU を選択する."""
        with patch("core.stt.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            stt = WhisperSTT()
            assert stt.device == "cuda"
            assert stt._compute_type == "float16"

    @pytest.mark.unit
    @patch("core.stt.WhisperModel")
    def test_load_model(self, mock_model_cls: MagicMock) -> None:
        """モデルが正しくロードされる."""
        with patch("core.stt.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            stt = WhisperSTT(model_size="base")
            stt.load_model()

            mock_model_cls.assert_called_once_with(
                "base", device="cpu", compute_type="int8",
            )
            assert stt._model is not None

    @pytest.mark.unit
    @patch("core.stt.WhisperModel")
    def test_is_available_success(self, mock_model_cls: MagicMock) -> None:
        """モデルが正常にロードできれば True."""
        with patch("core.stt.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_model_cls.return_value = MagicMock()
            stt = WhisperSTT()
            assert stt.is_available() is True

    @pytest.mark.unit
    @patch("core.stt.WhisperModel")
    def test_is_available_failure(self, mock_model_cls: MagicMock) -> None:
        """モデルロード失敗時は False."""
        with patch("core.stt.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_model_cls.side_effect = RuntimeError("model load failed")
            stt = WhisperSTT()
            assert stt.is_available() is False

    @pytest.mark.unit
    def test_transcribe_success(self) -> None:
        """正常に文字起こしできる."""
        with patch("core.stt.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            stt = WhisperSTT()

            # モデルをモック
            mock_model = MagicMock()
            stt._model = mock_model

            # segments のモック（イテラブル）
            seg1 = MagicMock()
            seg1.text = "こんにちは"
            seg2 = MagicMock()
            seg2.text = "世界"
            mock_info = MagicMock()
            mock_model.transcribe.return_value = (iter([seg1, seg2]), mock_info)

            audio = np.random.randn(16000).astype(np.float32)
            result = stt.transcribe(audio)

            assert result == "こんにちは世界"
            mock_model.transcribe.assert_called_once_with(
                audio,
                language="ja",
                beam_size=5,
                vad_filter=False,
            )

    @pytest.mark.unit
    def test_transcribe_empty_audio(self) -> None:
        """空の音声データは空文字列を返す."""
        with patch("core.stt.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            stt = WhisperSTT()
            stt._model = MagicMock()

            audio = np.array([], dtype=np.float32)
            result = stt.transcribe(audio)

            assert result == ""

    @pytest.mark.unit
    def test_transcribe_no_model_raises(self) -> None:
        """モデル未ロードで transcribe を呼ぶと RuntimeError."""
        with patch("core.stt.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            stt = WhisperSTT()

            audio = np.zeros(16000, dtype=np.float32)
            with pytest.raises(RuntimeError, match="Whisperモデルが未ロード"):
                stt.transcribe(audio)

    @pytest.mark.unit
    def test_transcribe_single_segment(self) -> None:
        """単一セグメントの文字起こし."""
        with patch("core.stt.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            stt = WhisperSTT()
            mock_model = MagicMock()
            stt._model = mock_model

            seg = MagicMock()
            seg.text = "テスト音声です"
            mock_info = MagicMock()
            mock_model.transcribe.return_value = (iter([seg]), mock_info)

            audio = np.zeros(16000, dtype=np.float32)
            result = stt.transcribe(audio)

            assert result == "テスト音声です"

    @pytest.mark.unit
    def test_transcribe_no_segments(self) -> None:
        """セグメントが空の場合は空文字列."""
        with patch("core.stt.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            stt = WhisperSTT()
            mock_model = MagicMock()
            stt._model = mock_model

            mock_info = MagicMock()
            mock_model.transcribe.return_value = (iter([]), mock_info)

            audio = np.zeros(16000, dtype=np.float32)
            result = stt.transcribe(audio)

            assert result == ""
