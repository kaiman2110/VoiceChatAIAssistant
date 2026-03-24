"""core/audio.py のユニットテスト."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from config import Settings
from core.audio import AudioRecorder, VADDetector


# ---------------------------------------------------------------------------
# VADDetector テスト
# ---------------------------------------------------------------------------


class TestVADDetector:
    """VADDetector クラスのテスト."""

    @pytest.mark.unit
    def test_init_defaults(self) -> None:
        """デフォルトパラメータで初期化できる."""
        vad = VADDetector()
        assert vad._threshold == 0.5
        assert vad._silence_duration_ms == 1000
        assert vad._sample_rate == 16000
        assert vad.is_speaking is False

    @pytest.mark.unit
    def test_init_custom_params(self) -> None:
        """カスタムパラメータで初期化できる."""
        vad = VADDetector(threshold=0.7, silence_duration_ms=500, sample_rate=8000)
        assert vad._threshold == 0.7
        assert vad._silence_duration_ms == 500
        assert vad._sample_rate == 8000

    @pytest.mark.unit
    @patch("core.audio._silero_vad")
    def test_is_available_success(self, mock_silero: MagicMock) -> None:
        """silero-vad モデルが正常にロードできれば True."""
        mock_silero.load_silero_vad.return_value = MagicMock()
        vad = VADDetector()
        assert vad.is_available() is True

    @pytest.mark.unit
    @patch("core.audio._silero_vad")
    def test_is_available_failure(self, mock_silero: MagicMock) -> None:
        """silero-vad ロード失敗時は False."""
        mock_silero.load_silero_vad.side_effect = RuntimeError("load failed")
        vad = VADDetector()
        assert vad.is_available() is False

    @pytest.mark.unit
    def test_process_chunk_speech_start(self) -> None:
        """音声確率が閾値以上で発話開始を検出."""
        vad = VADDetector(threshold=0.5)
        mock_model = MagicMock()
        vad._model = mock_model

        # 発話確率 0.8 を返す
        with patch("core.audio.torch") as mock_torch:
            mock_tensor = MagicMock()
            mock_torch.from_numpy.return_value.float.return_value = mock_tensor
            mock_tensor.dim.return_value = 1
            mock_model.return_value.item.return_value = 0.8

            chunk = np.zeros(512, dtype=np.float32)
            result = vad.process_chunk(chunk)

            assert result["speech_start"] is True
            assert result["speech_end"] is False
            assert vad.is_speaking is True

    @pytest.mark.unit
    def test_process_chunk_speech_end(self) -> None:
        """無音が継続して発話終了を検出."""
        vad = VADDetector(threshold=0.5, silence_duration_ms=1000, sample_rate=16000)
        mock_model = MagicMock()
        vad._model = mock_model

        # 無音判定に必要なサンプル数: 16000 * 1000ms / 1000 = 16000
        with patch("core.audio.torch") as mock_torch:
            mock_tensor = MagicMock()
            mock_torch.from_numpy.return_value.float.return_value = mock_tensor
            mock_tensor.dim.return_value = 1

            # まず発話開始させる
            mock_model.return_value.item.return_value = 0.8
            chunk = np.zeros(512, dtype=np.float32)
            vad.process_chunk(chunk)
            assert vad.is_speaking is True

            # 無音を送り続ける（確率 0.1）
            mock_model.return_value.item.return_value = 0.1
            result = {"speech_end": False}
            # 16000 / 512 = 31.25 → 32チャンク必要
            for _ in range(32):
                result = vad.process_chunk(chunk)
                if result["speech_end"]:
                    break

            assert result["speech_end"] is True
            assert vad.is_speaking is False

    @pytest.mark.unit
    def test_process_chunk_no_model_raises(self) -> None:
        """モデル未ロードで process_chunk を呼ぶと RuntimeError."""
        vad = VADDetector()
        chunk = np.zeros(512, dtype=np.float32)
        with pytest.raises(RuntimeError, match="VADモデルが未ロード"):
            vad.process_chunk(chunk)

    @pytest.mark.unit
    def test_reset(self) -> None:
        """reset で状態がクリアされる."""
        vad = VADDetector()
        mock_model = MagicMock()
        vad._model = mock_model
        vad._is_speaking = True
        vad._silence_samples = 5000

        vad.reset()

        assert vad.is_speaking is False
        assert vad._silence_samples == 0
        mock_model.reset_states.assert_called_once()

    @pytest.mark.unit
    def test_continued_speech_no_duplicate_start(self) -> None:
        """連続した発話で speech_start は1回だけ."""
        vad = VADDetector(threshold=0.5)
        mock_model = MagicMock()
        vad._model = mock_model

        with patch("core.audio.torch") as mock_torch:
            mock_tensor = MagicMock()
            mock_torch.from_numpy.return_value.float.return_value = mock_tensor
            mock_tensor.dim.return_value = 1
            mock_model.return_value.item.return_value = 0.8

            chunk = np.zeros(512, dtype=np.float32)

            result1 = vad.process_chunk(chunk)
            assert result1["speech_start"] is True

            result2 = vad.process_chunk(chunk)
            assert result2["speech_start"] is False
            assert vad.is_speaking is True


# ---------------------------------------------------------------------------
# AudioRecorder テスト
# ---------------------------------------------------------------------------


class TestAudioRecorder:
    """AudioRecorder クラスのテスト."""

    @pytest.mark.unit
    def test_init(self) -> None:
        """初期化できる."""
        vad = VADDetector()
        recorder = AudioRecorder(vad, sample_rate=16000, chunk_size=512)
        assert recorder.is_recording is False
        assert recorder._sample_rate == 16000
        assert recorder._chunk_size == 512

    @pytest.mark.unit
    @patch("core.audio.sd.InputStream")
    def test_start_recording(self, mock_input_stream: MagicMock) -> None:
        """録音開始で InputStream が作成・開始される."""
        vad = VADDetector()
        vad._model = MagicMock()  # モデルロード済みを想定
        recorder = AudioRecorder(vad)
        callback = MagicMock()

        recorder.start(on_speech_end=callback)

        assert recorder.is_recording is True
        mock_input_stream.assert_called_once()
        mock_input_stream.return_value.start.assert_called_once()

    @pytest.mark.unit
    @patch("core.audio.sd.InputStream")
    def test_stop_recording(self, mock_input_stream: MagicMock) -> None:
        """録音停止で InputStream が停止・クローズされる."""
        vad = VADDetector()
        vad._model = MagicMock()
        recorder = AudioRecorder(vad)

        recorder.start(on_speech_end=MagicMock())
        recorder.stop()

        assert recorder.is_recording is False
        mock_input_stream.return_value.stop.assert_called_once()
        mock_input_stream.return_value.close.assert_called_once()

    @pytest.mark.unit
    @patch("core.audio.sd.InputStream")
    def test_start_twice_ignored(self, mock_input_stream: MagicMock) -> None:
        """二重開始は無視される."""
        vad = VADDetector()
        vad._model = MagicMock()
        recorder = AudioRecorder(vad)

        recorder.start(on_speech_end=MagicMock())
        recorder.start(on_speech_end=MagicMock())

        # InputStream は1回だけ作成
        assert mock_input_stream.call_count == 1

    @pytest.mark.unit
    def test_on_audio_chunk_callback(self) -> None:
        """各チャンクで on_audio_chunk コールバックが呼ばれる."""
        vad = VADDetector(threshold=0.5)
        mock_model = MagicMock()
        vad._model = mock_model
        recorder = AudioRecorder(vad)
        chunk_callback = MagicMock()
        recorder._on_speech_end = MagicMock()
        recorder._on_audio_chunk = chunk_callback
        recorder._recording = True

        with patch("core.audio.torch") as mock_torch:
            mock_tensor = MagicMock()
            mock_torch.from_numpy.return_value.float.return_value = mock_tensor
            mock_tensor.dim.return_value = 1
            mock_model.return_value.item.return_value = 0.1

            indata = np.ones((512, 1), dtype=np.float32) * 0.1
            recorder._audio_callback(indata, 512, None, MagicMock(spec=False))

            chunk_callback.assert_called_once()
            chunk_data = chunk_callback.call_args[0][0]
            assert isinstance(chunk_data, np.ndarray)
            assert len(chunk_data) == 512

    @pytest.mark.unit
    def test_audio_callback_collects_speech(self) -> None:
        """発話中の音声データがバッファに蓄積される."""
        vad = VADDetector(threshold=0.5)
        mock_model = MagicMock()
        vad._model = mock_model
        recorder = AudioRecorder(vad)
        recorder._on_speech_end = MagicMock()
        recorder._recording = True

        with patch("core.audio.torch") as mock_torch:
            mock_tensor = MagicMock()
            mock_torch.from_numpy.return_value.float.return_value = mock_tensor
            mock_tensor.dim.return_value = 1
            mock_model.return_value.item.return_value = 0.8

            # 発話チャンクを送信
            indata = np.ones((512, 1), dtype=np.float32) * 0.5
            recorder._audio_callback(indata, 512, None, MagicMock(spec=False))

            assert len(recorder._audio_buffer) == 1

    @pytest.mark.unit
    def test_audio_callback_triggers_on_speech_end(self) -> None:
        """発話終了時にコールバックが呼ばれる."""
        vad = VADDetector(
            threshold=0.5, silence_duration_ms=100, sample_rate=16000,
        )
        mock_model = MagicMock()
        vad._model = mock_model
        recorder = AudioRecorder(vad, chunk_size=512)
        callback = MagicMock()
        recorder._on_speech_end = callback
        recorder._recording = True

        with patch("core.audio.torch") as mock_torch:
            mock_tensor = MagicMock()
            mock_torch.from_numpy.return_value.float.return_value = mock_tensor
            mock_tensor.dim.return_value = 1

            # 発話開始
            mock_model.return_value.item.return_value = 0.8
            indata = np.ones((512, 1), dtype=np.float32) * 0.5
            recorder._audio_callback(indata, 512, None, MagicMock(spec=False))
            assert vad.is_speaking is True

            # 無音送信（100ms = 1600サンプル → 512*4=2048 > 1600）
            mock_model.return_value.item.return_value = 0.1
            for _ in range(4):
                recorder._audio_callback(indata, 512, None, MagicMock(spec=False))

            callback.assert_called_once()
            speech_data = callback.call_args[0][0]
            assert isinstance(speech_data, np.ndarray)
            assert len(speech_data) > 0


# ---------------------------------------------------------------------------
# config.py VAD 設定テスト
# ---------------------------------------------------------------------------


class TestVADSettings:
    """config.py の VAD 設定テスト."""

    @pytest.mark.unit
    def test_default_vad_settings(self) -> None:
        """VAD 設定のデフォルト値."""
        s = Settings(_env_file=None)
        assert s.vad_threshold == 0.5
        assert s.vad_silence_duration_ms == 1000
        assert s.audio_sample_rate == 16000

    @pytest.mark.unit
    def test_custom_vad_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """環境変数から VAD 設定を上書きできる."""
        monkeypatch.setenv("VAD_THRESHOLD", "0.7")
        monkeypatch.setenv("VAD_SILENCE_DURATION_MS", "500")
        monkeypatch.setenv("AUDIO_SAMPLE_RATE", "8000")
        s = Settings(_env_file=None)
        assert s.vad_threshold == 0.7
        assert s.vad_silence_duration_ms == 500
        assert s.audio_sample_rate == 8000
