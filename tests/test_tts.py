"""core/tts.py のユニットテスト."""

from __future__ import annotations

import io
import struct
import time
import wave
from unittest.mock import MagicMock, call, patch

import pytest
import requests

from core.tts import VoicevoxTTS


def _make_wav_bytes(n_samples: int = 100, sample_rate: int = 44100) -> bytes:
    """テスト用の WAV バイナリを生成."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = struct.pack(f"<{n_samples}h", *([1000] * n_samples))
        wf.writeframes(frames)
    return buf.getvalue()


class TestVoicevoxTTS:
    """VoicevoxTTS クラスのテスト."""

    @pytest.mark.unit
    def test_is_available_success(self) -> None:
        """VOICEVOX Engine 稼働中は True を返す."""
        tts = VoicevoxTTS()
        with patch("core.tts.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            assert tts.is_available() is True

    @pytest.mark.unit
    def test_is_available_connection_error(self) -> None:
        """接続エラー時は False を返す."""
        tts = VoicevoxTTS()
        with patch("core.tts.requests.get") as mock_get:
            mock_get.side_effect = requests.ConnectionError()
            assert tts.is_available() is False

    @pytest.mark.unit
    def test_is_available_timeout(self) -> None:
        """タイムアウト時は False を返す."""
        tts = VoicevoxTTS()
        with patch("core.tts.requests.get") as mock_get:
            mock_get.side_effect = requests.Timeout()
            assert tts.is_available() is False

    @pytest.mark.unit
    def test_synthesize(self) -> None:
        """audio_query → synthesis の2段 API 呼び出しを検証."""
        tts = VoicevoxTTS(host="http://test:50021")
        wav_data = _make_wav_bytes()

        with patch("core.tts.requests.post") as mock_post:
            # audio_query のレスポンス
            query_resp = MagicMock()
            query_resp.json.return_value = {"accent_phrases": []}
            query_resp.raise_for_status = MagicMock()

            # synthesis のレスポンス
            synth_resp = MagicMock()
            synth_resp.content = wav_data
            synth_resp.raise_for_status = MagicMock()

            mock_post.side_effect = [query_resp, synth_resp]

            result = tts.synthesize("テストなのだ", speaker_id=3)

            assert result == wav_data
            assert mock_post.call_count == 2

            # audio_query の呼び出し確認
            first_call = mock_post.call_args_list[0]
            assert "audio_query" in first_call.args[0]
            assert first_call.kwargs["params"]["text"] == "テストなのだ"
            assert first_call.kwargs["params"]["speaker"] == 3

            # synthesis の呼び出し確認
            second_call = mock_post.call_args_list[1]
            assert "synthesis" in second_call.args[0]
            assert second_call.kwargs["params"]["speaker"] == 3

    @pytest.mark.unit
    def test_play_audio(self) -> None:
        """sounddevice.play が正しく呼び出される."""
        tts = VoicevoxTTS()
        wav_data = _make_wav_bytes()

        with patch("core.tts.sd.play") as mock_play, patch(
            "core.tts.sd.wait"
        ) as mock_wait:
            tts.play_audio(wav_data)

            mock_play.assert_called_once()
            mock_wait.assert_called_once()

            # 再生データの検証
            call_args = mock_play.call_args
            assert call_args.kwargs["samplerate"] == 44100

    @pytest.mark.unit
    def test_speak(self) -> None:
        """speak は synthesize + play_audio を呼ぶ."""
        tts = VoicevoxTTS()
        wav_data = _make_wav_bytes()

        with (
            patch.object(tts, "synthesize", return_value=wav_data) as mock_synth,
            patch.object(tts, "play_audio") as mock_play,
        ):
            tts.speak("テスト", speaker_id=2)

            mock_synth.assert_called_once_with("テスト", 2)
            mock_play.assert_called_once_with(wav_data)

    @pytest.mark.unit
    def test_get_speakers(self) -> None:
        """スピーカー一覧を取得できる."""
        tts = VoicevoxTTS()
        speakers_data = [
            {"name": "四国めたん", "speaker_uuid": "xxx", "styles": []},
            {"name": "ずんだもん", "speaker_uuid": "yyy", "styles": []},
        ]

        with patch("core.tts.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = speakers_data
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            result = tts.get_speakers()

            assert len(result) == 2
            assert result[1]["name"] == "ずんだもん"

    @pytest.mark.unit
    def test_custom_host(self) -> None:
        """カスタムホストが正しく使用される."""
        tts = VoicevoxTTS(host="http://custom:9999")
        wav_data = _make_wav_bytes()

        with patch("core.tts.requests.post") as mock_post:
            query_resp = MagicMock()
            query_resp.json.return_value = {}
            query_resp.raise_for_status = MagicMock()
            synth_resp = MagicMock()
            synth_resp.content = wav_data
            synth_resp.raise_for_status = MagicMock()
            mock_post.side_effect = [query_resp, synth_resp]

            tts.synthesize("テスト")

            first_call = mock_post.call_args_list[0]
            assert "http://custom:9999" in first_call.args[0]


class TestSpeakStreaming:
    """speak_streaming のテスト."""

    @pytest.mark.unit
    def test_streaming_plays_in_order(self) -> None:
        """複数文が順序通りに再生される."""
        tts = VoicevoxTTS()
        wav1 = _make_wav_bytes(n_samples=50)
        wav2 = _make_wav_bytes(n_samples=100)
        wav3 = _make_wav_bytes(n_samples=150)

        play_order: list[int] = []

        with (
            patch.object(
                tts, "synthesize", side_effect=[wav1, wav2, wav3],
            ) as mock_synth,
            patch.object(
                tts,
                "play_audio",
                side_effect=lambda wav: play_order.append(len(wav)),
            ),
        ):
            sentences = iter(["一番目の文。", "二番目の文。", "三番目の文。"])
            tts.speak_streaming(sentences, speaker_id=3)

            assert mock_synth.call_count == 3
            assert play_order == [len(wav1), len(wav2), len(wav3)]

    @pytest.mark.unit
    def test_streaming_skips_empty(self) -> None:
        """空文字列やスペースのみの文はスキップされる."""
        tts = VoicevoxTTS()
        wav = _make_wav_bytes()

        with (
            patch.object(tts, "synthesize", return_value=wav) as mock_synth,
            patch.object(tts, "play_audio"),
        ):
            sentences = iter(["有効な文。", "", "  ", "もう一文。"])
            tts.speak_streaming(sentences, speaker_id=3)

            assert mock_synth.call_count == 2

    @pytest.mark.unit
    def test_streaming_synthesis_error_continues(self) -> None:
        """合成失敗しても次の文の再生は継続する."""
        tts = VoicevoxTTS()
        wav_ok = _make_wav_bytes()

        with (
            patch.object(
                tts,
                "synthesize",
                side_effect=[wav_ok, RuntimeError("合成失敗"), wav_ok],
            ),
            patch.object(tts, "play_audio") as mock_play,
        ):
            sentences = iter(["成功。", "失敗。", "成功。"])
            tts.speak_streaming(sentences, speaker_id=3)

            # 失敗分を除いて2回再生
            assert mock_play.call_count == 2

    @pytest.mark.unit
    def test_streaming_single_sentence(self) -> None:
        """単一文でも正常に動作する."""
        tts = VoicevoxTTS()
        wav = _make_wav_bytes()

        with (
            patch.object(tts, "synthesize", return_value=wav) as mock_synth,
            patch.object(tts, "play_audio") as mock_play,
        ):
            tts.speak_streaming(iter(["一文だけ。"]), speaker_id=2)

            mock_synth.assert_called_once_with("一文だけ。", 2)
            mock_play.assert_called_once_with(wav)

    @pytest.mark.unit
    def test_streaming_empty_iterator(self) -> None:
        """空のイテレータでもエラーにならない."""
        tts = VoicevoxTTS()

        with (
            patch.object(tts, "synthesize") as mock_synth,
            patch.object(tts, "play_audio") as mock_play,
        ):
            tts.speak_streaming(iter([]), speaker_id=3)

            mock_synth.assert_not_called()
            mock_play.assert_not_called()
