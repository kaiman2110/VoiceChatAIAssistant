"""音声合成（VOICEVOX Engine連携）."""

from __future__ import annotations

import io
import logging
import queue
import threading
import wave
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import requests
import sounddevice as sd

logger = logging.getLogger(__name__)


class VoicevoxTTS:
    """VOICEVOX Engine を利用した音声合成クライアント."""

    def __init__(self, host: str = "http://localhost:50021") -> None:
        self._host: str = host

    def is_available(self) -> bool:
        """VOICEVOX Engine が稼働中か確認."""
        try:
            resp = requests.get(f"{self._host}/version", timeout=2)
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def synthesize(self, text: str, speaker_id: int = 3) -> bytes:
        """テキストから WAV バイナリを生成.

        Args:
            text: 合成するテキスト
            speaker_id: VOICEVOXキャラクターID (デフォルト: 3=ずんだもん)

        Returns:
            WAV 形式のバイナリデータ
        """
        # 1. audio_query でクエリ生成
        query_resp = requests.post(
            f"{self._host}/audio_query",
            params={"text": text, "speaker": speaker_id},
            timeout=10,
        )
        query_resp.raise_for_status()
        query = query_resp.json()

        # 2. synthesis で音声合成
        synth_resp = requests.post(
            f"{self._host}/synthesis",
            params={"speaker": speaker_id},
            json=query,
            timeout=30,
        )
        synth_resp.raise_for_status()
        return synth_resp.content

    def play_audio(self, wav_data: bytes) -> None:
        """WAV バイナリを再生.

        Args:
            wav_data: WAV 形式のバイナリデータ
        """
        with wave.open(io.BytesIO(wav_data), "rb") as wf:
            sample_rate: int = wf.getframerate()
            n_channels: int = wf.getnchannels()
            frames: bytes = wf.readframes(wf.getnframes())

        audio_array: np.ndarray = np.frombuffer(frames, dtype=np.int16)
        if n_channels > 1:
            audio_array = audio_array.reshape(-1, n_channels)

        sd.play(audio_array, samplerate=sample_rate)
        sd.wait()

    def speak(self, text: str, speaker_id: int = 3) -> None:
        """テキストを音声合成して再生.

        Args:
            text: 読み上げるテキスト
            speaker_id: VOICEVOXキャラクターID
        """
        wav_data: bytes = self.synthesize(text, speaker_id)
        self.play_audio(wav_data)

    def speak_streaming(
        self,
        sentences: Iterator[str],
        speaker_id: int = 3,
    ) -> None:
        """文単位で並列合成し、順序を保って再生する.

        Args:
            sentences: 文のイテレータ（generate_stream() の出力など）
            speaker_id: VOICEVOXキャラクターID
        """
        playback_queue: queue.Queue[Future[bytes] | None] = queue.Queue()

        def _play_worker() -> None:
            """再生ワーカー: キューから順番に取り出して再生."""
            while True:
                future = playback_queue.get()
                if future is None:
                    break
                try:
                    wav_data: bytes = future.result()
                    self.play_audio(wav_data)
                except Exception as e:
                    logger.warning("TTS再生に失敗: %s", e)

        with ThreadPoolExecutor(max_workers=2) as executor:
            player = threading.Thread(target=_play_worker, daemon=True)
            player.start()

            for sentence in sentences:
                if sentence.strip():
                    future: Future[bytes] = executor.submit(
                        self.synthesize, sentence, speaker_id,
                    )
                    playback_queue.put(future)

            playback_queue.put(None)  # 終了シグナル
            player.join()

    def get_speakers(self) -> list[dict]:
        """利用可能なキャラクター一覧を取得.

        Returns:
            キャラクター情報のリスト
        """
        resp = requests.get(f"{self._host}/speakers", timeout=5)
        resp.raise_for_status()
        return resp.json()
