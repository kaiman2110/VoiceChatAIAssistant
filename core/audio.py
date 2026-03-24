"""音声入力と発話区間検出 (VAD + マイク入力)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import sounddevice as sd
import torch

import silero_vad as _silero_vad

logger = logging.getLogger(__name__)


class VADDetector:
    """silero-vad による発話区間検出.

    音声チャンクを逐次入力し、発話開始/終了を検出する。
    """

    def __init__(
        self,
        threshold: float = 0.5,
        silence_duration_ms: int = 1000,
        sample_rate: int = 16000,
    ) -> None:
        self._threshold: float = threshold
        self._silence_duration_ms: int = silence_duration_ms
        self._sample_rate: int = sample_rate
        self._model: Any = None
        self._is_speaking: bool = False
        self._silence_samples: int = 0
        self._silence_limit: int = int(sample_rate * silence_duration_ms / 1000)

    def load_model(self) -> None:
        """silero-vad モデルをロードする."""
        self._model = _silero_vad.load_silero_vad()

    def is_available(self) -> bool:
        """VAD モデルが利用可能か確認."""
        try:
            if self._model is None:
                self.load_model()
            return self._model is not None
        except Exception:
            return False

    def reset(self) -> None:
        """検出状態をリセットする."""
        self._is_speaking = False
        self._silence_samples = 0
        if self._model is not None:
            self._model.reset_states()

    @property
    def is_speaking(self) -> bool:
        """現在発話中かどうか."""
        return self._is_speaking

    def process_chunk(self, audio_chunk: np.ndarray) -> dict[str, bool]:
        """音声チャンクを処理し、発話状態の変化を返す.

        Args:
            audio_chunk: 16kHz モノラルの音声データ (float32, -1.0〜1.0)

        Returns:
            {"speech_start": bool, "speech_end": bool} 状態変化を示す辞書
        """
        if self._model is None:
            raise RuntimeError("VADモデルが未ロード。load_model() を先に呼んでください")

        # numpy -> torch tensor
        tensor: torch.Tensor = torch.from_numpy(audio_chunk).float()
        if tensor.dim() > 1:
            tensor = tensor.squeeze()

        # silero-vad で音声確率を推定
        speech_prob: float = self._model(tensor, self._sample_rate).item()

        result: dict[str, bool] = {"speech_start": False, "speech_end": False}

        if speech_prob >= self._threshold:
            self._silence_samples = 0
            if not self._is_speaking:
                self._is_speaking = True
                result["speech_start"] = True
                logger.info("発話開始を検出 (確率: %.2f)", speech_prob)
        else:
            if self._is_speaking:
                self._silence_samples += len(audio_chunk)
                if self._silence_samples >= self._silence_limit:
                    self._is_speaking = False
                    result["speech_end"] = True
                    self._silence_samples = 0
                    logger.info("発話終了を検出 (無音 %dms)", self._silence_duration_ms)

        return result


class AudioRecorder:
    """sounddevice を利用したリアルタイムマイク録音.

    VADDetector と組み合わせ、発話終了時にコールバックで音声データを返す。
    """

    def __init__(
        self,
        vad: VADDetector,
        sample_rate: int = 16000,
        chunk_size: int = 512,
    ) -> None:
        self._vad: VADDetector = vad
        self._sample_rate: int = sample_rate
        self._chunk_size: int = chunk_size
        self._stream: sd.InputStream | None = None
        self._recording: bool = False
        self._audio_buffer: list[np.ndarray] = []
        self._on_speech_end: Callable[[np.ndarray], None] | None = None
        self._on_audio_chunk: Callable[[np.ndarray], None] | None = None
        self._lock: threading.Lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        """録音中かどうか."""
        return self._recording

    def start(
        self,
        on_speech_end: Callable[[np.ndarray], None],
        on_audio_chunk: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        """マイク録音を開始する.

        Args:
            on_speech_end: 発話終了時に呼ばれるコールバック。
                          引数は発話区間の音声データ (numpy array, float32)。
            on_audio_chunk: 各オーディオチャンク受信時のコールバック（オプション）。
                           Wake Word 検知などに使用。
        """
        if self._recording:
            return

        self._on_speech_end = on_speech_end
        self._on_audio_chunk = on_audio_chunk
        self._audio_buffer = []
        self._vad.reset()

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._chunk_size,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._recording = True
        logger.info("マイク録音を開始 (SR=%d, chunk=%d)", self._sample_rate, self._chunk_size)

    def stop(self) -> None:
        """マイク録音を停止する."""
        self._recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            self._audio_buffer = []
        self._vad.reset()
        logger.info("マイク録音を停止")

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice のコールバック（別スレッドで呼ばれる）."""
        if status:
            logger.warning("Audio callback status: %s", status)

        chunk: np.ndarray = indata[:, 0].copy()

        if self._on_audio_chunk is not None:
            self._on_audio_chunk(chunk)

        result: dict[str, bool] = self._vad.process_chunk(chunk)

        with self._lock:
            if self._vad.is_speaking or result["speech_end"]:
                self._audio_buffer.append(chunk)

            if result["speech_end"] and self._audio_buffer:
                speech_data: np.ndarray = np.concatenate(self._audio_buffer)
                self._audio_buffer = []
                if self._on_speech_end is not None:
                    self._on_speech_end(speech_data)
