"""Voice Chat AI Assistant — メインエントリーポイント（Gradio UI）."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any

import gradio as gr
import numpy as np

from config import Settings
from core.audio import AudioRecorder, VADDetector
from core.llm import GeminiClient
from core.stt import WhisperSTT
from core.tts import VoicevoxTTS

logger = logging.getLogger(__name__)


def create_app() -> gr.Blocks:
    """Gradio アプリケーションを構築して返す."""
    settings = Settings()
    llm = GeminiClient(settings)
    tts = VoicevoxTTS(host=settings.voicevox_host)
    stt = WhisperSTT(model_size=settings.whisper_model)
    vad = VADDetector(
        threshold=settings.vad_threshold,
        silence_duration_ms=settings.vad_silence_duration_ms,
        sample_rate=settings.audio_sample_rate,
    )
    recorder = AudioRecorder(
        vad,
        sample_rate=settings.audio_sample_rate,
        chunk_size=512,
    )

    # 外部サービスの稼働状態を確認
    voicevox_available: bool = tts.is_available()

    # 音声モデルを起動時にプリロード（UIブロック回避）
    logger.info("VAD モデルをロード中...")
    vad.load_model()
    logger.info("Whisper モデルをロード中 (%s)...", settings.whisper_model)
    stt.load_model()
    logger.info("モデルのロード完了 — 起動準備OK")

    # 音声パイプライン用キュー（AudioRecorderコールバック → メインスレッド）
    speech_queue: queue.Queue[np.ndarray] = queue.Queue()

    def on_speech_end(audio_data: np.ndarray) -> None:
        """AudioRecorder コールバック: 発話終了時にキューに投入."""
        speech_queue.put(audio_data)

    def start_mic() -> str:
        """マイク録音を開始する."""
        if not vad.is_available():
            return "VADモデルのロードに失敗"
        if not stt.is_available():
            return "Whisperモデルのロードに失敗"

        recorder.start(on_speech_end=on_speech_end)
        return "🔴 聴取中..."

    def stop_mic() -> str:
        """マイク録音を停止する."""
        recorder.stop()
        return "待機中"

    def process_speech(
        history: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str, str]:
        """キューに溜まった音声データを処理する.

        Gradio の every= で定期的に呼び出される。
        """
        try:
            audio_data: np.ndarray = speech_queue.get_nowait()
        except queue.Empty:
            return history, gr.skip(), gr.skip()  # type: ignore[return-value]

        # STT（音声認識）
        logger.info("音声認識中...")
        text: str = stt.transcribe(audio_data, sample_rate=settings.audio_sample_rate)
        if not text.strip():
            return history, "認識結果なし", f"LLM: {llm.model_name}"

        # ユーザーメッセージを履歴に追加
        history.append({"role": "user", "content": text})

        # LLM応答生成
        try:
            response: str = llm.generate(text)
        except Exception as e:
            response = f"エラーが発生したのだ: {e}"

        history.append({"role": "assistant", "content": response})

        # TTS再生（バックグラウンド）
        tts_status: str = "応答完了"
        if voicevox_available:
            tts_status = "音声合成中..."

            def _speak() -> None:
                try:
                    tts.speak(response, speaker_id=settings.default_speaker_id)
                except Exception:
                    pass

            threading.Thread(target=_speak, daemon=True).start()

        return history, tts_status, f"LLM: {llm.model_name}"

    def chat_response(
        message: str,
        history: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str, str]:
        """テキストチャットメッセージに応答する."""
        if not message.strip():
            return history, "", f"LLM: {llm.model_name}"

        history.append({"role": "user", "content": message})

        try:
            response: str = llm.generate(message)
        except Exception as e:
            response = f"エラーが発生したのだ: {e}"

        history.append({"role": "assistant", "content": response})

        # TTS再生（バックグラウンド）
        tts_status: str = ""
        if voicevox_available:
            tts_status = "音声合成中..."

            def _speak() -> None:
                try:
                    tts.speak(response, speaker_id=settings.default_speaker_id)
                except Exception:
                    pass

            threading.Thread(target=_speak, daemon=True).start()
        else:
            tts_status = "VOICEVOX未起動（テキストのみ）"

        return history, tts_status, f"LLM: {llm.model_name}"

    def clear_chat() -> tuple[list[Any], str, str]:
        """チャット履歴をクリア."""
        llm.history.clear()
        return [], "", f"LLM: {llm.model_name}"

    # --- UI レイアウト ---
    with gr.Blocks(title="Voice Chat AI Assistant") as app:
        gr.Markdown("# Voice Chat AI Assistant")
        gr.Markdown("作業中の雑談・壁打ち・設計相談を音声で行えるAIアシスタント")

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="チャット",
                    height=500,
                    type="messages",
                )

                # マイク制御
                with gr.Row():
                    mic_start_btn = gr.Button(
                        "🎙️ マイク ON", scale=2, variant="primary",
                    )
                    mic_stop_btn = gr.Button("⏹️ マイク OFF", scale=2)

                # テキスト入力
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="メッセージ",
                        placeholder="テキスト入力もできます...",
                        scale=4,
                        show_label=False,
                    )
                    send_btn = gr.Button("送信", scale=1, variant="primary")
                clear_btn = gr.Button("クリア")

            with gr.Column(scale=1):
                mic_status = gr.Textbox(
                    label="マイク",
                    value="待機中",
                    interactive=False,
                )
                llm_info = gr.Textbox(
                    label="LLM",
                    value=f"LLM: {llm.model_name}",
                    interactive=False,
                )
                status_display = gr.Textbox(
                    label="ステータス",
                    value="VOICEVOX: OK" if voicevox_available else "VOICEVOX: 未起動",
                    interactive=False,
                )

        # --- イベントハンドラー ---

        # マイク制御
        mic_start_btn.click(
            fn=start_mic,
            outputs=[mic_status],
        )
        mic_stop_btn.click(
            fn=stop_mic,
            outputs=[mic_status],
        )

        # 音声パイプライン: 定期ポーリングでキューを処理
        speech_timer = gr.Timer(value=0.5)
        speech_timer.tick(
            fn=process_speech,
            inputs=[chatbot],
            outputs=[chatbot, status_display, llm_info],
        )

        # テキスト入力
        send_btn.click(
            fn=chat_response,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, status_display, llm_info],
        ).then(
            fn=lambda: "",
            outputs=[msg_input],
        )

        msg_input.submit(
            fn=chat_response,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, status_display, llm_info],
        ).then(
            fn=lambda: "",
            outputs=[msg_input],
        )

        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, status_display, llm_info],
        )

    return app


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = create_app()
    app.launch()
