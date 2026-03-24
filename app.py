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
from core.llm import LLMManager
from core.logger import ConversationLogger
from core.stt import WhisperSTT
from core.tts import VoicevoxTTS

logger = logging.getLogger(__name__)


def create_app() -> gr.Blocks:
    """Gradio アプリケーションを構築して返す."""
    settings = Settings()
    llm = LLMManager(settings)
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

    # 会話ログ
    conv_logger = ConversationLogger(
        llm_name=llm.model_name,
        character="ずんだもん",
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
        conv_logger.add_entry("user", text)

        # LLM応答生成（ストリーミング → 文単位で収集）
        try:
            sentences: list[str] = list(llm.generate_stream(text))
            response: str = "".join(sentences)
        except Exception as e:
            response = f"エラーが発生したのだ: {e}"
            sentences = []

        history.append({"role": "assistant", "content": response})
        conv_logger.add_entry("assistant", response)
        conv_logger.set_llm_name(llm.model_name)

        # TTS再生（バックグラウンド・文単位並列合成）
        tts_status: str = "応答完了"
        if voicevox_available and sentences:
            tts_status = "音声合成中..."

            def _speak() -> None:
                try:
                    tts.speak_streaming(
                        iter(sentences),
                        speaker_id=settings.default_speaker_id,
                    )
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
        conv_logger.add_entry("user", message)

        # LLM応答生成（ストリーミング → 文単位で収集）
        try:
            sentences: list[str] = list(llm.generate_stream(message))
            response: str = "".join(sentences)
        except Exception as e:
            response = f"エラーが発生したのだ: {e}"
            sentences = []

        history.append({"role": "assistant", "content": response})
        conv_logger.add_entry("assistant", response)
        conv_logger.set_llm_name(llm.model_name)

        # TTS再生（バックグラウンド・文単位並列合成）
        tts_status: str = ""
        if voicevox_available and sentences:
            tts_status = "音声合成中..."

            def _speak() -> None:
                try:
                    tts.speak_streaming(
                        iter(sentences),
                        speaker_id=settings.default_speaker_id,
                    )
                except Exception:
                    pass

            threading.Thread(target=_speak, daemon=True).start()
        elif not voicevox_available:
            tts_status = "VOICEVOX未起動（テキストのみ）"

        return history, tts_status, f"LLM: {llm.model_name}"

    # モード名 → プロンプトファイル名のマッピング
    mode_map: dict[str, str] = {
        "雑談": "casual",
        "コード相談": "code_review",
        "進捗報告": "progress",
    }

    def change_mode(
        selected: str,
        history: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str, str]:
        """会話モードを切り替える（履歴クリア＋プロンプト変更）."""
        mode_key = mode_map.get(selected, "casual")
        llm.load_prompt_file(f"{mode_key}.txt")
        conv_logger.set_mode(mode_key)

        # モード変更時は履歴をクリア（ログ保存後）
        conv_logger.save_and_reset()
        llm.history.clear()

        return [], f"モード: {selected}", f"LLM: {llm.model_name}"

    def clear_chat() -> tuple[list[Any], str, str]:
        """チャット履歴をクリア（ログ保存後）."""
        conv_logger.save_and_reset()
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
                mode_radio = gr.Radio(
                    choices=["雑談", "コード相談", "進捗報告"],
                    value="雑談",
                    label="会話モード",
                )
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

        # モード切替
        mode_radio.change(
            fn=change_mode,
            inputs=[mode_radio, chatbot],
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
