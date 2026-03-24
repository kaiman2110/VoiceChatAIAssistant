"""Voice Chat AI Assistant — メインエントリーポイント（Gradio UI）."""

from __future__ import annotations

import threading
from typing import Any

import gradio as gr

from config import Settings
from core.llm import GeminiClient
from core.tts import VoicevoxTTS


def create_app() -> gr.Blocks:
    """Gradio アプリケーションを構築して返す."""
    settings = Settings()
    llm = GeminiClient(settings)
    tts = VoicevoxTTS(host=settings.voicevox_host)

    # VOICEVOX の稼働状態を確認
    voicevox_available: bool = tts.is_available()

    def chat_response(
        message: str,
        history: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str, str]:
        """チャットメッセージに応答する.

        Returns:
            (更新された履歴, ステータス, LLM情報)
        """
        if not message.strip():
            return history, "", f"LLM: {llm.model_name}"

        # 履歴にユーザーメッセージを追加
        history.append({"role": "user", "content": message})

        # LLM応答生成
        status = "応答生成中..."
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
                    pass  # TTS失敗はサイレントに無視

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
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="メッセージ",
                        placeholder="メッセージを入力...",
                        scale=4,
                        show_label=False,
                    )
                    send_btn = gr.Button("送信", scale=1, variant="primary")
                clear_btn = gr.Button("クリア")

            with gr.Column(scale=1):
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

        # イベントハンドラー
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
    app = create_app()
    app.launch()
