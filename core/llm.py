"""LLMクライアント（Gemini API / Ollama連携）."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests
from google import genai
from google.genai.types import GenerateContentResponse

from config import Settings, BASE_DIR

logger = logging.getLogger(__name__)


def _load_prompt(filename: str) -> str:
    """prompts/ からプロンプトファイルを読み込む."""
    prompt_path: Path = BASE_DIR / "prompts" / filename
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


class ChatHistory:
    """会話履歴を管理するクラス.

    直近 max_turns 往復を保持し、超過分は要約圧縮する。
    要約コールバックが未設定の場合は単純に切り捨てる。
    """

    SUMMARY_PROMPT: str = (
        "以下の会話を3〜5文で簡潔に要約してください。"
        "重要なトピックと結論のみを残してください:\n\n"
    )

    def __init__(self, max_turns: int = 20) -> None:
        self._max_turns: int = max_turns
        self._messages: list[dict[str, str]] = []
        self._summary: str = ""
        self._summarizer: Any = None

    @property
    def messages(self) -> list[dict[str, str]]:
        """現在の会話履歴を返す（要約があれば先頭に含む）."""
        result: list[dict[str, str]] = []
        if self._summary:
            result.append({
                "role": "user",
                "parts": [{"text": f"[これまでの会話の要約]: {self._summary}"}],
            })
            result.append({
                "role": "model",
                "parts": [{"text": "了解なのだ。要約の内容を踏まえて会話を続けるのだ。"}],
            })
        result.extend(self._messages)
        return result

    @property
    def summary(self) -> str:
        """現在の要約を返す."""
        return self._summary

    def set_summarizer(self, callback: Any) -> None:
        """要約コールバックを設定.

        Args:
            callback: テキストを受け取り要約テキストを返す callable。
        """
        self._summarizer = callback

    def add_user(self, message: str) -> None:
        """ユーザーメッセージを追加."""
        self._messages.append({"role": "user", "parts": [{"text": message}]})
        self._trim()

    def add_assistant(self, message: str) -> None:
        """アシスタントメッセージを追加."""
        self._messages.append({"role": "model", "parts": [{"text": message}]})
        self._trim()

    def clear(self) -> None:
        """履歴をクリア."""
        self._messages.clear()
        self._summary = ""

    def _trim(self) -> None:
        """最大往復数を超えた古い履歴を圧縮する."""
        max_messages = self._max_turns * 2
        if len(self._messages) <= max_messages:
            return

        # 超過分を取得
        overflow = self._messages[:-max_messages]
        self._messages = self._messages[-max_messages:]

        # 要約コールバックがあれば要約を生成
        if self._summarizer is not None:
            overflow_text = self._format_for_summary(overflow)
            try:
                new_summary = self._summarizer(
                    self.SUMMARY_PROMPT + overflow_text,
                )
                if self._summary:
                    self._summary = f"{self._summary}\n{new_summary}"
                else:
                    self._summary = new_summary
                logger.info("会話履歴を要約圧縮しました")
            except Exception as e:
                logger.warning("要約生成に失敗: %s", e)

    @staticmethod
    def _format_for_summary(messages: list[dict[str, str]]) -> str:
        """メッセージリストを要約用テキストに変換."""
        lines: list[str] = []
        for msg in messages:
            role = "ユーザー" if msg["role"] == "user" else "AI"
            text = msg["parts"][0]["text"]
            lines.append(f"{role}: {text}")
        return "\n".join(lines)


class GeminiClient:
    """Gemini API クライアント."""

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self._client: genai.Client = genai.Client(api_key=settings.gemini_api_key)
        self._model: str = "gemini-2.5-flash"
        self._history: ChatHistory = ChatHistory(
            max_turns=settings.max_chat_history,
        )
        self._system_prompt: str = _load_prompt("casual.txt")

    @property
    def history(self) -> ChatHistory:
        """会話履歴を返す."""
        return self._history

    @property
    def model_name(self) -> str:
        """現在のモデル名を返す."""
        return self._model

    def set_system_prompt(self, prompt: str) -> None:
        """システムプロンプトを設定."""
        self._system_prompt = prompt

    def load_prompt_file(self, filename: str) -> None:
        """prompts/ ディレクトリからプロンプトファイルを読み込んで設定."""
        self._system_prompt = _load_prompt(filename)

    def generate(self, user_message: str) -> str:
        """ユーザーメッセージに対する応答を生成."""
        self._history.add_user(user_message)

        response: GenerateContentResponse = self._client.models.generate_content(
            model=self._model,
            contents=self._history.messages,
            config={
                "temperature": 0.7,
                "system_instruction": self._system_prompt,
            },
        )

        assistant_text: str = response.text or ""
        self._history.add_assistant(assistant_text)
        return assistant_text

    def generate_stream(self, user_message: str) -> Iterator[str]:
        """ユーザーメッセージに対する応答をストリーミングで生成.

        句点（。）で区切って文単位で yield する。
        """
        self._history.add_user(user_message)

        response_stream: Any = self._client.models.generate_content_stream(
            model=self._model,
            contents=self._history.messages,
            config={
                "temperature": 0.7,
                "system_instruction": self._system_prompt,
            },
        )

        full_text: str = ""
        buffer: str = ""

        for chunk in response_stream:
            chunk_text: str = chunk.text or ""
            buffer += chunk_text
            full_text += chunk_text

            # 句点で分割して文単位で yield
            while "。" in buffer:
                sentence, buffer = buffer.split("。", 1)
                yield sentence + "。"

        # 残りのバッファを yield
        if buffer.strip():
            yield buffer

        self._history.add_assistant(full_text)


class OllamaClient:
    """Ollama REST API クライアント."""

    def __init__(self, settings: Settings) -> None:
        self._host: str = settings.ollama_host
        self._model: str = settings.ollama_model
        self._history: ChatHistory = ChatHistory(
            max_turns=settings.max_chat_history,
        )
        self._system_prompt: str = _load_prompt("casual.txt")

    @property
    def history(self) -> ChatHistory:
        """会話履歴を返す."""
        return self._history

    @property
    def model_name(self) -> str:
        """現在のモデル名を返す."""
        return f"ollama/{self._model}"

    def is_available(self) -> bool:
        """Ollama サーバーが稼働中か確認."""
        try:
            resp = requests.get(f"{self._host}/api/tags", timeout=2)
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def set_system_prompt(self, prompt: str) -> None:
        """システムプロンプトを設定."""
        self._system_prompt = prompt

    def load_prompt_file(self, filename: str) -> None:
        """prompts/ ディレクトリからプロンプトファイルを読み込んで設定."""
        self._system_prompt = _load_prompt(filename)

    def _build_messages(self) -> list[dict[str, str]]:
        """ChatHistory を Ollama 形式のメッセージリストに変換."""
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        for msg in self._history.messages:
            role = "assistant" if msg["role"] == "model" else msg["role"]
            text = msg["parts"][0]["text"]
            messages.append({"role": role, "content": text})
        return messages

    def generate(self, user_message: str) -> str:
        """ユーザーメッセージに対する応答を生成."""
        self._history.add_user(user_message)

        resp = requests.post(
            f"{self._host}/api/chat",
            json={
                "model": self._model,
                "messages": self._build_messages(),
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()

        assistant_text: str = resp.json()["message"]["content"]
        self._history.add_assistant(assistant_text)
        return assistant_text

    def generate_stream(self, user_message: str) -> Iterator[str]:
        """ユーザーメッセージに対する応答をストリーミングで生成.

        句点（。）で区切って文単位で yield する。
        """
        self._history.add_user(user_message)

        resp = requests.post(
            f"{self._host}/api/chat",
            json={
                "model": self._model,
                "messages": self._build_messages(),
                "stream": True,
            },
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()

        full_text: str = ""
        buffer: str = ""

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            data = json.loads(line)
            chunk_text: str = data.get("message", {}).get("content", "")
            buffer += chunk_text
            full_text += chunk_text

            while "。" in buffer:
                sentence, buffer = buffer.split("。", 1)
                yield sentence + "。"

        if buffer.strip():
            yield buffer

        self._history.add_assistant(full_text)


class LLMManager:
    """LLMクライアント管理（Gemini優先、Ollama自動フォールバック、手動切替対応）."""

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self._gemini: GeminiClient | None = None
        self._ollama: OllamaClient | None = None
        self._active: GeminiClient | OllamaClient

        # 両プロバイダを初期化（利用可能なもの）
        if settings.gemini_api_key:
            self._gemini = GeminiClient(settings)
        else:
            logger.warning("Gemini APIキー未設定")

        self._ollama = OllamaClient(settings)

        # Gemini があれば優先、なければ Ollama
        if self._gemini:
            self._active = self._gemini
        else:
            self._active = self._ollama

        # 要約コールバックを設定
        self._active.history.set_summarizer(self._summarize)

    @property
    def history(self) -> ChatHistory:
        """現在アクティブなクライアントの会話履歴を返す."""
        return self._active.history

    @property
    def model_name(self) -> str:
        """現在使用中のモデル名を返す."""
        return self._active.model_name

    @property
    def active_provider(self) -> str:
        """現在アクティブなプロバイダ名を返す."""
        if isinstance(self._active, GeminiClient):
            return "gemini"
        return "ollama"

    @property
    def available_providers(self) -> list[str]:
        """利用可能なプロバイダ名一覧を返す."""
        providers: list[str] = []
        if self._gemini is not None:
            providers.append("gemini")
        if self._ollama is not None:
            providers.append("ollama")
        return providers

    def switch_provider(self, provider_name: str) -> None:
        """プロバイダを手動切替する。会話履歴は新プロバイダへ引き継ぐ.

        Args:
            provider_name: "gemini" または "ollama"

        Raises:
            ValueError: 不明なプロバイダ名または利用不可の場合
        """
        provider_name = provider_name.lower()

        if provider_name == self.active_provider:
            logger.info("既に %s を使用中", provider_name)
            return

        target = self._resolve_provider(provider_name)
        source = self._active

        # 履歴を引き継ぐ
        self._transfer_history(source, target)

        self._active = target
        self._active.history.set_summarizer(self._summarize)
        logger.info("プロバイダを %s に切替", provider_name)

    def _resolve_provider(
        self, provider_name: str
    ) -> GeminiClient | OllamaClient:
        """プロバイダ名からクライアントインスタンスを解決する."""
        if provider_name == "gemini":
            if self._gemini is None:
                raise ValueError("Gemini は利用不可（APIキー未設定）")
            return self._gemini
        if provider_name == "ollama":
            if self._ollama is None:
                raise ValueError("Ollama は利用不可")
            return self._ollama
        raise ValueError(f"不明なプロバイダ: {provider_name}")

    @staticmethod
    def _transfer_history(
        source: GeminiClient | OllamaClient,
        target: GeminiClient | OllamaClient,
    ) -> None:
        """履歴とシステムプロンプトを source から target へ引き継ぐ."""
        target.history.clear()
        # 要約を引き継ぐ
        target.history._summary = source.history._summary
        # メッセージを引き継ぐ
        for msg in source.history._messages:
            text = msg["parts"][0]["text"]
            if msg["role"] == "user":
                target.history.add_user(text)
            else:
                target.history.add_assistant(text)
        # システムプロンプトを引き継ぐ
        target.set_system_prompt(source._system_prompt)

    def set_system_prompt(self, prompt: str) -> None:
        """全クライアントのシステムプロンプトを設定."""
        if self._gemini:
            self._gemini.set_system_prompt(prompt)
        if self._ollama:
            self._ollama.set_system_prompt(prompt)

    def load_prompt_file(self, filename: str) -> None:
        """全クライアントのプロンプトファイルを読み込んで設定."""
        if self._gemini:
            self._gemini.load_prompt_file(filename)
        if self._ollama:
            self._ollama.load_prompt_file(filename)

    def generate(self, user_message: str) -> str:
        """応答を生成（エラー時はフォールバック）."""
        try:
            return self._active.generate(user_message)
        except Exception as e:
            fallback = self._try_fallback(e)
            if fallback is None:
                raise
            return fallback.generate(user_message)

    def generate_stream(self, user_message: str) -> Iterator[str]:
        """ストリーミング応答を生成（エラー時はフォールバック）."""
        try:
            yield from self._active.generate_stream(user_message)
        except Exception as e:
            fallback = self._try_fallback(e)
            if fallback is None:
                raise
            yield from fallback.generate_stream(user_message)

    def _try_fallback(
        self, error: Exception
    ) -> GeminiClient | OllamaClient | None:
        """エラー時にフォールバック先を返す。切替不可なら None."""
        if not isinstance(self._active, GeminiClient):
            return None

        if self._ollama is None:
            return None

        logger.warning("Geminiエラー: %s → Ollamaへフォールバック", error)

        self._transfer_history(self._active, self._ollama)
        self._active = self._ollama
        self._ollama.history.set_summarizer(self._summarize)
        return self._ollama

    def _summarize(self, text: str) -> str:
        """要約テキストを生成する（内部用）."""
        return self._active.generate(text)
