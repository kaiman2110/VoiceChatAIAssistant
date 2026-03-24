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


class ChatHistory:
    """会話履歴を管理するクラス.

    直近 max_turns 往復を保持し、超過分は切り捨てる。
    """

    def __init__(self, max_turns: int = 10) -> None:
        self._max_turns: int = max_turns
        self._messages: list[dict[str, str]] = []

    @property
    def messages(self) -> list[dict[str, str]]:
        """現在の会話履歴を返す."""
        return list(self._messages)

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

    def _trim(self) -> None:
        """最大往復数を超えた古い履歴を切り捨てる."""
        max_messages = self._max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]


class GeminiClient:
    """Gemini API クライアント."""

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self._client: genai.Client = genai.Client(api_key=settings.gemini_api_key)
        self._model: str = "gemini-2.5-flash"
        self._history: ChatHistory = ChatHistory(
            max_turns=settings.max_chat_history,
        )
        self._system_prompt: str = self._load_prompt("casual.txt")

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
        self._system_prompt = self._load_prompt(filename)

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

    @staticmethod
    def _load_prompt(filename: str) -> str:
        """prompts/ からプロンプトファイルを読み込む."""
        prompt_path: Path = BASE_DIR / "prompts" / filename
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return ""


class OllamaClient:
    """Ollama REST API クライアント."""

    def __init__(self, settings: Settings) -> None:
        self._host: str = settings.ollama_host
        self._model: str = settings.ollama_model
        self._history: ChatHistory = ChatHistory(
            max_turns=settings.max_chat_history,
        )
        self._system_prompt: str = self._load_prompt("casual.txt")

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
        self._system_prompt = self._load_prompt(filename)

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

    @staticmethod
    def _load_prompt(filename: str) -> str:
        """prompts/ からプロンプトファイルを読み込む."""
        prompt_path: Path = BASE_DIR / "prompts" / filename
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return ""
