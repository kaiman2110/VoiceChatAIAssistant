"""core/llm.py のユニットテスト."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from config import Settings
from core.llm import ChatHistory, GeminiClient, LLMManager, OllamaClient


class TestChatHistory:
    """ChatHistory クラスのテスト."""

    @pytest.mark.unit
    def test_add_user_message(self) -> None:
        """ユーザーメッセージを追加できる."""
        history = ChatHistory()
        history.add_user("こんにちは")

        assert len(history.messages) == 1
        assert history.messages[0]["role"] == "user"
        assert history.messages[0]["parts"] == [{"text": "こんにちは"}]

    @pytest.mark.unit
    def test_add_assistant_message(self) -> None:
        """アシスタントメッセージを追加できる."""
        history = ChatHistory()
        history.add_assistant("こんにちはなのだ")

        assert len(history.messages) == 1
        assert history.messages[0]["role"] == "model"
        assert history.messages[0]["parts"] == [{"text": "こんにちはなのだ"}]

    @pytest.mark.unit
    def test_conversation_flow(self) -> None:
        """ユーザーとアシスタントの会話が正しく記録される."""
        history = ChatHistory()
        history.add_user("やぁ")
        history.add_assistant("やぁなのだ")
        history.add_user("元気？")
        history.add_assistant("元気なのだ！")

        assert len(history.messages) == 4
        assert history.messages[0]["role"] == "user"
        assert history.messages[1]["role"] == "model"
        assert history.messages[2]["role"] == "user"
        assert history.messages[3]["role"] == "model"

    @pytest.mark.unit
    def test_trim_on_exceed(self) -> None:
        """max_turns を超えた場合、古い履歴が切り捨てられる."""
        history = ChatHistory(max_turns=2)

        # 3往復追加（max_turnsは2）
        for i in range(3):
            history.add_user(f"ユーザー{i}")
            history.add_assistant(f"アシスタント{i}")

        # 2往復 = 4メッセージのみ残る
        assert len(history.messages) == 4
        # 最も古い往復（0番目）は切り捨てられている
        assert history.messages[0]["parts"] == [{"text": "ユーザー1"}]
        assert history.messages[1]["parts"] == [{"text": "アシスタント1"}]

    @pytest.mark.unit
    def test_clear(self) -> None:
        """履歴をクリアできる."""
        history = ChatHistory()
        history.add_user("テスト")
        history.add_assistant("テストなのだ")
        history.clear()

        assert len(history.messages) == 0

    @pytest.mark.unit
    def test_messages_returns_copy(self) -> None:
        """messages プロパティはコピーを返す."""
        history = ChatHistory()
        history.add_user("テスト")

        msgs = history.messages
        msgs.clear()

        # 元の履歴は変わらない
        assert len(history.messages) == 1


class TestGeminiClient:
    """GeminiClient クラスのテスト."""

    @pytest.mark.unit
    @patch("core.llm.genai.Client")
    def test_generate(
        self, mock_client_cls: MagicMock, settings: Settings
    ) -> None:
        """generate で応答テキストを取得できる."""
        # モックの設定
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "こんにちはなのだ！"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiClient(settings)
        result = client.generate("こんにちは")

        assert result == "こんにちはなのだ！"
        mock_client.models.generate_content.assert_called_once()

        # 履歴に追加されている
        assert len(client.history.messages) == 2
        assert client.history.messages[0]["role"] == "user"
        assert client.history.messages[1]["role"] == "model"

    @pytest.mark.unit
    @patch("core.llm.genai.Client")
    def test_generate_stream(
        self, mock_client_cls: MagicMock, settings: Settings
    ) -> None:
        """generate_stream で文単位のストリーミング応答を取得できる."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # ストリーミングレスポンスをシミュレート
        chunk1 = MagicMock()
        chunk1.text = "これは最初の文"
        chunk2 = MagicMock()
        chunk2.text = "。次の文なの"
        chunk3 = MagicMock()
        chunk3.text = "だ。最後"
        chunk4 = MagicMock()
        chunk4.text = ""

        mock_client.models.generate_content_stream.return_value = [
            chunk1,
            chunk2,
            chunk3,
            chunk4,
        ]

        client = GeminiClient(settings)
        sentences = list(client.generate_stream("テスト"))

        assert sentences == ["これは最初の文。", "次の文なのだ。", "最後"]

        # 履歴にフルテキストが追加されている
        assert len(client.history.messages) == 2
        full_text = client.history.messages[1]["parts"][0]["text"]
        assert full_text == "これは最初の文。次の文なのだ。最後"

    @pytest.mark.unit
    @patch("core.llm.genai.Client")
    def test_set_system_prompt(
        self, mock_client_cls: MagicMock, settings: Settings
    ) -> None:
        """システムプロンプトを変更できる."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "応答"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiClient(settings)
        client.set_system_prompt("カスタムプロンプト")
        client.generate("テスト")

        call_kwargs = mock_client.models.generate_content.call_args
        assert call_kwargs.kwargs["config"]["system_instruction"] == "カスタムプロンプト"

    @pytest.mark.unit
    @patch("core.llm.genai.Client")
    def test_load_prompt_file(
        self, mock_client_cls: MagicMock, settings: Settings
    ) -> None:
        """prompts/ からプロンプトファイルを読み込める."""
        mock_client_cls.return_value = MagicMock()

        client = GeminiClient(settings)
        client.load_prompt_file("casual.txt")

        # casual.txt の内容が読み込まれている（空でない）
        assert "ずんだもん" in client._system_prompt

    @pytest.mark.unit
    @patch("core.llm.genai.Client")
    def test_model_name(
        self, mock_client_cls: MagicMock, settings: Settings
    ) -> None:
        """モデル名を取得できる."""
        mock_client_cls.return_value = MagicMock()
        client = GeminiClient(settings)
        assert client.model_name == "gemini-2.5-flash"

    @pytest.mark.unit
    @patch("core.llm.genai.Client")
    def test_history_maintained_across_calls(
        self, mock_client_cls: MagicMock, settings: Settings
    ) -> None:
        """複数回の generate 呼び出しで履歴が維持される."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "応答"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiClient(settings)
        client.generate("1つ目")
        client.generate("2つ目")

        assert len(client.history.messages) == 4

        # 2回目の呼び出しで履歴が含まれている
        second_call = mock_client.models.generate_content.call_args_list[1]
        contents = second_call.kwargs["contents"]
        assert len(contents) == 3  # user1, model1, user2


class TestOllamaClient:
    """OllamaClient クラスのテスト."""

    @pytest.mark.unit
    @patch("core.llm.requests.get")
    def test_is_available_success(
        self, mock_get: MagicMock, settings: Settings
    ) -> None:
        """Ollama サーバー稼働中は True を返す."""
        mock_get.return_value = MagicMock(status_code=200)
        client = OllamaClient(settings)
        assert client.is_available() is True

    @pytest.mark.unit
    @patch("core.llm.requests.get")
    def test_is_available_connection_error(
        self, mock_get: MagicMock, settings: Settings
    ) -> None:
        """接続エラー時は False を返す."""
        mock_get.side_effect = requests.ConnectionError()
        client = OllamaClient(settings)
        assert client.is_available() is False

    @pytest.mark.unit
    @patch("core.llm.requests.get")
    def test_is_available_timeout(
        self, mock_get: MagicMock, settings: Settings
    ) -> None:
        """タイムアウト時は False を返す."""
        mock_get.side_effect = requests.Timeout()
        client = OllamaClient(settings)
        assert client.is_available() is False

    @pytest.mark.unit
    @patch("core.llm.requests.post")
    def test_generate(
        self, mock_post: MagicMock, settings: Settings
    ) -> None:
        """Ollama REST API 経由で応答を生成できる."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "こんにちはなのだ！"},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = OllamaClient(settings)
        result = client.generate("こんにちは")

        assert result == "こんにちはなのだ！"
        mock_post.assert_called_once()

        # 履歴に追加されている
        assert len(client.history.messages) == 2
        assert client.history.messages[0]["role"] == "user"
        assert client.history.messages[1]["role"] == "model"

    @pytest.mark.unit
    @patch("core.llm.requests.post")
    def test_generate_stream(
        self, mock_post: MagicMock, settings: Settings
    ) -> None:
        """ストリーミング応答を文単位で取得できる."""
        lines = [
            '{"message":{"content":"これは最初の文"}}',
            '{"message":{"content":"。次の文なの"}}',
            '{"message":{"content":"だ。最後"}}',
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = lines
        mock_post.return_value = mock_resp

        client = OllamaClient(settings)
        sentences = list(client.generate_stream("テスト"))

        assert sentences == ["これは最初の文。", "次の文なのだ。", "最後"]

        # 履歴にフルテキストが追加されている
        assert len(client.history.messages) == 2
        full_text = client.history.messages[1]["parts"][0]["text"]
        assert full_text == "これは最初の文。次の文なのだ。最後"

    @pytest.mark.unit
    @patch("core.llm.requests.post")
    def test_model_name(
        self, mock_post: MagicMock, settings: Settings
    ) -> None:
        """モデル名を取得できる."""
        client = OllamaClient(settings)
        assert client.model_name == "ollama/gemma3"

    @pytest.mark.unit
    @patch("core.llm.requests.post")
    def test_set_system_prompt(
        self, mock_post: MagicMock, settings: Settings
    ) -> None:
        """システムプロンプトを変更できる."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "応答"},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = OllamaClient(settings)
        client.set_system_prompt("カスタムプロンプト")
        client.generate("テスト")

        call_args = mock_post.call_args
        messages = call_args.kwargs["json"]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "カスタムプロンプト"


class TestLLMManager:
    """LLMManager クラスのテスト."""

    @pytest.mark.unit
    @patch("core.llm.genai.Client")
    def test_uses_gemini_when_api_key_set(
        self, mock_client_cls: MagicMock, settings: Settings
    ) -> None:
        """APIキー設定時は Gemini を使用する."""
        mock_client_cls.return_value = MagicMock()
        manager = LLMManager(settings)
        assert manager.model_name == "gemini-2.5-flash"

    @pytest.mark.unit
    def test_uses_ollama_when_no_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """APIキー未設定時は Ollama を使用する."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        s = Settings(_env_file=None)
        manager = LLMManager(s)
        assert manager.model_name == "ollama/gemma3"

    @pytest.mark.unit
    @patch("core.llm.genai.Client")
    def test_generate_delegates_to_active(
        self, mock_client_cls: MagicMock, settings: Settings
    ) -> None:
        """generate は アクティブクライアントに委譲する."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "応答なのだ"
        mock_client.models.generate_content.return_value = mock_response

        manager = LLMManager(settings)
        result = manager.generate("テスト")

        assert result == "応答なのだ"

    @pytest.mark.unit
    @patch("core.llm.requests.post")
    @patch("core.llm.genai.Client")
    def test_fallback_on_gemini_error(
        self,
        mock_gemini_cls: MagicMock,
        mock_post: MagicMock,
        settings: Settings,
    ) -> None:
        """Gemini エラー時に Ollama へ自動フォールバックする."""
        # Gemini: エラー発生
        mock_gemini = MagicMock()
        mock_gemini_cls.return_value = mock_gemini
        mock_gemini.models.generate_content.side_effect = Exception(
            "429 Too Many Requests"
        )

        # Ollama: 成功
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "Ollamaからの応答なのだ"},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        manager = LLMManager(settings)
        result = manager.generate("テスト")

        assert result == "Ollamaからの応答なのだ"
        assert manager.model_name == "ollama/gemma3"

    @pytest.mark.unit
    @patch("core.llm.requests.post")
    @patch("core.llm.genai.Client")
    def test_fallback_stream_on_gemini_error(
        self,
        mock_gemini_cls: MagicMock,
        mock_post: MagicMock,
        settings: Settings,
    ) -> None:
        """Gemini ストリーミングエラー時に Ollama へフォールバック."""
        mock_gemini = MagicMock()
        mock_gemini_cls.return_value = mock_gemini
        mock_gemini.models.generate_content_stream.side_effect = Exception(
            "429 Too Many Requests"
        )

        # Ollama ストリーミング
        lines = ['{"message":{"content":"フォールバック応答。"}}']
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = lines
        mock_post.return_value = mock_resp

        manager = LLMManager(settings)
        sentences = list(manager.generate_stream("テスト"))

        assert sentences == ["フォールバック応答。"]

    @pytest.mark.unit
    def test_ollama_error_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ollama がアクティブ時のエラーは再送出される."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        s = Settings(_env_file=None)
        manager = LLMManager(s)

        with patch("core.llm.requests.post") as mock_post:
            mock_post.side_effect = requests.ConnectionError("接続失敗")
            with pytest.raises(requests.ConnectionError):
                manager.generate("テスト")

    @pytest.mark.unit
    @patch("core.llm.genai.Client")
    def test_history_returns_active_history(
        self, mock_client_cls: MagicMock, settings: Settings
    ) -> None:
        """history は アクティブクライアントの履歴を返す."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "応答"
        mock_client.models.generate_content.return_value = mock_response

        manager = LLMManager(settings)
        manager.generate("テスト")

        assert len(manager.history.messages) == 2

    @pytest.mark.unit
    @patch("core.llm.genai.Client")
    def test_set_system_prompt_applies_to_active(
        self, mock_client_cls: MagicMock, settings: Settings
    ) -> None:
        """set_system_prompt はアクティブクライアントに適用される."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "応答"
        mock_client.models.generate_content.return_value = mock_response

        manager = LLMManager(settings)
        manager.set_system_prompt("カスタム")
        manager.generate("テスト")

        call_kwargs = mock_client.models.generate_content.call_args
        assert call_kwargs.kwargs["config"]["system_instruction"] == "カスタム"
