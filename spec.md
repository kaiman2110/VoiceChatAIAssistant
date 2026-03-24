# Voice Chat AI Assistant — 設計ドキュメント

> 作業中の雑談・壁打ち・設計相談を音声で行えるAIアシスタント
> Version 1.1 | 2026-03-24

---

## 1. プロジェクト概要

### 1.1 目的

作業中に音声でAIと対話できるデスクトップアプリケーション。雑談や壁打ち、コード・設計の相談、進捗記録などを音声で行える「作業用AIバディ」を実現する。

### 1.2 主要機能

- **雑談・壁打ち相手**: 作業中の独り言やアイデアの壁打ちに応答
- **コード・設計相談**: 技術的な質問や設計の議論に対応
- **進捗記録**: 会話ログを自動保存し、作業履歴として参照可能
- **キャラクター音声**: VOICEVOXによるずんだもん等のキャラ音声で応答
- **Wake Word検知**: 呼びかけ語で起動するモードと常時リスニングモードの切替

### 1.3 ターゲット環境

- OS: Windows 10/11
- GPU: VRAM 8GB以上（NVIDIA推奨）
- ランタイム: Python 3.10+
- 外部サービス: VOICEVOX Engine（ローカル）

---

## 2. アーキテクチャ

### 2.1 全体パイプライン

```
マイク → openWakeWord(待機) ─[検知]→ VAD(silero-vad) → Whisper → LLM → VOICEVOX → スピーカー
                                                                       └→ ログ保存
```

| ステージ | コンポーネント | 役割 |
|----------|----------------|------|
| 0. Wake Word検知 | openWakeWord | 呼びかけ語を検出し会話モードを起動 |
| 1. 音声取得 | silero-vad + sounddevice | マイクから発話区間を検出 |
| 2. 音声認識 (STT) | faster-whisper (local) | 音声→テキスト変換 |
| 3. LLM推論 | Gemini Flash API / Ollama | テキスト応答生成 |
| 4. 音声合成 (TTS) | VOICEVOX Engine | テキスト→音声変換 |
| 5. 音声再生 | sounddevice / pygame | スピーカー出力 |

### 2.2 リスニングモード

| モード | 動作 | 用途 |
|--------|------|------|
| Wake Wordモード | 待機中はopenWakeWordのみ稼働。検知後に会話モードに遷移し、一定時間無言で待機に戻る | 作業に集中しつつ、必要な時だけ呼びかけたい時 |
| 常時リスニング | VADが発話を検出したら即座にWhisperに流す。Wake Word不要 | 雑談や壁打ちを継続的に行いたい時 |
| ミュート | マイク入力を完全に停止 | 会議中や電話中など |

### 2.3 プラットフォーム

- **プロトタイプ**: Python + Gradio。単一プロセスで全コンポーネントを統合し、ブラウザ上でUIを提供する
- **将来的な移行先**: FastAPI + React（UIの自由度が必要になった場合）

---

## 3. コンポーネント詳細

### 3.1 音声認識 (STT)

#### 技術選定

| 項目 | 詳細 |
|------|------|
| ライブラリ | faster-whisper |
| モデル | large-v3 推奨（VRAMに余裕があれば）、mediumでも実用的 |
| 言語 | ja (日本語指定) |
| 推論デバイス | CUDA (GPU) 優先、フォールバックとしてCPU |

#### VAD（発話区間検出）

- `silero-vad` を使用し、発話の開始・終了をリアルタイムで検出する
- 無音が一定時間（約1秒）続いた時点で発話終了と判定し、その区間の音声をWhisperに渡す
- パラメータ調整ポイント: 無音判定の閾値を調整し、キーボード打鍵音やBGMを誤検出しないようにする

### 3.2 LLM（大規模言語モデル）

#### デュアル構成

コストと利便性のバランスから、以下の2系統を切り替え可能にする。

| プロバイダー | モデル | 用途 | コスト |
|-------------|--------|------|--------|
| Google Gemini API | Gemini 2.5 Flash | メイン（日常利用） | 無料枠内 |
| Ollama (ローカル) | Qwen 2.5 7B | フォールバック | 完全無料 |

#### Gemini API 無料枠のポイント

- クレジットカード登録不要。Google AI StudioでAPIキーを発行するだけ
- 無料枠を超えた場合はHTTP 429エラーが返るだけで、勝手に課金されることはない
- 自分でCloud Billingを有効化しない限り安全
- レートリミット: 約1分あたり5〜15リクエスト、1日約1,000リクエスト
- 注意: Cloud Billingを有効化すると全使用が課金対象になり、無料枠はなくなる

#### フォールバック戦略

Gemini APIがレートリミットに達した場合、またはネットワーク接続がない場合に、自動的にOllama（ローカル）に切り替える。ユーザーには現在どちらを使用中かをUI上で表示する。

#### 会話管理

- **スライディングウィンドウ**: 直近20往復を保持し、それ以前は要約してコンテキストを圧縮
- **システムプロンプト**: モード別（雑談/コード相談/進捗報告）に切り替え可能
- **ストリーミング**: 文単位でTTSに流し、体感レイテンシを削減

### 3.3 音声合成 (TTS)

#### VOICEVOX Engine

| 項目 | 詳細 |
|------|------|
| エンドポイント | `http://localhost:50021` |
| APIフロー | `POST /audio_query` → `POST /synthesis` |
| デフォルトキャラ | ずんだもん (`speaker_id: 3`) |
| 出力形式 | WAV (44.1kHz) |

#### リアルタイム再生の工夫

- LLMの出力を句点（。）や読点（、）で分割し、文ごとにTTSリクエストを並列で送信
- 最初の文の音声が生成された時点で再生開始、裏で次の文を生成（パイプライン化）
- `speaker_id` を切り替えればキャラ変更も可能（四国めたん=2、ずんだもん=3 等）

### 3.4 UI（ユーザーインターフェース）

#### Gradioベースのプロトタイプ

GradioのChatInterfaceをベースに、以下のコンポーネントを配置する。

- **チャットエリア**: 会話履歴の表示
- **マイクボタン**: 音声入力の開始/停止（トグル式 または 常時リスニング）
- **モード切替**: 雑談 / コード相談 / 進捗報告
- **キャラクター選択**: VOICEVOXのspeaker_id切替
- **LLM表示**: 現在使用中のLLM（Gemini/Ollama）の表示
- **ステータス表示**: 音声認識中/応答生成中/音声合成中
- **リスニングモード切替**: Wake Wordモード / 常時リスニング / ミュート

### 3.5 Wake Word検知

#### 技術選定

| 項目 | 詳細 |
|------|------|
| ライブラリ | openWakeWord |
| ライセンス | Apache 2.0（完全オープンソース） |
| 推論デバイス | CPUのみ（軽量、GPU不要） |
| フレームサイズ | 80ms単位でストリーミング処理 |
| 対応プラットフォーム | Windows / Linux / macOS |

#### 選定理由

- 完全無料・オープンソースでライセンス制約なし
- TTS合成音声だけでカスタムWake Wordの学習が可能（Google Colabで1時間以内）
- CPU処理のみで軽量なため、WhisperやOllamaとのGPUリソース競合が発生しない
- 共有の特徴抽出バックボーンにより、複数Wake Wordを追加しても負荷がほぼ増えない

#### カスタムWake Wordの作成手順

1. ターゲットフレーズを決定（例: 「ねぇずんだもん」）
2. TTSで合成音声データを大量生成（数百〜数千クリップ）
3. Google Colab上で小規模モデルを学習（1時間以内）
4. 生成された `.tflite` / `.onnx` モデルをアプリに組み込み

VOICEVOXをTTSデータ生成にも流用できるため、既存の環境とのシナジーが高い。

---

## 4. データフローとログ

### 4.1 会話ログの保存

会話履歴を自動的にMarkdownファイルとして保存し、進捗記録として参照できるようにする。

| 項目 | 詳細 |
|------|------|
| フォーマット | Markdown (.md) |
| ファイル名 | `logs/YYYY-MM-DD_HHmmss.md` |
| 内容 | タイムスタンプ付きの発言履歴（ユーザー/AI） |
| メタデータ | 使用LLM、モード、キャラクター名 |

### 4.2 ログフォーマット例

```markdown
# Voice Chat Log - 2026-03-24 14:30:00
## Meta: mode=雑談, llm=gemini-flash, character=ずんだもん

[14:30:05] **User**: 今日のタスク終わらないなぁ
[14:30:08] **AI**: お疲れなのだ！どんなタスクをやってるのだ？
```

---

## 5. 依存関係とセットアップ

### 5.1 Pythonパッケージ

| パッケージ | 用途 |
|-----------|------|
| gradio | Web UIフレームワーク |
| faster-whisper | ローカル音声認識 |
| openwakeword | Wake Word検知 |
| silero-vad (torch) | 発話区間検出 |
| sounddevice | マイク入力 / 音声再生 |
| numpy | 音声データ処理 |
| google-genai | Gemini APIクライアント (GA SDK) |
| requests | VOICEVOX API / Ollama API呼び出し |
| httpx (任意) | 非同期 HTTPクライアント |

### 5.2 外部ソフトウェア

| ソフトウェア | 備考 |
|-------------|------|
| VOICEVOX Engine | https://voicevox.hiroshiba.jp/ からインストール。起動時にlocalhost:50021で待ち受け |
| Ollama | https://ollama.ai/ からインストール。`ollama pull qwen2.5:7b` でモデル取得 |

### 5.3 環境変数

| 変数名 | 説明 | 必須 |
|--------|------|------|
| GEMINI_API_KEY | Google AI Studioで発行したAPIキー | ○ |
| VOICEVOX_HOST | VOICEVOXのホスト (デフォルト: localhost:50021) | — |
| OLLAMA_HOST | Ollamaのホスト (デフォルト: localhost:11434) | — |
| WHISPER_MODEL | Whisperモデルサイズ (デフォルト: large-v3) | — |
| WAKEWORD_MODEL | カスタムWake Wordモデルのパス (.tflite/.onnx) | — |

### 5.4 セットアップ手順

```bash
# 1. リポジトリをクローン
git clone <repo-url>
cd voice-chat-ai

# 2. 仮想環境を作成
python -m venv .venv
.venv\Scripts\activate  # Windows

# 3. 依存パッケージをインストール
pip install -r requirements.txt

# 4. 環境変数を設定
copy .env.example .env
# .env を編集して GEMINI_API_KEY を設定

# 5. VOICEVOX Engineを起動（別プロセス）

# 6. アプリを起動
python app.py
```

---

## 6. 開発プラン

### 6.1 フェーズ1: 最小構成プロトタイプ

- テキスト入力 → Gemini API → テキスト応答のGradioチャット
- VOICEVOXによる音声出力の組み込み
- 基本的な会話履歴管理

### 6.2 フェーズ2: 音声入力統合

- Whisper + silero-vad による音声入力パイプライン
- TTSのパイプライン化（文単位の並列生成・再生）
- VADの閾値調整（BGM・打鍵音の誤検出防止）

### 6.3 フェーズ3: 機能拡充

- Wake Word検知の組み込み（openWakeWord）
- リスニングモード切替（Wake Word / 常時リスニング / ミュート）
- カスタムWake Wordモデルの学習・組み込み
- モード切替（雑談/コード相談/進捗報告）
- Ollamaフォールバックの実装
- 会話ログの自動保存
- キャラクター切替 UI

### 6.4 フェーズ4: 発展（将来）

- FastAPI + Reactへの移行検討
- タスクトレイ常駐化
- ホットキーでのマイクトグル
- 進捗ログのサマリー自動生成

---

## 7. リスクと課題

| リスク | 影響 | 対策 |
|--------|------|------|
| Gemini APIの無料枠変更 | 中 | Ollamaフォールバックで完全ローカル運用も可能 |
| Whisperのレイテンシ | 中 | VADで発話区間を最小化、mediumモデルで高速化 |
| BGM・打鍵音の誤検出 | 高 | VAD閾値調整、プッシュトゥトーク方式も検討 |
| VRAM不足 | 中 | Whisper medium + Ollama 7BでVRAM分散、またはGeminiに寄せる |
| VOICEVOXの応答速度 | 低 | 文単位並列生成で体感待ち時間を削減 |
| Wake Wordの誤検出/見落とし | 中 | 閾値調整、カスタムモデル学習で精度向上 |

---

## 8. ディレクトリ構成（想定）

```
voice-chat-ai/
├── app.py                  # メインエントリーポイント（Gradio UI）
├── requirements.txt
├── .env.example
├── .env
├── config.py               # 設定管理（環境変数読み込み）
├── core/
│   ├── __init__.py
│   ├── stt.py              # Whisper + VAD（音声認識）
│   ├── llm.py              # LLMクライアント（Gemini / Ollama切替）
│   ├── tts.py              # VOICEVOX連携（音声合成）
│   ├── wakeword.py         # openWakeWord連携
│   └── audio.py            # マイク入力・音声再生
├── prompts/
│   ├── casual.txt          # 雑談モード用システムプロンプト
│   ├── code_review.txt     # コード相談モード用
│   └── progress.txt        # 進捗報告モード用
├── models/
│   └── wakeword/           # カスタムWake Wordモデル (.tflite/.onnx)
├── logs/                   # 会話ログ出力先
│   └── YYYY-MM-DD_HHmmss.md
└── tests/
    ├── test_stt.py
    ├── test_llm.py
    ├── test_tts.py
    └── test_wakeword.py
```

---

## 9. 実装メモ

### Gemini API呼び出し（google-genai SDK）

```python
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=messages,
    config={"temperature": 0.7},
)
```

### VOICEVOX音声合成

```python
import requests

VOICEVOX = "http://localhost:50021"
SPEAKER_ID = 3  # ずんだもん

# 1. クエリ生成
query = requests.post(
    f"{VOICEVOX}/audio_query",
    params={"text": text, "speaker": SPEAKER_ID},
).json()

# 2. 音声合成
wav = requests.post(
    f"{VOICEVOX}/synthesis",
    params={"speaker": SPEAKER_ID},
    json=query,
).content
```

### openWakeWord基本使用

```python
import openwakeword
from openwakeword.model import Model

openwakeword.utils.download_models()
model = Model(wakeword_models=["path/to/custom_model.onnx"])

# 80msフレーム単位で予測
prediction = model.predict(audio_frame)
for wake_word, score in prediction.items():
    if score > 0.5:
        activate_conversation_mode()
```

### Ollama呼び出し

```python
import requests

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "qwen2.5:7b",
        "messages": messages,
        "stream": True,
    },
    stream=True,
)
```