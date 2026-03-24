# VoiceChatAIAssistant

## プロジェクト概要
- エンジン/フレームワーク: Python 3.10+ / Gradio / google-genai / VOICEVOX
- リポジトリ: https://github.com/kaiman2110/VoiceChatAIAssistant
- テスト: `.venv/Scripts/pytest tests/ -v`

## アーキテクチャ原則
- パイプライン構成: マイク → VAD → Whisper → LLM → VOICEVOX → スピーカー
- 各コンポーネントは `core/` 配下に独立クラスとして実装
- VOICEVOX未起動時はテキストのみで動作（グレースフルデグレード）
- 設定は `config.py` (pydantic-settings) で一元管理

## 既知の落とし穴
<!-- 詳細は references/gotchas.md に記載。頻出のみここに残す -->

| 問題 | 対処法 |
|------|--------|
| gradio 6.x は Python 3.13 で起動時ハング | `gradio>=5.0,<6.0` に固定 |
| テストで .env の値が混入する | `Settings(_env_file=None)` でデフォルト値テスト |

## グローバル状態一覧
<!-- Autoload / シングルトン / グローバル変数 等 -->

## 参照テーブル

| ドキュメント | パス | 内容 |
|-------------|------|------|
| フォルダ構造 | `references/folder-structure.md` | ディレクトリ構成 |
| 既知の落とし穴（詳細） | `references/gotchas.md` | フレームワーク固有の問題と対処法 |
| Hook 一覧 | `references/hooks.md` | 自動 Hook の動作・設定 |
| スキル・ワークフロー | `references/skills.md` | 全スキル・Agent teams・セッションの流れ |
| ADR | `docs/adr/` | アーキテクチャ決定記録 |

## 実装時の原則
- **自律実行**: 実装→テスト→コミットを自律的に回す。途中でユーザーに聞かない
- **ステップ分割**: 一度にすべて実装しない。ステップごとに実装→テスト→コミット
- **自己検証**: エラーは自分で直す。設計判断はアーキテクチャ原則に従う
- **型ヒント**: 新規・変更コードには必ず型ヒントを付ける
- **検索は Grep/Glob 優先**: `Read` で探索的にファイルを開かない
- **Issue完了時は必ず `/close-issue`** / セッション終了は `/session-end`

## クイックリファレンス: スキル

| スキル | 用途 |
|--------|------|
| `/start-ms <N>` | MS単位で全Issue一括実装 |
| `/start-issue <N>` | Issue 実装開始 → 自律実装 |
| `/close-issue <N>` | Issue 完了（PR・マージ・メモリ更新） |
| `/session-start` | セッション開始コンテキスト読み込み |
| `/session-end` | セッション終了メモリ保存 |
| `/harness-review` | Hook・スキル・CLAUDE.md の定期点検 |

> 全スキル詳細・ワークフロー図は `references/skills.md` を参照
