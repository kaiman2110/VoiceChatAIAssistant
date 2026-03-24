# スキル・ワークフロー — VoiceChatAIAssistant

## メインスキル

| スキル | 用途 |
|--------|------|
| `/start-ms <N>` | MS単位で全Issue一括実装（依存関係順に自動実行） |
| `/start-issue <N>` | Issue 実装開始 → 自律実装 → 完了時のみ動作確認依頼 |
| `/close-issue <N>` | Issue 完了（PR作成・マージ・クリーンアップ） |
| `/session-start` | セッション開始時のコンテキスト自動読み込み |
| `/session-end` | セッション終了時のメモリ・ドキュメント更新・引継ぎ準備 |
| `/design <テーマ>` | 仕様の設計・実装計画の作成 |
| `/milestone-plan <テーマ>` | マイルストーン計画→Issue分割→ボード登録 |
| `/deep-analyze [focus]` | Agent teams 並列解析（構造・品質・シグナル・コンテンツ） |
| `/research-game <テーマ>` | 参考事例の設計を Deep Research → 適用案 |
| `/idea <内容>` | アイデアを docs/ideas.md に即メモ |
| `/fix-errors` | エラーログ自動解析・修正 |
| `/analyze-sessions` | セッション履歴分析→改善候補特定 |
| `/parallel-refactor` | Agent teams + worktree で並列リファクタリング |
| `/harness-review` | Hook・スキル・CLAUDE.md の定期点検 |
| `/sync-template` | スキル・Hook・CLAUDE.md をテンプレートと比較・同期 |

## 自動トリガー専用スキル（手動呼び出し不要）

| スキル | トリガー条件 |
|--------|------------|
| `/close-issue` | `/start-issue` フェーズ4 で ok |
| `/fix-errors` | テスト失敗時・エラーログ貼付時 |

## セッションの流れ

```
/session-start → コンテキスト読み込み + 推奨アクション提示
  ├─ /start-ms <N> → MS全Issue一括実装（推奨）
  ├─ /start-issue <N> → 単体Issue実装
  ├─ Issue 間の隙間時間:
  │    ├─ /research-game で参考事例調査 → /design に連携
  │    ├─ /parallel-refactor で技術的負債を並列解消
  │    ├─ /deep-analyze でプロジェクト健全性チェック
  │    ├─ /design で次MSの先行設計
  │    └─ /harness-review で定期点検
  └─ /session-end → メモリ保存 → 引継ぎ
```

## Agent teams 活用

| スキル | 並列 Agent 数 | タイミング |
|--------|-------------|-----------|
| `/start-issue` | Explore x3 | コードベース調査 |
| `/close-issue` | x3 | メモリ更新/次Issue分析/品質チェック |
| `/session-start` | Explore x3 | 依存関係/健全性/設計状況 |
| `/session-end` | general-purpose x3 | MEMORY/CLAUDE.md/知見保存 |
| `/deep-analyze` | Explore x4 | 構造/品質/シグナル/コンテンツ |
| `/parallel-refactor` | worktree 分離 xN | 独立リファクタを並列実行 |

### 並列実行の判断基準
- **読み取り専用（Explore）**: 常に並列OK。積極的に使う
- **書き込みあり（general-purpose）**: 変更ファイル重複チェック後に判断
- **worktree 分離**: ファイル衝突リスクがある場合に使用
