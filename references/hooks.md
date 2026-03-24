# Hook 一覧 — VoiceChatAIAssistant

## 自動 Hook

| Hook | トリガー | 動作 |
|------|---------|------|
| `block-push-main.sh` | Bash（git push） | main/master への直接 push をブロック |
| `pre-commit-check.sh` | Bash（git commit） | プロジェクト固有のコミット前チェック |
| `protect-files.sh` | Edit/Write | 保護対象ファイルの編集をブロック |
| `post-edit-quality-check.sh` | Edit/Write（PostToolUse） | 編集後の即時品質チェック |
| `session-start-context.sh` | SessionStart | ブランチ・Issue・MS情報を自動表示 |
| `stop-run-tests.sh` | Stop | ソース変更時にテスト自動実行 + セッション終了リマインダー |

## Hook 設定

設定ファイル: `.claude/settings.json`

### PreToolUse Hook
- **Bash matcher**: `block-push-main.sh` + `pre-commit-check.sh`
- **Edit|Write matcher**: `protect-files.sh`

### PostToolUse Hook
- **Edit|Write matcher**: `post-edit-quality-check.sh`

### Stop Hook
- `stop-run-tests.sh`（タイムアウト: 120秒）

### SessionStart Hook
- `session-start-context.sh`

## カスタマイズ

### pre-commit-check.sh にチェック追加
```bash
# === プロジェクト固有チェックをここに追加 ===
HITS=$(echo "$STAGED" | xargs grep -ln 'FORBIDDEN_PATTERN' 2>/dev/null)
if [ -n "$HITS" ]; then
  ERRORS="${ERRORS}\n- FORBIDDEN_PATTERN 使用禁止: $HITS"
fi
```

### protect-files.sh の保護対象変更
```bash
if echo "$FILE_PATH" | grep -qE "NEVER_MATCH_PLACEHOLDER"; then
```

### post-edit-quality-check.sh にチェック追加
```bash
# === プロジェクト固有チェック ===
# ファイル種別に応じたパターンチェックを追加
```
