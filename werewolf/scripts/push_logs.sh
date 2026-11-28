#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
cp logs/*.jsonl ../../werewolf-logs/ 2>/dev/null
cp agent/lessons.md ../../werewolf-logs/ 2>/dev/null
cd ../../werewolf-logs || { echo "找不到 ~/Desktop/werewolf-logs"; exit 1; }
git add -A && git commit -qm "logs $(date +%m%d-%H%M)" && git push -q && echo "✓ logs 已上傳"
