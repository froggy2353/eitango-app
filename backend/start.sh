#!/bin/bash
# 発音評価バックエンドを起動するスクリプト
# 依存パッケージのインストール: pip install -r requirements.txt
# また、WebM/mp4 音声を読み込むために ffmpeg が必要:
#   macOS: brew install ffmpeg
#   Ubuntu: sudo apt install ffmpeg
# 起動:
#   cd backend && bash start.sh

cd "$(dirname "$0")"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
