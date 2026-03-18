"""
発音評価バックエンド: FastAPI + torchaudio Wav2Vec2 強制アラインメント
音声ファイルと目標単語を受け取り、文字ごとの発音スコアを返す
"""
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
import io
from contextlib import asynccontextmanager

# ===== モデル読み込み（起動時に一度だけ） =====
model_state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Wav2Vec2 モデルを読み込み中...")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    m = bundle.get_model()
    m.eval()
    model_state["model"] = m
    model_state["labels"] = bundle.get_labels()
    model_state["sample_rate"] = bundle.sample_rate
    print(f"モデル読み込み完了 (サンプルレート: {bundle.sample_rate}Hz)")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def log_prob_to_score(lp: float) -> int:
    """
    log確率 (-inf, 0] を 0-100 スコアに変換。
    Wav2Vec2の出力は log_softmax 後の値。
    実験的な閾値: -4.0 → 0点, 0.0 → 100点
    """
    return max(0, min(100, round((lp + 4.0) / 4.0 * 100)))


def compute_char_scores(waveform: torch.Tensor, word: str) -> list[dict]:
    """
    強制アラインメントで文字ごとの発音スコアを計算。
    Returns: [{"char": "C", "score": 85}, ...]
    """
    model = model_state["model"]
    labels = model_state["labels"]
    label_map = {l: i for i, l in enumerate(labels)}

    # モノラルに変換
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)

    # 発音確率行列を取得 (1, T, C)
    with torch.no_grad():
        emission, _ = model(waveform)

    log_probs = torch.log_softmax(emission, dim=-1)  # (1, T, C)

    # 目標文字列をトークンIDに変換（スペースはスキップ）
    word_upper = word.upper()
    target_ids = []
    non_space_chars = []
    for c in word_upper:
        if c == " ":
            non_space_chars.append(None)  # スペースはそのまま記録
        elif c in label_map:
            target_ids.append(label_map[c])
            non_space_chars.append(c)

    if not target_ids:
        return [{"char": c, "score": 50} for c in word]

    targets = torch.tensor([target_ids], dtype=torch.long)
    input_lengths = torch.tensor([log_probs.shape[1]], dtype=torch.long)
    target_lengths = torch.tensor([len(target_ids)], dtype=torch.long)

    # 強制アラインメント実行
    try:
        paths, scores = torchaudio.functional.forced_align(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )
    except Exception as e:
        print(f"forced_align エラー: {e}")
        return [{"char": c, "score": 50} for c in word]

    path = paths[0].tolist()    # (T,)
    score_seq = scores[0].tolist()  # (T,)

    # フレームを文字ごとにグループ化 (CTC マージ)
    char_frame_scores: list[list[float]] = [[] for _ in range(len(target_ids))]
    char_idx = -1
    prev_token = 0  # 最初はブランク

    for token, s in zip(path, score_seq):
        if token == 0:  # ブランク
            prev_token = 0
        else:
            if token != prev_token:
                char_idx += 1
            if 0 <= char_idx < len(target_ids):
                char_frame_scores[char_idx].append(s)
            prev_token = token

    # 文字ごとの平均スコア算出
    char_avg: list[float] = []
    for frame_scores in char_frame_scores:
        if frame_scores:
            char_avg.append(sum(frame_scores) / len(frame_scores))
        else:
            char_avg.append(-4.0)  # フレームなし → 0点相当

    # 元の単語（スペース含む）にマッピング
    result = []
    scored_idx = 0
    for c in word:
        if c == " ":
            continue
        lp = char_avg[scored_idx] if scored_idx < len(char_avg) else -4.0
        result.append({"char": c, "score": log_prob_to_score(lp)})
        scored_idx += 1

    return result


@app.post("/assess")
async def assess_pronunciation(audio: UploadFile, word: str = Form()):
    try:
        audio_bytes = await audio.read()

        # 音声読み込み
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

        # 16kHz にリサンプリング
        target_sr = model_state["sample_rate"]
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)

        char_scores = compute_char_scores(waveform, word)

        # 総合スコア (0-100)
        total = sum(c["score"] for c in char_scores)
        overall = round(total / len(char_scores)) if char_scores else 0

        return {"chars": char_scores, "score": overall}

    except Exception as e:
        print(f"エラー: {e}")
        import traceback; traceback.print_exc()
        return {"error": str(e), "chars": [], "score": 0}


@app.get("/health")
async def health():
    return {"status": "ok", "model": "wav2vec2-base-960h"}
