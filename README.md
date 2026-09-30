# VenoDec Backend

FastAPI service that runs **FULL AASIST** anti-spoofing inference over audio streamed from the
VenoDec Android app.

There is no mock model, no heuristic scoring, and no hardcoded probability anywhere in this
service. If the pretrained checkpoint cannot be loaded, startup fails loudly rather than
degrading to a fake result.

---

## Setup

```bash
cd backend
pip install -r requirements.txt

# One-time: vendor FULL AASIST (model code + pretrained checkpoint) from the official repo
python scripts/download_aasist.py
```

`download_aasist.py` refuses to write the checkpoint unless it is exactly **1,281,532 bytes** —
the FULL AASIST release. AASIST-L (426,428 bytes) is explicitly rejected.

## Run

```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` matters: a physical phone must reach the laptop over the LAN.

Override the port with `--port` if 8000 is taken.

## Test

```bash
cd backend
pytest                                    # 47 tests
python scripts/validate_samples.py        # score demo_audio/, print real numbers
python scripts/e2e_client.py              # drive a running server over a real socket
```

---

## API

### `GET /health`

```json
{
  "status": "ok",
  "model": {
    "name": "AASIST", "variant": "FULL", "full_aasist": true, "is_mock": false,
    "num_parameters": 297866, "checkpoint_bytes": 1281532,
    "checkpoint_sha256": "51d2d9cf0738172f...", "device": "cpu",
    "sample_rate": 16000, "input_samples": 64600
  },
  "signals": [ ... ],
  "privacy_notice": "Audio is processed for analysis and is not permanently stored."
}
```

### `WS /ws/analyze`

**Mode A — live microphone**

```
-> {"type":"start","mode":"live","format":"pcm16","sample_rate":16000}
<- {"type":"ready", "model": {...}, "privacy_notice": "..."}
-> <binary>   raw PCM16LE mono 16 kHz, ~3 s per frame
<- {"type":"result", ...}        one result per chunk
-> {"type":"stop"}
```

**Mode B — test audio (WAV)**

```
-> {"type":"start","mode":"file","format":"wav"}
<- {"type":"ready", ...}
-> <binary> ... <binary>         WAV bytes, may span frames
-> {"type":"end"}
<- {"type":"result", ...}        one aggregated result
```

Failures come back as `{"type":"error","message":"..."}`.

### `POST /analyze`

Multipart WAV upload through the same pipeline, for curl/debugging:

```bash
curl -F "file=@demo_audio/genuine/timit_ldc93s1.wav" http://127.0.0.1:8000/analyze
```

### Result payload

```json
{
  "synthetic_probability": 0.9998,
  "voice_authenticity": 0.0002,
  "risk_score": 100,
  "risk_level": "HIGH",
  "recommended_action": "BLOCK",
  "confidence": 1.0,
  "confidence_label": "High",
  "summary": "AI-generated voice detected",
  "reasons": ["High probability of synthetic speech (100%)", "..."],
  "signals": [ ... ],
  "windows_analyzed": 3,
  "model": "AASIST (FULL)",
  "mode": "file"
}
```

---

## How the model is used

**Checkpoint** — `app/models/aasist/weights/AASIST.pth`, 1,281,532 bytes,
sha256 `51d2d9cf0738172f...`, from [clovaai/aasist](https://github.com/clovaai/aasist) (MIT).

**Architecture** — `app/models/aasist/aasist_model.py` is an unmodified copy of upstream
`models/AASIST.py`. It is instantiated from `aasist_config.json`, a verbatim copy of the
`model_config` block of upstream `config/AASIST.conf`:

```
nb_samp 64600, first_conv 128, gat_dims [64, 32],
filts [70,[1,32],[32,32],[32,64],[64,64]],
pool_ratios [0.5,0.7,0.5,0.5], temperatures [2.0,2.0,100.0,100.0]
```

Weights load with `strict=True`. That is load-bearing: AASIST-L uses `gat_dims [24,24]` and
different `filts`, so its weights cannot load against this config. Loaded model reports
**297,866 parameters**, matching FULL AASIST (AASIST-L is ~85k).

**Preprocessing** (`app/audio.py`, identical for both modes):

1. decode — `soundfile`/libsndfile for WAV; raw PCM16LE ÷ 32768 for microphone frames
2. downmix to mono by averaging channels
3. resample to 16 kHz with `scipy.signal.resample_poly`
4. keep the native `[-1, 1]` float range — no extra normalisation, matching upstream `sf.read`
5. window to exactly 64600 samples; short audio is tile-repeated by a direct port of upstream
   `pad()` from `data_utils.py`
6. `torch.from_numpy(...).float()` → shape `(batch, 64600)`

**Inference and scoring** — `_, logits = model(x)` gives `(batch, 2)`; `softmax` over dim 1.
Upstream `main.py:307` uses `batch_out[:, 1]` as the bonafide score, so **index 1 = bonafide,
index 0 = spoof**. `synthetic_probability = softmax[:, 0]`, averaged across windows.

> `librosa` is not used. `scipy.signal.resample_poly` does the resampling — it is already a
> dependency, it is lighter, and it avoids librosa's numba/numpy version constraints.

### Live-mode windowing

The app sends ~3 s chunks as specified, but AASIST's native input is 64600 samples (4.04 s).
`LiveAudioBuffer` keeps a rolling window of the most recent 64600 samples, so every inference
after the first sees real contiguous audio rather than tile-padding, while still emitting one
result per chunk. Only the very first chunk of a session is padded.

---

## Risk engine

`app/risk_engine.py` carries the full planned signal set with its design weights, but **only
implemented signals contribute**:

| Signal | Design weight | Status |
| --- | --- | --- |
| Voice authenticity (FULL AASIST) | 0.35 | **active** |
| Speaker similarity | 0.25 | Not active in prototype |
| Conversation risk | 0.25 | Not active in prototype |
| Context | 0.15 | Not active in prototype |

Weights are renormalised over active signals, so in the prototype AASIST carries the whole score
and `risk_score = round(100 × synthetic_probability)`. Inactive signals are returned with
`"active": false` and `"value": null` — no placeholder number is invented for a model that does
not exist yet.

Bands `0–30 LOW / 31–60 MEDIUM / 61–100 HIGH`; actions `ALLOW / WARN / BLOCK`.
`confidence = 2 × |p − 0.5|` — distance from the decision boundary, derived from the model output.

---

## Privacy

Audio is held only in per-connection in-memory buffers and discarded as soon as it is analysed or
the socket closes. Nothing is written to disk, and no audio bytes are logged. There is no audio
database. Only the Android client stores anything, and only analysis metadata.
