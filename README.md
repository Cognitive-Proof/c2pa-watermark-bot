# c2pa-watermark-bot

An HTTP API that takes an image and looks for C2PA soft bindings:

- **Watermarks**, using every algorithm in the
  [C2PA soft-binding algorithm list](https://github.com/c2pa-org/softbinding-algorithm-list)
  that has a public decoder.
- **Embedded C2PA manifests**, reporting any `c2pa.soft-binding` assertions and
  whether a decoded watermark matches them.

Image algorithms in the list that can't be checked locally are listed in the
response with the reason and any resolution APIs the list gives for them.

## Coverage

| `alg` | Decoder | Notes |
|---|---|---|
| `com.adobe.trustmark.Q` | [`trustmark`](https://crates.io/crates/trustmark) (MIT) | |
| `com.adobe.trustmark.C` | same | |
| `com.adobe.trustmark.P` | same | |
| `com.adobe.trustmark.F` | none | no public model |
| `me.reconize.videoseal.1` | Meta [VideoSeal](https://github.com/facebookresearch/videoseal) 1.0 (MIT), via ONNX | Reconize says it uses VideoSeal but not which checkpoint; unconfirmed against a real Reconize image |
| `com.aiwatermark.pixelseal.1` | Meta PixelSeal (MIT), via ONNX | AIWatermark says it uses PixelSeal but not whether it changed the model; unconfirmed |
| `com.microsoft.invismark.1` | Microsoft [InvisMark](https://github.com/microsoft/InvisMark) paper checkpoint (MIT), via ONNX | Nothing ties the paper checkpoint to the registered model; unconfirmed |

No other image algorithm in the list has public decoder code as of 2026-11.
Detectors implement `detectors::Detector`, so more can be added.

### TrustMark false positives

TrustMark's decoder tries every BCH schema, and BCH_3 corrects up to three bit
flips, so unmarked images sometimes decode to a valid-looking payload. The
detector decodes the image and a copy downscaled to 90%, and reports a
watermark only when both agree. On the 24-image Kodak set this removed every
false positive (2 of 72 decodes) and kept every real watermark (72 of 72).
Rerun the measurement with:

```sh
cargo run --release --example fp_probe -- path/to/images/*.png
```

### VideoSeal and PixelSeal detection

The released checkpoints have no trained detection channel and no error
correction, so every image decodes to 256 bits. The detector reports a
watermark when the mean absolute message logit (`score`) is at least 4.0, and
returns the score either way.

| Images | VideoSeal score | PixelSeal score |
|---|---|---|
| 240 clean (Kodak with crops, rotations, JPEG 30, downscales, grayscale) and 6 synthetic | max 1.5 | max 2.6 |
| Watermarked, PNG | min 8.4 | min 9.6 |
| Watermarked, JPEG 80 | min 6.6 | min 6.9 |
| Watermarked, JPEG 30 | min 4.2 | min 3.8 |
| Watermarked, downscaled 4x | min 4.8 | min 2.8 |
| Watermarked, then cropped to 50% | ~0.4 (not detectable) | ~0.4 |

So heavily degraded marks can be missed, and crops are not supported: the
detector resizes the whole image to 256x256, which changes the scale of a
cropped watermark. Bit accuracy on detected marks was 95–100%. A payload
matches a manifest value when at least 90% of bits agree (base64, hex or
bitstring, MSB first). Reproduce with `scripts/calibrate_videoseal.py`.

### InvisMark detection

The paper checkpoint decodes 100 bits with no error correction and no
detection output. The score is the mean of |2p − 1| over the bit
probabilities, from 0 (all 0.5) to 1 (all certain); a watermark is reported at
0.8 or more.

| Images | Score |
|---|---|
| 216 clean (Kodak with crops, rotations, JPEG 30, downscales, grayscale) | max 0.66 |
| Watermarked, PNG, or downscaled 2x or 4x | min 0.99 |
| Watermarked, JPEG 80 | min 0.93 |
| Watermarked, brightness +30% | min 0.90 |
| Watermarked, JPEG 50 | median 0.72 (mostly missed) |
| Watermarked, then cropped to 50% | ~0.28 (not detectable) |

The checkpoint was trained on the first 100 bits of a UUID4, and decodes
random bits worse (94% on PNG against 100% for UUIDs). Its decoder outputs the
UUID4 version and variant bits on clean images too, so they can't serve as a
check. Manifest values may be a UUID, hex, base64 or a bitstring and match on
at least 90% of the first 100 bits. Reproduce with
`scripts/calibrate_invismark.py`.

## Running

```sh
./scripts/fetch-models.sh        # TrustMark C, P and Q models, ~170 MB, into ./models
cargo run --release
```

The VideoSeal and PixelSeal detectors need ONNX exports (~130 MB each), made
with Python from Meta's repo:

```sh
git clone https://github.com/facebookresearch/videoseal /tmp/videoseal
python -m venv .venv && .venv/bin/pip install torch torchvision omegaconf einops \
  timm==0.9.16 safetensors scipy onnx onnxruntime requests tqdm av \
  opencv-python-headless pycocotools pillow
.venv/bin/python scripts/export_videoseal_onnx.py --videoseal /tmp/videoseal --out models
```

The InvisMark detector needs an ONNX export (~350 MB) of the paper checkpoint.
Download `ckpt_paper.zip` (1.06 GB) from the OneDrive folder linked in
Microsoft's README; it needs a browser. The checkpoint is loaded with
`weights_only=True`, so it can't run code.

```sh
git clone https://github.com/microsoft/InvisMark /tmp/invismark
unzip ckpt_paper.zip -d /tmp/invismark-ckpt
.venv/bin/pip install kornia bchlib
.venv/bin/python scripts/export_invismark_onnx.py --invismark /tmp/invismark \
  --ckpt /tmp/invismark-ckpt/paper.ckpt --out models
```

Detectors whose model files are missing are skipped and listed in `not_checked`.

| Variable | Default | |
|---|---|---|
| `BIND_ADDR` | `127.0.0.1:8080` | |
| `MODELS_DIR` | `models` | Missing models disable that detector |
| `MAX_UPLOAD_MB` | `25` | |
| `RUST_LOG` | `info,ort=warn` | |

## API

### `POST /v1/detect`

Send the image as the raw body, or as the `image` field of a multipart form.
The format is detected from the bytes.

```sh
curl --data-binary @photo.jpg http://127.0.0.1:8080/v1/detect
curl -F image=@photo.jpg http://127.0.0.1:8080/v1/detect
```

```jsonc
{
  "media_type": "image/png",
  "width": 512,
  "height": 384,
  "found": ["com.adobe.trustmark.Q"],
  "watermarks": [
    {
      "alg": "com.adobe.trustmark.Q",
      "found": true,
      "payload": { "bits": "0110…", "schema": "BCH_5" },
      "matches_manifest": true,   // absent when the manifest has no binding for this alg
      "elapsed_ms": 70
    },
    {
      "alg": "me.reconize.videoseal.1",
      "found": false,
      "score": 0.41,              // found at >= 4.0
      "note": "Meta's public VideoSeal 1.0 checkpoint. …",
      "elapsed_ms": 60
    }
  ],
  "c2pa": {
    "present": true,
    "validation_state": "Valid",
    "active_manifest": "urn:uuid:…",
    "soft_bindings": [
      { "manifest": "urn:uuid:…", "alg": "com.adobe.trustmark.Q", "values": ["2*0110…"] }
    ],
    "soft_binding_algorithms": [
      { "alg": "com.adobe.trustmark.Q", "registered": true, "local_decoder": true, "resolution_apis": [] }
    ]
  },
  "not_checked": [
    { "alg": "com.digimarc.validate.1", "type": "watermark", "reason": "no public decoder", "resolution_apis": [] }
  ]
}
```

TrustMark block values written as `<schema>*<bits>` (as in Adobe's C2PA
example) are matched on the bits after `*`.

### `GET /v1/algorithms`

Every entry in the vendored algorithm list, with whether it applies to images
and whether a local decoder is loaded.

### `GET /health`

## Updating the algorithm list

The list is vendored at `data/softbinding-algorithm-list.json` and compiled in.

```sh
gh api repos/c2pa-org/softbinding-algorithm-list/contents/softbinding-algorithm-list.json \
  -H "Accept: application/vnd.github.raw" > data/softbinding-algorithm-list.json
```

## Notes

- `c2pa` is pinned to 0.90.x because 0.91 and later need Rust 1.96.
- `Trustmark::new` loads the encoder as well as the decoder, so both model files
  are required per variant.
