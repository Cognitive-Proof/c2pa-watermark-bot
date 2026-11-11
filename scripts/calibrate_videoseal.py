"""Measures how the exported VideoSeal/PixelSeal detectors separate clean and
watermarked images, to choose the detection threshold used in
`src/detectors/videoseal.rs`.

The checkpoints have no trained detection channel and no error correction, so
detection relies on how confident the 256 message logits are. For each image
this embeds a random message with the PyTorch model, saves it as PNG and as
JPEG (quality 80), and runs the ONNX detector on the clean and marked copies
with the same preprocessing as the Rust code (bilinear downscale to 256x256).

    python scripts/calibrate_videoseal.py --videoseal path/to/videoseal \
        --models-dir models --fixtures tests/fixtures images/*.png
"""

import argparse
import io
import json
import os
import sys

import numpy as np
from PIL import Image

CARDS = {"videoseal": "videoseal_1.0", "pixelseal": "pixelseal"}


def logits(session, img):
    x = np.asarray(img.convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]
    return session.run(None, {"image": x})[0][0, 1:]


def score(lg):
    """Mean absolute logit: the statistic the Rust detector thresholds."""
    return float(np.abs(lg).mean())


def jpeg(img, quality=80):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videoseal", required=True)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--fixtures", help="write one watermarked PNG per model here")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    images = [os.path.abspath(p) for p in args.images]
    models_dir = os.path.abspath(args.models_dir)
    fixtures = os.path.abspath(args.fixtures) if args.fixtures else None
    sys.path.insert(0, os.path.abspath(args.videoseal))
    os.chdir(args.videoseal)

    import onnxruntime as ort
    import torch
    import videoseal

    torch.manual_seed(0)
    sessions = {
        name: ort.InferenceSession(os.path.join(models_dir, f"{name}_detector.onnx"))
        for name in CARDS
    }
    results = {name: {"clean": [], "png": [], "jpeg": [], "cross": [], "acc_png": [], "acc_jpeg": []} for name in CARDS}

    for name, card in CARDS.items():
        model = videoseal.load(card).eval()
        other = next(n for n in CARDS if n != name)
        for path in images:
            img = Image.open(path).convert("RGB")
            results[name]["clean"].append(score(logits(sessions[name], img)))

            msg = torch.randint(0, 2, (1, 256)).float()
            x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
            with torch.no_grad():
                marked = model.embed(x, msgs=msg, is_video=False)["imgs_w"][0]
            marked = Image.fromarray(
                (marked.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            )
            bits = msg[0].numpy().astype(bool)
            for kind, im in (("png", marked), ("jpeg", jpeg(marked))):
                lg = logits(sessions[name], im)
                results[name][kind].append(score(lg))
                results[name][f"acc_{kind}"].append(float(((lg > 0) == bits).mean()))
            # The other model's detector on this model's watermark.
            results[other]["cross"].append(score(logits(sessions[other], marked)))

            if fixtures and path == images[0]:
                os.makedirs(fixtures, exist_ok=True)
                # Downscaled rather than cropped: the detector sees the whole
                # image at 256x256, so a crop changes the watermark's scale.
                small = marked.resize((marked.width * 256 // marked.height, 256), Image.BILINEAR)
                small.save(os.path.join(fixtures, f"{name}_marked.png"))
                lg = logits(sessions[name], small)
                with open(os.path.join(fixtures, f"{name}_marked.json"), "w") as f:
                    json.dump(
                        {
                            "embedded_bits": "".join("1" if b else "0" for b in bits),
                            "bit_accuracy_on_fixture": float(((lg > 0) == bits).mean()),
                            "score_on_fixture": score(lg),
                        },
                        f,
                        indent=2,
                    )

    for name, r in results.items():
        print(f"\n{name}  (n={len(images)})  score = mean |logit|")
        for kind in ("clean", "cross", "png", "jpeg"):
            v = np.array(r[kind])
            print(f"  {kind:6s} min {v.min():6.2f}  median {np.median(v):6.2f}  max {v.max():6.2f}")
        for kind in ("acc_png", "acc_jpeg"):
            v = np.array(r[kind])
            print(f"  {kind:8s} bit accuracy min {v.min():.3f}  mean {v.mean():.3f}")


if __name__ == "__main__":
    main()
