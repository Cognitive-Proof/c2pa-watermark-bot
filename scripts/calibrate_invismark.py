"""Measures how the exported InvisMark decoder separates clean and watermarked
images, to choose the detection threshold used in `src/detectors/invismark.rs`.

The paper checkpoint has no error correction and no detection output, so
detection relies on how confident the 100 bit probabilities are: the score is
mean |2p - 1|, from 0 (all 0.5) to 1 (all certain). For each image this embeds
a random UUID4 (first 100 bits) with the PyTorch encoder (as `Watermark._encode` does), and
runs the ONNX decoder on clean, watermarked and attacked copies with the same
preprocessing as the Rust code (bilinear downscale to 256x256, scale to [-1, 1]).

    python scripts/calibrate_invismark.py --invismark path/to/InvisMark \
        --ckpt path/to/paper.ckpt --models-dir models --fixtures tests/fixtures images/*.png
"""

import argparse
import io
import json
import os
import sys

import numpy as np
from PIL import Image


def probs(session, img):
    x = np.asarray(img.convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.float32) / 127.5 - 1
    return session.run(None, {"image": x.transpose(2, 0, 1)[None]})[0][0]


def score(p):
    return float(np.abs(2 * p - 1).mean())


def jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


ATTACKS = {
    "png": lambda i: i,
    "jpeg80": lambda i: jpeg(i, 80),
    "jpeg50": lambda i: jpeg(i, 50),
    "jpeg30": lambda i: jpeg(i, 30),
    "resize50": lambda i: i.resize((i.width // 2, i.height // 2), Image.BILINEAR),
    "resize25": lambda i: i.resize((i.width // 4, i.height // 4), Image.BILINEAR),
    "crop50": lambda i: i.crop((0, 0, i.width // 2, i.height // 2)),
    "bright": lambda i: Image.fromarray(np.clip(np.asarray(i, dtype=np.float32) * 1.3, 0, 255).astype(np.uint8)),
}

CLEAN_TRANSFORMS = {
    "id": lambda i: i,
    "crop": lambda i: i.crop((0, 0, i.width // 2, i.height // 2)),
    "rot90": lambda i: i.rotate(90, expand=True),
    "flip": lambda i: i.transpose(Image.FLIP_LEFT_RIGHT),
    "jpeg30": lambda i: jpeg(i, 30),
    "small": lambda i: i.resize((i.width // 3, i.height // 3)),
    "gray": lambda i: i.convert("L").convert("RGB"),
    "dark": lambda i: Image.fromarray((np.asarray(i, dtype=np.float32) * 0.4).astype(np.uint8)),
    "square": lambda i: i.resize((300, 300)),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--invismark", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--fixtures", help="write a watermarked PNG of the first image here")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.invismark))
    import onnxruntime as ort
    import torch
    import torchvision
    import torchvision.transforms as transforms

    convnext_base = torchvision.models.convnext_base
    torchvision.models.convnext_base = lambda **kwargs: convnext_base(weights=None)
    import configs
    import model as invismark_model

    torch.serialization.add_safe_globals([configs.ModelConfig])
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    encoder = invismark_model.Encoder(state["config"])
    encoder.load_state_dict(state["encoder_state_dict"])
    encoder.eval()
    session = ort.InferenceSession(os.path.join(args.models_dir, "invismark_decoder.onnx"))

    torch.manual_seed(0)
    import uuid

    clean = []
    marked, acc = {k: [] for k in ATTACKS}, {k: [] for k in ATTACKS}
    for n, path in enumerate(args.images):
        img = Image.open(path).convert("RGB")
        clean += [score(probs(session, f(img))) for f in CLEAN_TRANSFORMS.values()]

        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
        # The checkpoint was trained on the first 100 bits of a UUID4
        # (`utils.uuid_to_bits`); random bits decode noticeably worse (94% on
        # PNG). Its decoder outputs the UUID4 version and variant bits on clean
        # images too, so they can't be used as a check.
        secret = torch.from_numpy(np.unpackbits(np.frombuffer(uuid.uuid4().bytes, dtype=np.uint8))[:100]).float()[None]
        with torch.no_grad():
            small = transforms.Resize((256, 256))(x)
            diff = transforms.Resize(x.shape[-2:])(encoder(small, secret) - small)
            out = torch.clamp(x + diff, -1, 1)[0]
        wm = Image.fromarray(((out.permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8))
        bits = secret[0].numpy().astype(bool)
        for k, f in ATTACKS.items():
            p = probs(session, f(wm))
            marked[k].append(score(p))
            acc[k].append(float(((p > 0.5) == bits).mean()))

        if args.fixtures and n == 0:
            os.makedirs(args.fixtures, exist_ok=True)
            fixture = wm.resize((wm.width * 256 // wm.height, 256), Image.BILINEAR)
            fixture.save(os.path.join(args.fixtures, "invismark_marked.png"))
            p = probs(session, fixture)
            with open(os.path.join(args.fixtures, "invismark_marked.json"), "w") as f:
                json.dump(
                    {
                        "embedded_bits": "".join("1" if b else "0" for b in bits),
                        "bit_accuracy_on_fixture": float(((p > 0.5) == bits).mean()),
                        "score_on_fixture": score(p),
                    },
                    f,
                    indent=2,
                )

    clean = np.array(clean)
    print(
        f"clean (n={len(clean)}): max {clean.max():.3f}  p99 {np.percentile(clean, 99):.3f}"
        f"  median {np.median(clean):.3f}"
    )
    for k in ATTACKS:
        print(
            f"marked {k:9s} score min {min(marked[k]):.3f} median {np.median(marked[k]):.3f}"
            f"  bit accuracy min {min(acc[k]):.3f} mean {np.mean(acc[k]):.3f}"
        )


if __name__ == "__main__":
    main()
