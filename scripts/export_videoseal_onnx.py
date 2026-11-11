"""Exports Meta's VideoSeal 1.0 and PixelSeal detectors to ONNX.

The detectors take a 1x3x256x256 RGB image in [0, 1] and return 1 + 256
logits: a detection channel (not trained in the released checkpoints) and the
256 message bits. Resizing to 256x256 happens in the caller, as in
`Videoseal.detect`.

    git clone https://github.com/facebookresearch/videoseal
    pip install torch torchvision omegaconf einops timm==0.9.16 safetensors scipy onnx onnxruntime \
        requests tqdm av opencv-python-headless pycocotools pillow
    python scripts/export_videoseal_onnx.py --videoseal path/to/videoseal --out models

Checkpoints are downloaded from dl.fbaipublicfiles.com (MIT licensed).
"""

import argparse
import os
import sys

import numpy as np

CARDS = {"videoseal": "videoseal_1.0", "pixelseal": "pixelseal"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videoseal", required=True, help="path to a clone of facebookresearch/videoseal")
    parser.add_argument("--out", default="models")
    parser.add_argument("--models", nargs="+", default=list(CARDS), choices=list(CARDS))
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.videoseal))
    # Model cards and configs are resolved relative to the working directory.
    out = os.path.abspath(args.out)
    os.chdir(args.videoseal)

    import onnxruntime as ort
    import torch
    import videoseal

    os.makedirs(out, exist_ok=True)
    for name in args.models:
        model = videoseal.load(CARDS[name]).eval()
        detector = model.detector.eval()
        dummy = torch.rand(1, 3, 256, 256)
        path = os.path.join(out, f"{name}_detector.onnx")
        torch.onnx.export(
            detector,
            (dummy,),
            path,
            input_names=["image"],
            output_names=["preds"],
            opset_version=17,
            dynamo=False,
        )

        session = ort.InferenceSession(path)
        with torch.no_grad():
            expected = detector(dummy).numpy()
        actual = session.run(None, {"image": dummy.numpy()})[0]
        diff = float(np.abs(expected - actual).max())
        print(f"{name}: wrote {path}, output {actual.shape}, max abs diff vs torch {diff:.2e}")
        if diff > 1e-3:
            sys.exit(f"{name}: ONNX output does not match torch")


if __name__ == "__main__":
    main()
