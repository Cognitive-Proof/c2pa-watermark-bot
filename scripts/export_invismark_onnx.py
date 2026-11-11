"""Exports the decoder of Microsoft's InvisMark paper checkpoint to ONNX.

The decoder takes a 1x3x256x256 RGB image in [-1, 1] and returns 100 bit
probabilities (sigmoid outputs). The checkpoint has no error correction.

The checkpoint is `ckpt_paper.zip` (1.06 GB) in the OneDrive folder linked from
https://github.com/microsoft/InvisMark, which needs a browser to download.

    git clone https://github.com/microsoft/InvisMark
    unzip ckpt_paper.zip            # paper.ckpt
    pip install torch torchvision onnx onnxruntime
    python scripts/export_invismark_onnx.py --invismark path/to/InvisMark \
        --ckpt path/to/paper.ckpt --out models

The checkpoint is a pickle. It is loaded with `weights_only=True`, which
refuses any object other than tensors, standard containers and InvisMark's
`ModelConfig` dataclass, so it cannot run code.
"""

import argparse
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--invismark", required=True, help="path to a clone of microsoft/InvisMark")
    parser.add_argument("--ckpt", required=True, help="path to paper.ckpt")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.invismark))

    import onnxruntime as ort
    import torch
    import torchvision

    # The extractor starts from ImageNet weights, which the checkpoint replaces;
    # skip downloading them.
    convnext_base = torchvision.models.convnext_base
    torchvision.models.convnext_base = lambda **kwargs: convnext_base(weights=None)

    import configs
    import model as invismark_model

    torch.serialization.add_safe_globals([configs.ModelConfig])
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    config = state["config"]
    print(f"config: {config.num_encoded_bits} bits, image {config.image_shape}, epoch {state['cur_epoch']}")

    decoder = invismark_model.Extractor(config)
    decoder.load_state_dict(state["decoder_state_dict"])
    decoder.eval()

    h, w = config.image_shape
    dummy = torch.rand(1, 3, h, w) * 2 - 1
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "invismark_decoder.onnx")
    torch.onnx.export(
        decoder,
        (dummy,),
        path,
        input_names=["image"],
        output_names=["bits"],
        opset_version=17,
        dynamo=False,
    )

    session = ort.InferenceSession(path)
    with torch.no_grad():
        expected = decoder(dummy).numpy()
    actual = session.run(None, {"image": dummy.numpy()})[0]
    diff = float(np.abs(expected - actual).max())
    print(f"wrote {path}, output {actual.shape}, max abs diff vs torch {diff:.2e}")
    if diff > 1e-3:
        sys.exit("ONNX output does not match torch")


if __name__ == "__main__":
    main()
