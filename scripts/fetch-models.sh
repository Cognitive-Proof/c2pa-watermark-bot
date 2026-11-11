#!/usr/bin/env bash
# Downloads the TrustMark ONNX models for the registered variants (C, P, Q)
# from Adobe's CDN, the same source as `cargo xtask fetch-models` in
# https://github.com/adobe/trustmark/tree/main/rust. About 170 MB in total.
set -euo pipefail

root="https://cai-watermark.adobe.net/watermarking/trustmark-models"
dest="${1:-models}"
mkdir -p "$dest"

for variant in C P Q; do
  for kind in encoder decoder; do
    file="${kind}_${variant}.onnx"
    if [[ -f "$dest/$file" ]]; then
      echo "skip $file (exists)"
      continue
    fi
    echo "fetch $file"
    curl -fsSL --retry 3 -o "$dest/$file.part" "$root/$file"
    mv "$dest/$file.part" "$dest/$file"
  done
done
