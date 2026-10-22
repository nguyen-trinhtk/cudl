#!/bin/bash
# Download and unpack MNIST into data/mnist (or the directory given as the first argument).
#
#   scripts/get_mnist.sh
#   scripts/get_mnist.sh /scratch/$USER/mnist
set -euo pipefail

dir=${1:-data/mnist}
mkdir -p "$dir"
mirrors=(
    https://ossci-datasets.s3.amazonaws.com/mnist
    https://storage.googleapis.com/cvdf-datasets/mnist
)

for name in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    if [[ -f "$dir/$name" ]]; then
        continue
    fi
    for m in "${mirrors[@]}"; do
        if curl -fsSL "$m/$name.gz" -o "$dir/$name.gz"; then
            break
        fi
        echo "mirror $m failed for $name, trying the next" >&2
    done
    [[ -f "$dir/$name.gz" ]] || { echo "could not download $name" >&2; exit 1; }
    gunzip -f "$dir/$name.gz"
done
ls -l "$dir"
