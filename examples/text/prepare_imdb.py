"""Convert Keras' IMDB dataset into flat binaries for bidirectional_lstm_imdb.exs.

`imdb.npz` stores each review as a pickled Python list inside a NumPy
object array, which Nx cannot read. This script applies exactly the
preprocessing that

    keras.datasets.imdb.load_data(num_words=20000)
    keras.utils.pad_sequences(..., maxlen=200)

performs, then writes the four resulting arrays as raw little-endian
binaries:

    x_train.bin, x_val.bin   int32, 25000 x 200
    y_train.bin, y_val.bin   float32, 25000

Usage:

    python prepare_imdb.py --out /tmp/imdb

Requires numpy. The 17MB source archive is downloaded once and cached in
the output directory.
"""

import argparse
import os
import urllib.request

import numpy as np

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz"

# keras.datasets.imdb.load_data defaults.
SEED = 113
START_CHAR = 1
OOV_CHAR = 2
INDEX_FROM = 3


def pad_pre(sequences, maxlen):
    """keras.utils.pad_sequences with padding='pre', truncating='pre', value=0."""
    padded = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for i, sequence in enumerate(sequences):
        if len(sequence) == 0:
            continue
        truncated = sequence[-maxlen:]
        padded[i, maxlen - len(truncated):] = truncated
    return padded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/imdb", help="output directory")
    parser.add_argument("--num-words", type=int, default=20000)
    parser.add_argument("--maxlen", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    archive = os.path.join(args.out, "imdb.npz")

    if not os.path.exists(archive):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, archive)

    with np.load(archive, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_val, y_val = f["x_test"], f["y_test"]

    # Keras shuffles both splits with a fixed seed before any filtering.
    rng = np.random.RandomState(SEED)

    for split in (0, 1):
        data, labels = (x_train, y_train) if split == 0 else (x_val, y_val)
        indices = np.arange(len(data))
        rng.shuffle(indices)
        if split == 0:
            x_train, y_train = data[indices], labels[indices]
        else:
            x_val, y_val = data[indices], labels[indices]

    def encode(sequences):
        # Prepend the start token and shift the stored indices, then map
        # anything outside the top num_words onto the OOV token.
        shifted = [[START_CHAR] + [w + INDEX_FROM for w in s] for s in sequences]
        return [[w if w < args.num_words else OOV_CHAR for w in s] for s in shifted]

    x_train = pad_pre(encode(x_train), args.maxlen)
    x_val = pad_pre(encode(x_val), args.maxlen)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    print(f"{len(x_train)} training sequences, {len(x_val)} validation sequences")
    print(f"vocabulary capped at {args.num_words}, sequences padded to {args.maxlen}")

    for name, array in [
        ("x_train", x_train),
        ("y_train", y_train),
        ("x_val", x_val),
        ("y_val", y_val),
    ]:
        path = os.path.join(args.out, f"{name}.bin")
        array.tofile(path)
        print(f"wrote {path} ({array.nbytes} bytes)")


if __name__ == "__main__":
    main()
