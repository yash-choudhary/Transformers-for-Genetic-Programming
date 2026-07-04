"""
One-time migration: convert a .weights.h5 checkpoint (saved by tf.keras on Mac)
to the portable .npy format that loads reliably on all platforms.

Run this on the machine where the .weights.h5 loads correctly (i.e., Mac):

    python convert_checkpoint.py checkpoints/tsgp_epoch_2.weights.h5 checkpoints/tsgp_epoch_2.npy

Then copy the .npy file to Windows and resume training normally.
"""
import sys
import numpy as np


def convert(h5_path: str, npy_path: str) -> None:
    from tsgp.transformer_model import create_model
    model = create_model()
    model.load_weights(h5_path)
    np.save(npy_path, np.array(model.get_weights(), dtype=object), allow_pickle=True)
    print(f"Converted  {h5_path}")
    print(f"       →   {npy_path}")
    print(f"Weights saved: {len(model.get_weights())} arrays")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_checkpoint.py <source.weights.h5> <dest.npy>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
