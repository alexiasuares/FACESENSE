"""Center-crop SMIC sequences as a cheap proxy for face crop.

Produces a new folder under data/smic/smic_cropped with X_train.npy, X_test.npy, y_train.npy, y_test.npy
by center-cropping each frame to a smaller size and resizing back to target.
"""
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data' / 'smic' / 'smic_processed'
OUT_DIR = ROOT / 'data' / 'smic' / 'smic_cropped'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def center_crop_frame(frame, out_h, out_w, margin=0.85):
    # frame: HxW (grayscale)
    H, W = frame.shape[:2]
    ch = int(H * margin)
    cw = int(W * margin)
    top = (H - ch) // 2
    left = (W - cw) // 2
    crop = frame[top:top+ch, left:left+cw]
    resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return resized


def process_file(name):
    p = DATA_DIR / name
    if not p.exists():
        alt = ROOT / 'data' / 'SMIC' / 'processed' / name
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(p)
    arr = np.load(p)
    # arr shape: (N, T, H, W) or (N,T,F)
    if arr.ndim != 4:
        print('Skipping non-image file', p)
        return None
    N, T, H, W = arr.shape
    out = np.zeros_like(arr)
    for i in range(N):
        for t in range(T):
            out[i, t] = center_crop_frame(arr[i, t], H, W)
    return out


def main():
    print('Processing X_train...')
    X_train = process_file('X_train.npy')
    print('Processing X_test...')
    X_test = process_file('X_test.npy')
    print('Copying labels...')
    y_train = np.load(DATA_DIR / 'y_train.npy')
    y_test = np.load(DATA_DIR / 'y_test.npy')
    print('Saving to', OUT_DIR)
    np.save(OUT_DIR / 'X_train.npy', X_train)
    np.save(OUT_DIR / 'X_test.npy', X_test)
    np.save(OUT_DIR / 'y_train.npy', y_train)
    np.save(OUT_DIR / 'y_test.npy', y_test)


if __name__ == '__main__':
    main()
