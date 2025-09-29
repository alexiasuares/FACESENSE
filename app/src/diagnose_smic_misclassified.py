"""Diagnóstico rápido para SMIC

Carrega X_train/y_train/X_test/y_test processados, treina um LSTM leve por poucas
épocas, avalia no conjunto de teste, gera `test_errors.csv` e salva algumas
amostras mal classificadas (frames ou vetores) em `runs/diagnostics/smic-<ts>`.

Use para inspecionar exemplos mal classificados antes de alterar o modelo.
"""
from pathlib import Path
import numpy as np
import os
from collections import Counter
from datetime import datetime
import json

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


def find_data_dir(root: Path):
    candidates = [
        root / 'facesence' / 'data' / 'smic',
        root / 'data' / 'smic',
        root / 'data' / 'SMIC' / 'processed',
        root / 'data' / 'processed_data',
        root / 'app' / 'data' / 'SMIC' / 'processed',
        root / 'app' / 'data' / 'SMIC' / 'processed' / 'smic_processed',
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback: search under data/
    for p in (root / 'data').rglob('*'):
        if p.is_dir() and p.name.lower() == 'smic':
            return p
    raise FileNotFoundError('Could not locate SMIC data directory (X_*.npy)')


def load_data(data_dir: Path):
    def lp(name):
        p = data_dir / name
        if p.exists():
            return np.load(p)
        # try common subdir
        alt = data_dir / 'smic_processed' / name
        if alt.exists():
            return np.load(alt)
        raise FileNotFoundError(f'Missing expected file: {p} (also checked {alt})')

    X_train = lp('X_train.npy')
    y_train = lp('y_train.npy')
    X_test = lp('X_test.npy')
    y_test = lp('y_test.npy')
    return X_train, y_train, X_test, y_test


def build_model(input_shape, n_classes, lstm_units=32):
    m = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        LSTM(lstm_units, return_sequences=False),
        Dropout(0.4),
        Dense(n_classes, activation='softmax')
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return m


def save_misclassified_samples(run_dir: Path, X_raw, X_proc, y_true, y_pred, probs, max_examples=20):
    import pandas as pd

    err_idx = np.where(y_true != y_pred)[0]
    rows = []
    for i in err_idx:
        rows.append({'index': int(i), 'true': int(y_true[i]), 'pred': int(y_pred[i])})
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / 'test_errors.csv', index=False)

    # save a few visualizations
    imgs_dir = run_dir / 'misclassified_samples'
    imgs_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i in err_idx[:max_examples]:
        # if raw is 4D (N,T,H,W) we save a frame grid; if 3D (N,T,F) we plot the feature vector
        xr = X_raw[i]
        xp = X_proc[i]
        true = int(y_true[i]); pred = int(y_pred[i])
        if xr.ndim == 3 and xr.shape[1] > 1 and xr.shape[2] > 1:
            # assume (T,H,W)
            T, H, W = xr.shape
            cols = min(8, T)
            rows = int(np.ceil(T / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols*1.6, rows*1.6))
            axes = np.array(axes).reshape(-1)
            for j in range(rows*cols):
                ax = axes[j]
                if j < T:
                    ax.imshow(xr[j], cmap='gray')
                    ax.set_title(f'f{j}')
                ax.axis('off')
            plt.suptitle(f'idx={i} true={true} pred={pred}')
            plt.tight_layout()
            plt.savefig(imgs_dir / f'err_{i}_grid.png', dpi=150)
            plt.close()
        else:
            # plot processed feature vector
            fig, ax = plt.subplots(figsize=(6,2))
            ax.plot(xp.flatten())
            ax.set_title(f'idx={i} true={true} pred={pred}')
            ax.set_xlabel('feature')
            ax.set_ylabel('value')
            plt.tight_layout()
            plt.savefig(imgs_dir / f'err_{i}_feat.png', dpi=150)
            plt.close()
        saved += 1
    return df, saved


def main():
    ROOT = Path(__file__).resolve().parents[2]
    data_dir = find_data_dir(ROOT)
    print('Using data_dir =', data_dir)

    X_train_raw, y_train, X_test_raw, y_test = load_data(data_dir)
    print('Loaded shapes:', X_train_raw.shape, y_train.shape, X_test_raw.shape, y_test.shape)

    # keep raw test for visualization
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    # If images (N,T,H,W) convert to (N,T,H,W) -> we want raw as (N,T,H,W) for viz
    is_image = False
    if X_train.ndim == 4:
        is_image = True
        # convert to (N,T,H,W) if channels present
        # existing preprocess likely had (N,T,H,W) already
        pass

    # If images, flatten frames for model input
    if X_train.ndim == 4:
        N, T, H, W = X_train.shape
        F = H * W
        X_train_proc = X_train.reshape((N, T, F))
        X_test_proc = X_test.reshape((X_test.shape[0], X_test.shape[1], -1))
    else:
        X_train_proc = X_train
        X_test_proc = X_test

    # remap labels
    all_y = np.concatenate([y_train, y_test])
    ul = sorted(np.unique(all_y))
    mapping = {old: new for new, old in enumerate(ul)}
    y_train_m = np.array([mapping[int(v)] for v in y_train])
    y_test_m = np.array([mapping[int(v)] for v in y_test])
    num_classes = len(ul)
    print('num_classes =', num_classes, 'label mapping sample:', mapping)

    # normalization computed on train (ignore padding zeros if present)
    N, T, F = X_train_proc.shape
    flat = X_train_proc.reshape(-1, F)
    mask = (flat != 0)
    feat_sum = (flat * mask).sum(axis=0)
    feat_count = mask.sum(axis=0).clip(min=1)
    feat_mean = feat_sum / feat_count
    feat_var = ((flat - feat_mean) ** 2 * mask).sum(axis=0) / feat_count
    feat_std = np.sqrt(feat_var)
    feat_std[feat_std == 0] = 1.0

    def apply_norm(X):
        Xf = X.reshape(-1, X.shape[-1])
        M = (Xf != 0)
        return (((Xf - feat_mean) / feat_std) * M).reshape(X.shape)

    X_train_proc = apply_norm(X_train_proc)
    X_test_proc = apply_norm(X_test_proc)

    # one-hot
    y_train_oh = to_categorical(y_train_m, num_classes=num_classes)

    # class weight
    cw = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_train_m)
    class_weight = {i: float(w) for i, w in enumerate(cw)}

    # small model + quick train
    input_shape = (X_train_proc.shape[1], X_train_proc.shape[2])
    model = build_model(input_shape, num_classes, lstm_units=32)

    run_dir = ROOT / 'runs' / 'diagnostics' / f'smic-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    run_dir.mkdir(parents=True, exist_ok=True)

    print('Training quick model (3-8 epochs) to obtain predictions...')
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)]
    model.fit(X_train_proc, y_train_oh, epochs=8, batch_size=16, class_weight=class_weight, callbacks=callbacks, verbose=1)

    proba = model.predict(X_test_proc, verbose=0)
    y_pred = np.argmax(proba, axis=1)
    acc = accuracy_score(y_test_m, y_pred)
    f1m = f1_score(y_test_m, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test_m, y_pred)

    results = {'acc': float(acc), 'f1_macro': float(f1m), 'cm': cm.tolist(), 'n_classes': int(num_classes)}
    with open(run_dir / 'quick_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    df_err, saved = save_misclassified_samples(run_dir, X_test_raw, X_test_proc, y_test_m, y_pred, proba, max_examples=24)
    print('Quick eval results:', results)
    print(f'Saved {len(df_err)} error rows and {saved} sample images/plots in', run_dir)


if __name__ == '__main__':
    main()
