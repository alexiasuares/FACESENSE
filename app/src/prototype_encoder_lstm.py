"""Protótipo: per-frame CNN encoder + LSTM

Cria um modelo que aplica um pequeno encoder CNN a cada frame (TimeDistributed)
e em seguida uma LSTM para classificar a sequência. Treina rápido e salva
resultados em `runs/encoder_prototype/<ts>/`.
"""
from pathlib import Path
from datetime import datetime
import numpy as np
import json
import os

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    TimeDistributed, Conv2D, MaxPool2D, GlobalAveragePooling2D,
    Dense, Dropout, LSTM, Masking, Reshape
)
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
    for p in (root / 'data').rglob('*'):
        if p.is_dir() and p.name.lower() == 'smic':
            return p
    raise FileNotFoundError('Could not locate SMIC data directory')


def load_arrays(data_dir: Path):
    def lp(name):
        p = data_dir / name
        if p.exists():
            return np.load(p)
        alt = data_dir / 'smic_processed' / name
        if alt.exists():
            return np.load(alt)
        raise FileNotFoundError(f'Missing {name} in {data_dir}')

    X_train = lp('X_train.npy')
    y_train = lp('y_train.npy')
    X_test = lp('X_test.npy')
    y_test = lp('y_test.npy')
    return X_train, y_train, X_test, y_test


def build_encoder_lstm(T, H, W, channels, enc_dim=128, lstm_units=32, n_classes=4):
    # Input shape: (T, H, W, C)
    inp = Input(shape=(T, H, W, channels), name='seq_input')
    x = TimeDistributed(Conv2D(16, 3, activation='relu', padding='same'))(inp)
    x = TimeDistributed(MaxPool2D(2))(x)
    x = TimeDistributed(Conv2D(32, 3, activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPool2D(2))(x)
    x = TimeDistributed(GlobalAveragePooling2D())(x)  # (T, features)
    x = TimeDistributed(Dense(enc_dim, activation='relu'))(x)  # per-frame encoding
    # optional masking if frames padded with zeros
    # LSTM expects (T, enc_dim)
    x = LSTM(lstm_units)(x)
    x = Dropout(0.4)(x)
    out = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def save_misclassified(run_dir: Path, X_raw, y_true, y_pred, max_examples=24):
    import matplotlib.pyplot as plt
    import pandas as pd
    err_idx = np.where(y_true != y_pred)[0]
    rows = []
    imgs_dir = run_dir / 'misclassified_samples'
    imgs_dir.mkdir(parents=True, exist_ok=True)
    for i in err_idx[:max_examples]:
        rows.append({'index': int(i), 'true': int(y_true[i]), 'pred': int(y_pred[i])})
        xr = X_raw[i]
        if xr.ndim == 3 and xr.shape[1] > 1 and xr.shape[2] > 1:
            T, H, W = xr.shape
            cols = min(8, T)
            rowsn = int(np.ceil(T / cols))
            fig, axes = plt.subplots(rowsn, cols, figsize=(cols*1.2, rowsn*1.2))
            axes = np.array(axes).reshape(-1)
            for j in range(rowsn*cols):
                ax = axes[j]
                if j < T:
                    ax.imshow(xr[j], cmap='gray')
                ax.axis('off')
            plt.suptitle(f'idx={i} true={int(y_true[i])} pred={int(y_pred[i])}')
            plt.tight_layout()
            plt.savefig(imgs_dir / f'err_{i}_grid.png', dpi=150)
            plt.close()
    pd.DataFrame(rows).to_csv(run_dir / 'test_errors.csv', index=False)


def main():
    ROOT = Path(__file__).resolve().parents[2]
    data_dir = find_data_dir(ROOT)
    print('Using data_dir =', data_dir)
    X_tr_raw, y_tr, X_te_raw, y_te = load_arrays(data_dir)
    print('Loaded shapes:', X_tr_raw.shape, y_tr.shape, X_te_raw.shape, y_te.shape)

    # ensure grayscale channel
    if X_tr_raw.ndim == 4:
        N, T, H, W = X_tr_raw.shape
        channels = 1
        X_tr = X_tr_raw.reshape((N, T, H, W))
        X_te = X_te_raw.reshape((X_te_raw.shape[0], X_te_raw.shape[1], X_te_raw.shape[2], X_te_raw.shape[3]))
    else:
        raise RuntimeError('Expected 4D image input (N,T,H,W)')

    # add channel axis
    X_tr = X_tr[..., np.newaxis]
    X_te = X_te[..., np.newaxis]

    # remap labels
    all_y = np.concatenate([y_tr, y_te])
    ul = sorted(np.unique(all_y))
    mapping = {old: new for new, old in enumerate(ul)}
    y_tr_m = np.array([mapping[int(v)] for v in y_tr])
    y_te_m = np.array([mapping[int(v)] for v in y_te])
    num_classes = len(ul)

    # normalization per feature (flatten spatial) ignoring zeros
    N, T, H, W, C = X_tr.shape
    flat = X_tr.reshape(-1, H*W*C)
    mask = (flat != 0)
    feat_sum = (flat * mask).sum(axis=0)
    feat_count = mask.sum(axis=0).clip(min=1)
    feat_mean = feat_sum / feat_count
    feat_var = ((flat - feat_mean) ** 2 * mask).sum(axis=0) / feat_count
    feat_std = np.sqrt(feat_var)
    feat_std[feat_std == 0] = 1.0

    def apply_norm_img(X):
        Xf = X.reshape(-1, H*W*C)
        M = (Xf != 0)
        Xn = (((Xf - feat_mean) / feat_std) * M).reshape(X.shape)
        return Xn

    X_tr = apply_norm_img(X_tr)
    X_te = apply_norm_img(X_te)

    y_tr_oh = to_categorical(y_tr_m, num_classes=num_classes)

    cw = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_tr_m)
    class_weight = {i: float(w) for i, w in enumerate(cw)}

    model = build_encoder_lstm(T, H, W, channels, enc_dim=64, lstm_units=64, n_classes=num_classes)
    print(model.summary())

    run_dir = ROOT / 'runs' / 'encoder_prototype' / f'smic-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    run_dir.mkdir(parents=True, exist_ok=True)

    # Augmentation & longer training
    import tensorflow as tf
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

    def augment_features(x, y):
        # small gaussian noise as a cheap augmentation in feature space
        x = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.01)
        return x, y

    ds = tf.data.Dataset.from_tensor_slices((X_tr.astype('float32'), y_tr_oh.astype('float32')))
    ds = ds.shuffle(512).map(lambda a, b: augment_features(a, b)).batch(16).prefetch(tf.data.AUTOTUNE)

    callbacks = [
        ModelCheckpoint(str(run_dir / 'best.keras'), monitor='loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor='loss', patience=6, restore_best_weights=True, verbose=1)
    ]

    model.fit(ds, epochs=20, callbacks=callbacks, verbose=1)

    proba = model.predict(X_te, verbose=0)
    preds = np.argmax(proba, axis=1)
    acc = accuracy_score(y_te_m, preds)
    f1m = f1_score(y_te_m, preds, average='macro', zero_division=0)
    cm = confusion_matrix(y_te_m, preds)

    results = {'acc': float(acc), 'f1_macro': float(f1m), 'cm': cm.tolist(), 'n_classes': int(num_classes)}
    with open(run_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    np.save(run_dir / 'proba.npy', proba)
    np.save(run_dir / 'preds.npy', preds)
    np.save(run_dir / 'y_test.npy', y_te_m)

    save_misclassified(run_dir, X_te.squeeze(-1), y_te_m, preds, max_examples=24)

    print('Results saved in', run_dir)
    print('Accuracy:', results['acc'], 'F1_macro:', results['f1_macro'])


if __name__ == '__main__':
    main()
