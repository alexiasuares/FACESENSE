"""MobileNetV2 per-frame encoder + LSTM prototype

Aplica MobileNetV2 (por-frame via TimeDistributed) seguido de LSTM. Tenta carregar
weights='imagenet' e cai para weights=None se não disponível. Treina rápido e salva
resultados em `runs/encoder_mobilenet/<ts>/`.
"""
from pathlib import Path
from datetime import datetime
import numpy as np
import json
import sys

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, LSTM
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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


def build_model_mobilenet(T, H, W, enc_dim, lstm_units, n_classes, weights='imagenet', freeze_base=True):
    # base MobileNetV2 per-frame
    try:
        base = MobileNetV2(include_top=False, pooling='avg', input_shape=(H, W, 3), weights=weights)
    except Exception as e:
        print('Failed to load imagenet weights, falling back to weights=None:', e)
        base = MobileNetV2(include_top=False, pooling='avg', input_shape=(H, W, 3), weights=None)

    if freeze_base:
        base.trainable = False

    inp = Input(shape=(T, H, W, 3), name='seq_input')
    x = TimeDistributed(base)(inp)          # (None, T, feat)
    x = TimeDistributed(Dense(enc_dim, activation='relu'))(x)
    x = LSTM(lstm_units)(x)
    x = Dropout(0.4)(x)
    out = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main(epochs=12, batch_size=8, enc_dim=128, lstm_units=64):
    ROOT = Path(__file__).resolve().parents[2]
    data_dir = find_data_dir(ROOT)
    print('Using data_dir =', data_dir)
    X_tr_raw, y_tr, X_te_raw, y_te = load_arrays(data_dir)
    print('Loaded shapes:', X_tr_raw.shape, y_tr.shape, X_te_raw.shape, y_te.shape)

    if X_tr_raw.ndim != 4:
        raise RuntimeError('Expected image tensors (N,T,H,W)')

    N, T, H, W = X_tr_raw.shape

    # convert grayscale to 3-ch by repeating
    X_tr = np.repeat(X_tr_raw[..., np.newaxis], 3, axis=-1)
    X_te = np.repeat(X_te_raw[..., np.newaxis], 3, axis=-1)

    # normalize to MobileNet expected range [-1,1]
    X_tr = (X_tr.astype('float32') / 127.5) - 1.0
    X_te = (X_te.astype('float32') / 127.5) - 1.0

    # remap labels
    all_y = np.concatenate([y_tr, y_te])
    ul = sorted(np.unique(all_y))
    mapping = {old: new for new, old in enumerate(ul)}
    y_tr_m = np.array([mapping[int(v)] for v in y_tr])
    y_te_m = np.array([mapping[int(v)] for v in y_te])
    num_classes = len(ul)

    y_tr_oh = to_categorical(y_tr_m, num_classes=num_classes)

    cw = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_tr_m)
    class_weight = {i: float(w) for i, w in enumerate(cw)}

    print('Building model (MobileNetV2 encoder) — this may be slow/large')
    model = build_model_mobilenet(T, H, W, enc_dim=enc_dim, lstm_units=lstm_units, n_classes=num_classes, weights='imagenet')
    model.summary()

    run_dir = ROOT / 'runs' / 'encoder_mobilenet' / f'smic-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    run_dir.mkdir(parents=True, exist_ok=True)

    # callbacks: checkpoint best, reduce LR, early stop
    from tensorflow.keras.callbacks import ModelCheckpoint
    callbacks = [
        ModelCheckpoint(str(run_dir / 'best.keras'), monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]

    history = model.fit(
        X_tr, y_tr_oh,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

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

    print('Saved results to', run_dir)
    print('Accuracy:', results['acc'], 'F1_macro:', results['f1_macro'])


if __name__ == '__main__':
    # default quick run
    main(epochs=4, batch_size=8, enc_dim=128, lstm_units=64)
