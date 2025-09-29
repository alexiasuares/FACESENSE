"""Run two-stage fine-tune experiments with focal loss and different unfreeze sizes.

This script will run two experiments:
 - unfreeze_last = 30
 - unfreeze_last = 15

For each experiment it will perform Stage1 (freeze base) and Stage2 (unfreeze last N)
using focal loss, then save results and a classification report.
"""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, LSTM
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = 1e-9
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        fl = weight * ce
        return tf.reduce_sum(fl, axis=-1)
    return loss


def load_resized(root: Path):
    cropped224 = root / 'data' / 'smic' / 'smic_cropped_224'
    if not cropped224.exists():
        raise FileNotFoundError('Please create resized dataset first (run two-stage script created it).')
    X_tr = np.load(cropped224 / 'X_train.npy')
    y_tr = np.load(cropped224 / 'y_train.npy')
    X_te = np.load(cropped224 / 'X_test.npy')
    y_te = np.load(cropped224 / 'y_test.npy')
    return X_tr, y_tr, X_te, y_te


def build_model(base, T, H, W, n_classes, enc_dim=128, lstm_units=64):
    inp = Input(shape=(T, H, W, 3))
    x = TimeDistributed(base)(inp)
    x = TimeDistributed(Dense(enc_dim, activation='relu'))(x)
    x = LSTM(lstm_units)(x)
    x = Dropout(0.4)(x)
    out = Dense(n_classes, activation='softmax')(x)
    return Model(inp, out)


def run_experiment(root: Path, unfreeze_last: int, gamma=2.0, alpha=0.25):
    X_tr, y_tr, X_te, y_te = load_resized(root)
    print('Loaded shapes:', X_tr.shape, y_tr.shape, X_te.shape, y_te.shape)

    X_tr = np.repeat(X_tr[..., np.newaxis], 3, axis=-1).astype('float32')
    X_te = np.repeat(X_te[..., np.newaxis], 3, axis=-1).astype('float32')
    X_tr = (X_tr / 127.5) - 1.0
    X_te = (X_te / 127.5) - 1.0

    all_y = np.concatenate([y_tr, y_te])
    ul = sorted(np.unique(all_y))
    mapping = {old: new for new, old in enumerate(ul)}
    y_tr_m = np.array([mapping[int(v)] for v in y_tr])
    y_te_m = np.array([mapping[int(v)] for v in y_te])
    num_classes = len(ul)
    y_tr_oh = tf.keras.utils.to_categorical(y_tr_m, num_classes=num_classes)

    cw = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_tr_m)
    class_weight = {i: float(w) for i, w in enumerate(cw)}

    H, W = X_tr.shape[2], X_tr.shape[3]
    base = MobileNetV2(include_top=False, pooling='avg', input_shape=(H, W, 3), weights='imagenet')

    run_dir = root / 'runs' / 'finetune_mobilenet_focal' / f'smic-{datetime.now().strftime("%Y%m%d-%H%M%S")}-unf{unfreeze_last}'
    run_dir.mkdir(parents=True, exist_ok=True)

    # Stage1: freeze all base
    for layer in base.layers:
        layer.trainable = False
    model1 = build_model(base, X_tr.shape[1], H, W, num_classes)
    model1.compile(optimizer=Adam(1e-4), loss=focal_loss(gamma, alpha), metrics=['accuracy'])
    print('Stage1 (focal) training, unfreeze_last=', unfreeze_last)
    cb1 = [
        tf.keras.callbacks.ModelCheckpoint(str(run_dir / 'stage1_best.keras'), monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ]
    model1.fit(X_tr, y_tr_oh, validation_split=0.15, epochs=12, batch_size=8, class_weight=class_weight, callbacks=cb1, verbose=1)

    # Stage2: unfreeze last N
    for layer in base.layers[:-unfreeze_last]:
        layer.trainable = False
    for layer in base.layers[-unfreeze_last:]:
        layer.trainable = True
    model2 = build_model(base, X_tr.shape[1], H, W, num_classes)
    # try to load stage1 weights
    st1 = run_dir / 'stage1_best.keras'
    if st1.exists():
        try:
            model2.load_weights(st1)
        except Exception as e:
            print('Could not load stage1 weights:', e)
    model2.compile(optimizer=Adam(1e-5), loss=focal_loss(gamma, alpha), metrics=['accuracy'])
    cb2 = [
        tf.keras.callbacks.ModelCheckpoint(str(run_dir / 'stage2_best.keras'), monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ]
    model2.fit(X_tr, y_tr_oh, validation_split=0.15, epochs=12, batch_size=8, class_weight=class_weight, callbacks=cb2, verbose=1)

    proba = model2.predict(X_te, verbose=0)
    preds = np.argmax(proba, axis=1)
    acc = accuracy_score(y_te_m, preds)
    f1m = f1_score(y_te_m, preds, average='macro', zero_division=0)
    cm = confusion_matrix(y_te_m, preds)
    crep = classification_report(y_te_m, preds, zero_division=0)

    results = {'acc': float(acc), 'f1_macro': float(f1m), 'cm': cm.tolist(), 'n_classes': int(num_classes)}
    with open(run_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    with open(run_dir / 'classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(crep)

    np.save(run_dir / 'proba.npy', proba)
    np.save(run_dir / 'preds.npy', preds)
    np.save(run_dir / 'y_test.npy', y_te_m)

    print('Saved experiment to', run_dir)
    return run_dir


def main():
    ROOT = Path(__file__).resolve().parents[2]
    # three experiments: 30, 20, 15
    runs = []
    for unf in [30, 20, 15]:
        runs.append(run_experiment(ROOT, unf))
    print('All experiments done. Runs:')
    for r in runs:
        print('-', r)


if __name__ == '__main__':
    main()
