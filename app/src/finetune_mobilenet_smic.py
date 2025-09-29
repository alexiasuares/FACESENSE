"""Fine-tune MobileNetV2 on SMIC with options to unfreeze last N layers and use focal loss.
Saves results to runs/finetune_mobilenet/<ts>.
"""
from pathlib import Path
from datetime import datetime
import numpy as np
import json

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, LSTM
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def load_data(root: Path):
    cropped = root / 'data' / 'smic' / 'smic_cropped'
    base = root / 'data' / 'smic' / 'smic_processed'
    if cropped.exists():
        data_dir = cropped
    elif base.exists():
        data_dir = base
    else:
        data_dir = root / 'data' / 'SMIC' / 'processed' / 'smic_processed'

    X_tr = np.load(data_dir / 'X_train.npy')
    y_tr = np.load(data_dir / 'y_train.npy')
    X_te = np.load(data_dir / 'X_test.npy')
    y_te = np.load(data_dir / 'y_test.npy')
    return X_tr, y_tr, X_te, y_te


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


def build_model(T, H, W, n_classes, enc_dim=128, lstm_units=64, weights='imagenet', freeze_base=True, unfreeze_last=0):
    base = MobileNetV2(include_top=False, pooling='avg', input_shape=(H, W, 3), weights=weights)
    if freeze_base:
        base.trainable = False
    else:
        base.trainable = True

    if unfreeze_last > 0:
        # unfreeze last unfreeze_last layers
        for layer in base.layers[-unfreeze_last:]:
            layer.trainable = True

    inp = Input(shape=(T, H, W, 3))
    x = TimeDistributed(base)(inp)
    x = TimeDistributed(Dense(enc_dim, activation='relu'))(x)
    x = LSTM(lstm_units)(x)
    x = Dropout(0.4)(x)
    out = Dense(n_classes, activation='softmax')(x)
    model = Model(inp, out)
    return model


def main():
    ROOT = Path(__file__).resolve().parents[2]
    X_tr, y_tr, X_te, y_te = load_data(ROOT)
    print('Loaded shapes:', X_tr.shape, y_tr.shape, X_te.shape, y_te.shape)

    # to 3-ch and normalize
    X_tr = np.repeat(X_tr[..., np.newaxis], 3, axis=-1).astype('float32')
    X_te = np.repeat(X_te[..., np.newaxis], 3, axis=-1).astype('float32')
    X_tr = (X_tr / 127.5) - 1.0
    X_te = (X_te / 127.5) - 1.0

    # remap labels
    all_y = np.concatenate([y_tr, y_te])
    ul = sorted(np.unique(all_y))
    mapping = {old: new for new, old in enumerate(ul)}
    y_tr_m = np.array([mapping[int(v)] for v in y_tr])
    y_te_m = np.array([mapping[int(v)] for v in y_te])
    num_classes = len(ul)
    y_tr_oh = tf.keras.utils.to_categorical(y_tr_m, num_classes=num_classes)

    cw = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_tr_m)
    class_weight = {i: float(w) for i, w in enumerate(cw)}

    model = build_model(X_tr.shape[1], X_tr.shape[2], X_tr.shape[3], num_classes, weights='imagenet', freeze_base=True, unfreeze_last=10)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    run_dir = ROOT / 'runs' / 'finetune_mobilenet' / f'smic-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    run_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(run_dir / 'best.keras'), monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(X_tr, y_tr_oh, validation_split=0.15, epochs=12, batch_size=8, class_weight=class_weight, callbacks=callbacks, verbose=1)

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

    print('Finetune results saved in', run_dir)


if __name__ == '__main__':
    main()
