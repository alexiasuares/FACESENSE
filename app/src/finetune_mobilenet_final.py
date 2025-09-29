"""Treino final MobileNetV2 LSTM SMIC: unfreeze_last=30, focal loss, 30 epochs.
Salva resultados completos em runs/finetune_mobilenet_final/<ts>.
"""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
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
        raise FileNotFoundError('Crie o dataset redimensionado primeiro.')
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

def main():
    ROOT = Path(__file__).resolve().parents[2]
    X_tr, y_tr, X_te, y_te = load_resized(ROOT)
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
    # Oversampling
    from imblearn.over_sampling import RandomOverSampler
    X_tr_rs = X_tr.reshape((X_tr.shape[0], -1))
    ros = RandomOverSampler(random_state=42)
    X_tr_rs, y_tr_m_rs = ros.fit_resample(X_tr_rs, y_tr_m)
    X_tr = X_tr_rs.reshape((-1, X_tr.shape[1], X_tr.shape[2], X_tr.shape[3], X_tr.shape[4]))
    y_tr_m = y_tr_m_rs
    y_tr_oh = tf.keras.utils.to_categorical(y_tr_m, num_classes=num_classes)
    cw = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_tr_m)
    class_weight = {i: float(w) for i, w in enumerate(cw)}
    H, W = X_tr.shape[2], X_tr.shape[3]
    base = MobileNetV2(include_top=False, pooling='avg', input_shape=(H, W, 3), weights='imagenet')
    # Stage1: freeze base
    for layer in base.layers:
        layer.trainable = False
    model1 = build_model(base, X_tr.shape[1], H, W, num_classes)
    model1.compile(optimizer=Adam(1e-4), loss=focal_loss(2.0, 0.25), metrics=['accuracy'])
    cb1 = [
        tf.keras.callbacks.ModelCheckpoint('stage1_best.keras', monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ]
    model1.fit(X_tr, y_tr_oh, validation_split=0.15, epochs=12, batch_size=8, class_weight=class_weight, callbacks=cb1, verbose=1)
    # Stage2: unfreeze last 30
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers[-30:]:
        layer.trainable = True
    model2 = build_model(base, X_tr.shape[1], H, W, num_classes)
    try:
        model2.load_weights('stage1_best.keras')
    except Exception as e:
        print('Could not load stage1 weights:', e)
    model2.compile(optimizer=Adam(1e-5), loss=focal_loss(2.0, 0.25), metrics=['accuracy'])
    cb2 = [
        tf.keras.callbacks.ModelCheckpoint('stage2_best.keras', monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    model2.fit(X_tr, y_tr_oh, validation_split=0.15, epochs=30, batch_size=8, class_weight=class_weight, callbacks=cb2, verbose=1)
    proba = model2.predict(X_te, verbose=0)
    preds = np.argmax(proba, axis=1)
    acc = accuracy_score(y_te_m, preds)
    f1m = f1_score(y_te_m, preds, average='macro', zero_division=0)
    cm = confusion_matrix(y_te_m, preds)
    crep = classification_report(y_te_m, preds, zero_division=0)
    run_dir = ROOT / 'runs' / 'finetune_mobilenet_final' / f'smic-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump({'acc': float(acc), 'f1_macro': float(f1m), 'cm': cm.tolist(), 'n_classes': int(num_classes)}, f, indent=2)
    with open(run_dir / 'classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(crep)
    np.save(run_dir / 'proba.npy', proba)
    np.save(run_dir / 'preds.npy', preds)
    np.save(run_dir / 'y_test.npy', y_te_m)
    print('Treino final salvo em', run_dir)

if __name__ == '__main__':
    main()
