"""Treino MobileNetV2 LSTM SMIC com data augmentation, focal loss ajustado, unfreeze_last=40.
Salva resultados em runs/finetune_mobilenet_augmented/<ts>.
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

def focal_loss(gamma=2.0, alpha=0.75):
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

def augment_batch(X, y):
    # X: (N, T, H, W), y: (N,)
    # Garante que todas as imagens fiquem com shape (224,224)
    X_aug = np.empty(X.shape, dtype=X.dtype)
    for i in range(X.shape[0]):
        for t in range(X.shape[1]):
            img = X[i, t]
            # Flip horizontal
            if np.random.rand() < 0.5:
                img = np.fliplr(img)
            # Random brightness
            if np.random.rand() < 0.5:
                img = np.clip(img + np.random.uniform(-20, 20), 0, 255)
            # Random crop (224x224 -> 200x200)
            if img.shape[0] > 200 and img.shape[1] > 200 and np.random.rand() < 0.5:
                y0 = np.random.randint(0, img.shape[0] - 200)
                x0 = np.random.randint(0, img.shape[1] - 200)
                img = img[y0:y0+200, x0:x0+200]
            # Resize sempre para (224,224)
            img = tf.image.resize(img[..., np.newaxis], (224, 224)).numpy().squeeze()
            X_aug[i, t] = img
    return X_aug, y

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
    print('Loaded shapes before augmentation:', X_tr.shape, y_tr.shape)
    # remap labels
    all_y = np.concatenate([y_tr, y_te])
    ul = sorted(np.unique(all_y))
    mapping = {old: new for new, old in enumerate(ul)}
    y_tr_m = np.array([mapping[int(v)] for v in y_tr])
    y_te_m = np.array([mapping[int(v)] for v in y_te])
    num_classes = len(ul)
    # Data augmentation
    X_tr_aug, y_tr_aug = augment_batch(X_tr, y_tr_m)
    print('Loaded shapes after augmentation:', X_tr_aug.shape, y_tr_aug.shape)
    # to 3-ch and normalize
    X_tr_aug = np.repeat(X_tr_aug[..., np.newaxis], 3, axis=-1).astype('float32')
    X_te = np.repeat(X_te[..., np.newaxis], 3, axis=-1).astype('float32')
    X_tr_aug = (X_tr_aug / 127.5) - 1.0
    X_te = (X_te / 127.5) - 1.0
    y_tr_oh = tf.keras.utils.to_categorical(y_tr_aug, num_classes=num_classes)
    cw = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_tr_aug)
    class_weight = {i: float(w) for i, w in enumerate(cw)}
    H, W = X_tr_aug.shape[2], X_tr_aug.shape[3]
    base = MobileNetV2(include_top=False, pooling='avg', input_shape=(H, W, 3), weights='imagenet')
    # Stage1: freeze base
    for layer in base.layers:
        layer.trainable = False
    model1 = build_model(base, X_tr_aug.shape[1], H, W, num_classes)
    model1.compile(optimizer=Adam(1e-4), loss=focal_loss(2.0, 0.75), metrics=['accuracy'])
    cb1 = [
        tf.keras.callbacks.ModelCheckpoint('stage1_best.keras', monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    ]
    model1.fit(X_tr_aug, y_tr_oh, validation_split=0.15, epochs=10, batch_size=8, class_weight=class_weight, callbacks=cb1, verbose=1)
    # Stage2: unfreeze last 40
    for layer in base.layers[:-40]:
        layer.trainable = False
    for layer in base.layers[-40:]:
        layer.trainable = True
    model2 = build_model(base, X_tr_aug.shape[1], H, W, num_classes)
    try:
        model2.load_weights('stage1_best.keras')
    except Exception as e:
        print('Could not load stage1 weights:', e)
    model2.compile(optimizer=Adam(1e-5), loss=focal_loss(2.0, 0.75), metrics=['accuracy'])
    cb2 = [
        tf.keras.callbacks.ModelCheckpoint('stage2_best.keras', monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    ]
    model2.fit(X_tr_aug, y_tr_oh, validation_split=0.15, epochs=20, batch_size=8, class_weight=class_weight, callbacks=cb2, verbose=1)
    proba = model2.predict(X_te, verbose=0)
    preds = np.argmax(proba, axis=1)
    acc = accuracy_score(y_te_m, preds)
    f1m = f1_score(y_te_m, preds, average='macro', zero_division=0)
    cm = confusion_matrix(y_te_m, preds)
    crep = classification_report(y_te_m, preds, zero_division=0)
    run_dir = ROOT / 'runs' / 'finetune_mobilenet_augmented' / f'smic-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump({'acc': float(acc), 'f1_macro': float(f1m), 'cm': cm.tolist(), 'n_classes': int(num_classes)}, f, indent=2)
    with open(run_dir / 'classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(crep)
    np.save(run_dir / 'proba.npy', proba)
    np.save(run_dir / 'preds.npy', preds)
    np.save(run_dir / 'y_test.npy', y_te_m)
    print('Treino com augmentation salvo em', run_dir)

if __name__ == '__main__':
    main()
