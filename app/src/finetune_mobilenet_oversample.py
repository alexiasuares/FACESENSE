"""Treino MobileNetV2 LSTM SMIC com oversampling das classes minoritárias, focal loss ajustado, early stopping agressivo.
Salva resultados em runs/finetune_mobilenet_oversample/<ts>.
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

def focal_loss(gamma=1.5, alpha=0.5):
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

def oversample_minority(X, y, classes=[0,2,3]):
    # oversample classes in 'classes' to match the majority class
    counts = {c: np.sum(y==c) for c in np.unique(y)}
    max_count = max(counts.values())
    X_new, y_new = [X], [y]
    for c in classes:
        idx = np.where(y==c)[0]
        n_needed = max_count - len(idx)
        if n_needed > 0:
            reps = np.random.choice(idx, n_needed, replace=True)
            X_new.append(X[reps])
            y_new.append(y[reps])
    X_cat = np.concatenate(X_new, axis=0)
    y_cat = np.concatenate(y_new, axis=0)
    # shuffle
    perm = np.random.permutation(len(y_cat))
    return X_cat[perm], y_cat[perm]

def build_model(base, T, H, W, n_classes, enc_dim=128, lstm_units=64):
    inp = Input(shape=(T, H, W, 3))
    x = TimeDistributed(base)(inp)
    x = TimeDistributed(Dense(enc_dim, activation='relu'))(x)
    x = LSTM(lstm_units)(x)
    x = Dropout(0.4)(x)
    out = Dense(n_classes, activation='softmax')(x)
    return Model(inp, out)

def main():
    try:
        ROOT = Path(__file__).resolve().parents[2]
        X_tr, y_tr, X_te, y_te = load_resized(ROOT)
        print('Loaded shapes before oversample:', X_tr.shape, y_tr.shape)
        # remap labels
        all_y = np.concatenate([y_tr, y_te])
        ul = sorted(np.unique(all_y))
        mapping = {old: new for new, old in enumerate(ul)}
        y_tr_m = np.array([mapping[int(v)] for v in y_tr])
        y_te_m = np.array([mapping[int(v)] for v in y_te])
        num_classes = len(ul)
        print('Iniciando oversample...')
        X_tr_os, y_tr_os = oversample_minority(X_tr, y_tr_m, classes=[0, 2, 3])
        print('Loaded shapes after oversample:', X_tr_os.shape, y_tr_os.shape)
        # to 3-ch and normalize
        print('Normalizando dados...')
        X_tr_os = np.repeat(X_tr_os[..., np.newaxis], 3, axis=-1).astype('float32')
        X_te = np.repeat(X_te[..., np.newaxis], 3, axis=-1).astype('float32')
        X_tr_os = (X_tr_os / 127.5) - 1.0
        X_te = (X_te / 127.5) - 1.0
        y_tr_oh = tf.keras.utils.to_categorical(y_tr_os, num_classes=num_classes)
        print('Calculando class_weight...')
        cw = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_tr_os)
        class_weight = {i: float(w) for i, w in enumerate(cw)}
        H, W = X_tr_os.shape[2], X_tr_os.shape[3]
        print('Criando modelo base MobileNetV2...')
        base = MobileNetV2(include_top=False, pooling='avg', input_shape=(H, W, 3), weights='imagenet')
        # Stage1: freeze base
        for layer in base.layers:
            layer.trainable = False
        print('Compilando modelo stage1...')
        model1 = build_model(base, X_tr_os.shape[1], H, W, num_classes)
        model1.compile(optimizer=Adam(1e-4), loss=focal_loss(1.5, 0.5), metrics=['accuracy'])
        cb1 = [
            tf.keras.callbacks.ModelCheckpoint('stage1_best.keras', monitor='val_loss', save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]
        print('Treinando stage1...')
        model1.fit(X_tr_os, y_tr_oh, validation_split=0.15, epochs=12, batch_size=8, class_weight=class_weight, callbacks=cb1, verbose=1)
        print('Stage1 concluído. Preparando stage2...')
        for layer in base.layers[:-30]:
            layer.trainable = False
        for layer in base.layers[-30:]:
            layer.trainable = True
        model2 = build_model(base, X_tr_os.shape[1], H, W, num_classes)
        try:
            model2.load_weights('stage1_best.keras')
        except Exception as e:
            print('Could not load stage1 weights:', e)
        print('Compilando modelo stage2...')
        model2.compile(optimizer=Adam(1e-5), loss=focal_loss(1.5, 0.5), metrics=['accuracy'])
        cb2 = [
            tf.keras.callbacks.ModelCheckpoint('stage2_best.keras', monitor='val_loss', save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]
        print('Treinando stage2...')
        model2.fit(X_tr_os, y_tr_oh, validation_split=0.15, epochs=12, batch_size=8, class_weight=class_weight, callbacks=cb2, verbose=1)
        print('Stage2 concluído. Gerando métricas...')
        proba = model2.predict(X_te, verbose=0)
        preds = np.argmax(proba, axis=1)
        acc = accuracy_score(y_te_m, preds)
        f1m = f1_score(y_te_m, preds, average='macro', zero_division=0)
        cm = confusion_matrix(y_te_m, preds)
        crep = classification_report(y_te_m, preds, zero_division=0)
        run_dir = ROOT / 'runs' / 'finetune_mobilenet_oversample' / f'smic-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / 'results.json', 'w', encoding='utf-8') as f:
            json.dump({'acc': float(acc), 'f1_macro': float(f1m), 'cm': cm.tolist(), 'n_classes': int(num_classes)}, f, indent=2)
        with open(run_dir / 'classification_report.txt', 'w', encoding='utf-8') as f:
            f.write(crep)
        np.save(run_dir / 'proba.npy', proba)
        np.save(run_dir / 'preds.npy', preds)
        np.save(run_dir / 'y_test.npy', y_te_m)
        print('Treino oversample salvo em', run_dir)
    except Exception as err:
        print('ERRO durante execução:', err)

if __name__ == '__main__':
    main()
