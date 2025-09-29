"""Two-stage MobileNetV2 fine-tune for SMIC.

Stage 1: freeze MobileNet base, train projection + LSTM.
Stage 2: unfreeze last N layers of base and fine-tune at lower LR.

If needed, creates a resized dataset at data/smic/smic_cropped_224 from
data/smic/smic_cropped or the processed folder.
"""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, LSTM
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def ensure_resized(src_dir: Path, dst_dir: Path, size=224):
    if dst_dir.exists():
        print('Resized dir exists:', dst_dir)
        return
    print('Creating resized dataset at', dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for arr in ['X_train.npy', 'X_test.npy']:
        src = src_dir / arr
        if not src.exists():
            raise FileNotFoundError(f'missing {src}')
        X = np.load(src)
        N, T, H, W = X.shape
        out = np.zeros((N, T, size, size), dtype=X.dtype)
        for i in range(N):
            for t in range(T):
                out[i, t] = cv2.resize(X[i, t], (size, size), interpolation=cv2.INTER_LINEAR)
        np.save(dst_dir / arr, out)
    # copy labels
    for lbl in ['y_train.npy', 'y_test.npy']:
        src = src_dir / lbl
        if src.exists():
            dst = dst_dir / lbl
            np.save(dst, np.load(src))


def load_data(data_root: Path, prefer_224=True):
    cropped224 = data_root / 'data' / 'smic' / 'smic_cropped_224'
    cropped = data_root / 'data' / 'smic' / 'smic_cropped'
    processed = data_root / 'data' / 'smic' / 'smic_processed'

    if prefer_224 and cropped224.exists():
        data_dir = cropped224
    elif cropped.exists():
        data_dir = cropped
    elif processed.exists():
        data_dir = processed
    else:
        raise FileNotFoundError('No SMIC data found in expected locations')

    X_tr = np.load(data_dir / 'X_train.npy')
    y_tr = np.load(data_dir / 'y_train.npy')
    X_te = np.load(data_dir / 'X_test.npy')
    y_te = np.load(data_dir / 'y_test.npy')
    return X_tr, y_tr, X_te, y_te, data_dir


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
    # prepare resized dataset (224x224) if missing
    cropped_dir = ROOT / 'data' / 'smic' / 'smic_cropped'
    resized_dir = ROOT / 'data' / 'smic' / 'smic_cropped_224'
    if not resized_dir.exists():
        if cropped_dir.exists():
            ensure_resized(cropped_dir, resized_dir, size=224)
        else:
            # try to create cropped from processed first
            processed_dir = ROOT / 'data' / 'smic' / 'smic_processed'
            if processed_dir.exists():
                # center crop then resize
                Xtr = np.load(processed_dir / 'X_train.npy')
                Xte = np.load(processed_dir / 'X_test.npy')
                # create an intermediate cropped dir
                cropped_dir.mkdir(parents=True, exist_ok=True)
                # center crop to square using min(H,W)
                for name, X in [('X_train.npy', Xtr), ('X_test.npy', Xte)]:
                    N, T, H, W = X.shape
                    m = min(H, W)
                    out_sq = np.zeros((N, T, m, m), dtype=X.dtype)
                    for i in range(N):
                        for t in range(T):
                            frm = X[i, t]
                            y0 = (H - m) // 2
                            x0 = (W - m) // 2
                            out_sq[i, t] = frm[y0:y0+m, x0:x0+m]
                    np.save(cropped_dir / name, out_sq)
                # copy labels
                for lbl in ['y_train.npy', 'y_test.npy']:
                    np.save(cropped_dir / lbl, np.load(processed_dir / lbl))
                ensure_resized(cropped_dir, resized_dir, size=224)

    # load resized data
    X_tr, y_tr, X_te, y_te, used_dir = load_data(ROOT, prefer_224=True)
    print('Using data dir:', used_dir)
    print('Loaded shapes:', X_tr.shape, y_tr.shape, X_te.shape, y_te.shape)

    # to 3-ch and normalize to [-1,1]
    X_tr = np.repeat(X_tr[..., np.newaxis], 3, axis=-1).astype('float32')
    X_te = np.repeat(X_te[..., np.newaxis], 3, axis=-1).astype('float32')
    X_tr = (X_tr / 127.5) - 1.0
    X_te = (X_te / 127.5) - 1.0

    # remap labels to 0..C-1
    all_y = np.concatenate([y_tr, y_te])
    ul = sorted(np.unique(all_y))
    mapping = {old: new for new, old in enumerate(ul)}
    y_tr_m = np.array([mapping[int(v)] for v in y_tr])
    y_te_m = np.array([mapping[int(v)] for v in y_te])
    num_classes = len(ul)
    y_tr_oh = tf.keras.utils.to_categorical(y_tr_m, num_classes=num_classes)

    cw = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_tr_m)
    class_weight = {i: float(w) for i, w in enumerate(cw)}

    # base MobileNetV2
    H, W = X_tr.shape[2], X_tr.shape[3]
    base = MobileNetV2(include_top=False, pooling='avg', input_shape=(H, W, 3), weights='imagenet')

    run_dir = ROOT / 'runs' / 'finetune_mobilenet_two_stage' / f'smic-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    run_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: freeze base
    for layer in base.layers:
        layer.trainable = False
    model_stage1 = build_model(base, X_tr.shape[1], H, W, num_classes)
    model_stage1.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    print('Stage1 model summary:')
    model_stage1.summary()

    cb_stage1 = [
        tf.keras.callbacks.ModelCheckpoint(str(run_dir / 'stage1_best.keras'), monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ]

    model_stage1.fit(X_tr, y_tr_oh, validation_split=0.15, epochs=12, batch_size=8, class_weight=class_weight, callbacks=cb_stage1, verbose=1)

    # Stage 2: unfreeze last N layers of base
    unfreeze_last = 30
    for layer in base.layers[:-unfreeze_last]:
        layer.trainable = False
    for layer in base.layers[-unfreeze_last:]:
        layer.trainable = True

    model_stage2 = build_model(base, X_tr.shape[1], H, W, num_classes)
    # load stage1 weights if available
    stage1_ckpt = run_dir / 'stage1_best.keras'
    if stage1_ckpt.exists():
        try:
            model_stage2.load_weights(stage1_ckpt)
            print('Loaded stage1 weights into stage2 model')
        except Exception as e:
            print('Warning: could not load stage1 weights:', e)

    model_stage2.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    print('Stage2 model summary:')
    model_stage2.summary()

    cb_stage2 = [
        tf.keras.callbacks.ModelCheckpoint(str(run_dir / 'stage2_best.keras'), monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ]

    model_stage2.fit(X_tr, y_tr_oh, validation_split=0.15, epochs=12, batch_size=8, class_weight=class_weight, callbacks=cb_stage2, verbose=1)

    # evaluate
    proba = model_stage2.predict(X_te, verbose=0)
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

    print('Two-stage finetune results saved in', run_dir)


if __name__ == '__main__':
    main()
