"""Analisa distribuição de probabilidades preditas por classe verdadeira.

Roda um treino rápido (mesma configuração do diagnóstico), gera predições e salva
`test_proba.npy` e `test_preds.npy` em `runs/analysis/smic-<ts>/`, além de imprimir
as probabilidades médias por classe verdadeira e os top pares true->pred.
"""
from pathlib import Path
from datetime import datetime
import numpy as np
import json
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
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


def build_model(input_shape, n_classes, units=32):
    m = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        LSTM(units, return_sequences=False),
        Dropout(0.4),
        Dense(n_classes, activation='softmax')
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return m


def main():
    ROOT = Path(__file__).resolve().parents[2]
    data_dir = find_data_dir(ROOT)
    print('Using data_dir =', data_dir)
    X_tr_raw, y_tr, X_te_raw, y_te = load_arrays(data_dir)
    print('Loaded shapes:', X_tr_raw.shape, y_tr.shape, X_te_raw.shape, y_te.shape)

    # preprocess similar to diagnosis
    if X_tr_raw.ndim == 4:
        N, T, H, W = X_tr_raw.shape
        F = H * W
        X_tr = X_tr_raw.reshape((N, T, F))
        X_te = X_te_raw.reshape((X_te_raw.shape[0], X_te_raw.shape[1], -1))
    else:
        X_tr = X_tr_raw
        X_te = X_te_raw

    # remap labels
    all_y = np.concatenate([y_tr, y_te])
    ul = sorted(np.unique(all_y))
    mapping = {old: new for new, old in enumerate(ul)}
    y_tr_m = np.array([mapping[int(v)] for v in y_tr])
    y_te_m = np.array([mapping[int(v)] for v in y_te])
    num_classes = len(ul)

    # normalization
    N, T, F = X_tr.shape
    flat = X_tr.reshape(-1, F)
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

    X_tr = apply_norm(X_tr)
    X_te = apply_norm(X_te)

    y_tr_oh = to_categorical(y_tr_m := y_tr_m if 'y_tr_m' in locals() else y_tr, num_classes=num_classes)

    cw = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_tr_m)
    class_weight = {i: float(w) for i, w in enumerate(cw)}

    model = build_model((X_tr.shape[1], X_tr.shape[2]), num_classes, units=32)
    model.fit(X_tr, y_tr_oh, epochs=8, batch_size=16, class_weight=class_weight, verbose=1)

    proba = model.predict(X_te, verbose=0)
    preds = np.argmax(proba, axis=1)

    run_dir = ROOT / 'runs' / 'analysis' / f'smic-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / 'test_proba.npy', proba)
    np.save(run_dir / 'test_preds.npy', preds)
    np.save(run_dir / 'y_test.npy', y_te_m)

    # mean prob per true class
    mean_probs = {}
    for c in range(num_classes):
        idx = np.where(y_te_m == c)[0]
        if len(idx) == 0:
            mean = [0.0] * num_classes
        else:
            mean = proba[idx].mean(axis=0).tolist()
        mean_probs[int(c)] = mean

    # confusion pairs counts
    cm = confusion_matrix(y_te_m, preds)
    pairs = []
    for true in range(num_classes):
        for pred in range(num_classes):
            if true == pred:
                continue
            count = int(cm[true, pred])
            if count > 0:
                pairs.append({'true': int(true), 'pred': int(pred), 'count': count})
    pairs = sorted(pairs, key=lambda x: x['count'], reverse=True)

    out = {
        'mean_probs_by_true': mean_probs,
        'top_confusions': pairs[:20],
        'confusion_matrix': cm.tolist()
    }
    with open(run_dir / 'probs_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print('Saved analysis in', run_dir)
    print('Top confusions:')
    for p in pairs[:10]:
        print(f"  true {p['true']} -> pred {p['pred']} : {p['count']}")
    print('\nMean predicted probabilities per true class:')
    for c, v in mean_probs.items():
        print(f' class {c}:', ['{:.3f}'.format(x) for x in v])


if __name__ == '__main__':
    # Análise dos resultados do treino com augmentation
    import numpy as np
    import json
    from sklearn.metrics import confusion_matrix
    from pathlib import Path

    AUG_DIR = Path(__file__).resolve().parents[2] / 'runs' / 'finetune_mobilenet_augmented' / 'smic-20250926-221400'
    proba = np.load(AUG_DIR / 'proba.npy')
    preds = np.load(AUG_DIR / 'preds.npy')
    y_test = np.load(AUG_DIR / 'y_test.npy')

    num_classes = proba.shape[1]
    # mean prob per true class
    mean_probs = {}
    for c in range(num_classes):
        idx = np.where(y_test == c)[0]
        if len(idx) == 0:
            mean = [0.0] * num_classes
        else:
            mean = proba[idx].mean(axis=0).tolist()
        mean_probs[int(c)] = mean

    # confusion pairs counts
    cm = confusion_matrix(y_test, preds)
    pairs = []
    for true in range(num_classes):
        for pred in range(num_classes):
            if true == pred:
                continue
            count = int(cm[true, pred])
            if count > 0:
                pairs.append({'true': int(true), 'pred': int(pred), 'count': count})
    pairs = sorted(pairs, key=lambda x: x['count'], reverse=True)

    print('Matriz de confusão:')
    print(cm)
    print('\nTop confusões:')
    for p in pairs[:10]:
        print(f"  true {p['true']} -> pred {p['pred']} : {p['count']}")
    print('\nMédias de probabilidade predita por classe verdadeira:')
    for c, v in mean_probs.items():
        print(f' classe {c}:', ['{:.3f}'.format(x) for x in v])
