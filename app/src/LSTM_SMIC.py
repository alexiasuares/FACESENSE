# -*- coding: utf-8 -*-
"""
Treinamento LSTM (modo similar ao script iMiGUE) para dados SMIC pre-processados.
Espera encontrar arquivos Numpy em: <repo_root>/data/SMIC/processed/
 - X_train.npy  (N, T, H, W) ou (N, T, F)
 - y_train.npy  (N,)
 - X_test.npy
 - y_test.npy

Este script segue a mesma estrutura e callbacks do `app/src/LSTM.py` mas adapta entradas de imagem
convertendo cada frame em um vetor de features (flatten) caso necessário, preservando o uso de Masking
para ignorar padding (valores 0).

Para executar:
  python app/src/LSTM_smic.py

"""

from pathlib import Path
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import regularizers
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, RocCurveDisplay, PrecisionRecallDisplay
import csv
import pandas as pd

# ----- PATHS ROBUSTOS -----
ROOT = Path(__file__).resolve().parents[2]

# Candidate locations (ordered). Add the new facesence path and common fallbacks.
candidates = [
    ROOT / 'facesence' / 'data' / 'smic',
    ROOT / 'data' / 'smic',
    ROOT / 'data' / 'SMIC' / 'processed',
    ROOT / 'data' / 'processed_data',
    ROOT / 'app' / 'data' / 'SMIC' / 'processed',
    ROOT / 'app' / 'data' / 'SMIC' / 'processed' / 'smic_processed',
]

DATA_DIR = None
for c in candidates:
    if c.exists():
        DATA_DIR = c
        break

if DATA_DIR is None:
    # last resort: look for any folder named 'smic' or 'SMIC' under repo data/
    for p in (ROOT / 'data').rglob('*'):
        if p.is_dir() and p.name.lower() == 'smic':
            DATA_DIR = p
            break

if DATA_DIR is None:
    raise FileNotFoundError('Não foi possível localizar o diretório de dados SMIC. Verifique onde os .npy estão salvos.')

print('Using DATA_DIR =', DATA_DIR)

def load_np(name):
    p = DATA_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Esperado {p} mas não encontrado. Rode o pré-processamento SMIC primeiro.")
    return np.load(p)

# ----- Carregamento dos dados pré-processados -----
X_train = load_np('X_train.npy')
y_train = load_np('y_train.npy')
X_test = load_np('X_test.npy')
y_test = load_np('y_test.npy')

print('Shapes originais:')
print('X_train:', X_train.shape, 'y_train:', y_train.shape)
print('X_test :', X_test.shape, 'y_test :', y_test.shape)

# ----- Se os dados forem imagens (N,T,H,W), converte para (N,T,F) flattenando H*W por frame -----
if X_train.ndim == 4:
    N, T, H, W = X_train.shape
    F = H * W
    print(f"Detectado X em (N,T,H,W) -> flattenando frames para F={F} features por timestep")
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], -1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], -1))

# agora X_* deverá ter shape (N,T,F)
if X_train.ndim != 3:
    raise RuntimeError('Formato de X inesperado; esperado (N,T,F) após pré-processamento')

# ----- Remapeamento consistente (0..C-1) -----
all_y = np.concatenate([y_train, y_test])
unique_labels = sorted(np.unique(all_y))
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
y_train = np.array([label_mapping[label] for label in y_train])
y_test = np.array([label_mapping[label] for label in y_test])
num_classes = len(unique_labels)
print('\n--- Rótulos ---')
print(f'labels únicos: {unique_labels} -> mapeados para 0..{num_classes-1}')
print('dist y_train:', Counter(y_train))
print('dist y_test :', Counter(y_test))

# ----- Oversampling com SMOTE nas classes minoritárias -----
print('Antes do SMOTE:', Counter(y_train))
# SMOTE precisa de 2D por amostra -> reshape (N, T*F)
N, T, F = X_train.shape
try:
    smote = SMOTE(random_state=42)
    X_flat = X_train.reshape((N, -1))
    X_rs, y_rs = smote.fit_resample(X_flat, y_train)
    X_train = X_rs.reshape((-1, T, F))
    y_train = y_rs
    print('Após SMOTE:', Counter(y_train))
except Exception as e:
    print('SMOTE falhou (provavelmente poucas amostras/alto dimensionalidade). Continuando sem SMOTE. Erro:', e)

# ----- Normalização z-score por feature ignorando padding 0 -----
print('Calculando normalização (ignora padding 0)')
flat = X_train.reshape(-1, F)
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
    return ((Xf - feat_mean) / feat_std) * M

X_train = apply_norm(X_train).reshape(X_train.shape)
X_test = apply_norm(X_test).reshape(X_test.shape)

# ----- One-Hot -----
y_train_oh = to_categorical(y_train, num_classes=num_classes)
y_test_oh = to_categorical(y_test, num_classes=num_classes)

# ----- Pesos de classe (desbalanceamento) -----
cw = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_train)
class_weight = {i: float(w) for i, w in enumerate(cw)}
print('class_weight:', class_weight)

# ----- Função de construção do modelo (idêntica ao iMiGUE) -----
def build_model(input_shape, n_classes):
    m = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.5),
        Dense(
            n_classes,
            activation='softmax',
            kernel_regularizer=regularizers.l2(1e-4)
        )
    ])
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return m

# ----- Cross-validation (StratifiedKFold) e treino final -----

def run_cv_and_train(X, y, num_classes, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f'Fold {fold+1}/{n_splits}')
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Normalização por feature calculada a partir do fold (igual ao script iMiGUE)
        T, F = X_tr.shape[1], X_tr.shape[2]
        flat = X_tr.reshape(-1, F)
        mask = (flat != 0)
        feat_sum = (flat * mask).sum(axis=0)
        feat_count = mask.sum(axis=0).clip(min=1)
        feat_mean = feat_sum / feat_count
        flat_center = (flat - feat_mean) * mask
        feat_var = ((flat - feat_mean) ** 2 * mask).sum(axis=0) / feat_count
        feat_std = np.sqrt(feat_var)
        feat_std[feat_std == 0] = 1.0

        def apply_norm_cv(X):
            Xf = X.reshape(-1, X.shape[-1])
            M = (Xf != 0)
            return ((Xf - feat_mean) / feat_std) * M

        X_tr = apply_norm_cv(X_tr).reshape(X_tr.shape)
        X_val = apply_norm_cv(X_val).reshape(X_val.shape)

        # One-hot
        y_tr_oh = to_categorical(y_tr, num_classes=num_classes)
        y_val_oh = to_categorical(y_val, num_classes=num_classes)

        # Pesos de classe
        cw = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_tr)
        class_weight = {i: float(w) for i, w in enumerate(cw)}

        # Modelo
        input_shape = (X_tr.shape[1], X_tr.shape[2])
        model = build_model(input_shape, num_classes)

        # Callbacks e dirs
        RUN_DIR = ROOT / 'runs' / 'smic' / datetime.now().strftime('%Y%m%d-%H%M%S')
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        tb = TensorBoard(log_dir=str(RUN_DIR / 'tb'))
        csv = CSVLogger(str(RUN_DIR / 'training.csv'))
        ckpt = ModelCheckpoint(str(RUN_DIR / 'best.keras'), monitor='accuracy', mode='max', save_best_only=True, verbose=1)
        rlr = ReduceLROnPlateau(monitor='accuracy', mode='max', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
        es = EarlyStopping(monitor='accuracy', mode='max', patience=10, restore_best_weights=True, verbose=1)

        model.fit(
            X_tr, y_tr_oh,
            epochs=40, batch_size=16, shuffle=True,
            class_weight=class_weight,
            callbacks=[tb, csv, ckpt, rlr, es],
            verbose=1
        )

        # Avaliação
        proba = model.predict(X_val, verbose=0)
        y_pred = np.argmax(proba, axis=1)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        print(f'  Acc: {acc:.3f} | Macro F1: {f1:.3f}')
        accs.append(acc)
        f1s.append(f1)

    print(f"\nCV Média Acc: {np.mean(accs):.3f} (+/- {np.std(accs):.3f})")
    print(f"CV Média Macro F1: {np.mean(f1s):.3f} (+/- {np.std(f1s):.3f})")

    return RUN_DIR

# ----- Executa CV + treino final (usa conjunto inteiro para treino/val interno) -----
RUN_DIR = run_cv_and_train(X_train, y_train, num_classes, n_splits=5)

# Treino final no conjunto de treino inteiro e avaliação no teste (usa mesma normalização global calculada antes)
print('\nTreino final sobre todo o conjunto de treino e avaliação no teste')
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_model(input_shape, num_classes)
model.fit(
    X_train, y_train_oh,
    epochs=80, batch_size=16, shuffle=True,
    class_weight=class_weight,
    callbacks=[],
    verbose=1
)

# Avaliação no teste
proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(proba, axis=1)
acc = (y_pred == y_test).mean().item()
f1M = f1_score(y_test, y_pred, average='macro', zero_division=0)
f1m = f1_score(y_test, y_pred, average='micro', zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# métricas adicionais
try:
    yte_oh = label_binarize(y_test, classes=np.arange(num_classes))
    rocM = roc_auc_score(yte_oh, proba, average='macro', multi_class='ovr')
except Exception:
    rocM = float('nan')
try:
    prM = average_precision_score(yte_oh, proba, average='macro')
except Exception:
    prM = float('nan')

res = {
    'acc': float(acc), 'f1_macro': float(f1M), 'f1_micro': float(f1m),
    'roc_auc_macro': float(rocM), 'pr_auc_macro': float(prM),
    'cm': cm.tolist(), 'n_classes': int(num_classes)
}
with open(RUN_DIR / 'final_results.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

print('✅ Resultados:', res)
print('TensorBoard:', (RUN_DIR / 'tb').resolve())

# plots
figs_dir = RUN_DIR / 'figs'
figs_dir.mkdir(parents=True, exist_ok=True)

# Confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(values_format='d', xticks_rotation=45, cmap='Blues')
plt.title('Matriz de Confusão – Teste')
plt.tight_layout()
plt.savefig(figs_dir / 'cm.png', dpi=160)
plt.close()

# ROC/PR micro
RocCurveDisplay.from_predictions(yte_oh.ravel(), proba.ravel(), name='micro-average ROC')
plt.title('ROC – micro média')
plt.tight_layout()
plt.savefig(figs_dir / 'roc_micro.png', dpi=160)
plt.close()

PrecisionRecallDisplay.from_predictions(yte_oh.ravel(), proba.ravel(), name='micro-average PR')
plt.title('Precision–Recall – micro média')
plt.tight_layout()
plt.savefig(figs_dir / 'pr_micro.png', dpi=160)
plt.close()

# Classification report
report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
with open(RUN_DIR / 'classification_report.json', 'w', encoding='utf-8') as f:
    json.dump(report_dict, f, ensure_ascii=False, indent=2)

# CSV report
fields = ['label','precision','recall','f1-score','support']
with open(RUN_DIR / 'classification_report.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(fields)
    for k, v in report_dict.items():
        if isinstance(v, dict) and {'precision','recall','f1-score','support'} <= set(v.keys()):
            w.writerow([k, v['precision'], v['recall'], v['f1-score'], v['support']])

# Save errors
error_indices = np.where(y_pred != y_test)[0]
error_report = []
for idx in error_indices:
    error_report.append({'index': int(idx), 'true_label': int(y_test[idx]), 'pred_label': int(y_pred[idx])})
error_df = pd.DataFrame(error_report)
error_df.to_csv(RUN_DIR / 'test_errors.csv', index=False)
print(f"Relatório de erros salvo em {RUN_DIR / 'test_errors.csv'} ({len(error_report)} exemplos)")
