# -*- coding: utf-8 -*-
# Versão da Gabi com patches mínimos p/ caminhos, normalização, masking, class_weight e F1 macro.

from pathlib import Path
import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, Callback, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import regularizers
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
from collections import Counter
import zipfile

# ----- PATHS ROBUSTOS -----
ROOT = Path(__file__).resolve().parents[2]  # .../FACESENSE
ZIP_PATH = ROOT / "data" / "processed_data.zip"
EXTRACT_DIR = ROOT / "data" / "processed_data"          # destino final
DATA_DIR = EXTRACT_DIR                                  # onde estão os npy (sem repetir /processed_data)

# Descompacta só se a pasta ainda não existir
if ZIP_PATH.exists() and not EXTRACT_DIR.exists():
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

# Alguns zips criam nível extra "processed_data/processed_data"; detecta e corrige
dup_dir = EXTRACT_DIR / "processed_data"
if dup_dir.exists() and (dup_dir / "X_train.npy").exists():
    DATA_DIR = dup_dir  # usa o nível interno

# ----- Carregamento dos dados pré-processados -----
from imblearn.over_sampling import SMOTE
def load_np(name):
    return np.load(DATA_DIR / name)

X_train = load_np("X_train.npy")
y_train = load_np("y_train.npy")
X_test = load_np("X_test.npy")
y_test = load_np("y_test.npy")

print("Shapes originais:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test : {X_test.shape},  y_test : {y_test.shape}")

# ----- Remapeamento consistente (0..C-1) -----
all_y = np.concatenate([y_train, y_test])
unique_labels = sorted(np.unique(all_y))
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
y_train = np.array([label_mapping[label] for label in y_train])
y_test = np.array([label_mapping[label] for label in y_test])

num_classes = len(unique_labels)
print("\n--- Rótulos ---")
print(f"labels únicos: {unique_labels} -> mapeados para 0..{num_classes-1}")
print("dist y_train:", Counter(y_train))
print("dist y_test :", Counter(y_test))

# ----- Oversampling com SMOTE nas classes minoritárias -----
print("Antes do SMOTE:", Counter(y_train))
smote = SMOTE(random_state=42)
X_train_rs, y_train_rs = smote.fit_resample(X_train.reshape((X_train.shape[0], -1)), y_train)
X_train = X_train_rs.reshape((-1, X_train.shape[1], X_train.shape[2]))
y_train = y_train_rs
print("Após SMOTE:", Counter(y_train))
# ----- Normalização z-score por feature ignorando padding 0 -----
# ----- Função de construção do modelo -----
def build_model(input_shape, n_classes):
    m = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),  # ignora padding zero
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
# ----- Cross-validation (StratifiedKFold) -----
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

def run_cv(X, y, num_classes, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/{n_splits}")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        # Normalização
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
        cw = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_tr)
        class_weight = {i: float(w) for i, w in enumerate(cw)}
        # Modelo
        input_shape = (X_tr.shape[1], X_tr.shape[2])
        model = build_model(input_shape, num_classes)
        model.fit(
            X_tr, y_tr_oh,
            epochs=40, batch_size=16, shuffle=True,
            class_weight=class_weight,
            verbose=0
        )
        # Avaliação
        proba = model.predict(X_val, verbose=0)
        y_pred = np.argmax(proba, axis=1)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        print(f"  Acc: {acc:.3f} | Macro F1: {f1:.3f}")
        accs.append(acc)
        f1s.append(f1)
    print(f"\nCV Média Acc: {np.mean(accs):.3f} (+/- {np.std(accs):.3f})")
    print(f"CV Média Macro F1: {np.mean(f1s):.3f} (+/- {np.std(f1s):.3f})")

# Executar cross-validation
run_cv(X_train, y_train, num_classes, n_splits=5)
T, F = X_train.shape[1], X_train.shape[2]
flat = X_train.reshape(-1, F)
mask = (flat != 0)
feat_sum = (flat * mask).sum(axis=0)
feat_count = mask.sum(axis=0).clip(min=1)
feat_mean = feat_sum / feat_count
flat_center = (flat - feat_mean) * mask
feat_var = ((flat - feat_mean) ** 2 * mask).sum(axis=0) / feat_count
feat_std = np.sqrt(feat_var)
feat_std[feat_std == 0] = 1.0

def apply_norm(X):
    Xf = X.reshape(-1, X.shape[-1])
    M = (Xf != 0)
    return ((Xf - feat_mean) / feat_std) * M

X_train = apply_norm(X_train).reshape(X_train.shape)
X_test = apply_norm(X_test).reshape(X_test.shape)

def stats(X, tag):
    nz = (X != 0).sum()
    tot = np.prod(X.shape)
    percent = 100 * nz / tot
    print(f"[{tag}] nonzeros={nz}/{tot} ({percent:.2f}%) mean={X.mean():.4f} std={X.std():.4f}")

stats(X_train, "train")
stats(X_test, "test")

# ----- One-Hot -----
y_train_oh = to_categorical(y_train, num_classes=num_classes)
y_test_oh  = to_categorical(y_test,  num_classes=num_classes)

# ----- Pesos de classe (desbalanceamento) -----
cw = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_train)
class_weight = {i: float(w) for i, w in enumerate(cw)}
print("class_weight:", class_weight)

# ----- Modelo (com Masking e BiLSTM) -----
def build_model(input_shape, n_classes):
    m = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),  # ignora padding zero
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

# ----- F1 macro como callback de validação -----
class F1MacroCallback(tf.keras.callbacks.Callback):
    pass  # Callback disabled (no validation set)

# ----- Logs e checkpoint -----
RUN_DIR = ROOT / "runs" / "skeleton" / datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)
tb = TensorBoard(log_dir=str(RUN_DIR / "tb"))
csv = CSVLogger(str(RUN_DIR / "training.csv"))
ckpt = ModelCheckpoint(str(RUN_DIR / "best.keras"), monitor="accuracy", mode="max", save_best_only=True, verbose=1)
rlr = ReduceLROnPlateau(monitor="accuracy", mode="max", factor=0.5, patience=5, min_lr=1e-5, verbose=1)
es = EarlyStopping(monitor="accuracy", mode="max", patience=10, restore_best_weights=True, verbose=1)

# ----- Treino -----
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_model(input_shape, num_classes)
model.summary()

history = model.fit(
    X_train, y_train_oh,
    epochs=80, batch_size=16, shuffle=True,
    class_weight=class_weight,
    callbacks=[tb, csv, ckpt, rlr, es],
    verbose=1
)

# ----- Avaliação final no melhor checkpoint -----
best = tf.keras.models.load_model(RUN_DIR / "best.keras")
proba = best.predict(X_test, verbose=0)
y_pred = np.argmax(proba, axis=1)

acc = (y_pred == y_test).mean().item()
f1M = f1_score(y_test, y_pred, average="macro", zero_division=0)
f1m = f1_score(y_test, y_pred, average="micro", zero_division=0)
cm  = confusion_matrix(y_test, y_pred)

def one_hot(y, C):
    m = np.zeros((len(y), C), dtype=np.float32); m[np.arange(len(y)), y] = 1.0; return m
yte_oh = one_hot(y_test, num_classes)
try:
    rocM = roc_auc_score(yte_oh, proba, average="macro", multi_class="ovr")
except Exception:
    rocM = float("nan")
try:
    prM = average_precision_score(yte_oh, proba, average="macro")
except Exception:
    prM = float("nan")

res = {
    "acc": float(acc), "f1_macro": float(f1M), "f1_micro": float(f1m),
    "roc_auc_macro": float(rocM), "pr_auc_macro": float(prM),
    "cm": cm.tolist(), "n_classes": int(num_classes)
}
with open(RUN_DIR / "final_results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

print("✅ Resultados:", res)
print("TensorBoard:", (RUN_DIR / "tb").resolve())

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.preprocessing import label_binarize
import csv

figs_dir = RUN_DIR / "figs"
figs_dir.mkdir(parents=True, exist_ok=True)

# 1) Matriz de confusão (PNG)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(values_format='d', xticks_rotation=45, cmap='Blues')
plt.title('Matriz de Confusão – Teste')
plt.tight_layout()
plt.savefig(figs_dir / "cm.png", dpi=160)
plt.close()

# 2) Curvas ROC/PR (micro-avg)
Y_bin = label_binarize(y_test, classes=np.arange(num_classes))  # shape: (N, C)

# ROC micro
RocCurveDisplay.from_predictions(Y_bin.ravel(), proba.ravel(), name='micro-average ROC')
plt.title('ROC – micro média')
plt.tight_layout()
plt.savefig(figs_dir / "roc_micro.png", dpi=160)
plt.close()

# PR micro
PrecisionRecallDisplay.from_predictions(Y_bin.ravel(), proba.ravel(), name='micro-average PR')
plt.title('Precision–Recall – micro média')
plt.tight_layout()
plt.savefig(figs_dir / "pr_micro.png", dpi=160)
plt.close()

# 3) Classification report (JSON + CSV)
report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
with open(RUN_DIR / "classification_report.json", "w", encoding="utf-8") as f:
    json.dump(report_dict, f, ensure_ascii=False, indent=2)

# CSV (linhas = classes + agregados; colunas = precision, recall, f1-score, support)
fields = ["label","precision","recall","f1-score","support"]
with open(RUN_DIR / "classification_report.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(fields)
    for k, v in report_dict.items():
        if isinstance(v, dict) and {"precision","recall","f1-score","support"} <= set(v.keys()):
            w.writerow([k, v["precision"], v["recall"], v["f1-score"], v["support"]])

# 4) Curvas de treino (acc/loss) – salvas a partir do 'history'
# OBS: a F1 de validação foi impressa pelo callback; se quiser plotar, podemos salvar num CSV próprio.
train_history = history.history

# acc
plt.plot(train_history.get('accuracy', []), label='train_acc')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.title('Curva de Acurácia (train)')
plt.legend()
plt.tight_layout()
plt.savefig(figs_dir / "curve_accuracy.png", dpi=160)
plt.close()

# loss
plt.plot(train_history.get('loss', []), label='train_loss')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.title('Curva de Loss (train)')
plt.legend()
plt.tight_layout()
plt.savefig(figs_dir / "curve_loss.png", dpi=160)
plt.close()

# Após salvar resultados e imagens

# Relatório de erros: exemplos mal classificados
import pandas as pd
error_indices = np.where(y_pred != y_test)[0]
error_report = []
for idx in error_indices:
    error_report.append({
        'index': int(idx),
        'true_label': int(y_test[idx]),
        'pred_label': int(y_pred[idx]),
        # 'tensor': X_test[idx].tolist()  # Descomente para salvar o tensor completo
    })

# Salvar relatório em CSV
error_df = pd.DataFrame(error_report)
error_df.to_csv(RUN_DIR / 'test_errors.csv', index=False)
print(f"Relatório de erros salvo em {RUN_DIR / 'test_errors.csv'} ({len(error_report)} exemplos)")