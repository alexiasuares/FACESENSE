# experiments/train_fixedsplits.py (PATCH MELHORIA)

from pathlib import Path
import os, json, numpy as np, tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, Callback, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from datetime import datetime

# ---- PATHS ----
ROOT = Path(__file__).resolve().parents[1]               # .../FACESENSE
DATA_DIR = ROOT / "data" / "processed_data" / "skeleton" # troque para "facial" quando tiver
RUN_DIR  = ROOT / "runs" / "skeleton" / datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)

def load_np(name): return np.load(DATA_DIR / name)

# --- Carrega dataset pronto ---
Xtr, ytr = load_np("X_train.npy"), load_np("y_train.npy")
Xva, yva = load_np("X_val.npy"),   load_np("y_val.npy")
Xte, yte = load_np("X_test.npy"),  load_np("y_test.npy")

# ------- Diagnósticos rápidos -------
def stats(X, tag):
    nz = (X != 0).sum()
    tot = np.prod(X.shape)
    print(f"[{tag}] shape={X.shape}  nonzeros={nz}/{tot} ({100*nz/tot:.2f}%)  mean={X.mean():.4f}  std={X.std():.4f}")
stats(Xtr, "X_train"); stats(Xva, "X_val"); stats(Xte, "X_test")
print("y_train dist:", Counter(ytr))
print("y_val   dist:", Counter(yva))
print("y_test  dist:", Counter(yte))

# --- Remapeia rótulos para 0..C-1 se necessário (consistente entre splits)
uniq = np.unique(np.concatenate([ytr, yva, yte]))
map_ = {old:i for i, old in enumerate(uniq)}
ytr = np.array([map_[v] for v in ytr]); yva = np.array([map_[v] for v in yva]); yte = np.array([map_[v] for v in yte])
n_classes = len(uniq)
print(f"n_classes={n_classes}  labels={list(range(n_classes))}")

# ------- Normalização por feature (z-score) ignorando padding 0 -------
# Achata (N*T, F) e computa mean/std só nas posições != 0 (para não usar padding)
T, F = Xtr.shape[1], Xtr.shape[2]
flat = Xtr.reshape(-1, F)
mask = flat != 0
feat_sum   = (flat * mask).sum(axis=0)
feat_count = mask.sum(axis=0).clip(min=1)           # evita div/0
feat_mean  = feat_sum / feat_count

flat_center = (flat - feat_mean) * mask
feat_var  = (flat_center**2).sum(axis=0) / feat_count
feat_std  = np.sqrt(feat_var); feat_std[feat_std == 0] = 1.0

def apply_norm(X):
    Xf = X.reshape(-1, F)
    M  = (Xf != 0)
    Xf = np.where(M, (Xf - feat_mean) / feat_std, 0.0)  # mantém padding em 0
    return Xf.reshape(X.shape)

Xtr = apply_norm(Xtr); Xva = apply_norm(Xva); Xte = apply_norm(Xte)
# Reconfirma stats após normalização
stats(Xtr, "X_train norm"); stats(Xva, "X_val norm"); stats(Xte, "X_test norm")

# ------- One-hot -------
ytr_oh = to_categorical(ytr, num_classes=n_classes)
yva_oh = to_categorical(yva, num_classes=n_classes)
yte_oh = to_categorical(yte, num_classes=n_classes)

# ------- Pesos de classe -------
cw = compute_class_weight(class_weight="balanced", classes=np.arange(n_classes), y=ytr)
cw = {i: float(w) for i, w in enumerate(cw)}
print("class_weight:", cw)

# ------- Modelo -------
def build_model(input_shape, n_classes):
    m = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),         # ignora padding zeros
        Bidirectional(LSTM(128, return_sequences=False)),
        Dropout(0.5),
        Dense(n_classes, activation="softmax")
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
    return m

class F1MacroCallback(Callback):
    def __init__(self, Xv, yv_oh):
        super().__init__(); self.Xv, self.yv_oh = Xv, yv_oh
    def on_epoch_end(self, epoch, logs=None):
        proba = self.model.predict(self.Xv, verbose=0)
        y_pred = np.argmax(proba, axis=1)
        y_true = np.argmax(self.yv_oh, axis=1)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        logs = logs or {}; logs["val_f1_macro"] = f1m
        print(f"\n[Epoch {epoch+1}] val_f1_macro={f1m:.4f}")

tb   = tf.keras.callbacks.TensorBoard(log_dir=str(RUN_DIR / "tb"))
csv  = tf.keras.callbacks.CSVLogger(str(RUN_DIR / "training.csv"))
ckpt = tf.keras.callbacks.ModelCheckpoint(str(RUN_DIR / "best.keras"),
        monitor="val_f1_macro", mode="max", save_best_only=True, verbose=1)
rlr  = ReduceLROnPlateau(monitor="val_f1_macro", mode="max", factor=0.5, patience=5, min_lr=1e-5, verbose=1)
es   = EarlyStopping(monitor="val_f1_macro", mode="max", patience=10, restore_best_weights=True, verbose=1)
f1cb = F1MacroCallback(Xva, yva_oh)

model = build_model(input_shape=Xtr.shape[1:], n_classes=n_classes)
model.summary()

hist = model.fit(
    Xtr, ytr_oh,
    validation_data=(Xva, yva_oh),
    epochs=80, batch_size=32, shuffle=True,
    class_weight=cw,
    callbacks=[tb, csv, ckpt, rlr, es, f1cb],
    verbose=1
)

# ------- Avaliação final -------
best = tf.keras.models.load_model(RUN_DIR / "best.keras")
proba = best.predict(Xte, verbose=0)
y_pred = np.argmax(proba, axis=1)

acc = (y_pred == yte).mean().item()
f1M = f1_score(yte, y_pred, average="macro", zero_division=0)
f1m = f1_score(yte, y_pred, average="micro", zero_division=0)
cm  = confusion_matrix(yte, y_pred)

# ROC/PR macro (one-vs-rest)
def one_hot(y, C):
    m = np.zeros((len(y), C), dtype=np.float32); m[np.arange(len(y)), y] = 1.0; return m
yte_oh = one_hot(yte, n_classes)
try:
    rocM = roc_auc_score(yte_oh, proba, average="macro", multi_class="ovr")
except Exception:
    rocM = float("nan")
try:
    prM = average_precision_score(yte_oh, proba, average="macro")
except Exception:
    prM = float("nan")

res = {"acc": float(acc), "f1_macro": float(f1M), "f1_micro": float(f1m),
       "roc_auc_macro": float(rocM), "pr_auc_macro": float(prM),
       "cm": cm.tolist(), "n_classes": int(n_classes)}

with open(RUN_DIR / "final_results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

print("✅ Resultados:", res)
print("TensorBoard:", (RUN_DIR / "tb").resolve())
