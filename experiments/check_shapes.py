import os, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # sobe de experiments/ para FACESENSE/
DATA_DIR = ROOT / "data" / "processed_data" / "skeleton"   # troque para "facial" se for o caso

def load(name):
    return np.load(os.path.join(DATA_DIR, name))

Xtr, ytr = load("X_train.npy"), load("y_train.npy")
Xva, yva = load("X_val.npy"),   load("y_val.npy")
Xte, yte = load("X_test.npy"),  load("y_test.npy")

print("X_train:", Xtr.shape, "y_train:", ytr.shape, "y uniq:", np.unique(ytr))
print("X_val  :", Xva.shape, "y_val  :", yva.shape, "y uniq:", np.unique(yva))
print("X_test :", Xte.shape, "y_test :", yte.shape, "y uniq:", np.unique(yte))

# checa se é (N, T, F)
assert Xtr.ndim == 3, f"Esperado (N,T,F); veio {Xtr.shape}"
assert Xva.shape[1:] == Xtr.shape[1:] and Xte.shape[1:] == Xtr.shape[1:], "Shapes inconsistentes"
print("✅ Shapes OK")