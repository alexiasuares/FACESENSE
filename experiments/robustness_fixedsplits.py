import os, json, numpy as np, tensorflow as tf
from sklearn.metrics import f1_score
from datetime import datetime

DATA_DIR = r"..\FACESENSE\data\processed_data\skeleton"
RUN_DIR  = r"..\FACESENSE\runs\skeleton\YYYYMMDD-HHMMSS"  # passe a pasta do run

def add_gaussian_noise(X, sigma=0.02):
    return X + np.random.normal(0, sigma, size=X.shape).astype(X.dtype)

def random_frame_drop(X, p=0.10):
    X2 = X.copy()
    mask = np.random.rand(X.shape[1]) < p
    X2[:, mask, :] = 0.0
    return X2

def temporal_jitter(X, max_shift=3):
    X2 = np.zeros_like(X)
    for i in range(X.shape[0]):
        shift = np.random.randint(-max_shift, max_shift+1)
        if shift >= 0:
            X2[i, shift:, :] = X[i, :X.shape[1]-shift, :]
        else:
            X2[i, :shift, :] = X[i, -shift:, :]
    return X2

def eval_variant(model, X, y, tag):
    proba = model.predict(X, verbose=0)
    y_pred = np.argmax(proba, axis=1)
    return {"acc": float((y_pred==y).mean().item()),
            "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
            "f1_micro": float(f1_score(y, y_pred, average="micro", zero_division=0))}

if __name__ == "__main__":
    Xte = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    yte = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    best = tf.keras.models.load_model(os.path.join(RUN_DIR, "best.keras"))

    out = {
        "clean":          eval_variant(best, Xte, yte, "clean"),
        "noise@0.02":     eval_variant(best, add_gaussian_noise(Xte, 0.02), yte, "noise"),
        "frame_drop@10%": eval_variant(best, random_frame_drop(Xte, 0.10), yte, "drop"),
        "jitter@±3":      eval_variant(best, temporal_jitter(Xte, 3), yte, "jitter"),
    }
    with open(os.path.join(RUN_DIR, "robustness.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("✅ Robustez:", out)