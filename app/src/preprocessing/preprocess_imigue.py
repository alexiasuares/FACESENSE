# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import time
from collections import Counter

# Diretórios e parâmetros
SKELETON_DIR = "../../../data/iMIGUE/mg_skeleton_only"
LABELS_FILE = "../../../data/iMIGUE/Label/labels_20200831.csv"
OUTPUT_DIR = "../../../data/processed_data"
N_FRAMES = 300
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SKELETON_DIR = os.path.join(ROOT, 'data', 'iMIGUE', 'mg_skeleton_only')
LABELS_FILE = os.path.join(ROOT, 'data', 'iMIGUE', 'Label', 'labels_20200831.csv')
OUTPUT_DIR = os.path.join(ROOT, 'data', 'processed_data')    # onde salvar os npy

os.makedirs(OUTPUT_DIR, exist_ok=True)

def pad_or_truncate(sequence, target_len=N_FRAMES):
    n_frames, n_features = sequence.shape
    if n_frames > target_len:
        return sequence[:target_len, :]
    elif n_frames < target_len:
        padding = np.zeros((target_len - n_frames, n_features))
        return np.vstack([sequence, padding])
    return sequence

# === Carregar labels e agrupar ===
labels_df = pd.read_csv(LABELS_FILE)

map_dict = {
    1: "Head posture", 2: "Head posture", 3: "Face touching",
    4: "Face touching", 5: "Face touching", 6: "Face touching",
    7: "Face touching", 8: "Face touching", 9: "Face touching",
    10: "Face touching", 11: "Face touching", 12: "Face touching",
    13: "Face touching", 14: "Face touching", 15: "Face touching",
    28: "Head posture", 29: "Head posture",
    16: "Body movement", 18: "Body movement", 20: "Body movement",
    21: "Body movement", 31: "Body movement",
    17: "Arm gestures", 19: "Arm gestures", 22: "Arm gestures",
    23: "Arm gestures", 24: "Arm gestures", 25: "Arm gestures",
    26: "Arm gestures", 27: "Arm gestures", 30: "Arm gestures",
    99: "Neutral"
}

classes_sem_mapeamento = set(labels_df["class"].unique()) - set(map_dict.keys())
for classe in classes_sem_mapeamento:
    map_dict[classe] = "Other"

labels_df["grouped_class"] = labels_df["class"].map(map_dict)

unique_labels = sorted(labels_df["grouped_class"].dropna().unique())
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
int_to_label = {v: k for k, v in label_to_int.items()}

label_map = dict(zip(
    labels_df["video_id"].astype(str).apply(lambda x: x.split('_')[0].zfill(4)),
    labels_df["grouped_class"]
))

# Salvar mapeamento
mapping_info = {
    "map_dict": map_dict,
    "label_to_int": label_to_int,
    "int_to_label": int_to_label,
    "unique_labels": unique_labels
}
with open(os.path.join(OUTPUT_DIR, "class_mapping.json"), "w") as f:
    json.dump(mapping_info, f, indent=2)

# === Buscar vídeos RGB ===
RGB_DIRS = [
    "../../../data/iMIGUE/iMiGUE_RGB_Phase1/imigue_rgb_train",
    "../../../data/iMIGUE/iMiGUE_RGB_Phase1/imigue_rgb_validate",
    "../../../data/iMIGUE/iMiGUE_RGB_Phase2/imigue_rgb_test"
]
RGB_DIRS = [
    os.path.join(ROOT, 'data', 'iMIGUE', 'iMiGUE_RGB_Phase1', 'imigue_rgb_train'),
    os.path.join(ROOT, 'data', 'iMIGUE', 'iMiGUE_RGB_Phase1', 'imigue_rgb_validate'),
    os.path.join(ROOT, 'data', 'iMIGUE', 'iMiGUE_RGB_Phase2', 'imigue_rgb_test'),
]
rgb_videos = set()
for rgb_dir in RGB_DIRS:
    if os.path.exists(rgb_dir):
        for folder_name in os.listdir(rgb_dir):
            folder_path = os.path.join(rgb_dir, folder_name)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file in ['.DS_Store']:
                        continue
                    ext = os.path.splitext(file)[1].lower()
                    if ext in ['.mp4', '.avi', '.mov']:
                        video_id_rgb = os.path.splitext(file)[0]
                        if folder_name == video_id_rgb:
                            rgb_videos.add(video_id_rgb)

# === Processar arquivos de skeleton ===
X, y = [], []
for root, dirs, files in os.walk(SKELETON_DIR):
    for file in files:
        if not file.endswith(".xlsx"):
            continue
        video_id_full = os.path.splitext(file)[0]
        video_id = video_id_full.split('_')[0]
        if video_id not in label_map:
            continue
        if video_id not in rgb_videos:
            continue
        df = pd.read_excel(os.path.join(root, file))
        seq = np.nan_to_num(df.values.astype(np.float32))
        seq_fixed = pad_or_truncate(seq, target_len=N_FRAMES)
        X.append(seq_fixed)
        y.append(label_to_int[label_map[video_id]])

X = np.array(X)
y = np.array(y)
print("Shape final:", X.shape, y.shape)

# Filtrar classes com apenas 1 amostra
class_counts = Counter(y)
classes_validas = [cls for cls, count in class_counts.items() if count > 1]
mask = np.isin(y, classes_validas)
X = X[mask]
y = y[mask]

# Split direto train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Distribuição train:", Counter(y_train))
print("Distribuição test :", Counter(y_test))

# Salvar
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

# CSVs auxiliares
train_df = pd.DataFrame({"label_id": y_train, "label_name": [int_to_label[idx] for idx in y_train]})
test_df = pd.DataFrame({"label_id": y_test, "label_name": [int_to_label[idx] for idx in y_test]})

train_df.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

print("✅ Dados salvos em", OUTPUT_DIR)