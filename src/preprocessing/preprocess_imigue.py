import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# === Configurações ===
SKELETON_DIR = "data/iMIGUE/mg_skeleton_only"   # pasta onde estão os .xlsx
LABELS_FILE = "data/iMIGUE/Label/labels_20200831.csv"       # CSV com mapeamento vídeo -> label
OUTPUT_DIR = "data/processed_data"    # onde salvar os npy
N_FRAMES = 300                   # número fixo de frames por vídeo

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Funções auxiliares ===
def pad_or_truncate(sequence, target_len=N_FRAMES):
    """
    Ajusta a sequência para um comprimento fixo (padding com zeros ou truncagem).
    """
    n_frames, n_features = sequence.shape
    if n_frames > target_len:
        return sequence[:target_len, :]
    elif n_frames < target_len:
        padding = np.zeros((target_len - n_frames, n_features))
        return np.vstack([sequence, padding])
    return sequence

# === Carregar labels ===
labels_df = pd.read_csv(LABELS_FILE)
label_map = dict(zip(labels_df["video_id"].astype(str).apply(lambda x: x.split('_')[0]), labels_df["class"]))

# Codificação das classes
unique_labels = sorted(labels_df["class"].unique())
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

# === Buscar vídeos existentes nas pastas RGB ===
RGB_DIRS = [
    "data/iMIGUE/iMiGUE_RGB_Phase1/imigue_rgb_train",
    "data/iMIGUE/iMiGUE_RGB_Phase1/imigue_rgb_validate",
    "data/iMIGUE/iMiGUE_RGB_Phase2/imigue_rgb_test"
]
rgb_videos = set()
for rgb_dir in RGB_DIRS:
    if os.path.exists(rgb_dir):
        for root, dirs, files in os.walk(rgb_dir):
            for file in files:
                if file == '.DS_Store':
                    continue
                ext = os.path.splitext(file)[1].lower()
                if ext in ['.mp4', '.avi', '.mov']:
                    video_id_rgb = os.path.splitext(file)[0].lstrip('0')
                    rgb_videos.add(video_id_rgb)

# === Processar cada arquivo de skeleton ===
X, y = [], []
skeleton_ids = set()
for root, dirs, files in os.walk(SKELETON_DIR):
    for file in files:
        if file.endswith(".xlsx"):
            video_id_full = os.path.splitext(file)[0]  # ex.: "0001_2"
            video_id = video_id_full.split('_')[0].lstrip('0')  # remove zeros à esquerda
            skeleton_ids.add(video_id)
            
            if video_id not in label_map:
                continue
            if video_id not in rgb_videos:
                continue
            # Carregar skeleton
            df = pd.read_excel(os.path.join(root, file))
            seq = df.values.astype(np.float32)  # (n_frames, 411)

            # Normalizar coordenadas (0-1)
            seq = np.nan_to_num(seq)  # substitui NaNs por 0

            # Ajustar tamanho
            seq_fixed = pad_or_truncate(seq, target_len=N_FRAMES)

            # Adicionar ao dataset
            X.append(seq_fixed)
            y.append(label_to_int[label_map[video_id]])

X = np.array(X)  # (n_videos, N_FRAMES, 411)
y = np.array(y)

# === Divisão treino/val/test ===
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# === Salvar em npy ===
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

print("✅ Dados salvos em", OUTPUT_DIR)