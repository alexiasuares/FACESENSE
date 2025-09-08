import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Diretórios e parâmetros
SKELETON_DIR = "../../data/iMIGUE/mg_skeleton_only"   # pasta onde estão os .xlsx
LABELS_FILE = "../../data/iMIGUE/Label/labels_20200831.csv"       # CSV com mapeamento vídeo -> label
OUTPUT_DIR = "../../data/processed_data"    # onde salvar os npy
N_FRAMES = 300                   # número fixo de frames por vídeo

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# Carregar labels
labels_df = pd.read_csv(LABELS_FILE)
label_map = dict(zip(labels_df["video_id"].astype(str).apply(lambda x: x.split('_')[0].zfill(4)), labels_df["class"]))

# Codificação das classes
unique_labels = sorted(labels_df["class"].unique())
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

RGB_DIRS = [
    "../../data/iMIGUE/iMiGUE_RGB_Phase1/imigue_rgb_train",
    "../../data/iMIGUE/iMiGUE_RGB_Phase1/imigue_rgb_validate",
    "../../data/iMIGUE/iMiGUE_RGB_Phase2/imigue_rgb_test"
 ]
rgb_videos = set()
for rgb_dir in RGB_DIRS:
    if os.path.exists(rgb_dir):
        for folder_name in os.listdir(rgb_dir):
            folder_path = os.path.join(rgb_dir, folder_name)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file == '.DS_Store':
                        continue
                    ext = os.path.splitext(file)[1].lower()
                    if ext in ['.mp4', '.avi', '.mov']:
                        video_id_rgb = os.path.splitext(file)[0]  # preserva zeros à esquerda
                        # Só adiciona se o nome da pasta for igual ao nome do vídeo
                        if folder_name == video_id_rgb:
                            rgb_videos.add(video_id_rgb)
                            print(f'RGB encontrado: pasta={folder_name}, arquivo={file}, id={video_id_rgb}')
# Depuração: listar diretórios RGB e subpastas
for rgb_dir in RGB_DIRS:
    print(f'RGB_DIR: {rgb_dir}')
    if os.path.exists(rgb_dir):
        subfolders = [f for f in os.listdir(rgb_dir) if os.path.isdir(os.path.join(rgb_dir, f))]
        print(f'  Subpastas encontradas: {subfolders}')
        for folder_name in subfolders:
            folder_path = os.path.join(rgb_dir, folder_name)
            files = os.listdir(folder_path)
            print(f'    Arquivos em {folder_name}: {files}')

import time
X, y = [], []
skeleton_ids = set()
xlsx_files = []
for root, dirs, files in os.walk(SKELETON_DIR):
    for file in files:
        if file.endswith(".xlsx"):
            xlsx_files.append((root, file))
n_total = len(xlsx_files)
print(f"Processando {n_total} arquivos de skeleton...")
start_time = time.time()
for idx, (root, file) in enumerate(xlsx_files):
    video_id_full = os.path.splitext(file)[0]  # ex.: "0001_2"
    video_id = video_id_full.split('_')[0]  # preserva zeros à esquerda
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
    
    # Verificar shape antes de adicionar
    if seq_fixed.shape != (N_FRAMES, 411):
        print(f"Arquivo ignorado: {file} - shape encontrado: {seq_fixed.shape}")
        continue
    # Adicionar ao dataset apenas se válido
    X.append(seq_fixed)
    y.append(label_to_int[label_map[video_id]])
    
    # Print progresso a cada 5%
    if n_total > 0 and idx % max(1, n_total // 20) == 0:
        percent = int(100 * (idx + 1) / n_total)
        elapsed = time.time() - start_time
        print(f"Progresso: {percent}% ({idx+1}/{n_total}) - {elapsed:.1f}s")
X = np.array(X)  # (n_videos, N_FRAMES, 411)
y = np.array(y)

# Depuração: verificar shapes dos elementos de X
shapes = [x.shape for x in X]
print("Shapes únicos encontrados em X:", set(shapes))
if all(s == (N_FRAMES, 411) for s in shapes):
    X = np.array(X)
else:
    print("Atenção: Existem elementos em X com shape diferente de (300, 411). Não foi possível converter para array.")
if isinstance(y, list):
    y = np.array(y)
print("Shape de X:", getattr(X, 'shape', 'lista'))
print("Shape de y:", getattr(y, 'shape', 'lista'))
print("Quantidade de skeletons processados:", len(X))

# Depuração: listar arquivos de skeleton encontrados
print('Skeleton IDs encontrados:', sorted(list(skeleton_ids)))

# Depuração: listar IDs de vídeos RGB encontrados
print('RGB video IDs encontrados:', sorted(list(rgb_videos)))

# Depuração: listar IDs de labels encontrados
print('Label IDs encontrados:', sorted(list(label_map.keys())))

# Filtrar classes com apenas 1 amostra para evitar erro no split estratificado
from collections import Counter
class_counts = Counter(y)
classes_validas = [cls for cls, count in class_counts.items() if count > 1]
mask = np.isin(y, classes_validas)
X_filtrado = X[mask]
y_filtrado = y[mask]
print(f"Classes removidas (apenas 1 amostra): {[cls for cls, count in class_counts.items() if count == 1]}")
print(f"Novo shape de X: {X_filtrado.shape}")
print(f"Novo shape de y: {y_filtrado.shape}")

# Filtrar classes com apenas 1 amostra em y_temp antes do segundo split
from collections import Counter
class_counts_temp = Counter(y_temp)
classes_validas_temp = [cls for cls, count in class_counts_temp.items() if count > 1]
mask_temp = np.isin(y_temp, classes_validas_temp)
X_temp_filtrado = X_temp[mask_temp]
y_temp_filtrado = y_temp[mask_temp]
print(f"Classes removidas do temp (apenas 1 amostra): {[cls for cls, count in class_counts_temp.items() if count == 1]}")
print(f"Novo shape de X_temp: {X_temp_filtrado.shape}")
print(f"Novo shape de y_temp: {y_temp_filtrado.shape}")

# Segundo split estratificado
X_val, X_test, y_val, y_test = train_test_split(
    X_temp_filtrado, y_temp_filtrado, test_size=0.50, stratify=y_temp_filtrado, random_state=42
)
print(f"Shapes: X_val={X_val.shape}, X_test={X_test.shape}")
print(f"Shapes: y_val={y_val.shape}, y_test={y_test.shape}")

np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

print("✅ Dados salvos em", OUTPUT_DIR)