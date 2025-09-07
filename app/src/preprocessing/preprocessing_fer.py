import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Caminhos
DATA_DIR = "../../data/FER/archive"
OUTPUT_DIR = "../../data/FER/processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuração
IMG_SIZE = 48
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
label_to_int = {cls: idx for idx, cls in enumerate(CLASSES)}


def load_images_from_folder(folder, label):
    data, labels = [], []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # FER é grayscale
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img / 255.0)  # normaliza 0–1
        labels.append(label)
    return data, labels

# Carregar treino
X_train, y_train = [], []
for cls in CLASSES:
    folder = os.path.join(DATA_DIR, "train", cls)
    data, labels = load_images_from_folder(folder, label_to_int[cls])
    X_train.extend(data)
    y_train.extend(labels)

# Carregar teste
X_test, y_test = [], []
for cls in CLASSES:
    folder = os.path.join(DATA_DIR, "test", cls)
    data, labels = load_images_from_folder(folder, label_to_int[cls])
    X_test.extend(data)
    y_test.extend(labels)

# Converter para numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# Adicionar dimensão de canal (para CNNs)
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Criar conjunto de validação a partir do treino
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}, y_test: {y_test.shape}")

# Salvar em .npy
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

print("✅ FER pré-processado e salvo em", OUTPUT_DIR)