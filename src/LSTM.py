

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# descompactando os dados pre processados 
import zipfile
zip_path = "../FACESENSE/data/processed_data.zip"
extract_path = "../FACESENSE/data/processed_data"


with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


# Carregando os dados pré-processados
X_train = np.load("../FACESENSE/data/processed_data/processed_data/X_train.npy")
y_train = np.load("../FACESENSE/data/processed_data/processed_data/y_train.npy")
X_val = np.load("../FACESENSE/data/processed_data/processed_data/X_val.npy")
y_val = np.load("../FACESENSE/data/processed_data/processed_data/y_val.npy")
X_test = np.load("../FACESENSE/data/processed_data/processed_data/X_test.npy")
y_test = np.load("../FACESENSE/data/processed_data/processed_data/y_test.npy")


# Passo 1: Carregue os dados pré-processados da Aléxia
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from collections import Counter

# Ajuste o caminho se necessário.
processed_data_path = "..\FACESENSE\data\processed_data\processed_data"

X_train = np.load(os.path.join(processed_data_path, "X_train.npy"))
y_train = np.load(os.path.join(processed_data_path, "y_train.npy"))
X_val = np.load(os.path.join(processed_data_path, "X_val.npy"))
y_val = np.load(os.path.join(processed_data_path, "y_val.npy"))
X_test = np.load(os.path.join(processed_data_path, "X_test.npy"))
y_test = np.load(os.path.join(processed_data_path, "y_test.npy"))

print("Shapes originais:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Passo 2: Remaapeamento de rótulos
# Identifique todos os rótulos únicos em todos os conjuntos de dados.
all_y = np.concatenate([y_train, y_val, y_test])
unique_labels = sorted(np.unique(all_y))

# Crie um dicionário para mapear os rótulos antigos para os novos (0, 1, 2, ...).
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
num_classes = len(unique_labels)

print("\n--- Processamento de Rótulos ---")
print(f"Rótulos originais encontrados: {unique_labels}")
print(f"Mapeamento de rótulos: {label_mapping}")
print(f"Novo número de classes: {num_classes}")

# Aplique o mapeamento nos seus dados.
y_train_remapped = np.array([label_mapping[label] for label in y_train])
y_val_remapped = np.array([label_mapping[label] for label in y_val])
y_test_remapped = np.array([label_mapping[label] for label in y_test])

# Verifique se o remapeamento funcionou.
print(f"Rótulos remapeados no conjunto de treino: {np.unique(y_train_remapped)}")

# Passo 3: One-Hot Encoding
# Converta os rótulos remapeados para o formato one-hot.
y_train_one_hot = to_categorical(y_train_remapped, num_classes=num_classes)
y_val_one_hot = to_categorical(y_val_remapped, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test_remapped, num_classes=num_classes)

print("\n--- One-Hot Encoding ---")
print(f"Novo shape de y_train_one_hot: {y_train_one_hot.shape}")
print(f"Novo shape de y_val_one_hot: {y_val_one_hot.shape}")
print(f"Novo shape de y_test_one_hot: {y_test_one_hot.shape}")

# O shape de entrada para o modelo LSTM. 
# É o shape de uma única amostra de X_train (frames, features)
input_shape = (X_train.shape[1], X_train.shape[2])

# Construa o modelo
model = Sequential()

# Adicione a camada LSTM para processar as sequências.
model.add(LSTM(64, input_shape=input_shape))

# Adicione uma camada de Dropout para ajudar a evitar o overfitting.
model.add(Dropout(0.5))

# Adicione a camada de saída. O número de neurônios deve ser igual ao número de classes.
model.add(Dense(num_classes, activation='softmax'))

# Exiba um resumo da arquitetura do seu modelo.
model.summary()


# Compile o modelo.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Treine o modelo usando os dados processados.
history = model.fit(
    X_train, y_train_one_hot,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val_one_hot)
)


# Gráfico de Acurácia
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Acurácia do Modelo')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(['Treino', 'Validação'], loc='upper left')
plt.show()


# Gráfico de Perda (Loss)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Perda do Modelo')
plt.ylabel('Perda')
plt.xlabel('Época')
plt.legend(['Treino', 'Validação'], loc='upper left')
plt.show()




