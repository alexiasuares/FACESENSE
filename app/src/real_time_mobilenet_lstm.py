import cv2
import numpy as np

import tensorflow as tf
from collections import deque

# Caminho para o modelo salvo
default_model_path = 'runs/finetune_mobilenet_final/smic-20251002-052100/model_final.keras'

# Parâmetros do modelo
T = 16  # número de frames na sequência (ajuste conforme seu treino)
IMG_SIZE = 224

# Função focal_loss igual ao treino
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = 1e-9
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        fl = weight * ce
        return tf.reduce_sum(fl, axis=-1)
    return loss

# Carrega o modelo com custom_objects
model = tf.keras.models.load_model(
    default_model_path,
    custom_objects={'loss': focal_loss()}
)

# Fila para armazenar os últimos T frames
frame_buffer = deque(maxlen=T)

# Função de pré-processamento igual ao treino
def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    frame = frame.astype('float32')
    frame = (frame / 127.5) - 1.0
    return frame

# Inicia captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_proc = preprocess_frame(frame)
    frame_buffer.append(frame_proc)

    # Só faz predição se já tiver T frames
    if len(frame_buffer) == T:
        seq = np.stack(frame_buffer, axis=0)  # (T, 224, 224, 3)
        seq = np.expand_dims(seq, axis=0)     # (1, T, 224, 224, 3)
        proba = model.predict(seq)
        classe = int(np.argmax(proba, axis=1)[0])
        # Mostra probabilidades de todas as classes
        probas_str = ' | '.join([f'{i}: {p:.2f}' for i, p in enumerate(proba[0])])
        cv2.putText(frame, f'Classe: {classe}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f'Probs: {probas_str}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.imshow('Reconhecimento em tempo real', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
