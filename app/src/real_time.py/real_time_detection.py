import cv2
import mediapipe as mp
import numpy as np
import time

# Inicializar soluções do MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configurar
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Abrir webcam
cap = cv2.VideoCapture(0)

print("🎥 Pressione 'q' para sair.")

sequence = []   # aqui guardamos os frames como no iMiGUE
MAX_FRAMES = 300  # tamanho fixo da sequência (igual ao preprocessamento)

def extract_features(face_results, pose_results, hands_results):
    """
    Extrai um vetor de features [411 dimensões] similar ao iMiGUE:
    - 25 pontos do corpo
    - 70 pontos faciais
    - 21 da mão esquerda
    - 21 da mão direita
    Cada ponto = (x, y, confiança)
    """
    features = []

    # Pose (25 pontos principais)
    if pose_results.pose_landmarks:
        for lm in pose_results.pose_landmarks.landmark[:25]:
            features.extend([lm.x, lm.y, lm.visibility])
    else:
        features.extend([0.0, 0.0, 0.0] * 25)

    # Face (70 pontos)
    if face_results.multi_face_landmarks:
        for lm in list(face_results.multi_face_landmarks[0].landmark)[:70]:
            features.extend([lm.x, lm.y, 1.0])
    else:
        features.extend([0.0, 0.0, 0.0] * 70)

    # Mão esquerda (21 pontos)
    if hands_results.multi_hand_landmarks:
        if len(hands_results.multi_hand_landmarks) > 0:
            for lm in hands_results.multi_hand_landmarks[0].landmark[:21]:
                features.extend([lm.x, lm.y, 1.0])
        else:
            features.extend([0.0, 0.0, 0.0] * 21)
    else:
        features.extend([0.0, 0.0, 0.0] * 21)

    # Mão direita (21 pontos)
    if hands_results.multi_hand_landmarks and len(hands_results.multi_hand_landmarks) > 1:
        for lm in hands_results.multi_hand_landmarks[1].landmark[:21]:
            features.extend([lm.x, lm.y, 1.0])
    else:
        features.extend([0.0, 0.0, 0.0] * 21)

    return np.array(features, dtype=np.float32)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converter BGR para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar com MediaPipe
    face_results = face_mesh.process(rgb_frame)
    pose_results = pose.process(rgb_frame)
    hands_results = hands.process(rgb_frame)

    # Extrair features
    features = extract_features(face_results, pose_results, hands_results)
    sequence.append(features)

    # Manter sequência no tamanho máximo
    if len(sequence) > MAX_FRAMES:
        sequence = sequence[-MAX_FRAMES:]

    # Desenhar landmarks
    if face_results.multi_face_landmarks:
        for landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION)

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar frame
    cv2.imshow('MediaPipe + OpenCV (Features)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

sequence = np.array(sequence)  # (frames, 411)
print("✅ Sequência capturada com shape:", sequence.shape)