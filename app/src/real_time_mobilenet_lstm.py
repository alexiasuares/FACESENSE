import argparse
import csv
import datetime
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from collections import deque

CLASS_NAMES = [
    "Negative",
    "Positive",
    "Surprise",
    "Neutral",
]

# Caminho para o modelo salvo (ajuste conforme necessário)
DEFAULT_MODEL_PATH = (
    'runs/finetune_mobilenet_final/smic-20251002-052100'
    '/model_final.keras'
)


def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = 1e-9
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        fl = weight * ce
        return tf.reduce_sum(fl, axis=-1)

    return loss


def build_arg_parser():
    p = argparse.ArgumentParser(
        description=(
            'Real-time Mobilenet+LSTM inference with smoothing, ' 'stability and logging'
        )
    )
    p.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument('--s', type=int, default=8, help='S: janela de suavização')
    p.add_argument('--h', type=int, default=40, help='H: janela de histórico (percentuais)')
    p.add_argument('--neg-boost', type=float, default=1.0, help='Negative class boost')
    p.add_argument('--threshold', type=float, default=0.35, help='Probabilidade mínima para aceitar predição')
    p.add_argument('--stability', type=int, default=3, help='Consecutivos para aceitar mudança de classe')
    p.add_argument('--log', type=str, default='runs/realtime_predictions.csv', help='CSV log path')
    p.add_argument('--t', type=int, default=None, help='Número de frames usados na inferência (se menor que o T do modelo, será feito upsample por repetição)')
    p.add_argument('--buffer', type=int, default=None, help='Tamanho do buffer para sliding-window (se maior que MODEL_T, serão feitas várias previsões e agregadas)')
    return p


def preprocess_frame(frame, img_size=224):
    frame = cv2.resize(frame, (img_size, img_size))
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    frame = frame.astype('float32')
    frame = (frame / 127.5) - 1.0
    return frame


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    S = int(args.s)
    H = int(args.h)
    NEGATIVE_BOOST = float(args.neg_boost)
    THRESHOLD = float(args.threshold)
    STABILITY = int(args.stability)
    MODEL_PATH = args.model
    LOG_PATH = args.log

    # Model params
    IMG_SIZE = 224

    print('Loading model...', MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'loss': focal_loss()})
    print('Model loaded')

    # determine model temporal length and inference window
    try:
        MODEL_T = int(model.input_shape[1])
    except Exception:
        MODEL_T = 16
    # T is the inference window we will collect from camera; if smaller than MODEL_T
    # we will upsample by repeating frames to reach MODEL_T before calling predict.
    T = int(args.t) if args.t is not None else MODEL_T
    if T != MODEL_T:
        print(f'Inference frames T={T} different from model T={MODEL_T}; upsampling will be applied')

    # Buffers
    frame_buffer = deque(maxlen=T)
    # optional larger buffer for sliding-window aggregation
    buffer_len = int(args.buffer) if args.buffer is not None else T
    big_buffer = deque(maxlen=buffer_len)
    proba_buffer = deque(maxlen=S)
    pred_buffer = deque(maxlen=H)

    # CSV logging
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    csv_file = open(LOG_PATH, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if os.path.getsize(LOG_PATH) == 0:
        csv_writer.writerow([
            'timestamp',
            'frame_idx',
            'predicted_idx',
            'predicted_name',
            'display_idx',
            'display_name',
            'avg_probas',
        ])

    # Display / stability state
    display_idx = None
    display_name = 'Aguardando...'
    candidate_idx = None
    stability_counter = 0

    cap = cv2.VideoCapture(0)
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_proc = preprocess_frame(frame, IMG_SIZE)
            frame_buffer.append(frame_proc)
            big_buffer.append(frame_proc)

            probas_str = ''

            if len(frame_buffer) == T:
                # If big_buffer is larger than MODEL_T, run sliding-window batch predictions
                if len(big_buffer) >= MODEL_T and buffer_len > MODEL_T:
                    buf_frames = np.stack(list(big_buffer), axis=0)
                    n_windows = len(buf_frames) - MODEL_T + 1
                    batch = np.stack([buf_frames[i:i + MODEL_T] for i in range(n_windows)],
                                     axis=0)
                    # batch predict all windows and average probabilities (recent windows can be weighted)
                    probas = model.predict(batch)
                    # simple average across windows
                    avg_window_proba = np.mean(probas, axis=0)
                    proba_buffer.append(avg_window_proba)
                else:
                    # fallback: use frame_buffer content (with upsample if needed)
                    frames = np.stack(frame_buffer, axis=0)
                    if T == MODEL_T:
                        seq = np.expand_dims(frames, axis=0)
                        proba = model.predict(seq)
                        proba_buffer.append(proba[0])
                    elif T < MODEL_T:
                        repeat = MODEL_T // T
                        rem = MODEL_T - (repeat * T)
                        expanded = []
                        for i in range(T):
                            expanded.extend([frames[i]] * repeat)
                        if rem > 0:
                            expanded.extend([frames[-1]] * rem)
                        seq = np.expand_dims(np.stack(expanded, axis=0), axis=0)
                        proba = model.predict(seq)
                        proba_buffer.append(proba[0])
                    else:
                        seq = np.expand_dims(frames[-MODEL_T:], axis=0)
                        proba = model.predict(seq)
                        proba_buffer.append(proba[0])

                if len(proba_buffer) > 0:
                    avg_proba = np.mean(np.stack(list(proba_buffer), axis=0), axis=0)
                else:
                    avg_proba = proba[0]

                # apply negative boost
                if NEGATIVE_BOOST != 1.0:
                    avg_proba[0] = avg_proba[0] * float(NEGATIVE_BOOST)
                    s = np.sum(avg_proba) + 1e-9
                    avg_proba = avg_proba / s

                predicted_idx = int(np.argmax(avg_proba))
                predicted_name = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else str(predicted_idx)

                pred_buffer.append(predicted_idx)

                # stability + threshold logic
                prob_top = float(avg_proba[predicted_idx])
                if prob_top >= THRESHOLD:
                    if candidate_idx == predicted_idx:
                        stability_counter += 1
                    else:
                        candidate_idx = predicted_idx
                        stability_counter = 1

                    if stability_counter >= STABILITY:
                        display_idx = candidate_idx
                        display_name = (
                            CLASS_NAMES[display_idx]
                            if display_idx < len(CLASS_NAMES)
                            else str(display_idx)
                        )
                else:
                    candidate_idx = None
                    stability_counter = 0

                probas_list = [f'{i}: {p:.2f}' for i, p in enumerate(avg_proba)]
                probas_str = ' | '.join(probas_list)

                # CSV write
                ts = datetime.datetime.utcnow().isoformat()
                csv_writer.writerow([
                    ts,
                    frame_idx,
                    predicted_idx,
                    predicted_name,
                    display_idx if display_idx is not None else '',
                    display_name,
                    ';'.join([f'{p:.4f}' for p in avg_proba.tolist()]),
                ])
                csv_file.flush()

            else:
                display_name = 'Aguardando...'
                probas_str = 'Carregando...'

            texto_emocao = f'Classe: {display_name} (Emoção: {display_name})'
            cv2.putText(frame, texto_emocao, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Probs: {probas_str}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # distribution
            if len(pred_buffer) > 0:
                counts = np.bincount(list(pred_buffer), minlength=len(CLASS_NAMES))
                percents = (counts.astype(float) / len(pred_buffer)) * 100.0
                dist_list = [f'{name}: {percents[i]:.0f}%' for i, name in enumerate(CLASS_NAMES)]
                dist_str = ' | '.join(dist_list)
            else:
                dist_str = ''

            cv2.putText(frame, dist_str, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Negative quick log for debugging
            if len(pred_buffer) > 0 and pred_buffer[-1] == 0:
                try:
                    print(f"Negative detected (smoothed prob ~ {avg_proba[0]:.2f})")
                except Exception:
                    pass

            # show negative boost and controls
            cv2.putText(frame, f'NegBoost: {NEGATIVE_BOOST:.2f}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

            cv2.imshow('Reconhecimento em tempo real', frame)
            key = cv2.waitKey(1) & 0xFF
            # controls
            if key == ord('q'):
                break
            elif key == ord('=') or key == ord('+'):
                NEGATIVE_BOOST += 0.1
                print(f'NEGATIVE_BOOST increased to {NEGATIVE_BOOST:.2f}')
            elif key == ord('-'):
                NEGATIVE_BOOST = max(0.1, NEGATIVE_BOOST - 0.1)
                print(f'NEGATIVE_BOOST decreased to {NEGATIVE_BOOST:.2f}')
            elif key == ord('r'):
                pred_buffer.clear()
                proba_buffer.clear()
                candidate_idx = None
                stability_counter = 0
                print('Prediction and probability buffers reset')

            frame_idx += 1

    finally:
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

