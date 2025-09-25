# src/utils/data_generator.py
import numpy as np
import tensorflow as tf

def augment_jitter(batch_X, prob=0.8, sigma=0.01):
    if np.random.rand() > prob:
        return batch_X
    noise = np.random.normal(0.0, sigma, size=batch_X.shape).astype(np.float32)
    return batch_X + noise

def generator(X, y, batch_size=16, shuffle=True, augment=False, num_classes=None):
    n = len(X)
    idx = np.arange(n)
    while True:
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            Xb = X[batch_idx].astype(np.float32)
            yb = y[batch_idx].astype(np.int32)
            if augment:
                Xb = augment_jitter(Xb, prob=0.8, sigma=0.01)
            yb_oh = tf.keras.utils.to_categorical(yb, num_classes=num_classes)
            yield Xb, yb_oh