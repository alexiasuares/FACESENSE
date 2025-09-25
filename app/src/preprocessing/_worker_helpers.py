import cv2

# small helper function that is safe to pickle for ProcessPool

def read_resize_worker(path_str):
    try:
        img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return path_str, None
        img = cv2.resize(img, (112, 112))
        return path_str, img
    except Exception:
        return path_str, None
