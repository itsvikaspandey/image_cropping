import cv2
import base64
import numpy as np

def img_to_base64(img: np.ndarray) -> str:
    """Convert NumPy image to base64 string."""
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode()

def base64_to_img(b64_str: str) -> np.ndarray:
    """Convert base64 string to NumPy image."""
    img_data = base64.b64decode(b64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
