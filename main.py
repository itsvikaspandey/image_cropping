from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2

from core.orientation import detect_skew_angle, rotate_image
from core.cropping import crop_document
from core.utils import img_to_base64

app = FastAPI()

@app.post("/process-doc")
async def process_document(file: UploadFile):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    angle = detect_skew_angle(image)
    rotated = rotate_image(image, -angle)
    cropped = crop_document(rotated)
    
    return JSONResponse({
        "rotation_angle": angle,
        "image_base64": img_to_base64(cropped)
    })
