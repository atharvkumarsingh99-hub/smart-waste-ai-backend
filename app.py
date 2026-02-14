from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load model once
model = load_model("garbageclassifier.h5")

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/")
def home():
    return {"message": "Garbage AI API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        result = "Garbage"
    else:
        result = "Not Garbage"

    return {
        "prediction": result,
        "confidence": confidence
    }
