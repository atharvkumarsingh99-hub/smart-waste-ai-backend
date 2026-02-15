from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os

app = FastAPI()

MODEL_PATH = "garbage_classifier.h5"
model = None


# Lazy load model (safer for Render)
def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"{MODEL_PATH} not found in project root")
        model = load_model(MODEL_PATH)
    return model


def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.get("/")
def home():
    return {"message": "Garbage AI API Running Successfully"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        processed_image = preprocess_image(image)

        model = get_model()
        prediction = model.predict(processed_image)

        confidence = float(np.max(prediction))
        class_index = int(np.argmax(prediction))

        return {
            "class_index": class_index,
            "confidence": confidence
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# IMPORTANT for Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
