from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import torch
import torch.nn.functional as F
from torchvision import transforms
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# -------------------------------------------------------
# 1️⃣ FastAPI Initialization
# -------------------------------------------------------
app = FastAPI(
    title="PlantDocBot API 🌿",
    description="API for Plant Disease Detection using Image and Text Models",
    version="1.0"
)

# -------------------------------------------------------
# 2️⃣ Device Configuration
# -------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# 3️⃣ Image Classification Model
# -------------------------------------------------------
class PlantDiseaseModel(torch.nn.Module):
    def __init__(self, num_classes=38):
        super(PlantDiseaseModel, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 28 * 28, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load image model weights
cnn = PlantDiseaseModel(num_classes=38)
cnn.load_state_dict(torch.load("models/plant_disease_model.pth", map_location=device))
cnn.to(device)
cnn.eval()

# Image preprocessing
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4760, 0.5004, 0.4266],
                         std=[0.1775, 0.1509, 0.1960])
])

# Class labels
class_names = [
    "Apple Scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew","Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy",
    "Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy","Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot","Peach___healthy","Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy",
    "Potato___Early_blight","Potato___Late_blight","Potato___healthy","Raspberry___healthy","Soybean___healthy",
    "Squash___Powdery_mildew","Strawberry___Leaf_scorch","Strawberry___healthy",
    "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight",
    "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot","Tomato___Tomato_mosaic_virus","Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___healthy"
]

# -------------------------------------------------------
# 4️⃣ Text Classification Model
# -------------------------------------------------------
# ✅ Use resolved absolute path to the folder in same root
text_model_path = Path("plant_classifier_model").resolve()

# ✅ Load tokenizer and model safely from local files
tokenizer = AutoTokenizer.from_pretrained(text_model_path, local_files_only=True)
text_model = AutoModelForSequenceClassification.from_pretrained(text_model_path, local_files_only=True).to(device)
text_model.eval()

# ✅ Load label encoder
encoder = joblib.load(text_model_path / "label_encoder.pkl")

def predict_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        confidence, pred_id = torch.max(probs, dim=-1)
    
    label = encoder.inverse_transform([pred_id.item()])[0]
    confidence_score = round(confidence.item(), 4)
    recommendation = (
        "Your plant looks healthy 🌱"
        if "healthy" in label.lower()
        else "Your plant seems affected. Consider proper diagnosis and treatment."
    )

    return {"label": label, "confidence": confidence_score, "recommendation": recommendation}

# -------------------------------------------------------
# 5️⃣ Pydantic Models
# -------------------------------------------------------
class TextPredictionInputModel(BaseModel):
    input: str

# -------------------------------------------------------
# 6️⃣ Health Check
# -------------------------------------------------------
@app.get("/health-check")
def health_check():
    return {"status": "ok", "message": "PlantDocBot API is running 🚀"}

# -------------------------------------------------------
# 7️⃣ Image Prediction Endpoint
# -------------------------------------------------------
@app.post("/image-prediction")
async def image_predict(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read())).convert("RGB")
        img_tensor = image_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = cnn(img_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)

        label = class_names[pred_class.item()]
        confidence_score = round(confidence.item(), 4)
        recommendation = (
            "Your plant looks healthy 🌱"
            if "healthy" in label.lower()
            else "Your plant seems affected. Consider proper diagnosis and treatment."
        )

        return {
            "filename": file.filename,
            "label": label,
            "confidence": confidence_score,
            "recommendation": recommendation
        }

    except Exception as e:
        return {"error": str(e)}

# -------------------------------------------------------
# 8️⃣ Text Prediction Endpoint
# -------------------------------------------------------
@app.post("/text-prediction")
def text_predict_endpoint(input_data: TextPredictionInputModel):
    try:
        result = predict_text(input_data.input)
        return {"input_text": input_data.input, **result}
    except Exception as e:
        return {"error": str(e)}
