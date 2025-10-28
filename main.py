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
from fastapi.middleware.cors import CORSMiddleware


# FastAPI Initialization
app = FastAPI(
    title="PlantDocBot API 🌿",
    description="API for Plant Disease Detection using Image and Text Models",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Image Classification Model
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
    "Apple - Scab",
    "Apple - Black Rot",
    "Apple - Cedar Apple Rust",
    "Apple - Healthy",
    "Blueberry - Healthy",
    "Cherry - Powdery Mildew",
    "Cherry - Healthy",
    "Corn (Maize) - Cercospora Leaf Spot (Gray Leaf Spot)",
    "Corn (Maize) - Common Rust",
    "Corn (Maize) - Northern Leaf Blight",
    "Corn (Maize) - Healthy",
    "Grape - Black Rot",
    "Grape - Esca (Black Measles)",
    "Grape - Leaf Blight (Isariopsis Leaf Spot)",
    "Grape - Healthy",
    "Orange - Huanglongbing (Citrus Greening)",
    "Peach - Bacterial Spot",
    "Peach - Healthy",
    "Pepper (Bell) - Bacterial Spot",
    "Pepper (Bell) - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Raspberry - Healthy",
    "Soybean - Healthy",
    "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch",
    "Strawberry - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites (Two-Spotted Spider Mite)",
    "Tomato - Target Spot",
    "Tomato - Mosaic Virus",
    "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Healthy"
]



recommendations = {
    "Apple - Scab": "Prune infected leaves and apply fungicide (captan or mancozeb). Avoid overhead watering. Collect and destroy fallen leaves to prevent reinfection.",
    "Apple - Black Rot": "Remove infected fruits, twigs, and cankers. Use copper-based fungicides. Avoid wounding the tree during pruning.",
    "Apple - Cedar Apple Rust": "Eliminate nearby juniper trees, apply sulfur or myclobutanil sprays, and ensure good air circulation.",
    "Apple - Healthy": "Your apple plant looks healthy 🍎 Keep monitoring for leaf spots and maintain balanced fertilization.",
    
    "Blueberry - Healthy": "Your blueberry plant is thriving 🫐 Keep soil acidic (pH 4.5–5.5), mulch regularly, and avoid waterlogging.",
    
    "Cherry - Powdery Mildew": "Use sulfur sprays or neem oil. Prune to improve airflow and remove infected leaves early.",
    "Cherry - Healthy": "Your cherry tree is healthy 🍒 Keep soil moist but not soggy, and prune after harvest to prevent fungus buildup.",
    
    "Corn (Maize) - Cercospora Leaf Spot (Gray Leaf Spot)": "Rotate crops yearly, use resistant hybrids, and apply strobilurin-based fungicides if infection is severe.",
    "Corn (Maize) - Common Rust": "Plant resistant varieties, and apply fungicides like propiconazole at early signs. Maintain field sanitation.",
    "Corn (Maize) - Northern Leaf Blight": "Use disease-free seeds, resistant hybrids, and apply fungicides during early infection.",
    "Corn (Maize) - Healthy": "Corn looks healthy 🌽 Maintain weed control and consistent irrigation for strong growth.",
    
    "Grape - Black Rot": "Prune infected vines, apply fungicides (mancozeb or myclobutanil), and remove mummified berries after harvest.",
    "Grape - Esca (Black Measles)": "Remove and destroy infected wood. Avoid pruning during wet weather. Apply trunk protectants.",
    "Grape - Leaf Blight (Isariopsis Leaf Spot)": "Spray copper fungicides, ensure good spacing between vines, and manage canopy airflow.",
    "Grape - Healthy": "Your grapevine is healthy 🍇 Maintain balanced fertilization and regular pruning for airflow.",
    
    "Orange - Huanglongbing (Citrus Greening)": "Sadly, there’s no cure. Remove infected trees, control psyllid populations, and use certified disease-free saplings.",
    
    "Peach - Bacterial Spot": "Avoid overhead irrigation. Use copper sprays during early growth and plant resistant cultivars.",
    "Peach - Healthy": "Healthy peach tree 🍑 Keep soil well-drained and apply dormant sprays in winter to prevent fungal infection.",
    
    "Pepper (Bell) - Bacterial Spot": "Use disease-free seeds, rotate crops, and apply copper-based bactericides. Avoid handling wet leaves.",
    "Pepper (Bell) - Healthy": "Pepper plant is healthy 🌶️ Keep humidity moderate and fertilize regularly with potassium-rich feed.",
    
    "Potato - Early Blight": "Rotate crops, remove infected leaves, and spray chlorothalonil fungicide. Avoid overhead irrigation.",
    "Potato - Late Blight": "Destroy infected plants immediately. Apply metalaxyl fungicide and ensure proper field drainage.",
    "Potato - Healthy": "Your potato plant is healthy 🥔 Maintain soil health with compost and monitor for leaf lesions.",
    
    "Raspberry - Healthy": "Healthy raspberry bush 🍓 Prune dead canes, mulch, and water consistently to prevent stress.",
    
    "Soybean - Healthy": "Soybean crop looks healthy 🌱 Watch for aphids and rotate crops yearly to prevent soil pathogens.",
    
    "Squash - Powdery Mildew": "Spray neem oil or potassium bicarbonate weekly. Ensure good airflow and avoid crowding plants.",
    
    "Strawberry - Leaf Scorch": "Remove infected leaves, apply organic compost tea, and avoid wet foliage during irrigation.",
    "Strawberry - Healthy": "Healthy strawberry plant 🍓 Keep soil moist, ensure good sunlight, and watch for aphids or mites.",
    
    "Tomato - Bacterial Spot": "Use copper sprays, avoid working with wet plants, and plant resistant varieties if available.",
    "Tomato - Early Blight": "Apply fungicides like mancozeb or chlorothalonil. Remove lower leaves and stake plants for airflow.",
    "Tomato - Late Blight": "Destroy infected plants, disinfect tools, and apply preventive fungicides regularly.",
    "Tomato - Leaf Mold": "Ensure good ventilation, reduce humidity, and apply bio-fungicides like *Trichoderma*.",
    "Tomato - Septoria Leaf Spot": "Remove infected leaves and apply fungicide every 7–10 days. Rotate crops annually.",
    "Tomato - Spider Mites (Two-Spotted Spider Mite)": "Increase humidity, spray neem oil or insecticidal soap, and remove heavily infested leaves.",
    "Tomato - Target Spot": "Remove affected leaves, use copper fungicides, and maintain proper spacing between plants.",
    "Tomato - Mosaic Virus": "No cure. Remove infected plants and disinfect tools. Control aphids to reduce spread.",
    "Tomato - Yellow Leaf Curl Virus": "Remove infected plants, use insect-proof nets, and control whiteflies organically with neem oil.",
    "Tomato - Healthy": "Your tomato plant looks healthy 🍅 Maintain consistent watering, sunlight, and balanced nutrients."
}



# Text Classification Model
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


class TextPredictionInputModel(BaseModel):
    input: str


# Health Check
@app.get("/health-check")
def health_check():
    return {"status": "ok", "message": "PlantDocBot API is running 🚀"}

# Image Prediction Endpoint
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
        recommendation = recommendations.get(label, "No specific recommendation available for this disease.")


        return {
            "filename": file.filename,
            "label": label,
            "confidence": confidence_score,
            "recommendation": recommendation
        }

    except Exception as e:
        return {"error": str(e)}


# Text Prediction Endpoint
@app.post("/text-prediction")
def text_predict_endpoint(input_data: TextPredictionInputModel):
    try:
        result = predict_text(input_data.input)
        return {"input_text": input_data.input, **result}
    except Exception as e:
        return {"error": str(e)}