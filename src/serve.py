from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from model import get_model

app = FastAPI()

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


def load_model():
    global model

    checkpoint_path = Path("/app/checkpoints/classifier_v1.pt")

    if not checkpoint_path.exists():
        checkpoint_path = Path("checkpoints/classifier_v1.pt")

    model = get_model("simple_cnn", 10)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()


@app.on_event("startup")
def startup():
    load_model()


@app.get("/health")
def health():
    if model is None:
        return {"status": "not_ready"}

    return {"status": "healthy"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()

    img = Image.open(BytesIO(contents)).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)[0]

    return {
        "class_id": int(torch.argmax(probabilities).item()),
        "probabilities": probabilities.tolist()
    }
