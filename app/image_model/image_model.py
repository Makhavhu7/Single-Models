# image_model/serve_image.py
import io
from PIL import Image
import torch
from torchvision import transforms

# Example: Load your image model
# Replace this with your actual model loading code
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def process_image(file_bytes: bytes):
    """
    Process an image and return the model prediction.
    """
    try:
        # Load image
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = output.argmax(dim=1).item()

        return {
            "status": "success",
            "predicted_class": int(pred_class)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }