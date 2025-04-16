
import os
import torch
import numpy as np
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize
from flask_cors import CORS  

from Config import Config
from model import AesFA_test
from blocks import test_model_load

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # Allow all origins for /predict

# Initialize Model
config = Config()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load model globally
model = AesFA_test(config)
ckpt = os.path.join(config.ckpt_dir, 'main.pth')
model = test_model_load(checkpoint=ckpt, model=model)
model.to(device)
model.eval()

# Image preprocessing function
def do_transform(img, osize):
    transform = Compose([
        Resize(size=osize),
        CenterCrop(size=osize),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform(img).unsqueeze(0)

# Convert tensor to an image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach().numpy()
    image = image.transpose(0, 2, 3, 1)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = (image * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(image[0])

@app.route("/predict", methods=["POST"])
def predict():
    if "content" not in request.files or "style_high" not in request.files or "style_low" not in request.files:
        return jsonify({"error": "Missing one or more images"}), 400

    content_img = request.files["content"]
    style_high_img = request.files["style_high"]
    style_low_img = request.files["style_low"]

    print("Received images:", content_img.filename, style_high_img.filename, style_low_img.filename)

    # Load images
    content = Image.open(content_img).convert("RGB")
    style_high = Image.open(style_high_img).convert("RGB")
    style_low = Image.open(style_low_img).convert("RGB")

    # Convert images to tensors
    content_tensor = do_transform(content, (512, 512)).to(device)
    style_high_tensor = do_transform(style_high, (512, 512)).to(device)
    style_low_tensor = do_transform(style_low, (512, 512)).to(device)

    # Run style blending using the globally loaded model
    stylized_image, _ = model.style_blending(content_tensor, style_high_tensor, style_low_tensor)

    # Convert tensor to PIL image
    output_image = im_convert(stylized_image)

    # Convert image to Base64 for response
    buffered = BytesIO()
    output_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    print("Returning generated image...")
    return jsonify({"output": img_str})

if __name__ == '__main__':
    app.run(debug=True)
