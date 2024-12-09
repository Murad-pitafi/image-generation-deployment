# model.py
from transformers import AutoTokenizer
from torchvision import models, transforms
from PIL import Image
import torch
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

# Initialize the tokenizer and ResNet model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
resnet_model = models.resnet101(pretrained=True)
resnet_model.eval()

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Process the image and extract features
def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Generate the BLEU score and report
def generate_bleu_score_and_report(image_path, reference_reports):
    # Process image
    image = process_image(image_path)
    
    # Extract ResNet features (if needed, this can be extended to generate features)
    with torch.no_grad():
        features = resnet_model(image)
    
    # Here, you would generate the model's report (dummy example)
    generated_report = "The heart size is normal with no signs of disease."
    
    # Calculate BLEU score (you can use actual references)
    bleu_score = corpus_bleu(reference_reports, [generated_report])
    
    # Create a detailed report
    report = f"""
    Report for Uploaded X-ray Image: {generated_report}
    ---------------------------------
    BLEU Score: {bleu_score} 
    """
    
    return bleu_score, report
