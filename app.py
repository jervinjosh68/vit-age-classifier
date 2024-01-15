from PIL import Image
import gradio as gr
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch

# Init model, transforms
model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')

def predict(im):
    labels = {0:"0-2", 1: "3-9" , 2: "10-19", 3: "20-29", 4: "30-39", 5: "40-49", 6: "50-59", 7:"60-69",8:"more than 70"} 
# Transform our image and pass it through the model
    inputs = transforms(im, return_tensors='pt')
    output = model(**inputs)

# Predicted Class probabilities
    proba = output.logits.softmax(1)

# Predicted Classes
    preds = proba.argmax(1)
    values, indices = torch.topk(proba, k=5)
    
    

    return {labels[i.item()]: v.item() for i, v in zip(indices.numpy()[0], values.detach().numpy()[0])}

inputs = [
    gr.inputs.Image(type="pil", label="Input Image")
]



title = "ViT-Age-Classification"
description = "ViT-Age-Classification is used to categorize an individual's age using images"
article = " <a href='https://huggingface.co/nateraw/vit-age-classifier'>ViT Age Classification Model Repo on Hugging Face Model Hub</a>"
examples = ["stock_baby.webp","stock_teen.webp","stock_guy.jpg","stock_old_woman.jpg"]

gr.Interface(
    predict,
    inputs,
    outputs = 'label',
    title=title,
    description=description,
    article=article,
    examples=examples,
    theme="huggingface",
).launch(debug=True, enable_queue=True)