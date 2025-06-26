import torch
from transformers import CLIPTokenizer, CLIPTextModel
import torch.nn.functional as F

def get_text_encoder(model_name='openai/clip-vit-base-patch32', device='cuda' if torch.cuda.is_available() else 'cpu'):
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModel.from_pretrained(model_name).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer

def encode_texts(texts, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=77)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        features = outputs.last_hidden_state[:, 0, :]
        features = F.normalize(features, dim=-1)
    return features.cpu() 