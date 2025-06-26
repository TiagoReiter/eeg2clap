import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn.functional as F

def get_audio_encoder(model_name='facebook/wav2vec2-base-960h', device='cuda' if torch.cuda.is_available() else 'cpu'):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return processor, model

def encode_audios(audio_arrays, processor, model, device='cuda' if torch.cuda.is_available() else 'cpu', sampling_rate=16000):
    inputs = processor(audio_arrays, sampling_rate=sampling_rate, return_tensors='pt', padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)
    with torch.no_grad():
        outputs = model(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        sum_hidden = torch.sum(hidden_states * mask, dim=1)
        sum_mask = torch.sum(mask, dim=1)
        pooled_features = sum_hidden / (sum_mask + 1e-8)
        pooled_features = F.normalize(pooled_features, dim=-1)
    return pooled_features.cpu() 