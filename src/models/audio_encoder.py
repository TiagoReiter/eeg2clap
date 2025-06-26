import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn.functional as F

# Define the output dimension consistently as a constant
AUDIO_EMBEDDING_DIM = 512

def get_audio_encoder(model_name='facebook/wav2vec2-base-960h', device='cuda' if torch.cuda.is_available() else 'cpu'):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # No projection needed as convolutional features are already 512-dimensional
    # Return None as projection to indicate we're using conv features directly
    return processor, model, None

def encode_audios(audio_arrays, processor, model, device='cuda' if torch.cuda.is_available() else 'cpu', sampling_rate=16000, projection=None):
    inputs = processor(audio_arrays, sampling_rate=sampling_rate, return_tensors='pt', padding=True)
    input_values = inputs.input_values.to(device)
    
    # Handle missing attention_mask
    if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None:
        attention_mask = inputs.attention_mask.to(device)
    else:
        # Create simple attention mask of ones
        attention_mask = torch.ones(input_values.shape[0], input_values.shape[1], dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Extract convolutional features directly (similar to brainmagick's Wav2VecConvolution)
        outputs = model(input_values, attention_mask=attention_mask, output_hidden_states=True)
        
        # Access the convolutional features (extract_features)
        # This is equivalent to brainmagick's approach of using name="extract_features"
        conv_features = outputs.extract_features
        
        # Mean pooling over time dimension to get fixed-size embeddings
        pooled_features = conv_features.mean(dim=1)  # [batch_size, 512]
        
        # Normalize embeddings
        pooled_features = F.normalize(pooled_features, dim=-1)
    
    return pooled_features.cpu() 