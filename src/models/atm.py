# src/eeg2clap/models/atm.py
"""
Adaptive Thinking Mapper (ATM)
------------------------------------------------
An EEG-to-CLIP encoder inspired by Li et al. 2024 :contentReference[oaicite:0]{index=0}.
– Patchify raw EEG (shape B × C × T) into N fixed-length chunks
– Linear( C·patch_len → d_model )
– Add spatial-temporal position encodings
– TransformerEncoder (L layers, multi-head self-attn)
– CLS token → MLP projector → 512-d CLIP space
"""
import os
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

os.environ["WANDB_API_KEY"] = "fc2d1bc24195ed0dc256cf0d1a94a44630eff1e7"
os.environ["WANDB_MODE"] = 'online'
from itertools import combinations

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from eegdatasets_leaveone import EEGDataset

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from util import wandb_logger
import csv
from torch import Tensor
import itertools
import math
import re
from loss import ClipLoss
import argparse
from torch import nn
from torch.optim import AdamW
import datetime

from typing import Tuple, Optional

class EEGPatchEmbed(nn.Module):
    def __init__(self, patch_len: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(in_chans * patch_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, C, T
        B, C, T = x.shape
        assert T % self.patch_len == 0, "T must be divisible by patch_len"
        x = x.unfold(2, self.patch_len, self.patch_len)  # B, C, N, patch
        x = x.contiguous().view(B, C * self.patch_len, -1)  # B, C*patch, N
        x = x.transpose(1, 2)                              # B, N, C*patch
        return self.proj(x)                                # B, N, D

class ATM(nn.Module):
    def __init__(
        self,
        patch_len: int = 10,
        in_chans: int = 64,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        proj_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.patch_embed = EEGPatchEmbed(patch_len, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, embed_dim))  # +1 for CLS
        enc_layer = nn.TransformerEncoderLayer(embed_dim, num_heads,
                                               dim_feedforward=4*embed_dim,
                                               dropout=dropout,
                                               batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, depth)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(proj_dim)
        )
        # Add logit scale and loss function for training
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights properly"""
        # Initialize CLS token with small random values
        torch.nn.init.normal_(self.cls_token, std=0.02)
        # Initialize position embeddings with small random values  
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        # Initialize projection layer
        torch.nn.init.xavier_uniform_(self.projector[0].weight)
        torch.nn.init.zeros_(self.projector[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: float32  B × C × T   (eeg_words from your .npz)
        returns:    B × 512     512-d CLIP-aligned embedding
        """
        tok = self.patch_embed(x)                     # B × N × D
        cls = self.cls_token.expand(tok.size(0), -1, -1)
        tok = torch.cat([cls, tok], dim=1)            # prepend CLS
        pos = self.pos_embed[:, :tok.size(1), :]
        h = self.encoder(tok + pos)
        return self.projector(h[:, 0])      

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

def train_model(sub, eeg_model, dataloader, optimizer, device, text_features_all, audio_features_all, config):
    eeg_model.train()
    text_features_all = text_features_all.to(device).float()
    audio_features_all = (audio_features_all[::10]).to(device).float()  # Changed from img to audio
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.5  # Equal weighting for audio and text since we're not sure about categories yet
    features_list = []
    
    for batch_idx, (eeg_data, labels, text, text_features, audio, audio_features) in enumerate(dataloader):  # Changed img to audio
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        audio_features = audio_features.to(device).float()  # Changed from img to audio
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # ATM forward pass
        eeg_features = eeg_model(eeg_data).float()
        
        features_list.append(eeg_features)
        logit_scale = eeg_model.logit_scale
        
        # Use audio features instead of image features
        audio_loss = eeg_model.loss_func(eeg_features, audio_features, logit_scale)
        text_loss = eeg_model.loss_func(eeg_features, text_features, logit_scale)
        loss = alpha * audio_loss + (1 - alpha) * text_loss
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
        
        # Simple accuracy calculation using audio features
        logits_audio = logit_scale * eeg_features @ audio_features_all.T
        logits_single = logits_audio
        predicted = torch.argmax(logits_single, dim=1)

        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()
        del eeg_data, eeg_features, audio_features
        
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    return average_loss, accuracy, torch.cat(features_list, dim=0)

def evaluate_model(sub, eeg_model, dataloader, device, text_features_all, audio_features_all, config):
    """Simplified evaluation - just basic loss and accuracy for now"""
    eeg_model.eval()
    
    text_features_all = text_features_all.to(device).float()
    audio_features_all = audio_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.5  # Equal weighting
    
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, audio, audio_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            audio_features = audio_features.to(device).float()
            
            eeg_features = eeg_model(eeg_data)
        
            logit_scale = eeg_model.logit_scale 
            audio_loss = eeg_model.loss_func(eeg_features, audio_features, logit_scale)
            text_loss = eeg_model.loss_func(eeg_features, text_features, logit_scale)
            loss = audio_loss*alpha + text_loss*(1-alpha)
            
            total_loss += loss.item()
            
            # Simple accuracy using audio features
            logits_audio = logit_scale * eeg_features @ audio_features_all.T
            predicted = torch.argmax(logits_audio, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            del eeg_data, eeg_features, audio_features
            
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    return average_loss, accuracy

def main_train_loop(sub, current_time, eeg_model, train_dataloader, test_dataloader, optimizer, device, text_features_train_all, text_features_test_all, audio_features_train_all, audio_features_test_all, config, logger=None):
    logger = wandb_logger(config) if logger else None
    logger.watch(eeg_model, logger) 
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    best_accuracy = 0.0
    best_model_weights = None
    best_epoch_info = {}
    results = []
    
    for epoch in range(config.epochs):
        # Train the model
        train_loss, train_accuracy, features_tensor = train_model(sub, eeg_model, train_dataloader, optimizer, device, text_features_train_all, audio_features_train_all, config=config)
        
        # Save model checkpoints
        if (epoch + 1) % 5 == 0:                    
            if config.insubject == True:       
                os.makedirs(f"./models/atm/{sub}/{current_time}", exist_ok=True)             
                file_path = f"./models/atm/{sub}/{current_time}/{epoch+1}.pth"
                torch.save(eeg_model.state_dict(), file_path)            
            else:                
                os.makedirs(f"./models/atm/across/{current_time}", exist_ok=True)             
                file_path = f"./models/atm/across/{current_time}/{epoch+1}.pth"
                torch.save(eeg_model.state_dict(), file_path)
            print(f"model saved in {file_path}!")
            
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate the model (simplified)
        test_loss, test_accuracy = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, audio_features_test_all, config=config)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        }

        results.append(epoch_results)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy
            }
            
        logger.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy,
            "Epoch": epoch
        })

        print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
  
    # Create simplified plots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(test_losses, label='Test Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    axs[0, 1].plot(train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(test_accuracies, label='Test Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")

    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
                f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
                f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}")

    axs[1, 0].axis('off')  
    axs[1, 0].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[1, 0].transAxes)
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.suptitle('ATM Training Results', fontsize=16, y=1.02)
    plt.savefig('ATM_training_results')
    logger.finish()
    return results

def main():
    parser = argparse.ArgumentParser(description='ATM Training Script')
    parser.add_argument('--data_path', type=str, default=r"D:\Universität\Master\4. Semester\Forschungspraxis\fp_python\Data\Preprocessed_data_250Hz\ThingsEEG", help='Path to the EEG dataset')
    parser.add_argument('--output_dir', type=str, default=r'D:\Universität\Master\4. Semester\Forschungspraxis\fp_python\Data\Output', help='Directory to save output results')    
    parser.add_argument('--project', type=str, default="Forschungspraxis", help='WandB project name')
    parser.add_argument('--entity', type=str, default="t-reiter-technical-university-of-munich", help='WandB entity name')
    parser.add_argument('--name', type=str, default="ATM_lr=3e-4_audio", help='Experiment name')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--logger', type=bool, default=True, help='Enable WandB logging')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU device to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device to run on (cpu or gpu)')    
    parser.add_argument('--insubject', type=bool, default=True, help='In-subject mode or cross-subject mode')
    parser.add_argument('--encoder_type', type=str, default='ATM', help='Encoder type')
    parser.add_argument('--subjects', nargs='+', default=['sub-08'], help='List of subject IDs') 
    
    # ATM-specific parameters
    parser.add_argument('--patch_len', type=int, default=10, help='EEG patch length')
    parser.add_argument('--in_chans', type=int, default=63, help='Number of EEG channels')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--proj_dim', type=int, default=512, help='Projection dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    args = parser.parse_args()

    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device(args.gpu)
    else:
        device = torch.device('cpu')

    subjects = args.subjects        
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    for sub in subjects:
        # Create ATM model with command line parameters
        eeg_model = ATM(
            patch_len=args.patch_len,
            in_chans=args.in_chans,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            proj_dim=args.proj_dim,
            dropout=args.dropout
        )
        eeg_model.to(device)

        optimizer = AdamW(eeg_model.parameters(), lr=args.lr)

        if args.insubject:
            train_dataset = EEGDataset(args.data_path, subjects=[sub], train=True)
            test_dataset = EEGDataset(args.data_path, subjects=[sub], train=False)
        else:
            train_dataset = EEGDataset(args.data_path, exclude_subject=sub, subjects=subjects, train=True)
            test_dataset = EEGDataset(args.data_path, exclude_subject=sub, subjects=subjects, train=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

        text_features_train_all = train_dataset.text_features
        text_features_test_all = test_dataset.text_features
        # Changed from img_features to audio_features
        audio_features_train_all = train_dataset.img_features  # Assuming this contains audio features
        audio_features_test_all = test_dataset.img_features    # Will need to be updated when you have actual audio features

        results = main_train_loop(sub, current_time, eeg_model, train_loader, test_loader, optimizer, device, 
                                  text_features_train_all, text_features_test_all, audio_features_train_all, audio_features_test_all, config=args, logger=args.logger)

        results_dir = os.path.join(args.output_dir, args.encoder_type, sub, current_time)
        os.makedirs(results_dir, exist_ok=True)

        if args.insubject:
            results_file = f"{results_dir}/{args.encoder_type}_{sub}.csv"
        else:
            results_file = f"{results_dir}/{args.encoder_type}_cross_exclude_{sub}.csv"

        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            print(f'Results saved to {results_file}')

if __name__ == '__main__':
    main()      