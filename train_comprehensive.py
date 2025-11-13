"""
COMPREHENSIVE TRAINING - ALL DATASETS INCLUDED
Prevents catastrophic forgetting by including ASVspoof + ElevenLabs + LibriSpeech + Your Voice

This will create ONE model that detects:
- Modern neural TTS (ElevenLabs)
- Traditional TTS (ASVspoof)
- Real voices (LibriSpeech, Your Voice)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import librosa
import numpy as np
from pathlib import Path
import sys
import os
from tqdm import tqdm
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from models.aasist import get_model

# Configuration
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
NUM_WORKERS = 0  # Windows compatibility
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AudioDataset(Dataset):
    """Simple dataset without heavy augmentation"""
    def __init__(self, file_paths, labels, sample_rate=16000, max_length=64000):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_length = max_length
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load audio with librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Light augmentation (50% chance)
            if random.random() < 0.5:
                # Add small noise
                noise_amp = random.uniform(0.001, 0.005)
                audio = audio + noise_amp * np.random.randn(len(audio))
                audio = np.clip(audio, -1.0, 1.0)
            
            # Pad or truncate
            if len(audio) < self.max_length:
                audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')
            else:
                audio = audio[:self.max_length]
            
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # Add channel dim
            
            return audio_tensor, label
            
        except Exception as e:
            print(f"\nError loading {audio_path}: {e}")
            # Return silence on error
            return torch.zeros(1, self.max_length), label

def collect_asvspoof_samples(max_samples_per_class=3000):
    """Collect ASVspoof samples from training set"""
    print("\n📦 Collecting ASVspoof2019 samples...")
    
    protocol_file = Path("data/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
    audio_dir = Path("data/ASVspoof2019/LA/ASVspoof2019_LA_train/flac")
    
    if not protocol_file.exists() or not audio_dir.exists():
        print("  ⚠️  ASVspoof training data not found, skipping...")
        return [], [], [], []
    
    bonafide_files = []
    spoof_files = []
    
    # Read protocol
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                audio_id = parts[1]
                label_str = parts[4]
                
                audio_file = audio_dir / f"{audio_id}.flac"
                if audio_file.exists():
                    if label_str == 'bonafide':
                        bonafide_files.append(str(audio_file))
                    elif label_str == 'spoof':
                        spoof_files.append(str(audio_file))
    
    # Sample to balance with other datasets
    bonafide_files = bonafide_files[:max_samples_per_class]
    spoof_files = spoof_files[:max_samples_per_class]
    
    print(f"  ✓ ASVspoof bonafide (real): {len(bonafide_files)} samples")
    print(f"  ✓ ASVspoof spoof (fake): {len(spoof_files)} samples")
    
    bonafide_labels = [0] * len(bonafide_files)  # 0 = REAL
    spoof_labels = [1] * len(spoof_files)  # 1 = FAKE
    
    return bonafide_files, bonafide_labels, spoof_files, spoof_labels

def main():
    print("="*80)
    print("COMPREHENSIVE TRAINING - ALL DATASETS")
    print("="*80)
    print(f"\nDevice: {DEVICE}")
    print(f"Configuration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    # ===== COLLECT ALL DATASETS =====
    print("\n" + "="*80)
    print("STEP 1: COLLECTING ALL DATASETS")
    print("="*80)
    
    # 1. ElevenLabs (FAKE)
    print("\n📦 Collecting ElevenLabs samples...")
    eleven_dir = Path("data/modern_tts_samples/fake")
    eleven_files = [str(f) for f in eleven_dir.glob("*.mp3")]
    # Replicate to balance
    eleven_files = eleven_files * max(1, 3000 // len(eleven_files))
    eleven_files = eleven_files[:3000]
    eleven_labels = [1] * len(eleven_files)  # 1 = FAKE
    print(f"  ✓ ElevenLabs (fake): {len(eleven_files)} samples")
    
    # 2. LibriSpeech (REAL)
    print("\n📦 Collecting LibriSpeech samples...")
    libre_dir = Path("data/downloaded_dataset/librispeech")
    libre_files = [str(f) for f in libre_dir.rglob("*.flac")][:3000]
    libre_labels = [0] * len(libre_files)  # 0 = REAL
    print(f"  ✓ LibriSpeech (real): {len(libre_files)} samples")
    
    # 3. Your Voice (REAL)
    print("\n📦 Collecting Your Voice samples...")
    voice_dir = Path("data/your_voice_samples/real")
    voice_files = [str(f) for f in voice_dir.glob("*.flac")]
    # Replicate to have more samples
    voice_files = voice_files * max(1, 300 // len(voice_files))
    voice_files = voice_files[:300]
    voice_labels = [0] * len(voice_files)  # 0 = REAL
    print(f"  ✓ Your Voice (real): {len(voice_files)} samples")
    
    # 4. ASVspoof (REAL + FAKE)
    asv_real_files, asv_real_labels, asv_fake_files, asv_fake_labels = collect_asvspoof_samples(max_samples_per_class=3000)
    
    # ===== COMBINE ALL DATA =====
    print("\n" + "="*80)
    print("STEP 2: COMBINING DATASETS")
    print("="*80)
    
    all_real_files = libre_files + voice_files + asv_real_files
    all_real_labels = libre_labels + voice_labels + asv_real_labels
    
    all_fake_files = eleven_files + asv_fake_files
    all_fake_labels = eleven_labels + asv_fake_labels
    
    print(f"\nCombined Dataset:")
    print(f"  REAL samples: {len(all_real_files)}")
    print(f"    - LibriSpeech: {len(libre_files)}")
    print(f"    - Your Voice: {len(voice_files)}")
    print(f"    - ASVspoof bonafide: {len(asv_real_files)}")
    print(f"\n  FAKE samples: {len(all_fake_files)}")
    print(f"    - ElevenLabs: {len(eleven_files)}")
    print(f"    - ASVspoof spoof: {len(asv_fake_files)}")
    print(f"\n  Total: {len(all_real_files) + len(all_fake_files)} samples")
    print(f"  Balance: {len(all_real_files)/(len(all_real_files)+len(all_fake_files))*100:.1f}% real, {len(all_fake_files)/(len(all_real_files)+len(all_fake_files))*100:.1f}% fake")
    
    # Create datasets
    real_dataset = AudioDataset(all_real_files, all_real_labels)
    fake_dataset = AudioDataset(all_fake_files, all_fake_labels)
    
    # Combine
    full_dataset = ConcatDataset([real_dataset, fake_dataset])
    
    # Split train/val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"\nSplit:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # ===== LOAD BASE MODEL =====
    print("\n" + "="*80)
    print("STEP 3: LOADING BASE MODEL")
    print("="*80)
    
    model = get_model().to(DEVICE)
    
    # Load ASVspoof base model
    base_ckpt = Path("backend/checkpoints/best.pth")
    if base_ckpt.exists():
        print(f"\n✓ Loading base model: {base_ckpt.name}")
        checkpoint = torch.load(base_ckpt, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("  ✓ Base model loaded (already knows ASVspoof patterns)")
    else:
        print("\n⚠️  Base model not found, training from scratch...")
    
    # ===== SETUP TRAINING =====
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    best_val_acc = 0.0
    
    # ===== TRAINING LOOP =====
    print("\n" + "="*80)
    print("STEP 4: TRAINING")
    print("="*80)
    print("\nThis will take 5-8 hours with current settings...")
    print("Training on ALL datasets to prevent catastrophic forgetting!\n")
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        real_correct = 0
        real_total = 0
        fake_correct = 0
        fake_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch_idx, (audio, labels) in enumerate(train_bar):
            audio = audio.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                if labels[i] == 0:  # REAL
                    real_total += 1
                    if predicted[i] == 0:
                        real_correct += 1
                else:  # FAKE
                    fake_total += 1
                    if predicted[i] == 1:
                        fake_correct += 1
            
            # Update progress bar
            train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0
            real_acc = 100.0 * real_correct / real_total if real_total > 0 else 0
            fake_acc = 100.0 * fake_correct / fake_total if fake_total > 0 else 0
            
            train_bar.set_postfix({
                'loss': f'{train_loss/(batch_idx+1):.4f}',
                'acc': f'{train_acc:.2f}%',
                'real': f'{real_acc:.1f}%',
                'fake': f'{fake_acc:.1f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_real_correct = 0
        val_real_total = 0
        val_fake_correct = 0
        val_fake_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  ")
            for batch_idx, (audio, labels) in enumerate(val_bar):
                audio = audio.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(audio)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Per-class accuracy
                for i in range(len(labels)):
                    if labels[i] == 0:  # REAL
                        val_real_total += 1
                        if predicted[i] == 0:
                            val_real_correct += 1
                    else:  # FAKE
                        val_fake_total += 1
                        if predicted[i] == 1:
                            val_fake_correct += 1
                
                val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
                val_real_acc = 100.0 * val_real_correct / val_real_total if val_real_total > 0 else 0
                val_fake_acc = 100.0 * val_fake_correct / val_fake_total if val_fake_total > 0 else 0
                
                val_bar.set_postfix({
                    'loss': f'{val_loss/(batch_idx+1):.4f}',
                    'acc': f'{val_acc:.2f}%',
                    'real': f'{val_real_acc:.1f}%',
                    'fake': f'{val_fake_acc:.1f}%'
                })
        
        # Calculate balanced accuracy
        balanced_acc = (val_real_acc + val_fake_acc) / 2
        
        # Epoch summary
        print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Train: Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.2f}% (Real={real_acc:.1f}%, Fake={fake_acc:.1f}%)")
        print(f"  Val:   Loss={val_loss/len(val_loader):.4f}, Acc={val_acc:.2f}% (Real={val_real_acc:.1f}%, Fake={val_fake_acc:.1f}%)")
        print(f"  Balanced Accuracy: {balanced_acc:.2f}%")
        
        # Scheduler step
        scheduler.step(balanced_acc)
        
        # Save best model
        if balanced_acc > best_val_acc:
            best_val_acc = balanced_acc
            save_path = Path("backend/checkpoints/comprehensive_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'accuracy': balanced_acc,
                'real_accuracy': val_real_acc,
                'fake_accuracy': val_fake_acc,
                'optimizer': optimizer.state_dict(),
            }, save_path)
            print(f"  ✓ Saved best model: {save_path.name} (Balanced Acc: {balanced_acc:.2f}%)")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved: backend/checkpoints/comprehensive_model.pth")
    print("\nThis model should now detect:")
    print("  ✓ ElevenLabs (modern neural TTS)")
    print("  ✓ ASVspoof (traditional TTS)")
    print("  ✓ LibriSpeech (real voices)")
    print("  ✓ Your voice (real voices)")
    print("\nNo more catastrophic forgetting! 🎯")

if __name__ == "__main__":
    main()
