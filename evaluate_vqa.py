import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from dataloader import LSVQDataset
from VQA import SpearmanCorrelationLoss, VideoQualityModelSimpleFusion
import random
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd

# Configuration
class Config:
    FRAME_DIR = './LSVQ_extracted_frames'
    MOS_CSV = './LSVQ/labels_train_test.csv'
    SALIENCY_DIR = './LSVQ_saliency_maps'
    CHECKPOINT = 'checkpoint_epoch_291_plcc_0.7600_lsvq.pth'
    # FRAME_DIR = './KoNViD_1k_extracted_frames'
    # MOS_CSV = './KoNViD_1k/KoNViD_1k_mos.csv'
    # SALIENCY_DIR = './KoNViD_1k_saliency_maps'
    # CHECKPOINT = './checkpoint_epoch_991_plcc_0.3226_konvid_temp.pth'
    # FRAME_DIR = './LIVE_Netflix/Netflix_videos'
    # MOS_CSV = './LIVE_Netflix/aggregated_mos_scores.csv'
    # SALIENCY_DIR = './LIVE_Netflix_saliency_maps'
    # FRAME_DIR = './YouTube_UGC_extracted_frames/'
    # MOS_CSV = './YouTube-UGC/original_videos_MOS_for_YouTube_UGC_dataset.csv'
    # SALIENCY_DIR = './YouTube-UGC_saliency_maps/'
    # FRAME_DIR='./LIVE_VQC_extracted_frames'
    # MOS_CSV='./LIVE_VQC/data.mat'
    # SALIENCY_DIR='./LIVE_saliency_maps'
    # VIDEO_DIR='./LIVE_VQC/Video'
    # CHECKPOINT = './checkpoint_epoch_991_plcc_0.7488_live_alpha1.pth'
    BATCH_SIZE = 2
    LR = 0.000008
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 30
    DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    SPEARMAN_WEIGHT = 0.1  # Weight for Spearman loss
    PREDICTIONS_DIR = './predictions'

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

# Fine-tuning and evaluation
def fine_tune_and_evaluate():
    set_seed(Config.SEED)
    
    # Create predictions directory
    os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)

    # Data transformations
    data_transforms = transforms.Compose([
        transforms.Resize((224, 398)),
        transforms.ToTensor(),
    ])

    # Dataset and split
    dataset = LSVQDataset(
        frame_dir=Config.FRAME_DIR,
        mos_csv=Config.MOS_CSV,
        saliency_dir=Config.SALIENCY_DIR,
        transform=data_transforms,
        num_frames=8,
        alpha=0.5
    )

    # Get all dataset indices
    total_indices = list(range(len(dataset)))
    
    # ===== LOAD TEST INDICES FROM FILE =====
    indices_file = 'test_indices_seed42.npy'
    if os.path.exists(indices_file):
        # Load test indices saved by FineVQ
        test_indices = np.load(indices_file).tolist()
        print(f"✅ Loaded test indices from {indices_file}: {len(test_indices)} videos")

        # Create train indices (all except test)
        test_indices_set = set(test_indices)
        train_indices = [i for i in total_indices if i not in test_indices_set]
    else:
        # Fallback: create new split if file doesn't exist
        print(f"⚠️  {indices_file} not found. Creating new split with random_state=953")
        train_indices, test_indices = train_test_split(
            total_indices,
            test_size=0.2,
            random_state=42
        )
        # Save for FineVQ to use
        np.save(indices_file, np.array(test_indices))
        print(f"✅ Created and saved indices to {indices_file}")

    print(f"Train indices: {len(train_indices)}")
    print(f"Test indices: {len(test_indices)}")


    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=1)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=1, shuffle=False, num_workers=1)
    # Model setup
    model = VideoQualityModelSimpleFusion(
        spatial_feature_dim=2048,  # Default output dimension of the spatial analyzer
        temporal_feature_dim=2048,  # Default input dimension of the temporal analyzer
        device=Config.DEVICE
    ).to(Config.DEVICE)

    spearman_criterion = SpearmanCorrelationLoss()
    l1_criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    # Load checkpoint if available
    best_srocc = -1
    if os.path.exists(Config.CHECKPOINT):
        checkpoint = torch.load(Config.CHECKPOINT, map_location=Config.DEVICE)

        # Fix for DDP-wrapped state_dicts
        state_dict = checkpoint.get('model_state_dict', {})
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)

        optimizer_state = checkpoint.get('optimizer_state_dict', None)
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)

        best_srocc = checkpoint.get('best_srocc', -1)
        print("Checkpoint loaded.")
    else:
        print("No checkpoint found. Training from scratch.")
        
    # ADD THIS: Store all metrics for CSV
    all_epochs_data = []

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")

        # Training
        model.train()
        running_loss = 0.0

        for frames, mos_labels, _ in tqdm(train_loader, desc="Training", leave=False):
            frames, mos_labels = frames.to(Config.DEVICE), mos_labels.to(Config.DEVICE)

            optimizer.zero_grad()

            # Forward pass
            quality_scores = model(frames)

            # Regression loss
            regression_loss = l1_criterion(quality_scores.squeeze(), mos_labels.squeeze())

            # Spearman correlation loss
            spearman_loss = spearman_criterion(quality_scores.squeeze(), mos_labels.squeeze())

            # Combined loss (without contrastive loss)
            loss = regression_loss + Config.SPEARMAN_WEIGHT * spearman_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"Training Loss: {avg_loss:.4f}")

        model.eval()
        all_true, all_pred, all_video_names = [], [], []

        with torch.no_grad():
            for frames, mos_labels, video_info in tqdm(test_loader, desc="Evaluating", leave=False):
                frames, mos_labels = frames.to(Config.DEVICE), mos_labels.to(Config.DEVICE)
                predictions = model(frames).squeeze()
                if predictions.dim() == 0:  # If scalar, wrap in a list
                    predictions = predictions.unsqueeze(0)
                all_true.extend(mos_labels.cpu().numpy())
                all_pred.extend(predictions.cpu().numpy())
                
                # Extract video name from video_info (adjust based on your dataloader output)
                if isinstance(video_info, (list, tuple)):
                    all_video_names.extend(video_info)
                else:
                    all_video_names.append(video_info)

        # Metrics computation
        all_true, all_pred = np.array(all_true), np.array(all_pred)
        plcc, plcc_pvalue = pearsonr(all_true, all_pred)
        srocc, srocc_pvalue = spearmanr(all_true, all_pred)
        mae = mean_absolute_error(all_true, all_pred)

        print(f"PLCC: {plcc:.4f} (p-value: {plcc_pvalue:.4e})")
        print(f"SROCC: {srocc:.4f} (p-value: {srocc_pvalue:.4e})")
        print(f"MAE: {mae:.4f}")
        
        # ===== SAVE PREDICTIONS WITH VIDEO NAMES =====
        predictions_csv = os.path.join(Config.PREDICTIONS_DIR, f'predictions_epoch_{epoch + 1}.csv')
        predictions_df = pd.DataFrame({
            'video_name': all_video_names,
            'ground_truth_mos': all_true,
            'predicted_score': all_pred,
            'absolute_error': np.abs(all_true - all_pred)
        })
        predictions_df.to_csv(predictions_csv, index=False)
        print(f"Predictions saved to: {predictions_csv}")


if __name__ == "__main__":
    fine_tune_and_evaluate()
