import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
import csv
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR

from VQA import VideoQualityModelSimpleFusion, SpearmanCorrelationLoss
from dataloader import LSVQDataset, KonvidVQADataset


class CrossDBTrainer:
    def __init__(self, lsvq_frame_dir, lsvq_csv, lsvq_sal_dir,
                 konvid_frame_dir, konvid_csv, konvid_sal_dir,
                 device):

        self.device = device

        # Data transforms (same for both datasets)
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 398)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
        ])

        # Load datasets
        self.lsvq_dataset = LSVQDataset(
            frame_dir=lsvq_frame_dir,
            mos_csv=lsvq_csv,
            saliency_dir=lsvq_sal_dir,
            transform=self.data_transforms,
            num_frames=8
        )

        self.konvid_dataset = KonvidVQADataset(
            frame_dir=konvid_frame_dir,
            mos_csv=konvid_csv,
            saliency_dir=konvid_sal_dir,
            transform=self.data_transforms,
            num_frames=8,
            alpha=0.5
        )

        # Model
        self.model = VideoQualityModelSimpleFusion(
            device=self.device,
            spatial_feature_dim=2048,
            temporal_feature_dim=2048
        ).to(self.device)

        self.regression_criterion = nn.L1Loss()
        self.rank_criterion = SpearmanCorrelationLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-6, weight_decay=5e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-8)

        self.loss_log_file = 'loss_log_crossdb.csv'
        with open(self.loss_log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Fold', 'Epoch', 'Train Loss', 'Train PLCC',
                             'Val Loss', 'Val PLCC', 'Test Loss', 'Test PLCC'])

    def calculate_plcc(self, y_true, y_pred):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        return np.corrcoef(y_true, y_pred)[0, 1]

    def run_one_epoch(self, loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

        regression_loss_total = 0.0
        all_labels, all_preds = [], []

        with torch.set_grad_enabled(train):
            for frames, mos_labels, _ in loader:
                frames, mos_labels = frames.to(self.device), mos_labels.to(self.device)
                quality_scores = self.model(frames)

                regression_loss = self.regression_criterion(quality_scores.view(-1), mos_labels)
                corr_loss = self.rank_criterion(quality_scores.view(-1), mos_labels)
                combined_loss = regression_loss + 0.1 * corr_loss

                if train:
                    self.optimizer.zero_grad()
                    combined_loss.backward()
                    self.optimizer.step()

                regression_loss_total += regression_loss.item()
                all_labels.extend(mos_labels.cpu().numpy().flatten())
                all_preds.extend(quality_scores.detach().cpu().numpy().flatten())

        plcc = self.calculate_plcc(np.array(all_labels), np.array(all_preds))
        return regression_loss_total / len(loader), plcc

    def cross_validate(self, num_epochs=20, num_folds=5):
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=953)

        for fold, (val_idx, test_idx) in enumerate(kf.split(range(len(self.konvid_dataset)))):
            print(f"\n===== Fold {fold + 1}/{num_folds} =====")

            val_subset = Subset(self.konvid_dataset, val_idx)
            test_subset = Subset(self.konvid_dataset, test_idx)

            train_loader = DataLoader(self.lsvq_dataset, batch_size=16, shuffle=True, num_workers=5, drop_last=True)
            val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, num_workers=5)
            test_loader = DataLoader(test_subset, batch_size=4, shuffle=False, num_workers=5)

            for epoch in range(num_epochs):
                train_loss, train_plcc = self.run_one_epoch(train_loader, train=True)
                val_loss, val_plcc = self.run_one_epoch(val_loader, train=False)
                test_loss, test_plcc = self.run_one_epoch(test_loader, train=False)

                print(f"Fold {fold+1}, Epoch [{epoch+1}/{num_epochs}] "
                      f"Train Loss: {train_loss:.4f}, Train PLCC: {train_plcc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val PLCC: {val_plcc:.4f}, "
                      f"Test Loss: {test_loss:.4f}, Test PLCC: {test_plcc:.4f}")

                with open(self.loss_log_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([fold+1, epoch+1, train_loss, train_plcc, val_loss, val_plcc, test_loss, test_plcc])

                self.scheduler.step()

                # Save best per fold
                if epoch > 100 and epoch%10 == 0:
                    torch.save(
                        {"model_state_dict": self.model.state_dict()},
                        f"crossdb_fold{fold+1}_valplcc{val_plcc:.4f}_testplcc{test_plcc:.4f}.pth"
                    )


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = CrossDBTrainer(
        lsvq_frame_dir='./LSVQ_extracted_frames/',
        lsvq_csv='./LSVQ/labels_train_test.csv',
        lsvq_sal_dir='./LSVQ_saliency_maps/',
        konvid_frame_dir='./KoNViD_1k_extracted_frames/',
        konvid_csv='./KoNViD_1k/KoNViD_1k_mos.csv',
        konvid_sal_dir='./KoNViD_1k_saliency_maps/',
        device=device
    )
    trainer.cross_validate(num_epochs=150, num_folds=5)