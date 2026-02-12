import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import csv
from VQA import VideoQualityModelSpatialOnly, SpearmanCorrelationLoss
from dataloader import LSVQDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

class SiameseNetworkTrainer:
    def __init__(self, frame_dir, mos_csv, saliency_dir, device):
        self.device = device

        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 398)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
        ])

        self.dataset = LSVQDataset(
            frame_dir=frame_dir,
            mos_csv=mos_csv,
            saliency_dir=saliency_dir,
            transform=self.data_transforms,
            num_frames=8
        )

        total_videos = len(self.dataset)
        all_indices = list(range(total_videos))

        # 1) First split: train vs temp (val+test)
        train_indices, temp_indices = train_test_split(
            all_indices,
            test_size=0.2,  # 80% train, 20% temp
            random_state=42,
            shuffle=True
        )

        # 2) Second split: temp -> val and test (half-half of the 20%)
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=0.5,  # 10% val, 10% test
            random_state=42,
            shuffle=True
        )

        print(f"Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")

        # Save indices (so your eval code can reuse EXACT same test set)
        np.save("train_indices_seed42.npy", np.array(train_indices))
        np.save("val_indices_seed42.npy", np.array(val_indices))
        np.save("test_indices_seed42.npy", np.array(test_indices))  # name matches your eval script
        print("✅ Saved train/val/test indices .npy files")

        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)

        self.train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True, num_workers=5, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=2, shuffle=False, num_workers=5, drop_last=True)
        
        self.model = VideoQualityModelSpatialOnly(
            device=self.device,
            spatial_feature_dim=2048,
            combined_dim=2048  # use the same as in your ablation class definition
        ).to(self.device)


        self.regression_criterion = nn.L1Loss()
        self.rank_criterion = SpearmanCorrelationLoss()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-6, weight_decay=5e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-8)

        self.loss_log_file = 'loss_log_lsvq.csv'
        with open(self.loss_log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Regression Loss', 'Correlation Loss', 'PLCC', 'Val Regression Loss', 'Val Correlation Loss', 'Val PLCC'])

    def calculate_plcc(self, y_true, y_pred):
        y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
        y_pred = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
        return np.corrcoef(y_true, y_pred)[0, 1]

    def validate(self):
        self.model.eval()
        val_regression_loss, val_corr_loss = 0.0, 0.0
        all_mos_labels, all_predicted_scores = [], []

        with torch.no_grad():
            for frames, mos_labels, _ in self.val_loader:
                frames, mos_labels = frames.to(self.device), mos_labels.to(self.device)
                quality_scores = self.model(frames)
                regression_loss = self.regression_criterion(quality_scores.view(-1), mos_labels)
                val_regression_loss += regression_loss.item()
                all_mos_labels.extend(mos_labels.cpu().numpy().flatten())
                all_predicted_scores.extend(quality_scores.cpu().numpy().flatten())

        val_corr_loss = self.rank_criterion(torch.tensor(all_predicted_scores), torch.tensor(all_mos_labels))
        val_plcc = self.calculate_plcc(np.array(all_mos_labels), np.array(all_predicted_scores))
        return val_regression_loss / len(self.val_loader), val_corr_loss.item(), val_plcc

    def train(self, num_epochs=15):
        for epoch in range(num_epochs):
            self.model.train()
            running_regression_loss, running_corr_loss = 0.0, 0.0
            all_train_mos_labels, all_train_predicted_scores = [], []

            for frames, mos_labels, _ in self.train_loader:
                frames, mos_labels = frames.to(self.device), mos_labels.to(self.device)
                quality_scores = self.model(frames)
                regression_loss = self.regression_criterion(quality_scores.view(-1), mos_labels)
                corr_loss = self.rank_criterion(quality_scores.view(-1), mos_labels)
                combined_loss = regression_loss + 0.1 * corr_loss

                self.optimizer.zero_grad()
                combined_loss.backward()
                self.optimizer.step()

                running_regression_loss += regression_loss.item()
                running_corr_loss += corr_loss.item()
                all_train_mos_labels.extend(mos_labels.cpu().numpy().flatten())
                all_train_predicted_scores.extend(quality_scores.detach().cpu().numpy().flatten())

            train_plcc = self.calculate_plcc(np.array(all_train_mos_labels), np.array(all_train_predicted_scores))
            val_reg_loss, val_corr_loss, val_plcc = self.validate()

            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Regression Loss: {running_regression_loss / len(self.train_loader):.4f}, "
                  f"Train Correlation Loss: {running_corr_loss / len(self.train_loader):.4f}, "
                  f"Train PLCC: {train_plcc:.4f}, "
                  f"Val Regression Loss: {val_reg_loss:.4f}, "
                  f"Val Correlation Loss: {val_corr_loss:.4f}, "
                  f"Val PLCC: {val_plcc:.4f}")

            with open(self.loss_log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    epoch + 1,
                    running_regression_loss / len(self.train_loader),
                    running_corr_loss / len(self.train_loader),
                    train_plcc,
                    val_reg_loss,
                    val_corr_loss,
                    val_plcc
                ])

            if epoch % 10 == 0 and epoch > 150:
                torch.save({"model_state_dict": self.model.state_dict()}, f"checkpoint_epoch_{epoch + 1}_plcc_{val_plcc:.4f}_lsvq.pth")
                print(f"Checkpoint saved for epoch {epoch + 1}, PLCC: {val_plcc:.4f}")

            self.scheduler.step()

if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    trainer = SiameseNetworkTrainer(
        frame_dir='./LSVQ_extracted_frames/',
        mos_csv='./LSVQ/labels_train_test.csv',
        saliency_dir='./LSVQ_saliency_maps/',
        device=device
    )
    trainer.train(num_epochs=300)