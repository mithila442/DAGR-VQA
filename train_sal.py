import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sal_withreg import UNetWithRegisterTokens
from utilities import *
from extract_frame import extract_frame
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import csv
from PIL import Image, UnidentifiedImageError

def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    #print(f"Normalizing tensor: min={min_val}, max={max_val}")  # Debug output to verify normalization
    normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
    return normalized_tensor

class DIEMSaliencyDataset(Dataset):
    def __init__(self, diem_root, transform=None, expected_length=60):
        self.diem_root = diem_root
        self.video_folders = [os.path.join(diem_root, d) for d in os.listdir(diem_root)
                              if os.path.isdir(os.path.join(diem_root, d))]
        self.video_folders.sort()
        self.transform = transform
        self.expected_length = expected_length

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_dir = self.video_folders[idx]
        frame_list = sorted([os.path.join(video_dir, "extracted_frames", f)
                            for f in os.listdir(os.path.join(video_dir, "extracted_frames"))
                            if f.endswith(('.png', '.jpg', '.jpeg'))])
        sal_list = sorted([os.path.join(video_dir, "saliency", f)
                            for f in os.listdir(os.path.join(video_dir, "saliency"))
                            if f.endswith(('.png', '.jpg', '.jpeg'))])

        # Equidistant sampling of indices
        num_frames = min(len(frame_list), len(sal_list))
        if num_frames == 0:
            raise ValueError(f"No valid frames/maps in {video_dir}")

        if num_frames >= self.expected_length:
            idxs = np.linspace(0, num_frames - 1, self.expected_length, dtype=int)
        else:
            idxs = np.arange(num_frames)
        
        # Select and process images/maps by indices
        images = []
        sal_maps = []
        for i in idxs:
            try:
                img = Image.open(frame_list[i]).convert('RGB')
                sal = Image.open(sal_list[i]).convert('L')
                if self.transform:
                    img = self.transform(img)
                    sal = self.transform(sal)
                images.append(img)
                sal_maps.append(sal)
            except Exception as e:
                print(f"Skipping corrupted: {frame_list[i]} or {sal_list[i]} — {e}")
                # If needed, pad with black images/maps here to keep count

        # Pad if needed
        if len(images) < self.expected_length:
            if len(images) > 0:
                while len(images) < self.expected_length:
                    images.append(torch.zeros_like(images[0]))
                    sal_maps.append(torch.zeros_like(sal_maps[0]))
            else:
                raise ValueError(f"No good frames/maps in {video_dir}")

        images = torch.stack(images)
        sal_maps = torch.stack(sal_maps)
        sal_maps = (sal_maps - sal_maps.min()) / (sal_maps.max() - sal_maps.min() + 1e-8)
        return images, sal_maps, video_dir


# Dataset class
class SaliencyDataset(Dataset):
    def __init__(self, root_dir, selected_indices, transform=None):
        self.root_dir = root_dir
        self.video_folders = [os.path.join(root_dir, f'{i:03d}') for i in range(1, 501)]
        self.selected_indices = selected_indices
        self.transform = transform

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        images = sorted([os.path.join(video_folder, 'extracted_frames', img) for img in
                        os.listdir(os.path.join(video_folder, 'extracted_frames')) if
                        img.endswith(('.jpg', '.jpeg', '.png'))])

        maps = sorted([os.path.join(video_folder, 'map2', map_file) for map_file in
                    os.listdir(os.path.join(video_folder, 'map2')) if
                    map_file.endswith(('.jpg', '.jpeg', '.png'))])

        batch_images = []
        batch_maps = []

        for img_path, map_path in zip(images, maps):
            try:
                image = Image.open(img_path).convert('RGB')
                sal_map = Image.open(map_path).convert('L')

                if self.transform:
                    image = self.transform(image)
                    sal_map = self.transform(sal_map)

                # Check for NaNs/Infs
                if not torch.isfinite(image).all() or not torch.isfinite(sal_map).all():
                    print(f"⚠️ Skipping non-finite values in: {img_path} or {map_path}")
                    continue

                # Check if image or map is completely black
                if torch.max(image) == 0:
                    print(f"⚠️ Skipping black image: {img_path}")
                    continue
                if torch.max(sal_map) == 0:
                    print(f"⚠️ Skipping black saliency map: {map_path}")
                    continue

                batch_images.append(image)
                batch_maps.append(sal_map)

            except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
                print(f"⚠️ Skipping corrupted file: {img_path} or {map_path} — {e}")
                continue

        if len(batch_images) == 0 or len(batch_maps) == 0:
            raise ValueError(f"❌ No valid images or maps found in {video_folder}")

        batch_images = torch.stack(batch_images)
        batch_maps = torch.stack(batch_maps)
        batch_maps = normalize_tensor(batch_maps)

        return batch_images, batch_maps, video_folder

# Collate function for DataLoader
def collate_fn(batch):
    all_images = []
    all_maps = []
    video_folders = []

    for images, maps, video_folder in batch:
        all_images.append(images)
        all_maps.append(maps)
        video_folders.append(video_folder)

    max_length = max(images.size(0) for images in all_images)

    for i in range(len(all_images)):
        images = all_images[i]
        maps = all_maps[i]

        if images.size(0) < max_length:
            padding_size = max_length - images.size(0)
            padding_images = torch.zeros((padding_size, *images.shape[1:]), dtype=images.dtype)
            all_images[i] = torch.cat((images, padding_images), dim=0)

            padding_maps = torch.zeros((padding_size, *maps.shape[1:]), dtype=maps.dtype)
            all_maps[i] = torch.cat((maps, padding_maps), dim=0)

    all_images = torch.stack(all_images)
    all_maps = torch.stack(all_maps)

    return all_images, all_maps, video_folders


# Function to save checkpoint
def save_checkpoint(model, optimizer, epoch, loss, path='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


# Function to train the model
# def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs, accumulation_steps, sub_batch_size,
#                 device, root_dir, expected_length, label_csv):
def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs, accumulation_steps, sub_batch_size,
                 device, root_dir, expected_length):
    with open('epoch_loss.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss'])  # Write the header

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, (images, maps, video_folders) in enumerate(dataloader):
                print(f"Epoch {epoch + 1}, Batch {i + 1}: Processing videos {video_folders}")
                images = images.to(device)
                maps = maps.to(device)

                # Skip batch if invalid values found
                if torch.isnan(images).any() or torch.isinf(images).any():
                    print(f"❌ NaN/Inf in input images at Batch {i + 1}, skipping batch.")
                    continue
                if torch.isnan(maps).any() or torch.isinf(maps).any():
                    print(f"❌ NaN/Inf in saliency maps at Batch {i + 1}, skipping batch.")
                    continue

                outputs = []
                for j in range(0, expected_length, sub_batch_size):
                    sub_images = images[:, j:j + sub_batch_size]
                    sub_maps = maps[:, j:j + sub_batch_size]

                    if sub_images.size(1) == 0:
                        continue

                    output = model(sub_images)
                    output = torch.sigmoid(output)
                    output = normalize_tensor(output)
                    sub_maps = normalize_tensor(sub_maps)
                    output = output.permute(0, 2, 1, 3, 4)
                    outputs.append(output)

                    loss = criterion(output, sub_maps)

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"🚨 NaN/Inf loss at Epoch {epoch + 1}, Batch {i + 1}, Sub-batch {j // sub_batch_size + 1}, skipping...")
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    if (i + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    running_loss += loss.item() * sub_images.size(0)
                    print(f"Epoch {epoch + 1}, Batch {i + 1}, Sub-batch {j // sub_batch_size + 1}: Loss = {loss.item():.4f}")

                if len(outputs) > 0:
                    outputs = torch.cat(outputs, dim=1)
                    outputs = outputs.permute(1, 0, 2, 3, 4)  # [seq, batch, ch, h, w]

                for k, video_folder in enumerate(video_folders):
                    output_folder = os.path.join(video_folder, 'pred_sal')
                    os.makedirs(output_folder, exist_ok=True)

                    num_images = len(os.listdir(os.path.join(video_folder, 'extracted_frames')))
                    num_images = min(num_images, expected_length)

                    for m in range(num_images):
                        img_name = f"{m:03d}.png"
                        try:
                            saliency_map = outputs[m, k].cpu().detach().numpy()
                        except IndexError as e:
                            print(f"IndexError: {e}, m: {m}, k: {k}, size of outputs: {outputs.size(0)}")
                            break

                        if len(saliency_map.shape) > 2:
                            saliency_map = saliency_map.squeeze()

                        saliency_map = (saliency_map * 255).astype(np.uint8)
                        if saliency_map.ndim == 3:
                            saliency_map = saliency_map[0]

                        original_image_path = os.path.join(video_folder, 'extracted_frames', img_name)
                        original_image = Image.open(original_image_path)
                        saliency_map_pil = Image.fromarray(saliency_map)
                        saliency_map_pil.save(os.path.join(output_folder, img_name))

            epoch_loss = running_loss / len(dataloader.dataset)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
            writer.writerow([epoch + 1, epoch_loss])
            file.flush()
            scheduler.step()

            # if (epoch + 1) >= 100 and (epoch + 1) % 10 == 0:
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pth"
                print(f"💾 Saving checkpoint at epoch {epoch + 1} → {checkpoint_path}")
                save_checkpoint(model, optimizer, epoch + 1, epoch_loss, path=checkpoint_path)   

                # emb_path = f"token_embeddings_epoch_{epoch + 1}.npz"
                # print(f"🧠 Saving register token embeddings at epoch {epoch + 1} → {emb_path}")
                # collect_token_embeddings(model, dataloader, device, save_path=emb_path, label_csv=label_csv)
                

                
def get_content_labels(csv_path):
    df = pd.read_csv(csv_path, header=1)
    df.index = df.index + 1
    df['VideoID'] = df.index.astype(str).str.zfill(3)

    category_cols = [
        'Daliy activity',
        'Sport',
        'Social activity',
        'Artistic performance',
        'Animal',
        'Artifact',
        'Landscape'
    ]

    label_map = {}
    for _, row in df.iterrows():
        for col in category_cols:
            if int(row.get(col, 0)) == 1:
                label_map[row['VideoID']] = col
                break
        else:
            label_map[row['VideoID']] = "unknown"

    return label_map


def collect_token_embeddings(model, dataloader, device, save_path="token_embeddings.npz", label_csv="DHF1k_attribute-all.csv"):
    model.eval()
    all_tokens = []
    labels = []
    video_ids = []

    folder_to_label = get_content_labels(label_csv)

    for i, (images, _, video_folders) in enumerate(dataloader):
        images = images.to(device)

        with torch.no_grad():
            #tokens = model.module(images, return_tokens=True)  # (B, N)
            tokens = model(images, return_tokens=True)  # (B, N)

        tokens = tokens.cpu().numpy()

        for b in range(tokens.shape[0]):
            folder_id = os.path.basename(video_folders[b])
            all_tokens.append(tokens[b])
            labels.append(folder_to_label.get(folder_id, "unknown"))
            video_ids.append(folder_id)

        print(f"[Batch {i + 1}] Collected {tokens.shape[0]} embeddings")

    # Convert and save
    all_tokens = np.array(all_tokens)  # (300, N)
    labels = np.array(labels)          # (300,)
    video_ids = np.array(video_ids)    # (300,)

    print(f"✅ Total embeddings collected: {len(all_tokens)}")  # Should be 300
    np.savez(save_path, tokens=all_tokens, labels=labels, video_ids=video_ids)
    print(f"💾 Saved to {save_path}")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(torch.cuda.device_count(), "GPUs are available.")
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set"))

    phase = 'train'
    if phase == 'train':
        stateful = False
    else:
        stateful = True

    if phase == "train":
        num_epochs = 200
        learning_rate = 5e-3
        batch_size = 5  # Number of videos processed simultaneously
        accumulation_steps = 4
        sub_batch_size = 6  # Number of frames processed at once per sub-batch
        expected_length = 60

        train_data_folder = './dhf1k/'

        data_transforms = transforms.Compose([
            transforms.Resize((224, 398)),
            transforms.ToTensor(),
        ])

        # Hyperparameters for frame extraction
        size = 224  # Resize dimension for frames
        frames_per_second = 4  # Adjusted to extract 60 frames from 15s videos
        video_length_min = expected_length

        videos_dir = os.path.join(train_data_folder, 'videos')

        selected_indices = []

        for i in range(1, 201):
            video_name = f'{i:03d}.AVI'
            save_folder = os.path.join(train_data_folder, f'{i:03d}', 'extracted_frames')
            os.makedirs(save_folder, exist_ok=True)

            # Extract frames but do not expect a return value
            extract_frame(videos_dir, video_name, save_folder=save_folder, dataset_type='saliency')

            # Append a placeholder to maintain indexing consistency
            selected_indices.append(list(range(video_length_min)))

            print(f'Extracted {video_length_min} frames from {video_name} to {save_folder}')

        # Create dataset and dataloader
        train_dataset = SaliencyDataset(train_data_folder, selected_indices, transform=data_transforms)
        print(f"Found {len(train_dataset)} video files in {train_data_folder}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                  collate_fn=collate_fn))

        # Initialize the model and training process
        model = UNetWithRegisterTokens(in_channels=3, out_channels=1, num_register_tokens=4).to(device)
        #model = nn.DataParallel(model).to(device)
        criterion = CombinedLoss(alpha1=0.01, alpha2=0.1)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0)

        # Train the model
        train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=num_epochs,
            accumulation_steps=accumulation_steps, sub_batch_size=sub_batch_size, device=device,
            root_dir=train_data_folder, expected_length=expected_length, label_csv="DHF1k_attribute-all.csv")
        
        # === Save token embeddings after training ===
        collect_token_embeddings(
            model, 
            train_loader, 
            device, 
            save_path="token_embeddings.npz", 
            label_csv="DHF1k_attribute-all.csv"
        )





