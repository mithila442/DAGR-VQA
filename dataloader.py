import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import scipy.io
import random

class KonvidVQADataset(Dataset):
    def __init__(self, frame_dir, mos_csv, saliency_dir, transform=None, num_frames=8, selected_videos=None, alpha=1.0):
        """
        Dataset class that loads **pre-extracted frames** and aligns saliency maps.

        :param frame_dir: Path to folder with extracted frames.
        :param mos_csv: Path to MOS scores CSV.
        :param saliency_dir: Path to saliency maps.
        :param video_dir: Path to original videos (for automatic frame extraction).
        :param transform: Torchvision transform for frames.
        :param num_frames: Number of frames per video (must match extract_frame.py).
        :param selected_videos: Optional list of selected video filenames (without extensions).
        """
        self.frame_dir = frame_dir
        self.saliency_dir = saliency_dir
        self.transform = transform
        self.num_frames = num_frames
        self.selected_videos = selected_videos
        self.alpha=alpha

        self.saliency_transforms = transforms.Compose([
            # transforms.Resize((1080, 1920)),
            transforms.Resize((224, 398)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.available_videos = self._get_available_videos()
        #self.mos_df = pd.read_csv(mos_csv)
        self.mos_df = pd.read_csv(mos_csv, encoding='latin1')
        self.valid_indices = self._filter_valid_videos()

    def __len__(self):
        return len(self.valid_indices)

    def _get_available_videos(self):
        available_videos = set(os.listdir(self.frame_dir))
        if self.selected_videos is not None:
            available_videos = available_videos.intersection(self.selected_videos)
        return available_videos

    def _filter_valid_videos(self):
        valid_indices = []
        for idx in range(len(self.mos_df)):
            flickr_id = str(int(self.mos_df.iloc[idx]['flickr_id']))
            if flickr_id in self.available_videos:
                valid_indices.append(idx)
        return valid_indices

    def __getitem__(self, idx):
        valid_idx = self.valid_indices[idx]
        flickr_id = str(int(self.mos_df.iloc[valid_idx]['flickr_id']))
        mos_score = self.mos_df.iloc[valid_idx]['mos']

        # Normalize MOS score
        mos_min = self.mos_df['mos'].min()
        mos_max = self.mos_df['mos'].max()
        mos_score = (mos_score - mos_min) / (mos_max - mos_min)

        frame_folder = os.path.join(self.frame_dir, flickr_id)
        saliency_folder = os.path.join(self.saliency_dir, flickr_id)

        # Load pre-extracted frames from disk
        frame_files = sorted([os.path.join(frame_folder, img) for img in os.listdir(frame_folder) if img.endswith('.png')])

        # Adjust frames to exactly self.num_frames
        if len(frame_files) >= self.num_frames:
            frame_files = frame_files[:self.num_frames]        # take first N
        else:
            # pad with last frame if too few
            last = frame_files[-1]
            frame_files += [last] * (self.num_frames - len(frame_files))


        frames = []
        for frame_file in frame_files:
            frame = Image.open(frame_file).convert("RGB")
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        # Load and align saliency maps
        if not os.path.exists(saliency_folder):
            raise FileNotFoundError(f"Saliency folder does not exist: {saliency_folder}")

        saliency_maps = sorted([os.path.join(saliency_folder, img) for img in os.listdir(saliency_folder) if img.endswith('.png')])

        if len(saliency_maps) >= self.num_frames:
            saliency_indices = np.linspace(0, len(saliency_maps) - 1, self.num_frames, dtype=int)
            saliency_maps = [saliency_maps[i] for i in saliency_indices]
        else:
            saliency_maps.extend([saliency_maps[-1]] * (self.num_frames - len(saliency_maps)))

        frame_tensors = []
        for i, (frame, saliency_map) in enumerate(zip(frames, saliency_maps)):
            saliency = Image.open(saliency_map).convert('L')
            saliency = self.saliency_transforms(saliency)

            if torch.all(saliency == 0):
                print(f"Warning: Zero saliency map detected for {saliency_map}")
                saliency = torch.ones_like(saliency)

            saliency = saliency / max(torch.max(saliency), 1e-6)
            saliency = torch.clamp(saliency, min=0.1)

            weighted_frame = (1 - self.alpha) * frame + self.alpha * (frame * saliency)

            frame_tensors.append(weighted_frame)

        frames_tensor = torch.stack(frame_tensors, dim=1)  # Shape: (C, num_frames, H, W)

        return frames_tensor, torch.tensor(mos_score, dtype=torch.float32), flickr_id


class LiveVQADataset(Dataset):
    def __init__(self, frame_dir, mat_file, saliency_dir, transform=None, num_frames=8, alpha=0.5):
        """
        Dataset class that loads **pre-extracted frames** and aligns saliency maps.

        :param frame_dir: Path to folder with extracted frames.
        :param mat_file: Path to LIVE-VQA dataset metadata file (.mat).
        :param saliency_dir: Path to saliency maps.
        :param video_dir: Path to original videos (for automatic frame extraction).
        :param transform: Torchvision transform for frames.
        :param num_frames: Number of frames per video (must match extract_frame.py).
        """
        self.frame_dir = frame_dir
        self.saliency_dir = saliency_dir
        # self.video_dir = video_dir
        self.transform = transform
        self.num_frames = num_frames
        self.alpha = alpha

        self.saliency_transforms = transforms.Compose([
            transforms.Resize((224, 398)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Load LIVE-VQA dataset metadata
        mat_contents = scipy.io.loadmat(mat_file)
        self.video_list = [v[0][0] for v in mat_contents["video_list"]]  # Extract filenames
        self.mos_scores = mat_contents["mos"].flatten()  # Extract MOS scores

        self.available_videos = self._get_available_videos()
        self.valid_indices = self._filter_valid_videos()

    def __len__(self):
        return len(self.valid_indices)

    def _get_available_videos(self):
        available_videos = set(os.listdir(self.frame_dir))
        return available_videos

    def _filter_valid_videos(self):
        valid_indices = []
        for idx in range(len(self.video_list)):
            video_name = self.video_list[idx].split('.')[0]  # Remove extension
            if video_name in self.available_videos:
                valid_indices.append(idx)
        return valid_indices

    def __getitem__(self, idx):
        valid_idx = self.valid_indices[idx]
        video_name = self.video_list[valid_idx]
        mos_score = self.mos_scores[valid_idx]

        # Normalize MOS score
        mos_min = self.mos_scores.min()
        mos_max = self.mos_scores.max()
        mos_score = (mos_score - mos_min) / (mos_max - mos_min)

        frame_folder = os.path.join(self.frame_dir, video_name.split('.')[0])
        saliency_folder = os.path.join(self.saliency_dir, video_name.split('.')[0])

        # Load pre-extracted frames from disk
        frame_files = sorted([os.path.join(frame_folder, img) for img in os.listdir(frame_folder) if img.endswith('.png')])

        if len(frame_files) != self.num_frames:
            raise ValueError(f"Mismatch: Expected {self.num_frames} frames but found {len(frame_files)} for {video_name}")

        frames = []
        for frame_file in frame_files:
            frame = Image.open(frame_file).convert("RGB")
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        # Load and align saliency maps
        if not os.path.exists(saliency_folder):
            raise FileNotFoundError(f"Saliency folder does not exist: {saliency_folder}")

        saliency_maps = sorted([os.path.join(saliency_folder, img) for img in os.listdir(saliency_folder) if img.endswith('.png')])

        if len(saliency_maps) >= self.num_frames:
            saliency_indices = np.linspace(0, len(saliency_maps) - 1, self.num_frames, dtype=int)
            saliency_maps = [saliency_maps[i] for i in saliency_indices]
        else:
            saliency_maps.extend([saliency_maps[-1]] * (self.num_frames - len(saliency_maps)))

        frame_tensors = []
        for i, (frame, saliency_map) in enumerate(zip(frames, saliency_maps)):
            saliency = Image.open(saliency_map).convert('L')
            saliency = self.saliency_transforms(saliency)

            if torch.all(saliency == 0):
                print(f"Warning: Zero saliency map detected for {saliency_map}")
                saliency = torch.ones_like(saliency)

            saliency = saliency / max(torch.max(saliency), 1e-6)
            saliency = torch.clamp(saliency, min=0.1)

            weighted_frame = (1 - self.alpha) * frame + self.alpha * (frame * saliency)

            frame_tensors.append(weighted_frame)

        frames_tensor = torch.stack(frame_tensors, dim=1)  # Shape: (C, num_frames, H, W)

        return frames_tensor, torch.tensor(mos_score, dtype=torch.float32), video_name

class LiveNetflixVQADataset(Dataset):
    def __init__(self, video_dir, mos_csv, saliency_dir, transform=None, num_frames=40, selected_videos=None):
        self.video_dir = video_dir
        self.saliency_dir = saliency_dir
        self.transform = transform
        self.num_frames = num_frames
        self.selected_videos = selected_videos

        self.saliency_transforms = transforms.Compose([
            transforms.Resize((224, 398)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.available_videos = self._get_available_videos()
        self.mos_df = pd.read_csv(mos_csv)
        self.valid_indices = self._filter_valid_videos()

    def __len__(self):
        return len(self.valid_indices)

    def _get_available_videos(self):
        available_videos = set(os.listdir(self.video_dir))
        if self.selected_videos is not None:
            available_videos = available_videos.intersection(self.selected_videos)
        return available_videos

    def _filter_valid_videos(self):
        valid_indices = []
        for idx in range(len(self.mos_df)):
            video_name = self.mos_df.iloc[idx]['Video Name']
            if video_name in self.available_videos:
                valid_indices.append(idx)
        return valid_indices

    def __getitem__(self, idx):
        valid_idx = self.valid_indices[idx]

        video_name = self.mos_df.iloc[valid_idx]['Video Name']
        mos_score = self.mos_df.iloc[valid_idx]['MOS Value']

        mos_min = self.mos_df['MOS Value'].min()
        mos_max = self.mos_df['MOS Value'].max()
        mos_score = (mos_score - mos_min) / (mos_max - mos_min)

        video_path = os.path.join(self.video_dir, video_name)
        saliency_folder = os.path.join(self.saliency_dir, os.path.splitext(video_name)[0])

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int) if frame_count >= self.num_frames else np.arange(0, frame_count)
        if len(frame_indices) < self.num_frames:
            frame_indices = np.concatenate([frame_indices, np.full(self.num_frames - len(frame_indices), frame_count - 1)])

        current_frame = 0
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Failed to read frame {current_frame} from video {video_name}")
                break
            if current_frame in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            current_frame += 1
        cap.release()

        if len(frames) < self.num_frames:
            print(f"Video {video_name}: Missing frames. Padding with last frame.")
            frames.extend([frames[-1]] * (self.num_frames - len(frames)))

        if not os.path.exists(saliency_folder):
            raise FileNotFoundError(f"Saliency folder does not exist: {saliency_folder}")

        saliency_maps = sorted([os.path.join(saliency_folder, img) for img in os.listdir(saliency_folder) if img.endswith(('.jpg', '.jpeg', '.png'))])

        if len(saliency_maps) >= self.num_frames:
            saliency_maps = saliency_maps[:self.num_frames]
        else:
            saliency_maps.extend([saliency_maps[-1]] * (self.num_frames - len(saliency_maps)))

        if len(frames) != len(saliency_maps):
            raise ValueError(f"Mismatch: {len(frames)} frames and {len(saliency_maps)} saliency maps for video {video_name}")

        frame_tensors = []
        for i, (frame, saliency_map) in enumerate(zip(frames, saliency_maps)):
            saliency = Image.open(saliency_map).convert('L')
            if self.transform:
                saliency = self.transform(saliency)

            if torch.all(saliency == 0):
                print(f"Warning: Zero saliency map detected for {saliency_map}")
                saliency = torch.ones_like(saliency)

            saliency = saliency / max(torch.max(saliency), 1e-6)
            saliency = torch.clamp(saliency, min=0.1)

            alpha = 0.5
            weighted_frame = (1 - alpha) * frame + alpha * (frame * saliency)

            frame_tensors.append(weighted_frame)

        frames_tensor = torch.stack(frame_tensors, dim=1)

        if torch.all(frames_tensor == 0):
            raise ValueError(f"All-zero frames detected for video {video_name}")

        return frames_tensor, torch.tensor(mos_score, dtype=torch.float32), video_name


class YouTubeUGCVQADataset(Dataset):
    def __init__(self, frame_dir, mos_csv, saliency_dir, transform=None, num_frames=40, selected_videos=None):
        """
        Dataset class that loads **pre-extracted frames** and aligns saliency maps.

        :param frame_dir: Path to folder with extracted frames.
        :param mos_csv: Path to MOS scores CSV.
        :param saliency_dir: Path to saliency maps.
        :param video_dir: Path to original videos (for reference).
        :param transform: Torchvision transform for frames.
        :param num_frames: Number of frames per video.
        :param selected_videos: Optional list of selected video filenames (without extensions).
        """
        self.frame_dir = frame_dir
        self.saliency_dir = saliency_dir
        self.transform = transform
        self.num_frames = num_frames
        self.selected_videos = selected_videos

        self.saliency_transforms = transforms.Compose([
            transforms.Resize((224, 398)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.available_videos = self._get_available_videos()
        self.mos_df = pd.read_csv(mos_csv)
        self.valid_indices = self._filter_valid_videos()

    def __len__(self):
        return len(self.valid_indices)

    def _get_available_videos(self):
        available_videos = set(os.listdir(self.frame_dir))  # ✅ Now checking extracted frames
        if self.selected_videos is not None:
            available_videos = available_videos.intersection(self.selected_videos)
        return available_videos

    def _filter_valid_videos(self):
        valid_indices = []
        for idx in range(len(self.mos_df)):
            video_id = self.mos_df.iloc[idx]['vid']  # Extracted frame folders match video IDs
            if video_id in self.available_videos:
                valid_indices.append(idx)
        return valid_indices

    def __getitem__(self, idx):
        valid_idx = self.valid_indices[idx]
        video_id = self.mos_df.iloc[valid_idx]['vid']
        mos_score = self.mos_df.iloc[valid_idx]['MOS full']

        # Normalize MOS score
        mos_min = self.mos_df['MOS full'].min()
        mos_max = self.mos_df['MOS full'].max()
        mos_score = (mos_score - mos_min) / (mos_max - mos_min)

        frame_folder = os.path.join(self.frame_dir, video_id)
        saliency_folder = os.path.join(self.saliency_dir, video_id)

        # Load pre-extracted frames
        frame_files = sorted([os.path.join(frame_folder, img) for img in os.listdir(frame_folder) if img.endswith('.png')])

        if len(frame_files) != self.num_frames:
            raise ValueError(f"Mismatch: Expected {self.num_frames} frames but found {len(frame_files)} for {video_id}")

        frames = []
        for frame_file in frame_files:
            frame = Image.open(frame_file).convert("RGB")
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        # Load and align saliency maps
        if not os.path.exists(saliency_folder):
            raise FileNotFoundError(f"Saliency folder does not exist: {saliency_folder}")

        saliency_maps = sorted([os.path.join(saliency_folder, img) for img in os.listdir(saliency_folder) if img.endswith('.png')])

        if len(saliency_maps) >= self.num_frames:
            saliency_indices = np.linspace(0, len(saliency_maps) - 1, self.num_frames, dtype=int)
            saliency_maps = [saliency_maps[i] for i in saliency_indices]
        else:
            saliency_maps.extend([saliency_maps[-1]] * (self.num_frames - len(saliency_maps)))

        frame_tensors = []
        for i, (frame, saliency_map) in enumerate(zip(frames, saliency_maps)):
            saliency = Image.open(saliency_map).convert('L')
            saliency = self.saliency_transforms(saliency)

            if torch.all(saliency == 0):
                print(f"Warning: Zero saliency map detected for {saliency_map}")
                saliency = torch.ones_like(saliency)

            saliency = saliency / max(torch.max(saliency), 1e-6)
            saliency = torch.clamp(saliency, min=0.1)

            alpha = 0.5
            weighted_frame = (1 - alpha) * frame + alpha * (frame * saliency)

            frame_tensors.append(weighted_frame)

        frames_tensor = torch.stack(frame_tensors, dim=1)  # Shape: (C, num_frames, H, W)

        return frames_tensor, torch.tensor(mos_score, dtype=torch.float32), video_id
    
    
class LSVQDataset(Dataset):
    def __init__(self, frame_dir, mos_csv, saliency_dir, transform=None, num_frames=8, alpha=0):
        """
        Dataset class for LSVQ loading pre-extracted frames and saliency maps.

        :param frame_dir: Directory with video folders containing extracted frames.
        :param mos_csv: Path to CSV file containing MOS and video names.
        :param saliency_dir: Directory with video folders containing saliency maps.
        :param transform: Transform for video frames.
        :param num_frames: Number of frames to use per video.
        :param alpha: Fusion weight for saliency-guided frame blending.
        """
        self.frame_dir = frame_dir
        self.saliency_dir = saliency_dir
        self.transform = transform
        self.num_frames = num_frames
        self.alpha = alpha

        self.saliency_transforms = transforms.Compose([
            transforms.Resize((224, 398)),
            # transforms.Resize((1440, 2560)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Load video info from CSV
        self.df = pd.read_csv(mos_csv)
        self.df["video_folder"] = self.df["name"].apply(lambda x: os.path.basename(x))
        self.df["video_name"] = self.df["video_folder"].apply(lambda x: x.split("/")[-1])
        self.df["mos"] = self.df["mos"].astype(float)

        self.available_videos = set(os.listdir(self.frame_dir))
        self.valid_indices = self._filter_valid_videos()

    def __len__(self):
        return len(self.valid_indices)

    def _filter_valid_videos(self):
        valid = []
        for idx in range(len(self.df)):
            folder = self.df.iloc[idx]["video_folder"]
            if os.path.exists(os.path.join(self.frame_dir, folder)):
                valid.append(idx)
        return valid

    def __getitem__(self, idx):
        try:
            raw_idx = self.valid_indices[idx]
            row = self.df.iloc[raw_idx]
            video_folder = row["video_folder"]
            video_name = row["video_name"]
            mos_score = row["mos"]

            # Normalize MOS
            mos_min = self.df["mos"].min()
            mos_max = self.df["mos"].max()
            mos_score = (mos_score - mos_min) / (mos_max - mos_min)

            frame_path = os.path.join(self.frame_dir, video_folder)
            saliency_path = os.path.join(self.saliency_dir, video_name)

            frame_files = sorted([os.path.join(frame_path, f) for f in os.listdir(frame_path) if f.endswith(".png")])
            if len(frame_files) < self.num_frames:
                raise ValueError(f"Not enough frames: {len(frame_files)} in {frame_path}")

            frame_files = frame_files[:self.num_frames]
            frames = [self.transform(Image.open(f).convert("RGB")) if self.transform else transforms.ToTensor()(Image.open(f).convert("RGB")) for f in frame_files]

            if not os.path.exists(saliency_path):
                raise FileNotFoundError(f"Saliency folder not found: {saliency_path}")

            saliency_files = sorted([os.path.join(saliency_path, f) for f in os.listdir(saliency_path) if f.endswith(".png")])
            if len(saliency_files) < self.num_frames:
                saliency_files += [saliency_files[-1]] * (self.num_frames - len(saliency_files))
            else:
                saliency_files = saliency_files[:self.num_frames]

            frame_tensors = []
            for frame, sal_path in zip(frames, saliency_files):
                sal = Image.open(sal_path).convert("L")
                sal = self.saliency_transforms(sal)

                if torch.all(sal == 0):
                    print(f"Warning: Zero saliency at {sal_path}")
                    sal = torch.ones_like(sal)

                sal = sal / max(torch.max(sal), 1e-6)
                sal = torch.clamp(sal, min=0.1)

                weighted = (1 - self.alpha) * frame + self.alpha * (frame * sal)
                frame_tensors.append(weighted)

            frames_tensor = torch.stack(frame_tensors, dim=1)  # [C, T, H, W]
            return frames_tensor, torch.tensor(mos_score, dtype=torch.float32), video_name

        except Exception as e:
            # Fallback: try another sample
            print(f"[Skipping] Error at index {idx}: {e}")
            new_idx = random.randint(0, len(self.valid_indices) - 1)
            return self.__getitem__(new_idx)

