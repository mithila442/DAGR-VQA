import os
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from sal_withreg import UNetWithRegisterTokens

class BatchSaliencyMapGenerator:
    def __init__(self, model_checkpoint, device, seq_len=8, resize_size=(224, 398)):
        self.device = device
        self.seq_len = seq_len
        self.resize_size = resize_size
        self.model = UNetWithRegisterTokens(in_channels=3, out_channels=1, num_register_tokens=4).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(model_checkpoint, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

        self.frame_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor()
        ])

    def process_video(self, video_path, output_dir):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        os.makedirs(output_dir, exist_ok=True)

        frames_rgb = []
        frame_indices = []

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_rgb.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            frame_indices.append(idx)
            idx += 1
        cap.release()

        # Padding if needed
        if len(frames_rgb) < self.seq_len:
            n_to_pad = self.seq_len - len(frames_rgb)
            frames_rgb += [frames_rgb[-1]] * n_to_pad
            frame_indices += [frame_indices[-1]] * n_to_pad

        # Batch inference over temporal windows
        num_windows = int(np.ceil(len(frames_rgb)/self.seq_len))
        for w in range(num_windows):
            start_idx = w * self.seq_len
            end_idx = min((w + 1) * self.seq_len, len(frames_rgb))
            window = frames_rgb[start_idx:end_idx]

            # Pad last window if < seq_len
            if len(window) < self.seq_len:
                window += [window[-1]] * (self.seq_len - len(window))

            window_tensors = torch.stack([self.frame_transform(img) for img in window])  # [seq_len, 3, H, W]
            input_tensor = window_tensors.unsqueeze(0).to(self.device)                   # [1, seq_len, 3, H, W]
            
            #print("Input tensor min/max:", input_tensor.min().item(), input_tensor.max().item())

            with torch.no_grad():
                preds = self.model(input_tensor)  # [1, 1, seq_len, H, W]
                preds = torch.sigmoid(preds)
                #print("Output preds min/max:", preds.min().item(), preds.max().item())
                preds = preds.squeeze(0).squeeze(0)  # Now [seq_len, H, W]
                
                #print("preds shape after squeeze:", preds.shape)  # Should be (seq_len, H, W)

            for i in range(self.seq_len):
                out_idx = start_idx + i
                if out_idx >= len(frame_indices):
                    continue
                sal_map = preds[i]  # Should be (H, W)
                
                #print("sal_map.shape:", sal_map.shape)  # Should be (H, W)
                
                if sal_map.ndim != 2:
                    raise ValueError(f"Saliency map shape after index is invalid: {sal_map.shape}")
                sal_map_2d = sal_map.cpu().numpy()  # convert to np.ndarray (if not already)
                sal_map_2d = (sal_map_2d - sal_map_2d.min()) / (sal_map_2d.max() - sal_map_2d.min() + 1e-8)
                sal_map_uint8 = (sal_map_2d * 255.0).astype(np.uint8)
                out_path = os.path.join(output_dir, f"{frame_indices[out_idx]:04d}.png")
                Image.fromarray(sal_map_uint8).save(out_path)



if __name__ == "__main__":
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model_checkpoint = './checkpoint_epoch_190.pth'
    generator = BatchSaliencyMapGenerator(model_checkpoint, device, seq_len=8) # Set seq_len to your training value

    video_root_dir = './LSVQ/Videos'
    output_root_dir = './LSVQ_saliency_maps'
    # video_root_dir = './KoNViD_1k/KoNViD_1k_videos'  # Root directory containing videos
    # output_root_dir = './KoNViD_1k_saliency_maps'  # Root directory for saving saliency maps
    video_files = sorted([f for f in os.listdir(video_root_dir) if f.endswith('.mp4')])

    print(f"Found video files: {video_files}")

    for video_file in video_files:
        video_path = os.path.join(video_root_dir, video_file)
        output_dir = os.path.join(output_root_dir, video_file.replace('.mp4', ''))
        print(f"Processing video {video_file}...")
        generator.process_video(video_path, output_dir)
        print(f"Finished processing video {video_file}")