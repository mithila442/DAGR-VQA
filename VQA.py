import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
from torch.nn import SyncBatchNorm
from torchvision.models import ResNet50_Weights
from torchvision import transforms

class SpearmanCorrelationLoss(nn.Module):
    def __init__(self):
        super(SpearmanCorrelationLoss, self).__init__()

    def forward(self, predictions, targets):
        pred_ranks = torch.argsort(torch.argsort(predictions))
        target_ranks = torch.argsort(torch.argsort(targets))

        pred_ranks = pred_ranks.float()
        pred_ranks_centered = pred_ranks - pred_ranks.mean()
        target_ranks = target_ranks.float()
        target_ranks_centered = target_ranks - target_ranks.mean()

        numerator = torch.sum(pred_ranks_centered * target_ranks_centered)
        denominator = torch.sqrt(torch.sum(pred_ranks_centered**2) * torch.sum(target_ranks_centered**2))

        spearman_corr = numerator / (denominator + 1e-8)
        return 1 - spearman_corr

class SpatialQualityAnalyzer(nn.Module):
    def __init__(self, pretrained=True):
        super(SpatialQualityAnalyzer, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        children = list(resnet.children())  # Convert generator to list
        self.features = nn.Sequential(
            *[SyncBatchNorm.convert_sync_batchnorm(child) for child in children[:-2]]
        )
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        batch_size, channels, frames, height, width = x.size()
        x = x.reshape(batch_size * frames, channels, height, width)  # Combine batch and frames

        # Ensure input frames are normalized before passing to the SpatialQualityAnalyzer
        if torch.all(x == 0):
            raise ValueError("All-zero frames detected before normalization.")
        #x = self.normalize(x)
        features = self.features(x)  # Extract features with spatial dimensions intact
        features = features.view(batch_size, frames, features.size(1), features.size(2), features.size(3))
        # Reshape back to [batch_size, frames, feature_dim, height, width]
        return features

class TemporalQualityAnalyzer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.3):
        super(TemporalQualityAnalyzer, self).__init__()
        self.input_dim = input_dim
        self.positional_encoding = self._get_positional_encoding(length=8, d_model=input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _get_positional_encoding(self, length, d_model):
        position = torch.arange(0, length).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: [1, length, d_model]

    def forward(self, x):
        #forcombined
        # batch_size, frames, _ = x.size()  # [batch_size, num_frames, input_dim]
        #fortemporalonly
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # Expand [batch_size, num_frames] to [batch_size, num_frames, 1]
        batch_size, frames, input_dim = x.size()
        device = x.device

        # Dynamically adjust the positional encoding to match the number of frames
        positional_encoding = self.positional_encoding[:, :frames, :].to(device)

        # Add positional encoding
        x = x + positional_encoding

        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        # Mean pooling over frames
        x = x.mean(dim=1)
        return x

class TemporalQualityAnalyzer(nn.Module):
    def __init__(self, input_dim, output_dim=1024, hidden_dim=128, num_heads=1, num_layers=2, dropout=0.3):
        super(TemporalQualityAnalyzer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _get_positional_encoding(self, length, d_model):
        position = torch.arange(0, length).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:,1::2].shape[1]])
        return pe.unsqueeze(0)  # [1, length, d_model]

    def forward(self, x):
        # Accepts: [B, T, C, H, W], [B, T, C], or [B, T]
        if x.ndim == 5:
            x = x.mean(dim=[3,4])  # [B, T, C]
        elif x.ndim == 4:
            x = x.mean(dim=[2,3], keepdim=True)  # [B, T, 1]
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # [B, T]

        # Project to output_dim if needed
        x = self.proj(x)  # [B, T, output_dim]

        # Positional encoding
        batch_size, num_frames, output_dim = x.size()
        device = x.device
        positional_encoding = self._get_positional_encoding(length=num_frames, d_model=output_dim).to(device)
        x = x + positional_encoding

        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # mean pooling over frames
        return x  # [B, output_dim]

class QualityRegressor(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[256, 128], output_dim=1):
        super(QualityRegressor, self).__init__()
        layers = []
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.ReLU(inplace=False))
            layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(hidden_dims[-1], output_dim))  # Single scalar output
        self.regressor = nn.Sequential(*layers)

        # Initialize weights with Xavier uniform
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.regressor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Input: x of shape [batch_size, feature_dim]
        Output: single scalar per video [batch_size, 1]
        """
        return self.regressor(x)  # Single score for the video
  # Single score for the video
  
 
#spatial only
class VideoQualityModelSpatialOnly(nn.Module):
    def __init__(self, device, spatial_feature_dim=2048, combined_dim=2048):
        super(VideoQualityModelSpatialOnly, self).__init__()
        self.device = device

        # Only spatial analyzer
        self.spatial_analyzer = SpatialQualityAnalyzer().to(device)

        # Projection layer for spatial features only
        self.spatial_projector = nn.Linear(spatial_feature_dim, combined_dim).to(device)

        # Quality regressor
        self.regressor = QualityRegressor(input_dim=combined_dim).to(device)

    def forward(self, x):
        # Extract spatial features
        spatial_features = self.spatial_analyzer(
            x)  # Shape: [batch_size, num_frames, spatial_feature_dim, height, width]
        batch_size, num_frames, spatial_feature_dim, height, width = spatial_features.size()

        # Average over spatial dimensions using adaptive average pooling
        spatial_features_pooled = F.adaptive_avg_pool2d(
            spatial_features.view(-1, spatial_feature_dim, height, width), (1, 1)
        )  # Shape: [batch_size * num_frames, spatial_feature_dim, 1, 1]
        spatial_features_pooled = spatial_features_pooled.view(
            batch_size, num_frames, spatial_feature_dim
        )  # Reshape to [batch_size, num_frames, spatial_feature_dim]

        # Average spatial features over frames
        spatial_features_avg = spatial_features_pooled.mean(dim=1)  # Shape: [batch_size, spatial_feature_dim]

        # Project to match regressor input
        spatial_features_proj = self.spatial_projector(spatial_features_avg)  # [batch_size, combined_dim]

        # Predict final quality score
        quality_score = self.regressor(spatial_features_proj)
        return quality_score.squeeze(-1)


#spatial+temporaL
class VideoQualityModelSimpleFusion(nn.Module):
    def __init__(self, device, spatial_feature_dim=2048, temporal_feature_dim=2048, combined_dim=2048):
        super(VideoQualityModelSimpleFusion, self).__init__()
        self.device = device

        # Spatial and temporal analyzers
        self.spatial_analyzer = SpatialQualityAnalyzer().to(device)
        self.temporal_analyzer = TemporalQualityAnalyzer(input_dim=spatial_feature_dim).to(device)

        # Projection layer for combined features
        self.combined_projector = nn.Linear(spatial_feature_dim + temporal_feature_dim, combined_dim).to(device)

        # Quality regressor
        self.regressor = QualityRegressor(input_dim=combined_dim).to(device)

    def forward(self, x):
        # Extract spatial features
        spatial_features = self.spatial_analyzer(
            x)  # Shape: [batch_size, num_frames, spatial_feature_dim, height, width]
        batch_size, num_frames, spatial_feature_dim, height, width = spatial_features.size()

        # Average over spatial dimensions using adaptive average pooling
        spatial_features_pooled = F.adaptive_avg_pool2d(
            spatial_features.view(-1, spatial_feature_dim, height, width), (1, 1)
        )  # Shape: [batch_size * num_frames, spatial_feature_dim, 1, 1]
        spatial_features_pooled = spatial_features_pooled.view(
            batch_size, num_frames, spatial_feature_dim
        )  # Reshape to [batch_size, num_frames, spatial_feature_dim]

        # Extract temporal features
        temporal_features = self.temporal_analyzer(spatial_features_pooled)  # [batch_size, temporal_feature_dim]

        # Average spatial features over frames
        spatial_features_avg = spatial_features_pooled.mean(dim=1)  # Shape: [batch_size, spatial_feature_dim]

        # Concatenate spatial and temporal features
        combined_features = torch.cat([spatial_features_avg, temporal_features],
                                      dim=-1)  # [batch_size, spatial_feature_dim + temporal_feature_dim]

        # Project combined features to match regressor input
        combined_features_proj = self.combined_projector(combined_features)  # [batch_size, combined_dim]

        # Predict final quality score
        quality_score = self.regressor(combined_features_proj)
        return quality_score.squeeze(-1)


#temporal only
class VideoQualityModelTemporalOnly(nn.Module):
    def __init__(self, device, frame_feature_dim=3, transformer_output_dim=1024,
                 temporal_hidden_dim=128, reg_hidden_dims=[256, 128]):
        super(VideoQualityModelTemporalOnly, self).__init__()
        self.device = device
        self.temporal_analyzer = TemporalQualityAnalyzer(
            input_dim=frame_feature_dim,    # e.g. 3 for RGB, 1 for intensity
            output_dim=transformer_output_dim, # e.g. 1024 to match QualityRegressor's input_dim
            hidden_dim=temporal_hidden_dim,
            num_heads=1,
            num_layers=2
        ).to(device)
        self.regressor = QualityRegressor(
            input_dim=transformer_output_dim,  # should match transformer_output_dim!
            hidden_dims=reg_hidden_dims
        ).to(device)

    def forward(self, x):
        # Accepts [B, T, 3, H, W], [B, T, 3], [B, T, 1], [B, T]
        temporal_features = self.temporal_analyzer(x)  # [B, transformer_output_dim]
        quality_score = self.regressor(temporal_features)  # [B, 1]
        return quality_score.squeeze(-1)