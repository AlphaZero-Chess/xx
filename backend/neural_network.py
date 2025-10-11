import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from device_manager import device_manager
import logging

logger = logging.getLogger(__name__)

class ResidualBlock(nn.Module):
    """Residual block for AlphaZero network"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class AlphaZeroNetwork(nn.Module):
    """
    AlphaZero-style neural network with ResNet architecture
    Input: (batch, 14, 8, 8) board encoding
    Output: policy (batch, 4096), value (batch, 1)
    """
    
    def __init__(self, num_channels=128, num_res_blocks=6):
        super().__init__()
        
        # Initial convolution
        self.conv_input = nn.Conv2d(14, num_channels, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)
        
        # Value head
        self.value_conv = nn.Conv2d(num_channels, 32, 1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Move network to appropriate device
        self.to(device_manager.device)
        logger.info(f"Neural network initialized on {device_manager.device_name}")
        
    def forward(self, x):
        # Input shape: (batch, 14, 8, 8)
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual tower
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.reshape(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.reshape(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, board_encoding: np.ndarray):
        """
        Predict policy and value for a single position
        board_encoding: (8, 8, 14) numpy array
        Returns: policy (4096,), value (scalar)
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            x = torch.FloatTensor(board_encoding).permute(2, 0, 1).unsqueeze(0)
            x = device_manager.to_device(x)
            policy, value = self.forward(x)
            policy = torch.exp(policy).cpu().numpy()[0]
            value = value.cpu().numpy()[0, 0]
        return policy, value
    
    def predict_batch(self, board_encodings: list):
        """
        Predict policy and value for multiple positions (batch inference)
        board_encodings: list of (8, 8, 14) numpy arrays
        Returns: policies (batch, 4096), values (batch,)
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor batch
            batch = torch.stack([
                torch.FloatTensor(enc).permute(2, 0, 1) 
                for enc in board_encodings
            ])
            batch = device_manager.to_device(batch)
            policies, values = self.forward(batch)
            policies = torch.exp(policies).cpu().numpy()
            values = values.cpu().numpy().squeeze()
        return policies, values

class ModelManager:
    """Manage model saving and loading"""
    
    def __init__(self, model_dir="/app/backend/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def get_next_version(self):
        """Get next version number for model naming"""
        existing_models = self.list_models()
        version_numbers = []
        
        for model_name in existing_models:
            # Extract version number from model_v{X} pattern
            if model_name.startswith('model_v'):
                try:
                    version = int(model_name.split('_v')[1])
                    version_numbers.append(version)
                except (ValueError, IndexError):
                    continue
        
        if version_numbers:
            return max(version_numbers) + 1
        else:
            return 1
    
    def save_model(self, model, name="alphazero_model", metadata=None):
        """Save model checkpoint"""
        try:
            path = self.model_dir / f"{name}.pt"
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'metadata': metadata or {},
                'architecture': {
                    'num_channels': 128,
                    'num_res_blocks': 6
                }
            }
            torch.save(checkpoint, path)
            return str(path)
        except Exception as e:
            raise Exception(f"Failed to save model: {str(e)}")
    
    def save_versioned_model(self, model, metadata=None):
        """Save model with automatic version numbering"""
        version = self.get_next_version()
        name = f"model_v{version}"
        
        if metadata is None:
            metadata = {}
        metadata['version'] = version
        
        return self.save_model(model, name=name, metadata=metadata)
    
    def load_model(self, name="alphazero_model"):
        """Load model checkpoint"""
        try:
            path = self.model_dir / f"{name}.pt"
            if not path.exists():
                return None, None
            
            # Load to CPU first, then move to appropriate device
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            model = AlphaZeroNetwork()
            model.load_state_dict(checkpoint['model_state_dict'])
            # Model is automatically moved to device in __init__
            return model, checkpoint.get('metadata', {})
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def list_models(self):
        """List all saved models"""
        try:
            return [f.stem for f in self.model_dir.glob("*.pt")]
        except Exception as e:
            return []
    
    def get_model_info(self, name):
        """Get model metadata without loading full model"""
        try:
            path = self.model_dir / f"{name}.pt"
            if not path.exists():
                return None
            
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            return {
                'name': name,
                'metadata': checkpoint.get('metadata', {}),
                'architecture': checkpoint.get('architecture', {})
            }
        except Exception as e:
            return None
    
    def get_model_path(self, name="alphazero_model"):
        """Get full path to model file"""
        return self.model_dir / f"{name}.pt"
    
    def delete_model(self, name="alphazero_model"):
        """Delete a model checkpoint"""
        try:
            path = self.model_dir / f"{name}.pt"
            if path.exists():
                path.unlink()
                return True
            return False
        except Exception as e:
            raise Exception(f"Failed to delete model: {str(e)}")