"""
Model Export and Management System
Handles model export to different formats (PyTorch, ONNX) and metadata tracking
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, List
import logging

# Add backend directory to path if needed
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import torch
import torch.onnx
from neural_network import AlphaZeroNetwork, ModelManager
from device_manager import device_manager

logger = logging.getLogger(__name__)

class ModelExporter:
    """Handle model export operations"""
    
    def __init__(self, export_dir="/app/backend/exports"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.model_manager = ModelManager()
        
    def export_pytorch(self, model_name: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Export model in PyTorch format (.pt) with metadata
        Returns: export information dict
        """
        try:
            # Load the model
            model, existing_metadata = self.model_manager.load_model(model_name)
            if model is None:
                raise Exception(f"Model {model_name} not found")
            
            # Merge metadata
            export_metadata = existing_metadata.copy() if existing_metadata else {}
            if metadata:
                export_metadata.update(metadata)
            
            # Add export information
            export_metadata['export_date'] = datetime.now(timezone.utc).isoformat()
            export_metadata['export_format'] = 'pytorch'
            export_metadata['device_used'] = device_manager.device_name
            
            # Create export filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_filename = f"{model_name}_export_{timestamp}.pt"
            export_path = self.export_dir / export_filename
            
            # Save with full metadata
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'metadata': export_metadata,
                'architecture': {
                    'num_channels': 128,
                    'num_res_blocks': 6,
                    'input_shape': [14, 8, 8],
                    'output_shapes': {
                        'policy': 4096,
                        'value': 1
                    }
                }
            }
            
            torch.save(checkpoint, export_path)
            
            file_size = os.path.getsize(export_path) / (1024 * 1024)  # MB
            
            logger.info(f"Model exported to PyTorch format: {export_path}")
            
            return {
                'success': True,
                'format': 'pytorch',
                'filename': export_filename,
                'path': str(export_path),
                'file_size_mb': round(file_size, 2),
                'metadata': export_metadata
            }
            
        except Exception as e:
            logger.error(f"Error exporting PyTorch model: {str(e)}")
            raise Exception(f"Failed to export PyTorch model: {str(e)}")
    
    def export_onnx(self, model_name: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Export model in ONNX format for cross-platform deployment
        Returns: export information dict
        """
        try:
            # Load the model
            model, existing_metadata = self.model_manager.load_model(model_name)
            if model is None:
                raise Exception(f"Model {model_name} not found")
            
            model.eval()
            
            # Create dummy input for tracing
            dummy_input = torch.randn(1, 14, 8, 8).to(device_manager.device)
            
            # Create export filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_filename = f"{model_name}_export_{timestamp}.onnx"
            export_path = self.export_dir / export_filename
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(export_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['board_input'],
                output_names=['policy_output', 'value_output'],
                dynamic_axes={
                    'board_input': {0: 'batch_size'},
                    'policy_output': {0: 'batch_size'},
                    'value_output': {0: 'batch_size'}
                }
            )
            
            # Save metadata separately
            export_metadata = existing_metadata.copy() if existing_metadata else {}
            if metadata:
                export_metadata.update(metadata)
            
            export_metadata['export_date'] = datetime.now(timezone.utc).isoformat()
            export_metadata['export_format'] = 'onnx'
            export_metadata['device_used'] = device_manager.device_name
            export_metadata['opset_version'] = 11
            
            metadata_filename = f"{model_name}_export_{timestamp}.json"
            metadata_path = self.export_dir / metadata_filename
            
            with open(metadata_path, 'w') as f:
                json.dump(export_metadata, f, indent=2)
            
            file_size = os.path.getsize(export_path) / (1024 * 1024)  # MB
            
            logger.info(f"Model exported to ONNX format: {export_path}")
            
            return {
                'success': True,
                'format': 'onnx',
                'filename': export_filename,
                'metadata_filename': metadata_filename,
                'path': str(export_path),
                'file_size_mb': round(file_size, 2),
                'metadata': export_metadata
            }
            
        except Exception as e:
            logger.error(f"Error exporting ONNX model: {str(e)}")
            raise Exception(f"Failed to export ONNX model: {str(e)}")
    
    def list_exports(self) -> List[Dict]:
        """List all exported models"""
        try:
            exports = []
            
            # List PyTorch exports
            for pt_file in self.export_dir.glob("*.pt"):
                try:
                    checkpoint = torch.load(pt_file, map_location='cpu', weights_only=False)
                    metadata = checkpoint.get('metadata', {})
                    file_size = os.path.getsize(pt_file) / (1024 * 1024)
                    
                    exports.append({
                        'filename': pt_file.name,
                        'format': 'pytorch',
                        'file_size_mb': round(file_size, 2),
                        'export_date': metadata.get('export_date', 'unknown'),
                        'metadata': metadata
                    })
                except Exception as e:
                    logger.warning(f"Could not load metadata from {pt_file.name}: {e}")
            
            # List ONNX exports
            for onnx_file in self.export_dir.glob("*.onnx"):
                try:
                    file_size = os.path.getsize(onnx_file) / (1024 * 1024)
                    
                    # Try to load associated metadata
                    metadata = {}
                    metadata_file = onnx_file.with_suffix('.json')
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    
                    exports.append({
                        'filename': onnx_file.name,
                        'format': 'onnx',
                        'file_size_mb': round(file_size, 2),
                        'export_date': metadata.get('export_date', 'unknown'),
                        'metadata': metadata
                    })
                except Exception as e:
                    logger.warning(f"Could not load metadata from {onnx_file.name}: {e}")
            
            # Sort by export date (newest first)
            exports.sort(key=lambda x: x.get('export_date', ''), reverse=True)
            
            return exports
            
        except Exception as e:
            logger.error(f"Error listing exports: {str(e)}")
            return []
    
    def get_export_path(self, filename: str) -> Optional[Path]:
        """Get full path to an exported file"""
        path = self.export_dir / filename
        return path if path.exists() else None
    
    def cleanup_old_exports(self, keep_recent: int = 10):
        """Clean up old export files, keeping only recent ones"""
        try:
            exports = self.list_exports()
            if len(exports) > keep_recent:
                to_delete = exports[keep_recent:]
                for export in to_delete:
                    file_path = self.export_dir / export['filename']
                    if file_path.exists():
                        file_path.unlink()
                        logger.info(f"Deleted old export: {export['filename']}")
                    
                    # Delete associated metadata for ONNX
                    if export['format'] == 'onnx':
                        metadata_path = file_path.with_suffix('.json')
                        if metadata_path.exists():
                            metadata_path.unlink()
                
                return len(to_delete)
            return 0
        except Exception as e:
            logger.error(f"Error cleaning up exports: {str(e)}")
            return 0


class ModelLoader:
    """Handle model loading and management"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.current_model = None
        self.current_model_name = None
        self.current_model_metadata = None
    
    def load_model(self, model_name: str) -> Dict:
        """
        Load a specific model and make it active
        Returns: model information
        """
        try:
            model, metadata = self.model_manager.load_model(model_name)
            
            if model is None:
                raise Exception(f"Model {model_name} not found")
            
            # Verify architecture compatibility
            model_info = self.model_manager.get_model_info(model_name)
            if model_info:
                arch = model_info.get('architecture', {})
                expected_channels = arch.get('num_channels', 128)
                expected_blocks = arch.get('num_res_blocks', 6)
                
                # Basic compatibility check
                if expected_channels != 128 or expected_blocks != 6:
                    logger.warning(f"Model architecture mismatch. Expected 128 channels, 6 blocks. "
                                 f"Got {expected_channels} channels, {expected_blocks} blocks.")
            
            # Set as current model
            self.current_model = model
            self.current_model_name = model_name
            self.current_model_metadata = metadata or {}
            
            logger.info(f"Loaded model: {model_name} on {device_manager.device_name}")
            
            return {
                'success': True,
                'model_name': model_name,
                'metadata': metadata,
                'device': device_manager.device_name,
                'load_date': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise Exception(f"Failed to load model: {str(e)}")
    
    def get_current_model_info(self) -> Optional[Dict]:
        """Get information about currently loaded model"""
        if self.current_model is None:
            return None
        
        return {
            'model_name': self.current_model_name,
            'metadata': self.current_model_metadata,
            'device': device_manager.device_name,
            'device_info': device_manager.get_device_info()
        }
    
    def list_available_models(self) -> List[Dict]:
        """List all available models with metadata"""
        try:
            models = self.model_manager.list_models()
            model_list = []
            
            for model_name in models:
                model_info = self.model_manager.get_model_info(model_name)
                if model_info:
                    metadata = model_info.get('metadata', {})
                    model_path = self.model_manager.get_model_path(model_name)
                    file_size = os.path.getsize(model_path) / (1024 * 1024) if model_path.exists() else 0
                    
                    model_list.append({
                        'name': model_name,
                        'version': metadata.get('version', 'unknown'),
                        'training_date': metadata.get('training_date', 'unknown'),
                        'file_size_mb': round(file_size, 2),
                        'metadata': metadata,
                        'is_current': model_name == self.current_model_name
                    })
            
            # Sort by version or name
            model_list.sort(key=lambda x: x.get('version', 0) if isinstance(x.get('version'), int) else 0, reverse=True)
            
            return model_list
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []


# Global instances
model_exporter = ModelExporter()
model_loader = ModelLoader()
