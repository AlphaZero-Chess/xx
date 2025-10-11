"""
Device management utility for GPU/CPU detection and tensor operations
"""
import torch
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device selection and tensor operations across GPU/CPU"""
    
    _instance = None
    _device = None
    _device_name = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
            cls._instance._initialize_device()
        return cls._instance
    
    def _initialize_device(self):
        """Initialize and detect available device"""
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
            self._device_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {self._device_name}")
            logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            self._device = torch.device('cpu')
            self._device_name = "CPU"
            logger.info("No GPU detected. Using CPU.")
    
    @property
    def device(self):
        """Get current device"""
        return self._device
    
    @property
    def device_name(self):
        """Get device name"""
        return self._device_name
    
    @property
    def is_gpu(self):
        """Check if GPU is being used"""
        return self._device.type == 'cuda'
    
    def get_device_info(self):
        """Get detailed device information"""
        info = {
            'device_type': self._device.type,
            'device_name': self._device_name,
            'is_gpu': self.is_gpu,
        }
        
        if self.is_gpu:
            info.update({
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'gpu_memory_allocated': f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB",
                'gpu_memory_reserved': f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB",
            })
        
        return info
    
    def to_device(self, tensor):
        """Move tensor to current device"""
        return tensor.to(self._device)
    
    def empty_cache(self):
        """Empty GPU cache if using GPU"""
        if self.is_gpu:
            torch.cuda.empty_cache()


# Global device manager instance
device_manager = DeviceManager()
