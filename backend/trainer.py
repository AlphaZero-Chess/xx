import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict
import logging
from device_manager import device_manager
import time

logger = logging.getLogger(__name__)

class AlphaZeroTrainer:
    """Training system for AlphaZero network"""
    
    def __init__(self, neural_network, learning_rate=0.001):
        self.neural_network = neural_network
        self.optimizer = optim.Adam(neural_network.parameters(), lr=learning_rate)
        self.device = device_manager.device
        self.neural_network.to(self.device)
        logger.info(f"Trainer initialized on {device_manager.device_name}")
        
    def prepare_batch(self, training_data: List[Dict], batch_size=32):
        """Prepare training batch from self-play data"""
        if len(training_data) == 0:
            return []
        
        # Shuffle data
        np.random.shuffle(training_data)
        
        batches = []
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i + batch_size]
            
            # Extract positions, policies, and values
            positions = []
            target_policies = []
            target_values = []
            
            for entry in batch:
                positions.append(entry['position'])
                
                # Convert policy dict to array
                policy_array = np.zeros(4096, dtype=np.float32)
                if isinstance(entry['policy'], dict):
                    from chess_engine import ChessEngine
                    engine = ChessEngine()
                    for move, prob in entry['policy'].items():
                        idx = engine.move_to_index(move)
                        if idx < 4096:
                            policy_array[idx] = prob
                
                target_policies.append(policy_array)
                target_values.append(entry['value'])
            
            # Convert to tensors
            positions_tensor = torch.FloatTensor(np.array(positions)).permute(0, 3, 1, 2)
            policies_tensor = torch.FloatTensor(np.array(target_policies))
            values_tensor = torch.FloatTensor(np.array(target_values)).unsqueeze(1)
            
            batches.append({
                'positions': positions_tensor,
                'policies': policies_tensor,
                'values': values_tensor
            })
        
        return batches
    
    def train_epoch(self, training_data: List[Dict], batch_size=32):
        """Train for one epoch"""
        self.neural_network.train()
        
        batches = self.prepare_batch(training_data, batch_size)
        if not batches:
            return {'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for batch in batches:
            positions = batch['positions'].to(self.device)
            target_policies = batch['policies'].to(self.device)
            target_values = batch['values'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_log_policies, pred_values = self.neural_network(positions)
            
            # Calculate losses
            # Policy loss: cross entropy
            policy_loss = -torch.sum(target_policies * pred_log_policies) / positions.size(0)
            
            # Value loss: MSE
            value_loss = nn.MSELoss()(pred_values, target_values)
            
            # Combined loss
            loss = policy_loss + value_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        num_batches = len(batches)
        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'num_batches': num_batches
        }
    
    def train(self, training_data: List[Dict], num_epochs=10, batch_size=32):
        """Train for multiple epochs with performance timing"""
        training_history = []
        
        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_start = time.time()
            metrics = self.train_epoch(training_data, batch_size)
            epoch_time = time.time() - epoch_start
            
            metrics['epoch'] = epoch + 1
            metrics['timestamp'] = datetime.now(timezone.utc).isoformat()
            metrics['epoch_time'] = epoch_time
            metrics['device'] = device_manager.device_name
            training_history.append(metrics)
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {metrics['loss']:.4f}, "
                       f"Policy Loss: {metrics['policy_loss']:.4f}, Value Loss: {metrics['value_loss']:.4f}, "
                       f"Time: {epoch_time:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"Training complete. Total time: {total_time:.2f}s on {device_manager.device_name}")
        
        return training_history