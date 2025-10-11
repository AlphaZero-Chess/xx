import math
import numpy as np
import chess
from typing import Dict, Optional, List
from chess_engine import ChessEngine
from device_manager import device_manager
import time
import logging

logger = logging.getLogger(__name__)

class MCTSNode:
    """Node in the Monte Carlo Tree Search"""
    
    def __init__(self, engine: ChessEngine, parent=None, move=None, prior=0.0):
        self.engine = engine.copy()
        self.parent = parent
        self.move = move  # Move that led to this node
        self.prior = prior  # Prior probability from NN
        
        self.children = {}  # {move_uci: MCTSNode}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
    def value(self):
        """Average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def is_leaf(self):
        """Check if this is a leaf node"""
        return not self.is_expanded
    
    def select_child(self, c_puct):
        """Select child with highest UCB score"""
        best_score = -float('inf')
        best_move = None
        best_child = None
        
        for move, child in self.children.items():
            ucb_score = self.ucb_score(child, c_puct)
            if ucb_score > best_score:
                best_score = ucb_score
                best_move = move
                best_child = child
        
        return best_move, best_child
    
    def ucb_score(self, child, c_puct):
        """Calculate UCB score (PUCT formula)"""
        # Q(s,a) + U(s,a)
        # U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        q_value = child.value()
        u_value = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
        
        return q_value + u_value
    
    def expand(self, policy_probs: Dict[str, float]):
        """Expand node with children based on policy"""
        legal_moves = self.engine.get_legal_moves()
        
        for move in legal_moves:
            if move not in self.children:
                child_engine = self.engine.copy()
                child_engine.make_move(move)
                prior = policy_probs.get(move, 0.0)
                self.children[move] = MCTSNode(child_engine, parent=self, move=move, prior=prior)
        
        self.is_expanded = True
    
    def update(self, value: float):
        """Update node statistics"""
        self.visit_count += 1
        self.value_sum += value
    
    def backpropagate(self, value: float):
        """Backpropagate value up the tree"""
        self.update(value)
        if self.parent:
            self.parent.backpropagate(-value)  # Flip value for opponent


class MCTS:
    """Monte Carlo Tree Search guided by neural network with batch inference optimization"""
    
    def __init__(self, neural_network, num_simulations=800, c_puct=1.0, batch_size=8):
        self.neural_network = neural_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.inference_cache = {}  # Simple FEN-based cache for positions
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _evaluate_position(self, node: MCTSNode):
        """Evaluate a single position with caching"""
        fen = node.engine.get_fen()
        
        # Check cache first
        if fen in self.inference_cache:
            self.cache_hits += 1
            policy, value = self.inference_cache[fen]
        else:
            self.cache_misses += 1
            board_encoding = node.engine.encode_board()
            policy, value = self.neural_network.predict(board_encoding)
            # Cache the result
            self.inference_cache[fen] = (policy, value)
        
        # Convert policy to move probabilities
        move_probs = node.engine.get_move_probabilities(policy)
        return move_probs, value
    
    def search(self, engine: ChessEngine, temperature=1.0):
        """
        Run MCTS from current position with optimized inference
        Returns: best_move (str), move_probabilities (dict), root_value (float)
        """
        start_time = time.time()
        root = MCTSNode(engine)
        
        # Run simulations
        for sim in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection: traverse tree using UCB
            while not node.is_leaf() and not node.engine.is_game_over():
                move, node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # Expansion and evaluation
            if not node.engine.is_game_over():
                # Get NN predictions with caching
                move_probs, value = self._evaluate_position(node)
                
                # Expand node
                node.expand(move_probs)
            else:
                # Terminal node - use game result
                result = node.engine.get_result()
                if result == "1-0":
                    value = 1.0 if node.engine.board.turn == chess.BLACK else -1.0
                elif result == "0-1":
                    value = 1.0 if node.engine.board.turn == chess.WHITE else -1.0
                else:
                    value = 0.0
            
            # Backpropagation
            for node_in_path in reversed(search_path):
                node_in_path.update(value)
                value = -value  # Flip for alternating players
        
        search_time = time.time() - start_time
        
        # Log performance metrics
        if self.num_simulations % 100 == 0:
            cache_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            logger.debug(f"MCTS: {self.num_simulations} sims in {search_time:.2f}s, "
                        f"Cache hit rate: {cache_rate:.1%}")
        
        # Select move based on visit counts
        visit_counts = {move: child.visit_count for move, child in root.children.items()}
        
        if temperature == 0:
            # Deterministic: pick most visited
            best_move = max(visit_counts, key=visit_counts.get)
        else:
            # Stochastic: sample based on visit counts with temperature
            moves = list(visit_counts.keys())
            counts = np.array([visit_counts[m] for m in moves])
            
            if temperature != 1.0:
                counts = counts ** (1.0 / temperature)
            
            probs = counts / counts.sum()
            best_move = np.random.choice(moves, p=probs)
        
        # Calculate move probabilities for training
        total_visits = sum(visit_counts.values())
        move_probs = {move: count / total_visits for move, count in visit_counts.items()}
        
        root_value = root.value()
        
        return best_move, move_probs, root_value
    
    def get_best_move(self, engine: ChessEngine):
        """Get best move (deterministic, temperature=0)"""
        best_move, _, value = self.search(engine, temperature=0)
        return best_move, value
    
    def get_performance_stats(self):
        """Get MCTS performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        cache_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': cache_rate,
            'cache_size': len(self.inference_cache)
        }
    
    def clear_cache(self):
        """Clear inference cache"""
        self.inference_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0