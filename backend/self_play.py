import numpy as np
from typing import List, Tuple
from chess_engine import ChessEngine
from mcts import MCTS
import chess
import time
import logging

logger = logging.getLogger(__name__)

class SelfPlayGame:
    """Generate self-play games for training"""
    
    def __init__(self, neural_network, num_simulations=800, c_puct=1.0):
        self.neural_network = neural_network
        self.mcts = MCTS(neural_network, num_simulations, c_puct)
        
    def play_game(self, temperature_threshold=15, store_fen=False):
        """
        Play one self-play game with performance tracking
        Returns: list of (position, policy, outcome) tuples
        """
        start_time = time.time()
        engine = ChessEngine()
        game_history = []
        move_count = 0
        
        while not engine.is_game_over():
            # Encode current position
            board_encoding = engine.encode_board()
            current_fen = engine.get_fen() if store_fen else None
            
            # Use MCTS to get move and policy
            temperature = 1.0 if move_count < temperature_threshold else 0.0
            move, move_probs, _ = self.mcts.search(engine, temperature=temperature)
            
            # Store position and policy
            game_history.append({
                'position': board_encoding,
                'policy': move_probs,
                'player': engine.board.turn,
                'fen': current_fen,
                'move_number': move_count
            })
            
            # Make move
            engine.make_move(move)
            move_count += 1
            
            # Safety limit
            if move_count > 500:
                break
        
        game_time = time.time() - start_time
        
        # Get game outcome
        result = engine.get_result()
        if result == "1-0":
            outcome = 1.0
        elif result == "0-1":
            outcome = -1.0
        else:
            outcome = 0.0
        
        # Assign outcomes from each player's perspective
        training_data = []
        for entry in game_history:
            # Outcome from perspective of player to move
            if entry['player'] == chess.WHITE:
                value = outcome
            else:
                value = -outcome
            
            training_data.append({
                'position': entry['position'],
                'policy': entry['policy'],
                'value': value,
                'fen': entry['fen'],
                'move_number': entry['move_number']
            })
        
        logger.info(f"Self-play game complete: {result}, {move_count} moves in {game_time:.2f}s")
        
        return training_data, result, move_count

class SelfPlayManager:
    """Manage self-play game generation"""
    
    def __init__(self, neural_network, num_simulations=800):
        self.neural_network = neural_network
        self.num_simulations = num_simulations
        self.self_play = SelfPlayGame(neural_network, num_simulations=num_simulations)
        
    def generate_games(self, num_games=10, store_fen=False):
        """Generate multiple self-play games with performance tracking"""
        start_time = time.time()
        all_training_data = []
        game_results = []
        
        for i in range(num_games):
            logger.info(f"Starting self-play game {i + 1}/{num_games}")
            training_data, result, num_moves = self.self_play.play_game(store_fen=store_fen)
            all_training_data.extend(training_data)
            game_results.append({
                'game_num': i + 1,
                'result': result,
                'num_moves': num_moves,
                'num_positions': len(training_data)
            })
        
        total_time = time.time() - start_time
        logger.info(f"Generated {num_games} games with {len(all_training_data)} positions in {total_time:.2f}s")
        
        return all_training_data, game_results