import logging
from typing import Dict, Tuple
from chess_engine import ChessEngine
from mcts import MCTS
import chess
import time

logger = logging.getLogger(__name__)


class EvaluationMatch:
    """Run evaluation matches between two models"""
    
    def __init__(self, model1, model2, num_simulations=400):
        """
        Initialize evaluation match
        
        Args:
            model1: First neural network model
            model2: Second neural network model
            num_simulations: MCTS simulations per move (fewer than training for speed)
        """
        self.model1 = model1
        self.model2 = model2
        self.num_simulations = num_simulations
        self.mcts1 = MCTS(model1, num_simulations=num_simulations, c_puct=1.0)
        self.mcts2 = MCTS(model2, num_simulations=num_simulations, c_puct=1.0)
    
    def play_game(self, model1_plays_white: bool) -> str:
        """
        Play a single game between two models
        
        Args:
            model1_plays_white: If True, model1 plays as White, model2 as Black
        
        Returns:
            Game result: "1-0", "0-1", or "1/2-1/2"
        """
        engine = ChessEngine()
        move_count = 0
        max_moves = 500  # Safety limit
        
        while not engine.is_game_over() and move_count < max_moves:
            # Determine which model plays this turn
            if engine.board.turn == chess.WHITE:
                mcts = self.mcts1 if model1_plays_white else self.mcts2
            else:
                mcts = self.mcts2 if model1_plays_white else self.mcts1
            
            # Get best move (deterministic, temperature=0)
            try:
                best_move, _ = mcts.get_best_move(engine)
                engine.make_move(best_move)
                move_count += 1
            except Exception as e:
                logger.error(f"Error during move generation: {e}")
                # If move fails, return draw
                return "1/2-1/2"
        
        # Get game result
        if engine.is_game_over():
            return engine.get_result()
        else:
            # Reached move limit, consider it a draw
            return "1/2-1/2"
    
    def run_evaluation(self, num_games: int = 20) -> Dict:
        """
        Run evaluation match with alternating colors and performance tracking
        
        Args:
            num_games: Number of games to play (will be split evenly between colors)
        
        Returns:
            Dictionary with evaluation results
        """
        start_time = time.time()
        results = {
            'model1_wins': 0,
            'model2_wins': 0,
            'draws': 0,
            'games_played': 0,
            'model1_as_white_wins': 0,
            'model1_as_black_wins': 0,
            'model2_as_white_wins': 0,
            'model2_as_black_wins': 0
        }
        
        # Play games with alternating colors
        for i in range(num_games):
            model1_plays_white = (i % 2 == 0)
            
            logger.info(f"Playing evaluation game {i + 1}/{num_games} "
                       f"(Model1={'White' if model1_plays_white else 'Black'})")
            
            result = self.play_game(model1_plays_white)
            results['games_played'] += 1
            
            # Update statistics
            if result == "1-0":
                # White won
                if model1_plays_white:
                    results['model1_wins'] += 1
                    results['model1_as_white_wins'] += 1
                else:
                    results['model2_wins'] += 1
                    results['model2_as_white_wins'] += 1
            elif result == "0-1":
                # Black won
                if model1_plays_white:
                    results['model2_wins'] += 1
                    results['model2_as_black_wins'] += 1
                else:
                    results['model1_wins'] += 1
                    results['model1_as_black_wins'] += 1
            else:
                # Draw
                results['draws'] += 1
        
        eval_time = time.time() - start_time
        
        # Calculate win rates
        if results['games_played'] > 0:
            results['model1_win_rate'] = results['model1_wins'] / results['games_played']
            results['model2_win_rate'] = results['model2_wins'] / results['games_played']
            results['draw_rate'] = results['draws'] / results['games_played']
        else:
            results['model1_win_rate'] = 0.0
            results['model2_win_rate'] = 0.0
            results['draw_rate'] = 0.0
        
        results['evaluation_time'] = eval_time
        logger.info(f"Evaluation complete in {eval_time:.2f}s")
        
        return results


class ModelEvaluator:
    """Manage model evaluation and comparison"""
    
    def __init__(self, num_evaluation_games=20, num_simulations=400, win_threshold=0.55):
        """
        Initialize evaluator
        
        Args:
            num_evaluation_games: Number of games per evaluation
            num_simulations: MCTS simulations per move during evaluation
            win_threshold: Win rate threshold for model promotion (default 55%)
        """
        self.num_evaluation_games = num_evaluation_games
        self.num_simulations = num_simulations
        self.win_threshold = win_threshold
    
    def evaluate_models(self, challenger_model, champion_model, 
                       challenger_name: str, champion_name: str) -> Tuple[Dict, bool]:
        """
        Evaluate challenger model against champion
        
        Args:
            challenger_model: New model to evaluate
            champion_model: Current best model
            challenger_name: Name of challenger model
            champion_name: Name of champion model
        
        Returns:
            Tuple of (results dict, should_promote boolean)
        """
        logger.info(f"Starting evaluation: {challenger_name} vs {champion_name}")
        logger.info(f"Playing {self.num_evaluation_games} games with {self.num_simulations} sims/move")
        
        # Create evaluation match
        match = EvaluationMatch(challenger_model, champion_model, self.num_simulations)
        
        # Run evaluation
        results = match.run_evaluation(self.num_evaluation_games)
        
        # Add model names to results
        results['challenger_name'] = challenger_name
        results['champion_name'] = champion_name
        results['challenger_win_rate'] = results['model1_win_rate']
        results['champion_win_rate'] = results['model2_win_rate']
        
        # Determine if challenger should be promoted
        should_promote = results['challenger_win_rate'] >= self.win_threshold
        results['promoted'] = should_promote
        
        logger.info(f"Evaluation complete: Challenger win rate = {results['challenger_win_rate']:.1%}")
        logger.info(f"Promotion threshold = {self.win_threshold:.1%}, Promoted = {should_promote}")
        
        return results, should_promote
