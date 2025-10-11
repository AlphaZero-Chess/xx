"""
PGN Parser for AlphaZero Historical Memory
Parses PGN files and extracts game data, moves, and positions for memory reconstruction
"""

import chess
import chess.pgn
import io
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import numpy as np

logger = logging.getLogger(__name__)


class PGNParser:
    """Parser for PGN chess game files"""
    
    def __init__(self):
        self.games_parsed = 0
        self.errors = []
    
    def parse_pgn_file(self, pgn_content: str) -> List[Dict[str, Any]]:
        """
        Parse PGN file content and extract all games with their moves and positions.
        
        Args:
            pgn_content: String content of PGN file
            
        Returns:
            List of parsed game dictionaries
        """
        games = []
        pgn_stream = io.StringIO(pgn_content)
        
        game_number = 0
        while True:
            try:
                game = chess.pgn.read_game(pgn_stream)
                if game is None:
                    break
                
                game_number += 1
                parsed_game = self._parse_single_game(game, game_number)
                
                if parsed_game:
                    games.append(parsed_game)
                    self.games_parsed += 1
                    
                    if game_number % 50 == 0:
                        logger.info(f"Parsed {game_number} games...")
                
            except Exception as e:
                error_msg = f"Error parsing game {game_number}: {str(e)}"
                logger.error(error_msg)
                self.errors.append(error_msg)
                continue
        
        logger.info(f"Successfully parsed {self.games_parsed} games from PGN")
        return games
    
    def _parse_single_game(self, game: chess.pgn.Game, game_number: int) -> Optional[Dict[str, Any]]:
        """
        Parse a single chess game and extract all relevant information.
        
        Args:
            game: chess.pgn.Game object
            game_number: Sequential game number
            
        Returns:
            Dictionary with game data or None if parsing failed
        """
        try:
            headers = game.headers
            
            # Extract metadata
            game_data = {
                "game_id": f"azs-{game_number:03d}",
                "game_number": game_number,
                "event": headers.get("Event", "AlphaZero vs Stockfish 2018"),
                "white": headers.get("White", "AlphaZero"),
                "black": headers.get("Black", "Stockfish"),
                "result": headers.get("Result", "*"),
                "date": headers.get("Date", "2018.??.??"),
                "round": headers.get("Round", str(game_number)),
                "opening": headers.get("Opening", "Unknown"),
                "eco": headers.get("ECO", ""),
                "timestamp_recalled": datetime.now(timezone.utc).isoformat(),
            }
            
            # Determine winner
            result = game_data["result"]
            if result == "1-0":
                game_data["winner"] = "white"
                game_data["outcome"] = "AlphaZero wins"
            elif result == "0-1":
                game_data["winner"] = "black"
                game_data["outcome"] = "Stockfish wins"
            elif result == "1/2-1/2":
                game_data["winner"] = "draw"
                game_data["outcome"] = "Draw"
            else:
                game_data["winner"] = "unknown"
                game_data["outcome"] = "Unknown"
            
            # Extract moves and positions
            moves = []
            move_sequence = []
            positions = []
            evaluation_curve = []
            
            board = game.board()
            move_number = 0
            
            for node in game.mainline():
                move = node.move
                move_uci = move.uci()
                move_san = board.san(move)
                
                # Store position before move
                fen_before = board.fen()
                
                # Make the move
                board.push(move)
                move_number += 1
                
                # Store position after move
                fen_after = board.fen()
                
                # Extract position encoding for training
                position_encoding = self._encode_board_position(board)
                
                move_data = {
                    "move_number": move_number,
                    "move_uci": move_uci,
                    "move_san": move_san,
                    "fen_before": fen_before,
                    "fen_after": fen_after,
                    "turn": "white" if move_number % 2 == 1 else "black",
                }
                
                moves.append(move_data)
                move_sequence.append(move_uci)
                positions.append(position_encoding.tolist())
                
                # Simplified evaluation (would need actual engine evaluation for real values)
                # For now, use a mock evaluation based on material and position
                evaluation = self._evaluate_position(board)
                evaluation_curve.append(evaluation)
            
            game_data["moves"] = moves
            game_data["move_sequence"] = move_sequence
            game_data["move_count"] = len(moves)
            game_data["positions"] = positions  # For training
            game_data["evaluation_curve"] = evaluation_curve
            game_data["final_fen"] = board.fen()
            
            # Store full PGN for replay
            exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
            pgn_string = game.accept(exporter)
            game_data["pgn"] = pgn_string
            
            return game_data
            
        except Exception as e:
            logger.error(f"Error parsing game {game_number}: {e}")
            return None
    
    def _encode_board_position(self, board: chess.Board) -> np.ndarray:
        """
        Encode chess board position as numpy array for neural network training.
        Uses 8x8x12 representation (6 piece types x 2 colors).
        
        Args:
            board: chess.Board object
            
        Returns:
            numpy array of shape (8, 8, 12)
        """
        # 12 planes: 6 piece types (P, N, B, R, Q, K) x 2 colors (white, black)
        position = np.zeros((8, 8, 12), dtype=np.float32)
        
        piece_to_plane = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                
                plane = piece_to_plane[piece.piece_type]
                if not piece.color:  # Black
                    plane += 6
                
                position[rank][file][plane] = 1.0
        
        return position
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """
        Simple position evaluation for visualization.
        Returns a score from -1.0 (black advantage) to +1.0 (white advantage).
        
        Args:
            board: chess.Board object
            
        Returns:
            Evaluation score
        """
        if board.is_checkmate():
            return 1.0 if board.turn == chess.BLACK else -1.0
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        
        # Material count
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        # Normalize to [-1, 1]
        total_material = white_material + black_material
        if total_material == 0:
            return 0.0
        
        material_advantage = (white_material - black_material) / (total_material + 1)
        
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, material_advantage))
    
    def extract_training_positions(self, games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract training positions from parsed games for retraining.
        
        Args:
            games: List of parsed game dictionaries
            
        Returns:
            List of training position dictionaries
        """
        training_data = []
        
        for game in games:
            game_result = game["result"]
            
            # Convert result to value target
            if game_result == "1-0":
                game_value = 1.0  # White (AlphaZero) win
            elif game_result == "0-1":
                game_value = -1.0  # Black (Stockfish) win
            else:
                game_value = 0.0  # Draw
            
            # Extract each position as a training sample
            for i, position in enumerate(game["positions"]):
                move_data = game["moves"][i]
                
                # Alternate value based on turn (flip for black's perspective)
                turn_value = game_value if move_data["turn"] == "white" else -game_value
                
                training_sample = {
                    "game_id": game["game_id"],
                    "move_number": move_data["move_number"],
                    "position": position,
                    "fen": move_data["fen_after"],
                    "value": turn_value,
                    "move_played": move_data["move_uci"],
                    "source": "alphazero_stockfish_2018",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                training_data.append(training_sample)
        
        logger.info(f"Extracted {len(training_data)} training positions from {len(games)} games")
        return training_data
    
    def get_parsing_stats(self) -> Dict[str, Any]:
        """Get parsing statistics"""
        return {
            "games_parsed": self.games_parsed,
            "errors": self.errors,
            "error_count": len(self.errors)
        }
