import chess
import numpy as np
from typing import List, Tuple

class ChessEngine:
    """Chess engine with full legal move validation and board encoding"""
    
    def __init__(self):
        self.board = chess.Board()
        
    def reset(self):
        """Reset board to starting position"""
        self.board.reset()
        
    def make_move(self, move_uci: str) -> bool:
        """Make a move in UCI format (e.g., 'e2e4')"""
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                return True
            return False
        except:
            return False
    
    def get_legal_moves(self) -> List[str]:
        """Get all legal moves in UCI format"""
        return [move.uci() for move in self.board.legal_moves]
    
    def is_promotion_move(self, from_square: str, to_square: str) -> bool:
        """Check if a move would be a pawn promotion"""
        try:
            from_sq = chess.parse_square(from_square)
            to_sq = chess.parse_square(to_square)
            piece = self.board.piece_at(from_sq)
            
            if not piece or piece.piece_type != chess.PAWN:
                return False
            
            # Check if pawn is moving to last rank
            to_rank = chess.square_rank(to_sq)
            if piece.color == chess.WHITE and to_rank == 7:
                return True
            if piece.color == chess.BLACK and to_rank == 0:
                return True
            
            return False
        except:
            return False
    
    def get_promotion_moves(self, from_square: str, to_square: str) -> List[str]:
        """Get all possible promotion moves for a given pawn move"""
        if not self.is_promotion_move(from_square, to_square):
            return []
        
        promotion_pieces = ['q', 'r', 'b', 'n']  # Queen, Rook, Bishop, Knight
        base_move = from_square + to_square
        return [base_move + piece for piece in promotion_pieces]
    
    def is_game_over(self) -> bool:
        """Check if game is over"""
        return self.board.is_game_over()
    
    def get_result(self) -> str:
        """Get game result"""
        if self.board.is_checkmate():
            return "1-0" if self.board.turn == chess.BLACK else "0-1"
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return "1/2-1/2"
        return "*"
    
    def get_fen(self) -> str:
        """Get FEN representation of current position"""
        return self.board.fen()
    
    def set_fen(self, fen: str):
        """Set board from FEN string"""
        self.board.set_fen(fen)
    
    def encode_board(self) -> np.ndarray:
        """
        Encode board for neural network input
        Returns: (8, 8, 14) array
        - 12 planes for piece positions (6 types × 2 colors)
        - 1 plane for turn (all 1s for white, all 0s for black)
        - 1 plane for castling rights
        """
        board_array = np.zeros((8, 8, 14), dtype=np.float32)
        
        # Piece planes
        piece_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                piece_type_idx = piece_idx[piece.piece_type]
                color_offset = 0 if piece.color == chess.WHITE else 6
                board_array[rank, file, piece_type_idx + color_offset] = 1
        
        # Turn plane (plane 12)
        if self.board.turn == chess.WHITE:
            board_array[:, :, 12] = 1
        
        # Castling rights plane (plane 13)
        castling_value = 0
        if self.board.has_kingside_castling_rights(chess.WHITE):
            castling_value += 0.25
        if self.board.has_queenside_castling_rights(chess.WHITE):
            castling_value += 0.25
        if self.board.has_kingside_castling_rights(chess.BLACK):
            castling_value += 0.25
        if self.board.has_queenside_castling_rights(chess.BLACK):
            castling_value += 0.25
        board_array[:, :, 13] = castling_value
        
        return board_array
    
    def move_to_index(self, move_uci: str) -> int:
        """Convert UCI move to index (simplified policy space)"""
        move = chess.Move.from_uci(move_uci)
        from_square = move.from_square
        to_square = move.to_square
        return from_square * 64 + to_square
    
    def index_to_move(self, index: int) -> str:
        """Convert index to UCI move (simplified)"""
        from_square = index // 64
        to_square = index % 64
        move = chess.Move(from_square, to_square)
        return move.uci()
    
    def get_move_probabilities(self, policy_output: np.ndarray) -> dict:
        """
        Convert neural network policy output to legal move probabilities
        policy_output: (4096,) array
        Returns: {move_uci: probability}
        """
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return {}
        
        # Get policy values for legal moves
        move_probs = {}
        for move in legal_moves:
            idx = self.move_to_index(move.uci())
            if idx < len(policy_output):
                move_probs[move.uci()] = policy_output[idx]
            else:
                move_probs[move.uci()] = 0.0
        
        # Normalize probabilities
        total = sum(move_probs.values())
        if total > 0:
            move_probs = {k: v / total for k, v in move_probs.items()}
        else:
            # Uniform distribution if all zeros
            uniform_prob = 1.0 / len(legal_moves)
            move_probs = {move.uci(): uniform_prob for move in legal_moves}
        
        return move_probs
    
    def copy(self):
        """Create a copy of the engine"""
        new_engine = ChessEngine()
        new_engine.board = self.board.copy()
        return new_engine