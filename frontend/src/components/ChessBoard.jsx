import React, { useState, useEffect } from 'react';
import { Card } from './ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from './ui/dialog';
import { Button } from './ui/button';

const ChessBoard = ({ fen, legalMoves, onMove, gameOver }) => {
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [highlightedMoves, setHighlightedMoves] = useState([]);
  const [showPromotionDialog, setShowPromotionDialog] = useState(false);
  const [promotionMove, setPromotionMove] = useState(null);

  // Parse FEN to get piece positions
  const parseFEN = (fen) => {
    // Return empty board if FEN is not provided
    if (!fen) {
      console.error('No FEN provided to ChessBoard');
      return Array(8).fill().map(() => Array(8).fill(null));
    }
    
    try {
      const fenParts = fen.split(' ');
      const rows = fenParts[0].split('/');
      const board = [];

      for (let i = 0; i < 8; i++) {
        const row = [];
        let col = 0;
        
        for (const char of rows[i]) {
          if (isNaN(char)) {
            row.push(char);
            col++;
          } else {
            const emptySquares = parseInt(char);
            for (let j = 0; j < emptySquares; j++) {
              row.push(null);
              col++;
            }
          }
        }
        board.push(row);
      }

      return board;
    } catch (error) {
      console.error('Error parsing FEN:', error, 'FEN:', fen);
      // Return empty board on error
      return Array(8).fill().map(() => Array(8).fill(null));
    }
  };

  const board = parseFEN(fen);

  const getPieceSymbol = (piece) => {
    const symbols = {
      'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
      'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
    };
    return symbols[piece] || '';
  };

  const squareToIndex = (square) => {
    const file = square.charCodeAt(0) - 97; // a=0, b=1, ...
    const rank = 8 - parseInt(square[1]); // 8=0, 7=1, ...
    return { rank, file };
  };

  const indexToSquare = (rank, file) => {
    const files = 'abcdefgh';
    const ranks = '87654321';
    return files[file] + ranks[rank];
  };

  const handleSquareClick = (rank, file) => {
    if (gameOver) return;

    const square = indexToSquare(rank, file);
    const piece = board[rank][file];

    if (selectedSquare) {
      // Try to make a move
      const baseMove = selectedSquare + square;
      
      // Check if this is a pawn promotion
      const selectedPiece = board[squareToIndex(selectedSquare).rank][squareToIndex(selectedSquare).file];
      const isPromotion = checkIfPromotion(selectedPiece, selectedSquare, square);
      
      if (isPromotion) {
        // Check if any promotion move is legal
        const promotionMoves = ['q', 'r', 'b', 'n'].map(p => baseMove + p);
        const hasLegalPromotion = promotionMoves.some(m => legalMoves && legalMoves.includes(m));
        
        if (hasLegalPromotion) {
          // Show promotion dialog
          setPromotionMove(baseMove);
          setShowPromotionDialog(true);
          return;
        }
      }
      
      // Check if it's a legal move (non-promotion)
      if (legalMoves && legalMoves.includes(baseMove)) {
        onMove(baseMove);
        setSelectedSquare(null);
        setHighlightedMoves([]);
      } else {
        // Select new piece or deselect
        if (piece) {
          setSelectedSquare(square);
          const possibleMoves = legalMoves ? legalMoves.filter(m => m.startsWith(square)).map(m => m.substring(2, 4)) : [];
          setHighlightedMoves(possibleMoves);
        } else {
          setSelectedSquare(null);
          setHighlightedMoves([]);
        }
      }
    } else {
      // Select a piece
      if (piece) {
        setSelectedSquare(square);
        const possibleMoves = legalMoves ? legalMoves.filter(m => m.startsWith(square)).map(m => m.substring(2, 4)) : [];
        setHighlightedMoves(possibleMoves);
      }
    }
  };

  const checkIfPromotion = (piece, fromSquare, toSquare) => {
    if (!piece || (piece.toLowerCase() !== 'p')) return false;
    
    const toRank = toSquare[1];
    // White pawn to rank 8 or black pawn to rank 1
    if ((piece === 'P' && toRank === '8') || (piece === 'p' && toRank === '1')) {
      return true;
    }
    return false;
  };

  const handlePromotion = (promotionPiece) => {
    if (promotionMove) {
      const fullMove = promotionMove + promotionPiece;
      onMove(fullMove);
    }
    setShowPromotionDialog(false);
    setPromotionMove(null);
    setSelectedSquare(null);
    setHighlightedMoves([]);
  };

  const cancelPromotion = () => {
    setShowPromotionDialog(false);
    setPromotionMove(null);
    setSelectedSquare(null);
    setHighlightedMoves([]);
  };

  const isSquareHighlighted = (rank, file) => {
    const square = indexToSquare(rank, file);
    return highlightedMoves.includes(square);
  };

  const isSquareSelected = (rank, file) => {
    const square = indexToSquare(rank, file);
    return selectedSquare === square;
  };

  return (
    <Card className="p-6 bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 shadow-2xl" data-testid="chess-board-container">
      <div className="inline-block border-4 border-slate-700 rounded-lg overflow-hidden shadow-xl">
        {board.map((row, rankIdx) => (
          <div key={rankIdx} className="flex">
            {row.map((piece, fileIdx) => {
              const isLight = (rankIdx + fileIdx) % 2 === 0;
              const isSelected = isSquareSelected(rankIdx, fileIdx);
              const isHighlighted = isSquareHighlighted(rankIdx, fileIdx);
              
              return (
                <div
                  key={`${rankIdx}-${fileIdx}`}
                  data-testid={`square-${indexToSquare(rankIdx, fileIdx)}`}
                  onClick={() => handleSquareClick(rankIdx, fileIdx)}
                  className={`
                    w-16 h-16 sm:w-20 sm:h-20 flex items-center justify-center text-4xl sm:text-5xl cursor-pointer
                    transition-all duration-200 hover:brightness-95 relative
                    ${isLight ? 'bg-amber-100' : 'bg-amber-700'}
                    ${isSelected ? 'ring-4 ring-blue-500 ring-inset' : ''}
                    ${isHighlighted ? 'ring-4 ring-green-400 ring-inset' : ''}
                  `}
                >
                  {piece && (
                    <span 
                      className={`drop-shadow-lg select-none font-bold`}
                      style={{
                        color: piece === piece.toUpperCase() ? '#f8f8f8' : '#1a1a1a',
                        textShadow: piece === piece.toUpperCase() 
                          ? '0 0 3px rgba(0,0,0,0.8), 1px 1px 0 rgba(0,0,0,0.8)' 
                          : '0 0 3px rgba(255,255,255,0.8), 1px 1px 0 rgba(255,255,255,0.8)'
                      }}
                    >
                      {getPieceSymbol(piece)}
                    </span>
                  )}
                  {isHighlighted && !piece && (
                    <div className="w-4 h-4 rounded-full bg-green-400 opacity-60"></div>
                  )}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div className="mt-4 text-center text-sm text-slate-600 dark:text-slate-400">
        {!fen ? (
          <div>
            <span className="text-red-500 font-medium">⚠️ No board data available</span>
            <div className="text-xs mt-1">Expected FEN but received: {typeof fen}</div>
          </div>
        ) : gameOver ? (
          "Game Over"
        ) : (
          "Click a piece to move"
        )}
      </div>

      {/* Promotion Dialog */}
      <Dialog open={showPromotionDialog} onOpenChange={setShowPromotionDialog}>
        <DialogContent className="sm:max-w-md" data-testid="promotion-dialog">
          <DialogHeader>
            <DialogTitle>Choose Promotion Piece</DialogTitle>
            <DialogDescription>
              Select which piece you want to promote your pawn to.
            </DialogDescription>
          </DialogHeader>
          <div className="grid grid-cols-2 gap-4 py-4">
            <Button
              onClick={() => handlePromotion('q')}
              className="h-24 text-4xl bg-blue-600 hover:bg-blue-700"
              data-testid="promote-queen"
            >
              ♕
              <span className="text-sm ml-2">Queen</span>
            </Button>
            <Button
              onClick={() => handlePromotion('r')}
              className="h-24 text-4xl bg-green-600 hover:bg-green-700"
              data-testid="promote-rook"
            >
              ♖
              <span className="text-sm ml-2">Rook</span>
            </Button>
            <Button
              onClick={() => handlePromotion('b')}
              className="h-24 text-4xl bg-purple-600 hover:bg-purple-700"
              data-testid="promote-bishop"
            >
              ♗
              <span className="text-sm ml-2">Bishop</span>
            </Button>
            <Button
              onClick={() => handlePromotion('n')}
              className="h-24 text-4xl bg-orange-600 hover:bg-orange-700"
              data-testid="promote-knight"
            >
              ♘
              <span className="text-sm ml-2">Knight</span>
            </Button>
          </div>
          <Button
            onClick={cancelPromotion}
            variant="outline"
            className="w-full"
            data-testid="cancel-promotion"
          >
            Cancel
          </Button>
        </DialogContent>
      </Dialog>
    </Card>
  );
};

export default ChessBoard;