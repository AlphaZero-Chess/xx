import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Loader2 } from 'lucide-react';

const GamePanel = ({ gameState, onNewGame, onAIMove, aiThinking, gameResult }) => {
  const [numSimulations, setNumSimulations] = useState(800);

  return (
    <Card className="shadow-xl" data-testid="game-panel">
      <CardHeader className="bg-gradient-to-r from-slate-100 to-slate-200 dark:from-slate-800 dark:to-slate-700">
        <CardTitle className="flex items-center justify-between">
          <span>Game Controls</span>
          {gameState?.is_game_over && (
            <Badge variant="destructive" data-testid="game-over-badge">Game Over</Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 pt-6">
        <Button 
          onClick={onNewGame} 
          className="w-full bg-emerald-600 hover:bg-emerald-700"
          data-testid="new-game-btn"
        >
          New Game
        </Button>

        <div className="space-y-2">
          <label className="text-sm font-medium">MCTS Simulations: {numSimulations}</label>
          <input
            type="range"
            min="100"
            max="1600"
            step="100"
            value={numSimulations}
            onChange={(e) => setNumSimulations(parseInt(e.target.value))}
            className="w-full"
            data-testid="simulation-slider"
          />
        </div>

        <Button
          onClick={() => onAIMove(numSimulations)}
          disabled={aiThinking || gameState?.is_game_over}
          className="w-full bg-blue-600 hover:bg-blue-700"
          data-testid="ai-move-btn"
        >
          {aiThinking ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              AI Thinking...
            </>
          ) : (
            'Get AI Move'
          )}
        </Button>

        {gameResult && (
          <div className="mt-4 p-4 bg-slate-100 dark:bg-slate-800 rounded-lg" data-testid="game-result">
            <div className="text-sm font-medium mb-2">Game Result</div>
            <div className="text-lg font-bold">
              {gameResult === '1-0' ? 'White Wins!' : 
               gameResult === '0-1' ? 'Black Wins!' : 
               'Draw'}
            </div>
          </div>
        )}

        <div className="mt-4 p-4 bg-slate-50 dark:bg-slate-900 rounded-lg space-y-2">
          <div className="text-sm font-medium">Position Info</div>
          <div className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
            <div>Legal Moves: {gameState?.legal_moves?.length || 0}</div>
            <div className="font-mono text-xs break-all">
              FEN: {gameState?.fen?.substring(0, 30)}...
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default GamePanel;