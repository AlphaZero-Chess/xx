import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Loader2, Play, RotateCcw, GraduationCap } from 'lucide-react';
import ChessBoard from './ChessBoard';
import CoachingPanel from './CoachingPanel';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const GameTab = () => {
  const [gameId, setGameId] = useState(null);
  const [gameState, setGameState] = useState(null);
  const [aiThinking, setAiThinking] = useState(false);
  const [numSimulations, setNumSimulations] = useState(200);
  const [aiColor, setAiColor] = useState('black');
  const [loading, setLoading] = useState(false);
  const [llmExplanation, setLlmExplanation] = useState(null);
  const [llmLoading, setLlmLoading] = useState(false);
  const [coachingMode, setCoachingMode] = useState(false);

  const startNewGame = async () => {
    try {
      setLoading(true);
      const response = await axios.post(`${API}/game/new`, { ai_color: aiColor });
      
      setGameId(response.data.game_id);
      setGameState({
        fen: response.data.fen,
        legal_moves: response.data.legal_moves,
        is_game_over: response.data.is_game_over,
        result: response.data.result,
        history: response.data.history
      });
      
      toast.success('New game started!');
    } catch (error) {
      console.error('Error starting game:', error);
      toast.error('Failed to start game');
    } finally {
      setLoading(false);
    }
  };

  const makeMove = async (move) => {
    if (!gameId || aiThinking) return;

    try {
      const response = await axios.post(`${API}/game/move`, {
        game_id: gameId,
        move: move
      });

      setGameState({
        fen: response.data.fen,
        legal_moves: response.data.legal_moves,
        is_game_over: response.data.is_game_over,
        result: response.data.result,
        history: response.data.history
      });

      // If game is not over and it's AI's turn, get AI move
      if (!response.data.is_game_over) {
        await getAIMove();
      } else {
        toast.info(`Game Over: ${response.data.result}`);
      }
    } catch (error) {
      console.error('Error making move:', error);
      toast.error(error.response?.data?.detail || 'Invalid move');
    }
  };

  const getLLMExplanation = async (fen, lastMove) => {
    try {
      setLlmLoading(true);
      const response = await axios.post(`${API}/llm/explain`, {
        fen: fen,
        last_move: lastMove,
        context: "Explain this chess position and the last move played."
      });
      
      setLlmExplanation({
        text: response.data.explanation,
        move: lastMove,
        offline: response.data.offline || false
      });
    } catch (error) {
      console.error('Error getting LLM explanation:', error);
      setLlmExplanation({
        text: "Position analysis unavailable. The AI is learning from your games.",
        move: lastMove,
        offline: true
      });
    } finally {
      setLlmLoading(false);
    }
  };

  const getAIMove = async () => {
    if (!gameId) return;

    try {
      setAiThinking(true);
      const response = await axios.post(`${API}/game/ai-move`, {
        game_id: gameId,
        num_simulations: numSimulations
      });

      setGameState({
        fen: response.data.fen,
        legal_moves: response.data.legal_moves,
        is_game_over: response.data.is_game_over,
        result: response.data.result,
        history: response.data.history
      });

      toast.success(`AI played: ${response.data.move} (${response.data.computation_time.toFixed(2)}s)`);

      // Get LLM explanation for the AI's move
      await getLLMExplanation(response.data.fen, response.data.move);

      if (response.data.is_game_over) {
        toast.info(`Game Over: ${response.data.result}`);
      }
    } catch (error) {
      console.error('Error getting AI move:', error);
      toast.error('AI failed to make a move');
    } finally {
      setAiThinking(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Chess Board - Takes 2 columns */}
      <div className="lg:col-span-2">
        <Card className="shadow-xl bg-white/95 dark:bg-slate-800/95" data-testid="chess-game-card">
          <CardHeader className="bg-gradient-to-r from-slate-100 to-slate-200 dark:from-slate-800 dark:to-slate-700">
            <CardTitle className="flex items-center justify-between">
              <span>AlphaZero Chess Game</span>
              {gameState?.is_game_over && (
                <Badge variant="destructive" data-testid="game-over-badge">
                  Game Over
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-6">
            {gameState ? (
              <>
                <ChessBoard
                  fen={gameState.fen}
                  legalMoves={gameState.legal_moves || []}
                  onMove={makeMove}
                  gameOver={gameState.is_game_over || aiThinking}
                />
                {aiThinking && (
                  <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg flex items-center justify-center">
                    <Loader2 className="mr-2 h-5 w-5 animate-spin text-blue-600" />
                    <span className="text-blue-900 dark:text-blue-100 font-medium">
                      AI is thinking... ({numSimulations} simulations)
                    </span>
                  </div>
                )}
              </>
            ) : (
              <div className="flex flex-col items-center justify-center py-16 space-y-4">
                <div className="text-6xl">♟️</div>
                <p className="text-xl text-slate-600 dark:text-slate-400">
                  Start a new game to play against AlphaZero
                </p>
                <Button
                  onClick={startNewGame}
                  disabled={loading}
                  className="bg-emerald-600 hover:bg-emerald-700"
                  data-testid="start-game-btn"
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Starting...
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Start New Game
                    </>
                  )}
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Game Controls - Takes 1 column */}
      <div className="space-y-6">
        {/* Control Panel */}
        <Card className="shadow-xl" data-testid="game-controls">
          <CardHeader className="bg-gradient-to-r from-slate-100 to-slate-200 dark:from-slate-800 dark:to-slate-700">
            <CardTitle>Game Controls</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 pt-6">
            {/* Coaching Mode Toggle */}
            <div className="space-y-2 pb-4 border-b">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium flex items-center gap-2">
                  <GraduationCap className="h-4 w-4" />
                  Coaching Mode
                </label>
                <Button
                  variant={coachingMode ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setCoachingMode(!coachingMode)}
                  className={coachingMode ? 'bg-purple-600 hover:bg-purple-700' : ''}
                  data-testid="coaching-mode-toggle"
                >
                  {coachingMode ? 'ON' : 'OFF'}
                </Button>
              </div>
              {coachingMode && (
                <p className="text-xs text-slate-600 dark:text-slate-400">
                  Get real-time advice from your AI coach powered by AlphaZero
                </p>
              )}
            </div>

            {/* AI Color Selection */}
            <div className="space-y-2">
              <label className="text-sm font-medium">AI Plays As</label>
              <div className="flex gap-2">
                <Button
                  variant={aiColor === 'white' ? 'default' : 'outline'}
                  onClick={() => setAiColor('white')}
                  disabled={gameId !== null}
                  className="flex-1"
                  data-testid="ai-white-btn"
                >
                  ♔ White
                </Button>
                <Button
                  variant={aiColor === 'black' ? 'default' : 'outline'}
                  onClick={() => setAiColor('black')}
                  disabled={gameId !== null}
                  className="flex-1"
                  data-testid="ai-black-btn"
                >
                  ♚ Black
                </Button>
              </div>
            </div>

            {/* MCTS Simulations */}
            <div className="space-y-2">
              <label className="text-sm font-medium">
                AI Strength: {numSimulations} simulations
              </label>
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
              <div className="text-xs text-slate-500 flex justify-between">
                <span>Fast (100)</span>
                <span>Strong (1600)</span>
              </div>
            </div>

            {/* New Game Button */}
            <Button
              onClick={startNewGame}
              disabled={loading || aiThinking}
              className="w-full bg-emerald-600 hover:bg-emerald-700"
              data-testid="new-game-btn"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Starting...
                </>
              ) : (
                <>
                  <RotateCcw className="mr-2 h-4 w-4" />
                  New Game
                </>
              )}
            </Button>

            {/* Manual AI Move (for debugging) */}
            {gameState && !gameState.is_game_over && (
              <Button
                onClick={getAIMove}
                disabled={aiThinking}
                variant="outline"
                className="w-full"
                data-testid="manual-ai-move-btn"
              >
                {aiThinking ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    AI Thinking...
                  </>
                ) : (
                  'Request AI Move'
                )}
              </Button>
            )}
          </CardContent>
        </Card>

        {/* Game Info */}
        {gameState && (
          <Card className="shadow-xl" data-testid="game-info">
            <CardHeader className="bg-gradient-to-r from-slate-100 to-slate-200 dark:from-slate-800 dark:to-slate-700">
              <CardTitle>Game Information</CardTitle>
            </CardHeader>
            <CardContent className="pt-6 space-y-3">
              <div className="text-sm space-y-2">
                <div className="flex justify-between">
                  <span className="text-slate-600 dark:text-slate-400">Legal Moves:</span>
                  <span className="font-medium">{gameState.legal_moves?.length || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-600 dark:text-slate-400">Moves Played:</span>
                  <span className="font-medium">{gameState.history?.length || 0}</span>
                </div>
                {gameState.is_game_over && (
                  <div className="flex justify-between">
                    <span className="text-slate-600 dark:text-slate-400">Result:</span>
                    <span className="font-bold text-lg">{gameState.result}</span>
                  </div>
                )}
              </div>

              {/* Move History */}
              {gameState.history && gameState.history.length > 0 && (
                <div className="mt-4">
                  <div className="text-sm font-medium mb-2">Move History</div>
                  <div className="max-h-48 overflow-y-auto bg-slate-50 dark:bg-slate-900 rounded p-3 space-y-1">
                    {gameState.history.map((move, idx) => (
                      <div key={idx} className="text-xs font-mono flex">
                        <span className="text-slate-500 w-8">{idx + 1}.</span>
                        <span>{move}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* FEN Display */}
              <div className="mt-4">
                <div className="text-sm font-medium mb-2">Position (FEN)</div>
                <div className="text-xs font-mono bg-slate-50 dark:bg-slate-900 rounded p-2 break-all">
                  {gameState.fen}
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Coaching Panel - New Interactive Coaching Mode */}
        <CoachingPanel 
          gameId={gameId}
          gameState={gameState}
          isEnabled={coachingMode}
        />

        {/* LLM Explanation Panel - Legacy (shown when coaching mode is off) */}
        {!coachingMode && llmExplanation && (
          <Card className="shadow-xl" data-testid="llm-explanation">
            <CardHeader className="bg-gradient-to-r from-purple-100 to-purple-200 dark:from-purple-900 dark:to-purple-800">
              <CardTitle className="flex items-center gap-2">
                <span>🤖 AI Analysis</span>
                {llmExplanation.offline && (
                  <Badge variant="outline" className="text-xs">Offline</Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-6">
              {llmLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin text-purple-600" />
                  <span className="ml-2 text-slate-600 dark:text-slate-400">
                    Analyzing position...
                  </span>
                </div>
              ) : (
                <div className="space-y-3">
                  {llmExplanation.move && (
                    <div className="text-sm">
                      <span className="font-medium text-slate-600 dark:text-slate-400">
                        Move: 
                      </span>
                      <span className="ml-2 font-mono font-bold">
                        {llmExplanation.move}
                      </span>
                    </div>
                  )}
                  <div className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed">
                    {llmExplanation.text}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default GameTab;
