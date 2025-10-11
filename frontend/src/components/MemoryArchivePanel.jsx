/**
 * AlphaZero Memory Archive Panel
 * 
 * Displays and replays historical AlphaZero vs Stockfish matches from 2018.
 * Supports PGN upload, match browsing, move-by-move replay, and training from memory.
 */

import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, 
  Play, 
  Pause, 
  SkipBack, 
  SkipForward, 
  Download,
  Trophy,
  Clock,
  Hash,
  TrendingUp,
  Database,
  Brain,
  AlertCircle,
  CheckCircle,
  Search,
  Filter
} from 'lucide-react';
import axios from 'axios';
import { toast } from 'sonner';
import ChessBoard from './ChessBoard';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || import.meta.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const MemoryArchivePanel = () => {
  const [activeView, setActiveView] = useState('overview'); // overview, browse, replay
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);
  
  // Archive data
  const [stats, setStats] = useState(null);
  const [games, setGames] = useState([]);
  const [selectedGame, setSelectedGame] = useState(null);
  
  // Replay state
  const [currentMoveIndex, setCurrentMoveIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1000); // milliseconds per move
  const playbackInterval = useRef(null);
  
  // Filters
  const [filterWinner, setFilterWinner] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  
  // Load stats on mount
  useEffect(() => {
    loadStats();
    loadGames();
  }, []);
  
  // Auto-play functionality
  useEffect(() => {
    if (isPlaying && selectedGame) {
      playbackInterval.current = setInterval(() => {
        setCurrentMoveIndex(prev => {
          if (prev >= selectedGame.moves.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, playbackSpeed);
    } else {
      if (playbackInterval.current) {
        clearInterval(playbackInterval.current);
      }
    }
    
    return () => {
      if (playbackInterval.current) {
        clearInterval(playbackInterval.current);
      }
    };
  }, [isPlaying, selectedGame, playbackSpeed]);
  
  const loadStats = async () => {
    try {
      const response = await axios.get(`${API}/memory/stats`);
      if (response.data.success) {
        setStats(response.data);
      }
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  };
  
  const loadGames = async () => {
    try {
      setLoading(true);
      const winner = filterWinner === 'all' ? '' : filterWinner;
      const response = await axios.get(`${API}/memory/list`, {
        params: { limit: 220, winner }
      });
      
      if (response.data.success) {
        setGames(response.data.games);
      }
    } catch (error) {
      console.error('Error loading games:', error);
      toast.error('Failed to load games');
    } finally {
      setLoading(false);
    }
  };
  
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    try {
      setLoading(true);
      setUploadProgress({ stage: 'reading', percent: 30 });
      
      const content = await file.text();
      
      setUploadProgress({ stage: 'uploading', percent: 50 });
      
      const response = await axios.post(`${API}/memory/upload`, {
        pgn_content: content
      });
      
      if (response.data.success) {
        setUploadProgress({ stage: 'complete', percent: 100 });
        toast.success(`Successfully uploaded ${response.data.games_stored} games!`);
        
        // Reload stats and games
        await loadStats();
        await loadGames();
        
        setTimeout(() => setUploadProgress(null), 2000);
      }
    } catch (error) {
      console.error('Error uploading PGN:', error);
      toast.error('Failed to upload PGN file');
      setUploadProgress(null);
    } finally {
      setLoading(false);
    }
  };
  
  const handleGameSelect = async (gameId) => {
    try {
      setLoading(true);
      const response = await axios.get(`${API}/memory/replay/${gameId}`);
      
      if (response.data.success) {
        setSelectedGame(response.data.game);
        setCurrentMoveIndex(0);
        setActiveView('replay');
      }
    } catch (error) {
      console.error('Error loading game:', error);
      toast.error('Failed to load game');
    } finally {
      setLoading(false);
    }
  };
  
  const handleTrainFromMemory = async () => {
    if (!window.confirm(
      'This will train AlphaZero using all historical matches.\n' +
      'Training may take several minutes.\n\nContinue?'
    )) {
      return;
    }
    
    try {
      setLoading(true);
      const response = await axios.post(`${API}/train/from_memory`, {
        num_epochs: 5,
        batch_size: 64,
        learning_rate: 0.001
      });
      
      if (response.data.success) {
        toast.success(
          `Training started! Processing ${response.data.training_positions} positions from ${response.data.games_count} games.`
        );
      }
    } catch (error) {
      console.error('Error starting training:', error);
      toast.error('Failed to start training');
    } finally {
      setLoading(false);
    }
  };
  
  const getCurrentFEN = () => {
    if (!selectedGame || !selectedGame.moves[currentMoveIndex]) {
      return 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'; // Starting position
    }
    return selectedGame.moves[currentMoveIndex].fen_after;
  };
  
  const filteredGames = games.filter(game => {
    if (!searchQuery) return true;
    return (
      game.game_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      game.opening.toLowerCase().includes(searchQuery.toLowerCase()) ||
      game.outcome.toLowerCase().includes(searchQuery.toLowerCase())
    );
  });
  
  // Render Overview
  const renderOverview = () => (
    <div className="space-y-6" data-testid="memory-archive-overview">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-6 rounded-lg">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold flex items-center gap-3">
              <Brain className="w-10 h-10" />
              AlphaZero Memory Archive
            </h2>
            <p className="text-purple-100 mt-2">
              Historical matches: AlphaZero vs Stockfish 2018
            </p>
          </div>
          
          {stats && stats.total_games > 0 && (
            <div className="text-right">
              <div className="text-5xl font-bold">{stats.total_games}</div>
              <div className="text-sm text-purple-100">Stored Matches</div>
            </div>
          )}
        </div>
      </div>
      
      {/* Upload Section */}
      {(!stats || stats.total_games === 0) && (
        <div className="bg-white border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
          <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-900 mb-2">
            Upload Historical Memory
          </h3>
          <p className="text-gray-600 mb-4">
            Upload the alphazero_stockfish_2018.pgn file to begin memory reconstruction
          </p>
          
          <label className="inline-flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 cursor-pointer transition-colors">
            <Upload className="w-5 h-5" />
            Select PGN File
            <input
              type="file"
              accept=".pgn"
              onChange={handleFileUpload}
              className="hidden"
              disabled={loading}
            />
          </label>
          
          {uploadProgress && (
            <div className="mt-4">
              <div className="flex items-center justify-center gap-2 mb-2">
                <span className="text-sm text-gray-600">
                  {uploadProgress.stage === 'reading' && 'Reading file...'}
                  {uploadProgress.stage === 'uploading' && 'Parsing games...'}
                  {uploadProgress.stage === 'complete' && 'Complete!'}
                </span>
              </div>
              <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-purple-600 transition-all duration-300"
                  style={{ width: `${uploadProgress.percent}%` }}
                />
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Statistics */}
      {stats && stats.total_games > 0 && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white border rounded-lg p-6">
              <div className="flex items-center justify-between mb-2">
                <Trophy className="w-8 h-8 text-yellow-500" />
                <span className="text-3xl font-bold text-gray-900">
                  {stats.alphazero_wins}
                </span>
              </div>
              <div className="text-sm text-gray-600">AlphaZero Wins</div>
              <div className="text-xs text-gray-500 mt-1">
                {stats.win_rate.alphazero}% win rate
              </div>
            </div>
            
            <div className="bg-white border rounded-lg p-6">
              <div className="flex items-center justify-between mb-2">
                <Trophy className="w-8 h-8 text-gray-500" />
                <span className="text-3xl font-bold text-gray-900">
                  {stats.draws}
                </span>
              </div>
              <div className="text-sm text-gray-600">Draws</div>
              <div className="text-xs text-gray-500 mt-1">
                {stats.win_rate.draw}% of games
              </div>
            </div>
            
            <div className="bg-white border rounded-lg p-6">
              <div className="flex items-center justify-between mb-2">
                <Trophy className="w-8 h-8 text-red-500" />
                <span className="text-3xl font-bold text-gray-900">
                  {stats.stockfish_wins}
                </span>
              </div>
              <div className="text-sm text-gray-600">Stockfish Wins</div>
              <div className="text-xs text-gray-500 mt-1">
                {stats.win_rate.stockfish}% win rate
              </div>
            </div>
          </div>
          
          <div className="bg-white border rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-blue-600" />
              Move Statistics
            </h3>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <div className="text-2xl font-bold text-gray-900">
                  {stats.move_statistics.average}
                </div>
                <div className="text-sm text-gray-600">Average Moves</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-900">
                  {stats.move_statistics.minimum}
                </div>
                <div className="text-sm text-gray-600">Shortest Game</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-900">
                  {stats.move_statistics.maximum}
                </div>
                <div className="text-sm text-gray-600">Longest Game</div>
              </div>
            </div>
          </div>
          
          {/* Actions */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <button
              onClick={() => setActiveView('browse')}
              className="flex items-center justify-center gap-2 px-6 py-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              data-testid="browse-matches-button"
            >
              <Database className="w-5 h-5" />
              Browse All Matches
            </button>
            
            <button
              onClick={handleTrainFromMemory}
              disabled={loading}
              className="flex items-center justify-center gap-2 px-6 py-4 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              data-testid="train-from-memory-button"
            >
              <Brain className="w-5 h-5" />
              Train from Historical Memory
            </button>
          </div>
        </>
      )}
    </div>
  );
  
  // Render Browse View
  const renderBrowse = () => (
    <div className="space-y-6" data-testid="browse-matches-view">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">All Matches</h2>
        <button
          onClick={() => setActiveView('overview')}
          className="px-4 py-2 text-gray-600 hover:text-gray-900"
        >
          ← Back to Overview
        </button>
      </div>
      
      {/* Filters */}
      <div className="flex flex-wrap gap-4 items-center bg-gray-50 p-4 rounded-lg">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-gray-600" />
          <select
            value={filterWinner}
            onChange={(e) => {
              setFilterWinner(e.target.value);
              loadGames();
            }}
            className="px-3 py-1 border rounded text-sm"
          >
            <option value="all">All Results</option>
            <option value="white">AlphaZero Wins</option>
            <option value="black">Stockfish Wins</option>
            <option value="draw">Draws</option>
          </select>
        </div>
        
        <div className="flex-1">
          <div className="relative">
            <Search className="absolute left-3 top-2.5 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search by ID, opening, or outcome..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border rounded text-sm"
            />
          </div>
        </div>
        
        <div className="text-sm text-gray-600">
          {filteredGames.length} matches
        </div>
      </div>
      
      {/* Games Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {loading ? (
          <div className="col-span-3 text-center py-12">
            <p className="text-gray-600">Loading matches...</p>
          </div>
        ) : filteredGames.length === 0 ? (
          <div className="col-span-3 text-center py-12 bg-gray-50 rounded-lg">
            <Database className="w-12 h-12 text-gray-400 mx-auto mb-3" />
            <p className="text-gray-600">No matches found</p>
          </div>
        ) : (
          filteredGames.map((game) => (
            <div
              key={game.game_id}
              className="bg-white border rounded-lg p-4 hover:shadow-lg transition-shadow cursor-pointer"
              onClick={() => handleGameSelect(game.game_id)}
              data-testid={`game-card-${game.game_id}`}
            >
              <div className="flex items-center justify-between mb-3">
                <span className="font-mono text-sm text-gray-600">
                  {game.game_id}
                </span>
                {game.winner === 'white' && (
                  <Trophy className="w-5 h-5 text-yellow-500" />
                )}
                {game.winner === 'black' && (
                  <Trophy className="w-5 h-5 text-red-500" />
                )}
                {game.winner === 'draw' && (
                  <Trophy className="w-5 h-5 text-gray-400" />
                )}
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600">Result:</span>
                  <span className="font-medium text-gray-900">{game.result}</span>
                </div>
                
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600">Moves:</span>
                  <span className="font-medium text-gray-900">{game.move_count}</span>
                </div>
                
                {game.opening && (
                  <div className="text-xs text-gray-500 mt-2 truncate">
                    {game.opening}
                  </div>
                )}
                
                <div className="text-xs text-gray-400 mt-2">
                  {game.outcome}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
  
  // Render Replay View
  const renderReplay = () => {
    if (!selectedGame) return null;
    
    const currentMove = selectedGame.moves[currentMoveIndex];
    const progress = (currentMoveIndex / (selectedGame.moves.length - 1)) * 100;
    
    return (
      <div className="space-y-6" data-testid="replay-view">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">
              {selectedGame.game_id}
            </h2>
            <p className="text-sm text-gray-600">
              {selectedGame.white} vs {selectedGame.black}
            </p>
          </div>
          <button
            onClick={() => {
              setSelectedGame(null);
              setActiveView('browse');
              setIsPlaying(false);
            }}
            className="px-4 py-2 text-gray-600 hover:text-gray-900"
          >
            ← Back to Browse
          </button>
        </div>
        
        {/* Game Info */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white border rounded-lg p-3">
            <div className="text-sm text-gray-600">Result</div>
            <div className="text-lg font-bold text-gray-900">{selectedGame.result}</div>
          </div>
          <div className="bg-white border rounded-lg p-3">
            <div className="text-sm text-gray-600">Moves</div>
            <div className="text-lg font-bold text-gray-900">{selectedGame.move_count}</div>
          </div>
          <div className="bg-white border rounded-lg p-3">
            <div className="text-sm text-gray-600">Opening</div>
            <div className="text-sm font-medium text-gray-900 truncate">{selectedGame.opening}</div>
          </div>
          <div className="bg-white border rounded-lg p-3">
            <div className="text-sm text-gray-600">Outcome</div>
            <div className="text-sm font-medium text-gray-900">{selectedGame.outcome}</div>
          </div>
        </div>
        
        {/* Chess Board & Controls */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Chess Board */}
          <div className="lg:col-span-2 bg-white border rounded-lg p-6">
            <ChessBoard 
              fen={getCurrentFEN()}
              orientation="white"
              interactive={false}
            />
            
            {/* Progress Bar */}
            <div className="mt-6">
              <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
                <span>Move {currentMoveIndex + 1} of {selectedGame.moves.length}</span>
                <span>{currentMove ? currentMove.move_san : '-'}</span>
              </div>
              <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-purple-600 transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
            
            {/* Playback Controls */}
            <div className="mt-6 flex items-center justify-center gap-4">
              <button
                onClick={() => setCurrentMoveIndex(0)}
                className="p-2 rounded hover:bg-gray-100 disabled:opacity-50"
                disabled={currentMoveIndex === 0}
              >
                <SkipBack className="w-5 h-5" />
              </button>
              
              <button
                onClick={() => setCurrentMoveIndex(Math.max(0, currentMoveIndex - 1))}
                className="p-2 rounded hover:bg-gray-100 disabled:opacity-50"
                disabled={currentMoveIndex === 0}
              >
                ← Prev
              </button>
              
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="p-3 bg-purple-600 text-white rounded-full hover:bg-purple-700"
                data-testid="play-pause-button"
              >
                {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6" />}
              </button>
              
              <button
                onClick={() => setCurrentMoveIndex(Math.min(selectedGame.moves.length - 1, currentMoveIndex + 1))}
                className="p-2 rounded hover:bg-gray-100 disabled:opacity-50"
                disabled={currentMoveIndex >= selectedGame.moves.length - 1}
              >
                Next →
              </button>
              
              <button
                onClick={() => setCurrentMoveIndex(selectedGame.moves.length - 1)}
                className="p-2 rounded hover:bg-gray-100 disabled:opacity-50"
                disabled={currentMoveIndex >= selectedGame.moves.length - 1}
              >
                <SkipForward className="w-5 h-5" />
              </button>
            </div>
            
            {/* Speed Control */}
            <div className="mt-4 flex items-center justify-center gap-4">
              <span className="text-sm text-gray-600">Speed:</span>
              <select
                value={playbackSpeed}
                onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
                className="px-3 py-1 border rounded text-sm"
              >
                <option value={2000}>0.5x</option>
                <option value={1000}>1x</option>
                <option value={500}>2x</option>
                <option value={250}>4x</option>
              </select>
            </div>
          </div>
          
          {/* Move List */}
          <div className="bg-white border rounded-lg p-4">
            <h3 className="font-semibold mb-3">Move History</h3>
            <div className="space-y-1 max-h-[500px] overflow-y-auto">
              {selectedGame.moves.map((move, index) => (
                <div
                  key={index}
                  onClick={() => setCurrentMoveIndex(index)}
                  className={`flex items-center justify-between p-2 rounded cursor-pointer transition-colors ${
                    index === currentMoveIndex
                      ? 'bg-purple-100 text-purple-900'
                      : 'hover:bg-gray-100'
                  }`}
                >
                  <span className="text-sm font-mono">
                    {Math.floor(index / 2) + 1}.{index % 2 === 0 ? '' : '..'} {move.move_san}
                  </span>
                  <span className="text-xs text-gray-500">
                    {move.turn === 'white' ? '○' : '●'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };
  
  return (
    <div className="w-full bg-white rounded-lg shadow-lg p-6" data-testid="memory-archive-panel">
      {activeView === 'overview' && renderOverview()}
      {activeView === 'browse' && renderBrowse()}
      {activeView === 'replay' && renderReplay()}
    </div>
  );
};

export default MemoryArchivePanel;
