import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Loader2, ChevronDown, ChevronUp, MessageSquare, Lightbulb, TrendingUp, AlertCircle } from 'lucide-react';
import { toast } from 'sonner';
import FeedbackModule from './FeedbackModule';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const CoachingPanel = ({ gameId, gameState, isEnabled }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [coachingData, setCoachingData] = useState(null);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [userQuestion, setUserQuestion] = useState('');
  const [showFeedback, setShowFeedback] = useState({});
  const [lastResponseTime, setLastResponseTime] = useState(null);
  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [conversationHistory]);

  // Reset conversation when new game starts
  useEffect(() => {
    if (gameId && isEnabled) {
      setConversationHistory([]);
      setCoachingData(null);
    }
  }, [gameId]);

  const getSuggestion = async () => {
    if (!gameId || !isEnabled) return;

    try {
      setLoading(true);
      const startTime = Date.now();
      const response = await axios.post(`${API}/coaching/suggest`, {
        game_id: gameId,
        num_simulations: 400
      });

      const responseTime = (Date.now() - startTime) / 1000;
      setLastResponseTime(responseTime);

      if (response.data.success) {
        setCoachingData({
          topMoves: response.data.top_moves || [],
          positionValue: response.data.position_value || 0,
          coaching: response.data.coaching,
          offline: response.data.offline || false
        });

        // Add to conversation with unique ID for feedback
        const messageId = `coaching-${Date.now()}`;
        setConversationHistory(prev => [...prev, {
          id: messageId,
          type: 'coaching',
          content: response.data.coaching,
          topMoves: response.data.top_moves || [],
          positionValue: response.data.position_value || 0,
          responseTime: responseTime
        }]);
      }
    } catch (error) {
      console.error('Error getting coaching suggestion:', error);
      toast.error('Failed to get coaching suggestion');
    } finally {
      setLoading(false);
    }
  };

  const analyzeLastMove = async () => {
    if (!gameId || !gameState?.history?.length) {
      toast.error('No moves to analyze');
      return;
    }

    try {
      setLoading(true);
      const startTime = Date.now();
      const lastMove = gameState.history[gameState.history.length - 1];
      
      const response = await axios.post(`${API}/coaching/analyze-move`, {
        game_id: gameId,
        move: lastMove,
        num_simulations: 400
      });

      const responseTime = (Date.now() - startTime) / 1000;

      if (response.data.success) {
        const messageId = `analysis-${Date.now()}`;
        setConversationHistory(prev => [...prev, {
          id: messageId,
          type: 'analysis',
          move: lastMove,
          wasBest: response.data.was_best,
          wasInTop3: response.data.was_in_top_3,
          content: response.data.analysis,
          topMoves: response.data.top_moves || [],
          responseTime: responseTime
        }]);
      }
    } catch (error) {
      console.error('Error analyzing move:', error);
      toast.error('Failed to analyze move');
    } finally {
      setLoading(false);
    }
  };

  const explainPosition = async () => {
    await getSuggestion();
  };

  const askQuestion = async () => {
    if (!userQuestion.trim() || !gameId) return;

    const question = userQuestion.trim();
    setUserQuestion('');

    // Add user question to chat
    setConversationHistory(prev => [...prev, {
      type: 'question',
      content: question,
      isUser: true
    }]);

    try {
      setLoading(true);
      const startTime = Date.now();
      const response = await axios.post(`${API}/coaching/ask`, {
        game_id: gameId,
        question: question
      });

      const responseTime = (Date.now() - startTime) / 1000;

      if (response.data.success) {
        const messageId = `answer-${Date.now()}`;
        setConversationHistory(prev => [...prev, {
          id: messageId,
          type: 'answer',
          content: response.data.answer,
          offline: response.data.offline || false,
          responseTime: responseTime
        }]);
      }
    } catch (error) {
      console.error('Error asking question:', error);
      toast.error('Failed to get answer');
    } finally {
      setLoading(false);
    }
  };

  const getPositionValueColor = (value) => {
    if (value > 0.3) return 'text-green-600 dark:text-green-400';
    if (value > 0.1) return 'text-green-500 dark:text-green-500';
    if (value > -0.1) return 'text-gray-600 dark:text-gray-400';
    if (value > -0.3) return 'text-orange-500 dark:text-orange-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getPositionValueText = (value) => {
    if (value > 0.3) return `+${value.toFixed(2)} (White advantage)`;
    if (value > 0.1) return `+${value.toFixed(2)} (White better)`;
    if (value > -0.1) return `${value.toFixed(2)} (Equal)`;
    if (value > -0.3) return `${value.toFixed(2)} (Black better)`;
    return `${value.toFixed(2)} (Black advantage)`;
  };

  if (!isEnabled) {
    return null;
  }

  return (
    <Card className="shadow-xl" data-testid="coaching-panel">
      <CardHeader 
        className="bg-gradient-to-r from-purple-100 to-indigo-200 dark:from-purple-900 dark:to-indigo-800 cursor-pointer"
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            <span>🎓 AI Coach</span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              setIsCollapsed(!isCollapsed);
            }}
            data-testid="coaching-toggle"
          >
            {isCollapsed ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
          </Button>
        </CardTitle>
      </CardHeader>

      {!isCollapsed && (
        <CardContent className="pt-6 space-y-4">
          {/* Action Buttons */}
          <div className="grid grid-cols-2 gap-2">
            <Button
              onClick={explainPosition}
              disabled={loading || !gameState || gameState.is_game_over}
              variant="outline"
              className="w-full"
              data-testid="explain-position-btn"
            >
              <Lightbulb className="mr-2 h-4 w-4" />
              Explain Position
            </Button>
            <Button
              onClick={getSuggestion}
              disabled={loading || !gameState || gameState.is_game_over}
              variant="outline"
              className="w-full"
              data-testid="suggest-move-btn"
            >
              <TrendingUp className="mr-2 h-4 w-4" />
              Suggest Move
            </Button>
            <Button
              onClick={analyzeLastMove}
              disabled={loading || !gameState?.history?.length}
              variant="outline"
              className="w-full col-span-2"
              data-testid="analyze-last-move-btn"
            >
              <AlertCircle className="mr-2 h-4 w-4" />
              Analyze Last Move
            </Button>
          </div>

          {/* Current Evaluation Display */}
          {coachingData && coachingData.topMoves.length > 0 && (
            <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3 space-y-2">
              <div className="text-sm font-medium text-slate-700 dark:text-slate-300">
                AlphaZero Evaluation
              </div>
              <div className={`text-lg font-bold ${getPositionValueColor(coachingData.positionValue)}`}>
                {getPositionValueText(coachingData.positionValue)}
              </div>
              <div className="text-xs font-medium text-slate-600 dark:text-slate-400 mt-2">
                Top Recommended Moves:
              </div>
              {coachingData.topMoves.slice(0, 3).map((move, idx) => (
                <div key={idx} className="flex justify-between items-center text-sm">
                  <span className="font-mono font-medium">{idx + 1}. {move.move}</span>
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="text-xs">
                      {(move.probability * 100).toFixed(1)}%
                    </Badge>
                    <span className="text-xs text-slate-500">{move.visits} visits</span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Conversation Area */}
          <div className="border-t pt-4">
            <div className="text-sm font-medium mb-2 text-slate-700 dark:text-slate-300">
              Coaching Conversation
            </div>
            <div 
              className="max-h-64 overflow-y-auto bg-white dark:bg-slate-950 border rounded-lg p-3 space-y-3"
              data-testid="coaching-conversation"
            >
              {conversationHistory.length === 0 ? (
                <div className="text-center text-slate-500 dark:text-slate-400 text-sm py-4">
                  Click a button above to start getting coaching advice!
                </div>
              ) : (
                conversationHistory.map((msg, idx) => (
                  <div 
                    key={idx} 
                    className={`p-2 rounded ${
                      msg.isUser 
                        ? 'bg-blue-50 dark:bg-blue-900/20 ml-4' 
                        : 'bg-purple-50 dark:bg-purple-900/20 mr-4'
                    }`}
                  >
                    {msg.type === 'question' && (
                      <div className="text-sm">
                        <div className="font-medium text-blue-700 dark:text-blue-300 mb-1">You:</div>
                        <div className="text-slate-700 dark:text-slate-300">{msg.content}</div>
                      </div>
                    )}
                    
                    {(msg.type === 'coaching' || msg.type === 'answer') && (
                      <div className="space-y-3">
                        <div className="text-sm">
                          <div className="font-medium text-purple-700 dark:text-purple-300 mb-1 flex items-center gap-1">
                            🎓 Coach:
                            {msg.offline && <Badge variant="outline" className="text-xs">Offline</Badge>}
                          </div>
                          <div className="text-slate-700 dark:text-slate-300">{msg.content}</div>
                        </div>
                        
                        {/* Add feedback module for this response */}
                        {msg.id && !showFeedback[msg.id] && (
                          <FeedbackModule
                            sessionId={gameId || 'default'}
                            operationType="coaching"
                            responseTime={msg.responseTime || null}
                            compact={true}
                            onFeedbackSubmitted={(data) => {
                              setShowFeedback(prev => ({...prev, [msg.id]: true}));
                            }}
                          />
                        )}
                      </div>
                    )}

                    {msg.type === 'analysis' && (
                      <div className="space-y-3">
                        <div className="text-sm">
                          <div className="font-medium text-purple-700 dark:text-purple-300 mb-1">
                            🎓 Move Analysis: <span className="font-mono">{msg.move}</span>
                          </div>
                          {msg.wasBest && (
                            <Badge className="mb-2 bg-green-600">✓ Best Move!</Badge>
                          )}
                          {msg.wasInTop3 && !msg.wasBest && (
                            <Badge className="mb-2 bg-yellow-600">Good Move</Badge>
                          )}
                          {!msg.wasInTop3 && (
                            <Badge className="mb-2 bg-red-600">Not Optimal</Badge>
                          )}
                          <div className="text-slate-700 dark:text-slate-300 mt-1">{msg.content}</div>
                        </div>
                        
                        {/* Add feedback module for analysis responses */}
                        {msg.id && !showFeedback[msg.id] && (
                          <FeedbackModule
                            sessionId={gameId || 'default'}
                            operationType="coaching"
                            responseTime={msg.responseTime || null}
                            compact={true}
                            onFeedbackSubmitted={(data) => {
                              setShowFeedback(prev => ({...prev, [msg.id]: true}));
                            }}
                          />
                        )}
                      </div>
                    )}
                  </div>
                ))
              )}
              {loading && (
                <div className="flex items-center justify-center py-2">
                  <Loader2 className="h-4 w-4 animate-spin text-purple-600" />
                  <span className="ml-2 text-sm text-slate-600 dark:text-slate-400">
                    Coach is thinking...
                  </span>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Ask Question Input */}
            <div className="mt-3 flex gap-2">
              <input
                type="text"
                placeholder="Ask the coach a question..."
                value={userQuestion}
                onChange={(e) => setUserQuestion(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && askQuestion()}
                disabled={loading || !gameState}
                className="flex-1 px-3 py-2 text-sm border rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 dark:bg-slate-800 dark:border-slate-600"
                data-testid="coach-question-input"
              />
              <Button
                onClick={askQuestion}
                disabled={loading || !userQuestion.trim() || !gameState}
                size="sm"
                className="bg-purple-600 hover:bg-purple-700"
                data-testid="ask-coach-btn"
              >
                Ask
              </Button>
            </div>
          </div>
        </CardContent>
      )}
    </Card>
  );
};

export default CoachingPanel;
