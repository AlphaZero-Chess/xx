import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import { Textarea } from './ui/textarea';
import { toast } from 'sonner';
import axios from 'axios';
import {
  Network,
  Brain,
  Users,
  MessageSquare,
  TrendingUp,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Download,
  Play,
  Clock,
  Target,
  Lightbulb,
  BarChart3
} from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const MultiAgentCollaborationPanel = () => {
  const [sessions, setSessions] = useState([]);
  const [metaKnowledge, setMetaKnowledge] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [sessionRunning, setSessionRunning] = useState(false);
  const [selectedSession, setSelectedSession] = useState(null);
  const [showSessionDialog, setShowSessionDialog] = useState(false);
  
  // New session form
  const [taskInput, setTaskInput] = useState('');
  const [contextInput, setContextInput] = useState('');
  const [applyMetaLearning, setApplyMetaLearning] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [sessionsRes, knowledgeRes, metricsRes] = await Promise.all([
        axios.get(`${API}/llm/meta-collaboration/sessions?limit=10`),
        axios.get(`${API}/llm/meta-knowledge?limit=20`),
        axios.get(`${API}/llm/meta-learning/metrics`)
      ]);

      setSessions(sessionsRes.data.sessions || []);
      setMetaKnowledge(knowledgeRes.data.knowledge || []);
      setMetrics(metricsRes.data.metrics || null);
    } catch (error) {
      console.error('Error loading collaboration data:', error);
      toast.error('Failed to load collaboration data');
    } finally {
      setLoading(false);
    }
  };

  const runMultiAgentSession = async () => {
    if (!taskInput.trim()) {
      toast.error('Please enter a task description');
      return;
    }

    setSessionRunning(true);
    try {
      const response = await axios.post(`${API}/llm/meta-collaboration`, {
        task: taskInput,
        context: contextInput ? JSON.parse(contextInput) : null,
        apply_meta_learning: applyMetaLearning
      });

      if (response.data.success) {
        toast.success('Multi-agent reasoning completed!');
        setTaskInput('');
        setContextInput('');
        await loadData();
        
        // Show results dialog
        const sessionDetail = await axios.get(
          `${API}/llm/meta-collaboration/session/${response.data.session_id}`
        );
        setSelectedSession(sessionDetail.data.session);
        setShowSessionDialog(true);
      }
    } catch (error) {
      console.error('Error running multi-agent session:', error);
      toast.error('Failed to run multi-agent session');
    } finally {
      setSessionRunning(false);
    }
  };

  const viewSessionDetail = async (sessionId) => {
    try {
      const response = await axios.get(`${API}/llm/meta-collaboration/session/${sessionId}`);
      setSelectedSession(response.data.session);
      setShowSessionDialog(true);
    } catch (error) {
      console.error('Error fetching session detail:', error);
      toast.error('Failed to load session details');
    }
  };

  const applyHighConfidenceKnowledge = async () => {
    try {
      const response = await axios.post(`${API}/llm/meta-knowledge/apply`, {
        apply_all_high_confidence: true
      });

      if (response.data.success) {
        toast.success(`Applied ${response.data.applied_count} high-confidence insights`);
        await loadData();
      }
    } catch (error) {
      console.error('Error applying meta-knowledge:', error);
      toast.error('Failed to apply meta-knowledge');
    }
  };

  const exportSession = (session) => {
    const dataStr = JSON.stringify(session, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `meta-session-${session.session_id.substring(0, 8)}.json`;
    link.click();
    URL.revokeObjectURL(url);
    toast.success('Session exported!');
  };

  const getConfidenceBadge = (confidence) => {
    if (confidence >= 0.8) return <Badge className="bg-green-600">High: {(confidence * 100).toFixed(0)}%</Badge>;
    if (confidence >= 0.6) return <Badge className="bg-yellow-600">Medium: {(confidence * 100).toFixed(0)}%</Badge>;
    return <Badge className="bg-red-600">Low: {(confidence * 100).toFixed(0)}%</Badge>;
  };

  const getConsensusIcon = (reached) => {
    return reached ? (
      <CheckCircle2 className="text-green-500" size={20} />
    ) : (
      <XCircle className="text-red-500" size={20} />
    );
  };

  return (
    <div className="space-y-6" data-testid="multi-agent-collaboration-panel">
      {/* Header with Metrics Overview */}
      <Card className="bg-gradient-to-br from-indigo-900/40 to-purple-900/40 border-indigo-500/30">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Network className="text-indigo-400" size={28} />
              <div>
                <CardTitle className="text-2xl text-white">Multi-Agent Collaboration</CardTitle>
                <p className="text-indigo-300 text-sm mt-1">
                  Collaborative reasoning with Strategy, Evaluation, Forecast & Adaptation agents
                </p>
              </div>
            </div>
            <Button
              onClick={loadData}
              variant="outline"
              size="sm"
              className="text-indigo-300 border-indigo-500/50"
              data-testid="refresh-collaboration-btn"
            >
              <BarChart3 size={16} className="mr-2" />
              Refresh
            </Button>
          </div>
        </CardHeader>

        {metrics && (
          <CardContent>
            <div className="grid grid-cols-4 gap-4">
              <div className="bg-slate-800/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Users className="text-blue-400" size={20} />
                  <span className="text-slate-400 text-sm">Total Sessions</span>
                </div>
                <div className="text-3xl font-bold text-white">{metrics.total_sessions}</div>
              </div>

              <div className="bg-slate-800/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle2 className="text-green-400" size={20} />
                  <span className="text-slate-400 text-sm">Consensus Rate</span>
                </div>
                <div className="text-3xl font-bold text-white">{metrics.consensus_rate}%</div>
              </div>

              <div className="bg-slate-800/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="text-purple-400" size={20} />
                  <span className="text-slate-400 text-sm">Avg Confidence</span>
                </div>
                <div className="text-3xl font-bold text-white">
                  {(metrics.avg_confidence * 100).toFixed(0)}%
                </div>
              </div>

              <div className="bg-slate-800/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Lightbulb className="text-yellow-400" size={20} />
                  <span className="text-slate-400 text-sm">Total Insights</span>
                </div>
                <div className="text-3xl font-bold text-white">{metrics.total_insights}</div>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Main Tabs */}
      <Tabs defaultValue="new-session" className="w-full">
        <TabsList className="grid w-full grid-cols-3 bg-slate-800/50">
          <TabsTrigger value="new-session" data-testid="new-session-tab">
            <Play size={16} className="mr-2" />
            New Session
          </TabsTrigger>
          <TabsTrigger value="sessions" data-testid="sessions-tab">
            <MessageSquare size={16} className="mr-2" />
            Sessions
          </TabsTrigger>
          <TabsTrigger value="meta-knowledge" data-testid="meta-knowledge-tab">
            <Brain size={16} className="mr-2" />
            Meta-Knowledge
          </TabsTrigger>
        </TabsList>

        {/* New Session Tab */}
        <TabsContent value="new-session">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Play className="text-green-400" size={20} />
                Run Multi-Agent Reasoning Session
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-slate-300 text-sm font-medium mb-2 block">
                  Task Description *
                </label>
                <Textarea
                  placeholder="E.g., 'Recommend optimal MCTS depth for next training epoch' or 'Analyze win-rate plateau at epoch 50'"
                  value={taskInput}
                  onChange={(e) => setTaskInput(e.target.value)}
                  rows={4}
                  className="bg-slate-900 border-slate-600 text-white"
                  data-testid="task-input"
                />
              </div>

              <div>
                <label className="text-slate-300 text-sm font-medium mb-2 block">
                  Context (Optional JSON)
                </label>
                <Textarea
                  placeholder='{"current_mcts_depth": 800, "win_rate": 0.65}'
                  value={contextInput}
                  onChange={(e) => setContextInput(e.target.value)}
                  rows={3}
                  className="bg-slate-900 border-slate-600 text-white font-mono text-sm"
                  data-testid="context-input"
                />
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="meta-learning"
                  checked={applyMetaLearning}
                  onChange={(e) => setApplyMetaLearning(e.target.checked)}
                  className="w-4 h-4"
                  data-testid="meta-learning-checkbox"
                />
                <label htmlFor="meta-learning" className="text-slate-300 text-sm">
                  Apply meta-learning (extract and store insights)
                </label>
              </div>

              <Button
                onClick={runMultiAgentSession}
                disabled={sessionRunning || !taskInput.trim()}
                className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700"
                data-testid="run-session-btn"
              >
                {sessionRunning ? (
                  <>
                    <Clock className="animate-spin mr-2" size={16} />
                    Running Multi-Agent Session...
                  </>
                ) : (
                  <>
                    <Play size={16} className="mr-2" />
                    Run Multi-Agent Session
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Sessions History Tab */}
        <TabsContent value="sessions">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Recent Collaboration Sessions</CardTitle>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="text-slate-400 text-center py-8">Loading sessions...</div>
              ) : sessions.length === 0 ? (
                <div className="text-slate-400 text-center py-8">
                  No sessions yet. Run your first multi-agent reasoning session!
                </div>
              ) : (
                <div className="space-y-4">
                  {sessions.map((session, idx) => (
                    <div
                      key={idx}
                      className="bg-slate-900/50 rounded-lg p-4 border border-slate-700 hover:border-indigo-500/50 transition-all cursor-pointer"
                      onClick={() => viewSessionDetail(session.session_id)}
                      data-testid={`session-${idx}`}
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            {getConsensusIcon(session.consensus?.consensus_reached)}
                            <span className="text-white font-medium">
                              {session.task.substring(0, 80)}
                              {session.task.length > 80 ? '...' : ''}
                            </span>
                          </div>
                          <div className="text-sm text-slate-400">
                            Session ID: {session.session_id.substring(0, 12)}...
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            exportSession(session);
                          }}
                          data-testid={`export-session-${idx}`}
                        >
                          <Download size={16} />
                        </Button>
                      </div>

                      <div className="grid grid-cols-3 gap-4 mb-3">
                        <div>
                          <div className="text-xs text-slate-500 mb-1">Confidence</div>
                          {getConfidenceBadge(session.consensus?.confidence_score || 0)}
                        </div>
                        <div>
                          <div className="text-xs text-slate-500 mb-1">Consensus Level</div>
                          <Badge className="bg-blue-600">
                            {((session.consensus?.consensus_level || 0) * 100).toFixed(0)}%
                          </Badge>
                        </div>
                        <div>
                          <div className="text-xs text-slate-500 mb-1">Agents</div>
                          <Badge className="bg-purple-600">
                            {session.consensus?.participating_agents?.length || 0}
                          </Badge>
                        </div>
                      </div>

                      {session.consensus?.meta_insights && session.consensus.meta_insights.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-slate-700">
                          <div className="flex items-center gap-2 text-yellow-400 text-sm mb-2">
                            <Lightbulb size={14} />
                            <span>{session.consensus.meta_insights.length} Insights Extracted</span>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Meta-Knowledge Tab */}
        <TabsContent value="meta-knowledge">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-white">Accumulated Meta-Knowledge</CardTitle>
                <Button
                  onClick={applyHighConfidenceKnowledge}
                  className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700"
                  data-testid="apply-knowledge-btn"
                >
                  <TrendingUp size={16} className="mr-2" />
                  Apply High-Confidence Insights
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="text-slate-400 text-center py-8">Loading meta-knowledge...</div>
              ) : metaKnowledge.length === 0 ? (
                <div className="text-slate-400 text-center py-8">
                  No meta-knowledge yet. Run sessions with meta-learning enabled!
                </div>
              ) : (
                <div className="space-y-3">
                  {metaKnowledge.map((knowledge, idx) => (
                    <div
                      key={idx}
                      className="bg-slate-900/50 rounded-lg p-4 border border-slate-700"
                      data-testid={`knowledge-${idx}`}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <Badge className="bg-indigo-600 capitalize">{knowledge.category}</Badge>
                        {getConfidenceBadge(knowledge.confidence)}
                      </div>

                      <div className="text-white mb-2">{knowledge.insight}</div>

                      <div className="text-sm text-slate-400">
                        Applied: {knowledge.validation_count} times
                        {knowledge.success_rate > 0 && (
                          <> • Success Rate: {(knowledge.success_rate * 100).toFixed(0)}%</>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Session Detail Dialog */}
      <Dialog open={showSessionDialog} onOpenChange={setShowSessionDialog}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto bg-slate-900 border-slate-700">
          <DialogHeader>
            <DialogTitle className="text-white">Session Transcript</DialogTitle>
          </DialogHeader>
          
          {selectedSession && (
            <div className="space-y-4">
              <div className="bg-slate-800/50 rounded-lg p-4">
                <h3 className="text-white font-semibold mb-2">Task</h3>
                <p className="text-slate-300">{selectedSession.task}</p>
              </div>

              <div className="bg-slate-800/50 rounded-lg p-4">
                <h3 className="text-white font-semibold mb-3">Consensus Result</h3>
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    {getConsensusIcon(selectedSession.consensus?.consensus_reached)}
                    <span className="text-slate-300">
                      {selectedSession.consensus?.consensus_reached
                        ? 'Consensus Reached'
                        : 'No Consensus'}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-4 mt-3">
                    <div>
                      <span className="text-slate-500 text-sm">Confidence:</span>
                      <div className="mt-1">
                        {getConfidenceBadge(selectedSession.consensus?.confidence_score || 0)}
                      </div>
                    </div>
                    <div>
                      <span className="text-slate-500 text-sm">Consensus Level:</span>
                      <div className="mt-1">
                        <Badge className="bg-blue-600">
                          {((selectedSession.consensus?.consensus_level || 0) * 100).toFixed(0)}%
                        </Badge>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-slate-800/50 rounded-lg p-4">
                <h3 className="text-white font-semibold mb-3">Final Decision</h3>
                <p className="text-slate-300 whitespace-pre-wrap">
                  {selectedSession.consensus?.final_decision}
                </p>
              </div>

              {selectedSession.consensus?.individual_positions && (
                <div className="bg-slate-800/50 rounded-lg p-4">
                  <h3 className="text-white font-semibold mb-3">Agent Positions</h3>
                  <div className="space-y-3">
                    {selectedSession.consensus.individual_positions.map((position, idx) => (
                      <div key={idx} className="bg-slate-900/50 rounded p-3">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-indigo-400 font-medium">{position.agent}</span>
                          {getConfidenceBadge(position.confidence)}
                        </div>
                        <p className="text-slate-300 text-sm">{position.position}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {selectedSession.consensus?.meta_insights && selectedSession.consensus.meta_insights.length > 0 && (
                <div className="bg-slate-800/50 rounded-lg p-4">
                  <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
                    <Lightbulb className="text-yellow-400" size={20} />
                    Meta-Insights Extracted
                  </h3>
                  <ul className="space-y-2">
                    {selectedSession.consensus.meta_insights.map((insight, idx) => (
                      <li key={idx} className="text-slate-300 flex gap-2">
                        <span className="text-yellow-400">•</span>
                        <span>{insight}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default MultiAgentCollaborationPanel;
