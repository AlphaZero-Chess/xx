import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, Award, Target, Activity, RefreshCw, Sparkles, Brain, ChevronDown, ChevronUp, Settings } from 'lucide-react';
import axios from 'axios';
import { toast } from 'sonner';
import LLMSettingsPanel from './LLMSettingsPanel';
import AutoEvaluationCard from './AutoEvaluationCard';
import FeedbackModule from './FeedbackModule';
import KnowledgeDistillationPanel from './KnowledgeDistillationPanel';
import ForecastPanel from './ForecastPanel';
import AdaptiveForecastPanel from './AdaptiveForecastPanel';
import StrategyAdjustmentLog from './StrategyAdjustmentLog';
import StrategicInsightFusionPanel from './StrategicInsightFusionPanel';
import MultiAgentCollaborationPanel from './MultiAgentCollaborationPanel';
import ConsensusDashboard from './ConsensusDashboard';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const AnalyticsPanel = () => {
  const [trainingMetrics, setTrainingMetrics] = useState([]);
  const [trainingSummary, setTrainingSummary] = useState(null);
  const [evaluationSummary, setEvaluationSummary] = useState(null);
  const [modelHistory, setModelHistory] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // LLM Insights State
  const [llmInsights, setLlmInsights] = useState(null);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [insightsExpanded, setInsightsExpanded] = useState(true);
  const [insightsTimestamp, setInsightsTimestamp] = useState(null);
  const [insightsResponseTime, setInsightsResponseTime] = useState(null);
  const [showInsightsFeedback, setShowInsightsFeedback] = useState(false);
  
  // LLM Settings State
  const [showLLMSettings, setShowLLMSettings] = useState(false);

  useEffect(() => {
    loadAnalytics();
  }, []);

  const loadAnalytics = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Load all analytics data in parallel
      const [metricsRes, summaryRes, evalRes, modelRes] = await Promise.all([
        axios.get(`${API}/training/metrics?limit=50`),
        axios.get(`${API}/analytics/training-summary`),
        axios.get(`${API}/analytics/evaluation-summary?limit=20`),
        axios.get(`${API}/analytics/model-history`)
      ]);

      setTrainingMetrics(metricsRes.data.metrics || []);
      setTrainingSummary(summaryRes.data);
      setEvaluationSummary(evalRes.data);
      setModelHistory(modelRes.data);
    } catch (err) {
      console.error('Error loading analytics:', err);
      setError('Failed to load analytics data');
    } finally {
      setLoading(false);
    }
  };

  const generateInsights = async () => {
    setInsightsLoading(true);
    setShowInsightsFeedback(false);
    try {
      const startTime = Date.now();
      const response = await axios.post(`${API}/llm/insights`);
      const responseTime = (Date.now() - startTime) / 1000;
      
      if (response.data.success) {
        setLlmInsights(response.data.insights);
        setInsightsTimestamp(response.data.timestamp);
        setInsightsResponseTime(responseTime);
        setShowInsightsFeedback(true);
        toast.success('Insights generated successfully!');
      }
    } catch (err) {
      console.error('Error generating insights:', err);
      toast.error('Failed to generate insights. Please try again.');
    } finally {
      setInsightsLoading(false);
    }
  };

  // Format loss data for charts
  const getLossChartData = () => {
    if (!trainingMetrics || trainingMetrics.length === 0) return [];
    
    return trainingMetrics.map((metric, idx) => ({
      epoch: metric.epoch || idx + 1,
      totalLoss: metric.loss ? parseFloat(metric.loss.toFixed(4)) : 0,
      policyLoss: metric.policy_loss ? parseFloat(metric.policy_loss.toFixed(4)) : 0,
      valueLoss: metric.value_loss ? parseFloat(metric.value_loss.toFixed(4)) : 0,
    }));
  };

  // Format win rate progression data
  const getWinRateChartData = () => {
    if (!evaluationSummary || !evaluationSummary.win_rate_progression) return [];
    
    return evaluationSummary.win_rate_progression.map((eval_data, idx) => ({
      index: idx + 1,
      winRate: (eval_data.win_rate * 100).toFixed(1),
      challenger: eval_data.challenger,
      promoted: eval_data.promoted,
      games: eval_data.games_played
    }));
  };

  // Format evaluation comparison data
  const getEvaluationComparisonData = () => {
    if (!evaluationSummary || !evaluationSummary.evaluations) return [];
    
    return evaluationSummary.evaluations.slice(-10).map((eval_data, idx) => ({
      name: `${eval_data.challenger_name?.substring(0, 8)}...`,
      wins: eval_data.model1_wins || 0,
      losses: eval_data.model2_wins || 0,
      draws: eval_data.draws || 0,
      fullName: eval_data.challenger_name
    }));
  };

  const lossChartData = getLossChartData();
  const winRateChartData = getWinRateChartData();
  const evaluationComparisonData = getEvaluationComparisonData();

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12" data-testid="analytics-loading">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mx-auto mb-4"></div>
          <p className="text-slate-400">Loading analytics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 bg-red-900/20 border border-red-500 rounded-lg" data-testid="analytics-error">
        <p className="text-red-400">{error}</p>
        <Button onClick={loadAnalytics} className="mt-4 bg-red-600 hover:bg-red-700">
          <RefreshCw className="mr-2 h-4 w-4" />
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="analytics-panel">
      {/* LLM Settings Toggle Button */}
      <div className="flex justify-end">
        <Button
          onClick={() => setShowLLMSettings(!showLLMSettings)}
          variant="outline"
          className="border-indigo-500 text-indigo-300 hover:bg-indigo-900/30"
          data-testid="toggle-llm-settings-btn"
        >
          <Settings className="mr-2 h-4 w-4" />
          {showLLMSettings ? 'Hide' : 'Show'} LLM Settings
        </Button>
      </div>

      {/* LLM Settings Panel - Conditionally Rendered */}
      {showLLMSettings && <LLMSettingsPanel />}

      {/* Multi-Agent Collaboration Panel - Step 19 */}
      <MultiAgentCollaborationPanel />

      {/* Consensus Dashboard - Step 20 */}
      <ConsensusDashboard />

      {/* LLM Insights Panel - TOP PLACEMENT */}
      <Card className="bg-gradient-to-br from-indigo-900/50 via-purple-900/50 to-pink-900/50 border-purple-500 shadow-2xl" data-testid="insights-panel">
        <CardHeader 
          className="cursor-pointer hover:bg-white/5 transition-colors rounded-t-lg"
          onClick={() => setInsightsExpanded(!insightsExpanded)}
        >
          <CardTitle className="flex items-center justify-between text-white">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xl font-bold">AI Training Coach Insights</span>
                  <Badge className="bg-gradient-to-r from-yellow-500 to-orange-500 text-white border-0">
                    <Sparkles className="h-3 w-3 mr-1" />
                    Powered by GPT-4o-mini
                  </Badge>
                </div>
                {insightsTimestamp && (
                  <p className="text-xs text-purple-200 mt-1">
                    Last generated: {new Date(insightsTimestamp).toLocaleString()}
                  </p>
                )}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                className="text-white hover:bg-white/10"
                onClick={(e) => {
                  e.stopPropagation();
                  setInsightsExpanded(!insightsExpanded);
                }}
              >
                {insightsExpanded ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
              </Button>
            </div>
          </CardTitle>
        </CardHeader>

        {insightsExpanded && (
          <CardContent className="pt-6">
            {!llmInsights && !insightsLoading && (
              <div className="text-center py-12">
                <div className="mb-6">
                  <Brain className="h-16 w-16 text-purple-300 mx-auto mb-4 animate-pulse" />
                  <p className="text-white text-lg mb-2">
                    Get AI-Powered Training Insights
                  </p>
                  <p className="text-purple-200 text-sm max-w-2xl mx-auto">
                    Our AI coach will analyze your training metrics, evaluation results, and self-play data 
                    to provide comprehensive, actionable recommendations for improving your AlphaZero model.
                  </p>
                </div>
                <Button
                  onClick={generateInsights}
                  disabled={insightsLoading || (!trainingSummary && !evaluationSummary)}
                  className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white px-8 py-6 text-lg shadow-lg"
                  data-testid="generate-insights-btn"
                >
                  <Sparkles className="mr-2 h-5 w-5" />
                  Generate Insights
                </Button>
              </div>
            )}

            {insightsLoading && (
              <div className="flex flex-col items-center justify-center py-12">
                <div className="relative mb-6">
                  <div className="animate-spin rounded-full h-16 w-16 border-4 border-purple-500 border-t-transparent"></div>
                  <Brain className="h-8 w-8 text-purple-300 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 animate-pulse" />
                </div>
                <p className="text-white text-lg mb-2">Analyzing Your Training Data...</p>
                <p className="text-purple-200 text-sm">
                  The AI coach is reviewing metrics and generating detailed insights
                </p>
              </div>
            )}

            {llmInsights && !insightsLoading && (
              <div className="space-y-4">
                <div className="bg-slate-900/50 backdrop-blur-sm rounded-lg p-6 border border-purple-500/30">
                  <div className="prose prose-invert max-w-none">
                    <div className="text-slate-100 leading-relaxed whitespace-pre-wrap">
                      {llmInsights}
                    </div>
                  </div>
                </div>
                
                {/* Feedback Module for Analytics Insights */}
                {showInsightsFeedback && (
                  <div className="max-w-2xl mx-auto">
                    <FeedbackModule
                      sessionId="analytics-insights"
                      operationType="analytics"
                      responseTime={insightsResponseTime}
                      compact={false}
                      onFeedbackSubmitted={(data) => {
                        setShowInsightsFeedback(false);
                      }}
                    />
                  </div>
                )}
                
                <div className="flex justify-center gap-3">
                  <Button
                    onClick={generateInsights}
                    disabled={insightsLoading}
                    variant="outline"
                    className="border-purple-500 text-purple-300 hover:bg-purple-900/30"
                    data-testid="regenerate-insights-btn"
                  >
                    <RefreshCw className="mr-2 h-4 w-4" />
                    Regenerate Insights
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        )}
      </Card>

      {/* Summary Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-gradient-to-br from-purple-900/50 to-purple-800/50 border-purple-700" data-testid="total-sessions-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-purple-200">Training Sessions</p>
                <p className="text-3xl font-bold text-white mt-2">
                  {trainingSummary?.total_sessions || 0}
                </p>
              </div>
              <Activity className="h-10 w-10 text-purple-300" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-blue-900/50 to-blue-800/50 border-blue-700" data-testid="total-epochs-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-blue-200">Total Epochs</p>
                <p className="text-3xl font-bold text-white mt-2">
                  {trainingSummary?.total_epochs || 0}
                </p>
              </div>
              <TrendingUp className="h-10 w-10 text-blue-300" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-green-900/50 to-green-800/50 border-green-700" data-testid="evaluations-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-green-200">Evaluations</p>
                <p className="text-3xl font-bold text-white mt-2">
                  {evaluationSummary?.count || 0}
                </p>
              </div>
              <Target className="h-10 w-10 text-green-300" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-amber-900/50 to-amber-800/50 border-amber-700" data-testid="active-model-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-amber-200">Active Model</p>
                <p className="text-sm font-bold text-white mt-2 truncate" title={modelHistory?.active_model || 'None'}>
                  {modelHistory?.active_model ? modelHistory.active_model.substring(0, 12) + '...' : 'None'}
                </p>
              </div>
              <Award className="h-10 w-10 text-amber-300" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Auto-Evaluation Summary Card */}
      <AutoEvaluationCard />

      {/* Knowledge Distillation & Audit Panel (Step 15) */}
      <KnowledgeDistillationPanel />

      {/* Predictive Forecasting Panel (Step 16) */}
      <ForecastPanel />

      {/* Real-Time Adaptive Forecasting Panel (Step 17) */}
      <AdaptiveForecastPanel />

      {/* Strategy Adjustment Log (Step 17) */}
      <StrategyAdjustmentLog />

      {/* Strategic Insight Fusion Panel (Step 18) */}
      <StrategicInsightFusionPanel />

      {/* Training Loss Curves */}
      <Card className="bg-slate-800 border-slate-700" data-testid="loss-chart-card">
        <CardHeader>
          <CardTitle className="text-white flex items-center justify-between">
            <span>Training Loss Curves</span>
            <Badge variant="outline" className="text-slate-300">
              Last {lossChartData.length} Epochs
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {lossChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={lossChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                <XAxis 
                  dataKey="epoch" 
                  stroke="#94a3b8"
                  label={{ value: 'Epoch', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
                />
                <YAxis 
                  stroke="#94a3b8"
                  label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                  labelStyle={{ color: '#f1f5f9' }}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="totalLoss" 
                  stroke="#a855f7" 
                  strokeWidth={2}
                  name="Total Loss"
                  dot={{ r: 3 }}
                />
                <Line 
                  type="monotone" 
                  dataKey="policyLoss" 
                  stroke="#3b82f6" 
                  strokeWidth={2}
                  name="Policy Loss"
                  dot={{ r: 3 }}
                />
                <Line 
                  type="monotone" 
                  dataKey="valueLoss" 
                  stroke="#10b981" 
                  strokeWidth={2}
                  name="Value Loss"
                  dot={{ r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center py-12 text-slate-500">
              No training data available yet. Start training to see loss curves.
            </div>
          )}
        </CardContent>
      </Card>

      {/* Win Rate Progression */}
      <Card className="bg-slate-800 border-slate-700" data-testid="winrate-chart-card">
        <CardHeader>
          <CardTitle className="text-white flex items-center justify-between">
            <span>Win Rate Progression</span>
            <Badge variant="outline" className="text-slate-300">
              Last {winRateChartData.length} Evaluations
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {winRateChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={winRateChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                <XAxis 
                  dataKey="index" 
                  stroke="#94a3b8"
                  label={{ value: 'Evaluation #', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
                />
                <YAxis 
                  stroke="#94a3b8"
                  domain={[0, 100]}
                  label={{ value: 'Win Rate (%)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                  labelStyle={{ color: '#f1f5f9' }}
                  formatter={(value, name, props) => {
                    if (name === 'winRate') {
                      return [`${value}%`, `Win Rate`];
                    }
                    return [value, name];
                  }}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="winRate" 
                  stroke="#f59e0b" 
                  strokeWidth={3}
                  name="Win Rate (%)"
                  dot={(props) => {
                    const { cx, cy, payload } = props;
                    return (
                      <circle
                        cx={cx}
                        cy={cy}
                        r={payload.promoted ? 6 : 4}
                        fill={payload.promoted ? "#10b981" : "#f59e0b"}
                        stroke="#fff"
                        strokeWidth={2}
                      />
                    );
                  }}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center py-12 text-slate-500">
              No evaluation data available yet. Run evaluations to see win rate progression.
            </div>
          )}
          {winRateChartData.length > 0 && (
            <div className="mt-4 text-xs text-slate-400 flex items-center gap-2">
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 rounded-full bg-green-500 border-2 border-white"></span>
                Promoted Models
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 rounded-full bg-amber-500 border-2 border-white"></span>
                Not Promoted
              </span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Evaluation Results Comparison */}
      <Card className="bg-slate-800 border-slate-700" data-testid="evaluation-comparison-card">
        <CardHeader>
          <CardTitle className="text-white">Recent Evaluation Results</CardTitle>
        </CardHeader>
        <CardContent>
          {evaluationComparisonData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={evaluationComparisonData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                <XAxis 
                  dataKey="name" 
                  stroke="#94a3b8"
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis 
                  stroke="#94a3b8"
                  label={{ value: 'Games', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                  labelStyle={{ color: '#f1f5f9' }}
                  formatter={(value, name, props) => {
                    return [value, name.charAt(0).toUpperCase() + name.slice(1)];
                  }}
                />
                <Legend />
                <Bar dataKey="wins" fill="#10b981" name="Wins" />
                <Bar dataKey="losses" fill="#ef4444" name="Losses" />
                <Bar dataKey="draws" fill="#6b7280" name="Draws" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center py-12 text-slate-500">
              No evaluation results available yet.
            </div>
          )}
        </CardContent>
      </Card>

      {/* Model Promotion History */}
      {modelHistory && modelHistory.promotion_history && modelHistory.promotion_history.length > 0 && (
        <Card className="bg-slate-800 border-slate-700" data-testid="promotion-history-card">
          <CardHeader>
            <CardTitle className="text-white">Model Promotion History</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {modelHistory.promotion_history.map((promo, idx) => (
                <div 
                  key={idx} 
                  className="flex items-center justify-between p-3 bg-slate-700 rounded-lg"
                  data-testid={`promotion-${idx}`}
                >
                  <div className="flex items-center gap-3">
                    <Award className="h-5 w-5 text-green-400" />
                    <div>
                      <p className="text-sm font-semibold text-white">
                        {promo.model_name}
                      </p>
                      <p className="text-xs text-slate-400">
                        Defeated: {promo.defeated || 'N/A'}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <Badge className="bg-green-600 mb-1">
                      {(promo.win_rate * 100).toFixed(1)}% Win Rate
                    </Badge>
                    <p className="text-xs text-slate-400">
                      {new Date(promo.promoted_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Training Sessions Summary */}
      {trainingSummary && trainingSummary.training_sessions && trainingSummary.training_sessions.length > 0 && (
        <Card className="bg-slate-800 border-slate-700" data-testid="training-sessions-card">
          <CardHeader>
            <CardTitle className="text-white">Recent Training Sessions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {trainingSummary.training_sessions.slice(0, 10).map((session, idx) => (
                <div 
                  key={idx} 
                  className="flex items-center justify-between p-3 bg-slate-700 rounded-lg text-sm"
                  data-testid={`training-session-${idx}`}
                >
                  <div>
                    <p className="text-slate-300 font-medium">
                      Session: {session.session_id?.substring(0, 8)}...
                    </p>
                    <p className="text-xs text-slate-500">
                      {session.device || 'Unknown Device'}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-slate-300">
                      {session.epochs} epochs
                    </p>
                    <p className="text-xs text-slate-500">
                      Avg Loss: {session.avg_loss?.toFixed(4) || 'N/A'}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Refresh Button */}
      <div className="flex justify-center">
        <Button 
          onClick={loadAnalytics}
          className="bg-purple-600 hover:bg-purple-700"
          data-testid="refresh-analytics-btn"
        >
          <RefreshCw className="mr-2 h-4 w-4" />
          Refresh Analytics
        </Button>
      </div>
    </div>
  );
};

export default AnalyticsPanel;
