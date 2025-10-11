import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Progress } from './ui/progress';
import { 
  Brain, 
  Network, 
  TrendingUp, 
  Download, 
  RefreshCw, 
  Zap,
  CheckCircle2,
  AlertCircle,
  Activity,
  Target,
  GitBranch
} from 'lucide-react';
import { toast } from 'sonner';
import MetaOptimizationPanel from './MetaOptimizationPanel';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const CollectiveIntelligencePanel = () => {
  const [loading, setLoading] = useState(false);
  const [synthesizing, setSynthesizing] = useState(false);
  const [synthesisResult, setSynthesisResult] = useState(null);
  const [strategies, setStrategies] = useState([]);
  const [alignment, setAlignment] = useState(null);
  const [insightsSummary, setInsightsSummary] = useState(null);
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const strategyMapRef = useRef(null);

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        fetchInsightsSummary(),
        fetchStrategies(),
        fetchAlignment()
      ]);
    } catch (error) {
      console.error('Error loading data:', error);
      toast.error('Failed to load collective intelligence data');
    } finally {
      setLoading(false);
    }
  };

  const fetchInsightsSummary = async () => {
    try {
      const response = await axios.get(`${API}/llm/collective/insights-summary`);
      setInsightsSummary(response.data);
    } catch (error) {
      console.error('Error fetching insights summary:', error);
    }
  };

  const fetchStrategies = async () => {
    try {
      const response = await axios.get(`${API}/llm/collective/strategies`, {
        params: {
          min_confidence: 0.6,
          limit: 20,
          sort_by: 'confidence'
        }
      });
      setStrategies(response.data.strategies || []);
    } catch (error) {
      console.error('Error fetching strategies:', error);
    }
  };

  const fetchAlignment = async () => {
    try {
      const response = await axios.get(`${API}/llm/collective/alignment`, {
        params: { limit: 10 }
      });
      setAlignment(response.data);
    } catch (error) {
      console.error('Error fetching alignment:', error);
    }
  };

  const runSynthesis = async () => {
    setSynthesizing(true);
    try {
      toast.info('Starting collective intelligence synthesis...');
      const response = await axios.post(`${API}/llm/collective/synthesize`);
      setSynthesisResult(response.data);
      
      // Refresh data after synthesis
      await Promise.all([
        fetchInsightsSummary(),
        fetchStrategies(),
        fetchAlignment()
      ]);
      
      toast.success(`Synthesis complete: ${response.data.collective_strategy.strategy_archetype}`);
    } catch (error) {
      console.error('Error running synthesis:', error);
      toast.error('Synthesis failed: ' + (error.response?.data?.detail || error.message));
    } finally {
      setSynthesizing(false);
    }
  };

  const exportSynthesisReport = () => {
    if (!synthesisResult) {
      toast.error('No synthesis data to export');
      return;
    }

    const report = {
      title: 'Collective Intelligence Synthesis Report',
      timestamp: new Date().toISOString(),
      strategy: synthesisResult.collective_strategy,
      alignment: synthesisResult.alignment_metrics,
      insights: synthesisResult.aggregated_insights
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `collective-intelligence-report-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast.success('Report exported successfully');
  };

  const getAlignmentColor = (score) => {
    if (score >= 0.85) return 'text-green-500';
    if (score >= 0.7) return 'text-yellow-500';
    if (score >= 0.5) return 'text-orange-500';
    return 'text-red-500';
  };

  const getConsensusColor = (level) => {
    switch(level) {
      case 'high': return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'moderate': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'low': return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
      default: return 'bg-red-500/20 text-red-400 border-red-500/30';
    }
  };

  useEffect(() => {
    if (strategies.length > 0 && strategyMapRef.current) {
      renderStrategyMap();
    }
  }, [strategies]);

  const renderStrategyMap = () => {
    if (!strategyMapRef.current || strategies.length === 0) return;

    const canvas = strategyMapRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw background grid
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i < width; i += 50) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, height);
      ctx.stroke();
    }
    for (let i = 0; i < height; i += 50) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(width, i);
      ctx.stroke();
    }

    // Position strategies in a circular layout
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.35;

    strategies.forEach((strategy, index) => {
      const angle = (index / strategies.length) * 2 * Math.PI - Math.PI / 2;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      
      const confidence = strategy.confidence_score || 0;
      const nodeRadius = 10 + confidence * 20;
      
      // Draw connections to center
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(x, y);
      ctx.strokeStyle = `rgba(96, 165, 250, ${confidence * 0.5})`;
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Draw node
      ctx.beginPath();
      ctx.arc(x, y, nodeRadius, 0, 2 * Math.PI);
      ctx.fillStyle = `rgba(59, 130, 246, ${confidence})`;
      ctx.fill();
      ctx.strokeStyle = 'rgba(96, 165, 250, 1)';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Draw usage count badge
      if (strategy.usage_count > 1) {
        ctx.beginPath();
        ctx.arc(x + nodeRadius - 5, y - nodeRadius + 5, 8, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(34, 197, 94, 0.9)';
        ctx.fill();
        ctx.fillStyle = 'white';
        ctx.font = 'bold 10px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(strategy.usage_count, x + nodeRadius - 5, y - nodeRadius + 5);
      }
    });

    // Draw center hub
    ctx.beginPath();
    ctx.arc(centerX, centerY, 30, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(139, 92, 246, 0.8)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(167, 139, 250, 1)';
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Draw "CI" text in center
    ctx.fillStyle = 'white';
    ctx.font = 'bold 16px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('CI', centerX, centerY);
  };

  const handleCanvasClick = (e) => {
    if (!strategyMapRef.current || strategies.length === 0) return;
    
    const canvas = strategyMapRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.35;
    
    strategies.forEach((strategy, index) => {
      const angle = (index / strategies.length) * 2 * Math.PI - Math.PI / 2;
      const nodeX = centerX + radius * Math.cos(angle);
      const nodeY = centerY + radius * Math.sin(angle);
      const nodeRadius = 10 + (strategy.confidence_score || 0) * 20;
      
      const distance = Math.sqrt((x - nodeX) ** 2 + (y - nodeY) ** 2);
      if (distance <= nodeRadius) {
        setSelectedStrategy(strategy);
      }
    });
  };

  if (loading) {
    return (
      <Card className="bg-slate-800/50 border-slate-700" data-testid="collective-intelligence-panel">
        <CardContent className="flex items-center justify-center p-12">
          <div className="text-slate-400">Loading collective intelligence data...</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6" data-testid="collective-intelligence-panel">
      {/* Header with Status */}
      <Card className="bg-gradient-to-br from-purple-900/30 to-blue-900/30 border-purple-500/30">
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <CardTitle className="text-2xl text-white flex items-center gap-3">
                <Brain className="text-purple-400" size={32} />
                Collective Intelligence Layer
              </CardTitle>
              <CardDescription className="text-slate-300">
                Global Strategy Synthesis & Multi-Agent Alignment
              </CardDescription>
            </div>
            
            {insightsSummary && (
              <div className="text-right space-y-2">
                <div className="flex items-center gap-2">
                  <Activity size={16} className="text-purple-400" />
                  <span className="text-sm text-slate-300">System Health:</span>
                  <Badge className={
                    insightsSummary.collective_intelligence.system_health === 'optimal' 
                      ? 'bg-green-500/20 text-green-400' 
                      : insightsSummary.collective_intelligence.system_health === 'good'
                      ? 'bg-yellow-500/20 text-yellow-400'
                      : 'bg-orange-500/20 text-orange-400'
                  }>
                    {insightsSummary.collective_intelligence.system_health.toUpperCase()}
                  </Badge>
                </div>
                {alignment?.current_alignment && (
                  <div className="flex items-center gap-2">
                    <Target size={16} className="text-blue-400" />
                    <span className="text-sm text-slate-300">Alignment:</span>
                    <span className={`font-bold ${getAlignmentColor(alignment.current_alignment.overall_alignment_score)}`}>
                      {(alignment.current_alignment.overall_alignment_score * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
        </CardHeader>
        
        <CardContent>
          <div className="grid grid-cols-4 gap-4 mb-4">
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">Memory Experiences</div>
              <div className="text-2xl font-bold text-white">
                {insightsSummary?.subsystem_status.memory_experiences || 0}
              </div>
            </div>
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">Trust Calibrations</div>
              <div className="text-2xl font-bold text-white">
                {insightsSummary?.subsystem_status.trust_calibrations || 0}
              </div>
            </div>
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">Arbitrations</div>
              <div className="text-2xl font-bold text-white">
                {insightsSummary?.subsystem_status.arbitration_logs || 0}
              </div>
            </div>
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">Global Strategies</div>
              <div className="text-2xl font-bold text-white">
                {insightsSummary?.subsystem_status.global_strategies || 0}
              </div>
            </div>
          </div>
          
          <div className="flex gap-3">
            <Button
              onClick={runSynthesis}
              disabled={synthesizing}
              className="bg-purple-600 hover:bg-purple-700 text-white"
              data-testid="synthesize-btn"
            >
              {synthesizing ? (
                <>
                  <RefreshCw className="animate-spin mr-2" size={16} />
                  Synthesizing...
                </>
              ) : (
                <>
                  <Zap className="mr-2" size={16} />
                  Run Synthesis
                </>
              )}
            </Button>
            
            <Button
              onClick={exportSynthesisReport}
              disabled={!synthesisResult}
              variant="outline"
              className="border-slate-600 text-slate-300 hover:bg-slate-700"
              data-testid="export-report-btn"
            >
              <Download className="mr-2" size={16} />
              Export Report
            </Button>
            
            <Button
              onClick={loadInitialData}
              variant="outline"
              className="border-slate-600 text-slate-300 hover:bg-slate-700"
              data-testid="refresh-data-btn"
            >
              <RefreshCw className="mr-2" size={16} />
              Refresh
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Main Content Tabs */}
      <Tabs defaultValue="map" className="w-full">
        <TabsList className="grid w-full grid-cols-5 bg-slate-800/50">
          <TabsTrigger value="map" data-testid="tab-strategy-map">Strategy Map</TabsTrigger>
          <TabsTrigger value="alignment" data-testid="tab-alignment">Alignment</TabsTrigger>
          <TabsTrigger value="strategies" data-testid="tab-strategies">Strategies</TabsTrigger>
          <TabsTrigger value="synthesis" data-testid="tab-synthesis">Latest Synthesis</TabsTrigger>
          <TabsTrigger value="meta-optimization" data-testid="tab-meta-optimization">Self-Optimization</TabsTrigger>
        </TabsList>

        {/* Strategy Map View */}
        <TabsContent value="map" data-testid="strategy-map-content">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Network size={20} className="text-blue-400" />
                Interactive Strategy Map
              </CardTitle>
              <CardDescription className="text-slate-400">
                Nodes = Strategy Archetypes | Size = Confidence | Edges = Strategic Relations
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="relative">
                <canvas
                  ref={strategyMapRef}
                  width={800}
                  height={600}
                  onClick={handleCanvasClick}
                  className="w-full h-auto bg-slate-900/50 rounded-lg border border-slate-700 cursor-pointer"
                  data-testid="strategy-map-canvas"
                />
                {strategies.length === 0 && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-slate-400 text-center">
                      <Network size={48} className="mx-auto mb-4 opacity-30" />
                      <p>No strategies available</p>
                      <p className="text-sm mt-2">Run synthesis to generate strategies</p>
                    </div>
                  </div>
                )}
              </div>
              
              {selectedStrategy && (
                <Card className="mt-4 bg-slate-900/50 border-blue-500/30">
                  <CardHeader>
                    <CardTitle className="text-lg text-blue-400">
                      {selectedStrategy.strategy_archetype}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <p className="text-slate-300 text-sm">{selectedStrategy.description}</p>
                    <div className="grid grid-cols-3 gap-3">
                      <div className="bg-slate-800/50 p-3 rounded">
                        <div className="text-xs text-slate-400">Confidence</div>
                        <div className="text-lg font-bold text-white">
                          {(selectedStrategy.confidence_score * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-slate-800/50 p-3 rounded">
                        <div className="text-xs text-slate-400">Alignment</div>
                        <div className="text-lg font-bold text-white">
                          {(selectedStrategy.alignment_score * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-slate-800/50 p-3 rounded">
                        <div className="text-xs text-slate-400">Usage Count</div>
                        <div className="text-lg font-bold text-white">
                          {selectedStrategy.usage_count}
                        </div>
                      </div>
                    </div>
                    {selectedStrategy.strategic_recommendations && (
                      <div>
                        <div className="text-sm text-slate-400 mb-2">Recommendations:</div>
                        <ul className="space-y-1">
                          {selectedStrategy.strategic_recommendations.map((rec, idx) => (
                            <li key={idx} className="text-sm text-slate-300 flex items-start gap-2">
                              <CheckCircle2 size={14} className="text-green-400 mt-0.5 flex-shrink-0" />
                              <span>{rec}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Alignment Tab */}
        <TabsContent value="alignment" data-testid="alignment-content">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Target size={20} className="text-purple-400" />
                Collective Alignment Metrics
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {alignment?.current_alignment ? (
                <>
                  {/* Overall Alignment Gauge */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-300 font-medium">Overall Alignment Score</span>
                      <span className={`text-2xl font-bold ${getAlignmentColor(alignment.current_alignment.overall_alignment_score)}`}>
                        {(alignment.current_alignment.overall_alignment_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress 
                      value={alignment.current_alignment.overall_alignment_score * 100} 
                      className="h-4"
                    />
                    <div className="flex items-center gap-2">
                      <Badge className={getConsensusColor(alignment.current_alignment.consensus_level)}>
                        {alignment.current_alignment.consensus_level.toUpperCase()}
                      </Badge>
                      <span className="text-sm text-slate-400">
                        {alignment.current_alignment.unified_direction}
                      </span>
                    </div>
                  </div>

                  {/* Subsystem Scores */}
                  <div>
                    <h4 className="text-slate-300 font-medium mb-3">Subsystem Alignment</h4>
                    <div className="grid grid-cols-2 gap-4">
                      {Object.entries(alignment.current_alignment.subsystem_scores).map(([subsystem, score]) => (
                        <div key={subsystem} className="bg-slate-900/50 p-4 rounded-lg">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm text-slate-400 capitalize">{subsystem}</span>
                            <span className={`font-bold ${getAlignmentColor(score)}`}>
                              {(score * 100).toFixed(1)}%
                            </span>
                          </div>
                          <Progress value={score * 100} className="h-2" />
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Additional Metrics */}
                  <div className="grid grid-cols-3 gap-4">
                    <div className="bg-slate-900/50 p-4 rounded-lg">
                      <div className="text-xs text-slate-400 mb-1">Harmony Index</div>
                      <div className="text-xl font-bold text-white">
                        {(alignment.current_alignment.harmony_index * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="bg-slate-900/50 p-4 rounded-lg">
                      <div className="text-xs text-slate-400 mb-1">Strategic Coherence</div>
                      <div className="text-xl font-bold text-white">
                        {(alignment.current_alignment.strategic_coherence * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="bg-slate-900/50 p-4 rounded-lg">
                      <div className="text-xs text-slate-400 mb-1">Active Conflicts</div>
                      <div className="text-xl font-bold text-white">
                        {alignment.current_alignment.conflict_count}
                      </div>
                    </div>
                  </div>

                  {/* Trend Information */}
                  {alignment.trends && (
                    <div className="bg-slate-900/50 p-4 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <TrendingUp size={16} className="text-blue-400" />
                        <span className="text-sm font-medium text-slate-300">Alignment Trend</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <Badge className={
                          alignment.trends.overall_trend === 'improving' ? 'bg-green-500/20 text-green-400' :
                          alignment.trends.overall_trend === 'declining' ? 'bg-red-500/20 text-red-400' :
                          'bg-slate-500/20 text-slate-400'
                        }>
                          {alignment.trends.overall_trend.toUpperCase()}
                        </Badge>
                        <span className="text-sm text-slate-400">
                          {alignment.trends.change_percent >= 0 ? '+' : ''}{alignment.trends.change_percent.toFixed(2)}%
                        </span>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-center text-slate-400 py-12">
                  <AlertCircle size={48} className="mx-auto mb-4 opacity-30" />
                  <p>No alignment data available</p>
                  <p className="text-sm mt-2">Run synthesis to generate alignment metrics</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Strategies List Tab */}
        <TabsContent value="strategies" data-testid="strategies-content">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <GitBranch size={20} className="text-green-400" />
                Global Strategy Repository
              </CardTitle>
              <CardDescription className="text-slate-400">
                High-confidence strategies synthesized from collective intelligence
              </CardDescription>
            </CardHeader>
            <CardContent>
              {strategies.length > 0 ? (
                <div className="space-y-4">
                  {strategies.map((strategy) => (
                    <Card 
                      key={strategy.strategy_id} 
                      className="bg-slate-900/50 border-slate-700 hover:border-blue-500/50 transition-colors cursor-pointer"
                      onClick={() => setSelectedStrategy(strategy)}
                    >
                      <CardHeader>
                        <div className="flex items-start justify-between">
                          <CardTitle className="text-lg text-blue-400">
                            {strategy.strategy_archetype}
                          </CardTitle>
                          <div className="flex gap-2">
                            <Badge className="bg-purple-500/20 text-purple-400">
                              {(strategy.confidence_score * 100).toFixed(0)}% confidence
                            </Badge>
                            {strategy.usage_count > 1 && (
                              <Badge className="bg-green-500/20 text-green-400">
                                Used {strategy.usage_count}x
                              </Badge>
                            )}
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <p className="text-slate-300 text-sm">{strategy.description}</p>
                        
                        <div className="grid grid-cols-3 gap-2">
                          <div className="text-center p-2 bg-slate-800/50 rounded">
                            <div className="text-xs text-slate-400">Confidence</div>
                            <div className="text-sm font-bold text-white">
                              {(strategy.confidence_score * 100).toFixed(1)}%
                            </div>
                          </div>
                          <div className="text-center p-2 bg-slate-800/50 rounded">
                            <div className="text-xs text-slate-400">Alignment</div>
                            <div className="text-sm font-bold text-white">
                              {(strategy.alignment_score * 100).toFixed(1)}%
                            </div>
                          </div>
                          <div className="text-center p-2 bg-slate-800/50 rounded">
                            <div className="text-xs text-slate-400">Usage</div>
                            <div className="text-sm font-bold text-white">
                              {strategy.usage_count}
                            </div>
                          </div>
                        </div>

                        {strategy.strategic_recommendations && strategy.strategic_recommendations.length > 0 && (
                          <div>
                            <div className="text-xs text-slate-400 mb-2">Key Recommendations:</div>
                            <ul className="space-y-1">
                              {strategy.strategic_recommendations.slice(0, 3).map((rec, idx) => (
                                <li key={idx} className="text-xs text-slate-300 flex items-start gap-2">
                                  <CheckCircle2 size={12} className="text-green-400 mt-0.5 flex-shrink-0" />
                                  <span>{rec}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        <div className="text-xs text-slate-500">
                          Last updated: {new Date(strategy.last_updated).toLocaleString()}
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : (
                <div className="text-center text-slate-400 py-12">
                  <GitBranch size={48} className="mx-auto mb-4 opacity-30" />
                  <p>No strategies available</p>
                  <p className="text-sm mt-2">Run synthesis to generate global strategies</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Latest Synthesis Tab */}
        <TabsContent value="synthesis" data-testid="synthesis-content">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Brain size={20} className="text-purple-400" />
                Latest Synthesis Results
              </CardTitle>
            </CardHeader>
            <CardContent>
              {synthesisResult ? (
                <div className="space-y-6">
                  {/* Collective Strategy */}
                  <div className="bg-gradient-to-br from-purple-900/20 to-blue-900/20 p-6 rounded-lg border border-purple-500/30">
                    <h3 className="text-xl font-bold text-purple-400 mb-3">
                      {synthesisResult.collective_strategy.strategy_archetype}
                    </h3>
                    <p className="text-slate-300 mb-4">
                      {synthesisResult.collective_strategy.description}
                    </p>
                    
                    <div className="grid grid-cols-3 gap-4 mb-4">
                      <div className="bg-slate-900/50 p-3 rounded">
                        <div className="text-xs text-slate-400">Confidence</div>
                        <div className="text-lg font-bold text-white">
                          {(synthesisResult.collective_strategy.confidence_score * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-slate-900/50 p-3 rounded">
                        <div className="text-xs text-slate-400">Alignment</div>
                        <div className="text-lg font-bold text-white">
                          {(synthesisResult.collective_strategy.alignment_score * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-slate-900/50 p-3 rounded">
                        <div className="text-xs text-slate-400">Contributing Agents</div>
                        <div className="text-lg font-bold text-white">
                          {synthesisResult.collective_strategy.contributing_agents.length}
                        </div>
                      </div>
                    </div>

                    {synthesisResult.collective_strategy.strategic_recommendations && (
                      <div>
                        <h4 className="text-sm font-medium text-slate-300 mb-2">Strategic Recommendations:</h4>
                        <ul className="space-y-2">
                          {synthesisResult.collective_strategy.strategic_recommendations.map((rec, idx) => (
                            <li key={idx} className="text-sm text-slate-300 flex items-start gap-2 bg-slate-900/30 p-3 rounded">
                              <CheckCircle2 size={16} className="text-green-400 mt-0.5 flex-shrink-0" />
                              <span>{rec}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {synthesisResult.collective_strategy.performance_impact && (
                      <div className="mt-4 p-3 bg-blue-900/20 rounded border border-blue-500/30">
                        <div className="text-xs text-blue-400 font-medium mb-1">Performance Impact:</div>
                        <div className="text-sm text-slate-300">
                          {synthesisResult.collective_strategy.performance_impact}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Aggregated Insights */}
                  <div>
                    <h4 className="text-slate-300 font-medium mb-3">Data Sources</h4>
                    <div className="grid grid-cols-4 gap-3">
                      <div className="bg-slate-900/50 p-3 rounded text-center">
                        <div className="text-xs text-slate-400">Memory</div>
                        <div className="text-xl font-bold text-white">
                          {synthesisResult.data_sources.memory_count}
                        </div>
                      </div>
                      <div className="bg-slate-900/50 p-3 rounded text-center">
                        <div className="text-xs text-slate-400">Trust</div>
                        <div className="text-xl font-bold text-white">
                          {synthesisResult.data_sources.trust_count}
                        </div>
                      </div>
                      <div className="bg-slate-900/50 p-3 rounded text-center">
                        <div className="text-xs text-slate-400">Arbitration</div>
                        <div className="text-xl font-bold text-white">
                          {synthesisResult.data_sources.arbitration_count}
                        </div>
                      </div>
                      <div className="bg-slate-900/50 p-3 rounded text-center">
                        <div className="text-xs text-slate-400">Forecast</div>
                        <div className="text-xl font-bold text-white">
                          {synthesisResult.data_sources.forecast_available ? '✓' : '✗'}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Cross-System Patterns */}
                  {synthesisResult.aggregated_insights.cross_system_patterns.length > 0 && (
                    <div>
                      <h4 className="text-slate-300 font-medium mb-3">Cross-System Patterns Detected</h4>
                      <ul className="space-y-2">
                        {synthesisResult.aggregated_insights.cross_system_patterns.map((pattern, idx) => (
                          <li key={idx} className="text-sm text-slate-300 flex items-start gap-2 bg-slate-900/30 p-3 rounded">
                            <AlertCircle size={16} className="text-yellow-400 mt-0.5 flex-shrink-0" />
                            <span>{pattern}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <div className="text-xs text-slate-500 text-center">
                    Synthesis completed at {new Date(synthesisResult.timestamp).toLocaleString()}
                  </div>
                </div>
              ) : (
                <div className="text-center text-slate-400 py-12">
                  <Brain size={48} className="mx-auto mb-4 opacity-30" />
                  <p>No synthesis results available</p>
                  <p className="text-sm mt-2">Click "Run Synthesis" to generate collective intelligence</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Meta-Optimization Tab (Step 24) */}
        <TabsContent value="meta-optimization" data-testid="meta-optimization-content">
          <MetaOptimizationPanel />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default CollectiveIntelligencePanel;
