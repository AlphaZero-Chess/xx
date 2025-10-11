import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import {
  Brain,
  Zap,
  TrendingUp,
  TrendingDown,
  Activity,
  CheckCircle2,
  AlertCircle,
  Clock,
  Settings,
  BarChart3,
  Play,
  RefreshCw
} from 'lucide-react';
import { toast } from 'sonner';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const MetaOptimizationPanel = () => {
  const [loading, setLoading] = useState(false);
  const [optimizing, setOptimizing] = useState(false);
  const [status, setStatus] = useState(null);
  const [history, setHistory] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [selectedCycle, setSelectedCycle] = useState(null);

  useEffect(() => {
    loadData();
    // Auto-refresh every 30 seconds
    const interval = setInterval(() => {
      loadStatus();
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        loadStatus(),
        loadHistory(),
        loadMetrics()
      ]);
    } catch (error) {
      console.error('Error loading meta-optimization data:', error);
      toast.error('Failed to load optimization data');
    } finally {
      setLoading(false);
    }
  };

  const loadStatus = async () => {
    try {
      const response = await axios.get(`${API}/llm/meta/status`);
      setStatus(response.data);
    } catch (error) {
      console.error('Error loading status:', error);
    }
  };

  const loadHistory = async () => {
    try {
      const response = await axios.get(`${API}/llm/meta/history`, {
        params: { limit: 20 }
      });
      setHistory(response.data.cycles || []);
    } catch (error) {
      console.error('Error loading history:', error);
    }
  };

  const loadMetrics = async () => {
    try {
      const response = await axios.get(`${API}/llm/meta/metrics`, {
        params: { lookback_hours: 24 }
      });
      setMetrics(response.data.metrics);
    } catch (error) {
      console.error('Error loading metrics:', error);
    }
  };

  const runOptimization = async () => {
    setOptimizing(true);
    try {
      toast.info('Starting meta-optimization cycle...');
      const response = await axios.post(`${API}/llm/meta/optimize`);
      
      toast.success(
        response.data.applied 
          ? 'Optimization applied successfully!' 
          : 'Optimization pending approval'
      );
      
      // Reload data
      await loadData();
    } catch (error) {
      console.error('Error running optimization:', error);
      toast.error('Optimization failed');
    } finally {
      setOptimizing(false);
    }
  };

  const approveCycle = async (cycleId) => {
    try {
      toast.info('Approving optimization cycle...');
      const response = await axios.post(`${API}/llm/meta/approve/${cycleId}`);
      
      if (response.data.success) {
        toast.success('Optimization cycle approved and applied!');
        await loadData();
      } else {
        toast.error(response.data.message || 'Approval failed');
      }
    } catch (error) {
      console.error('Error approving cycle:', error);
      toast.error('Failed to approve cycle');
    }
  };

  const getHealthColor = (score) => {
    if (score >= 85) return 'text-green-500';
    if (score >= 70) return 'text-blue-500';
    if (score >= 50) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getHealthStatus = (score) => {
    if (score >= 85) return 'Excellent';
    if (score >= 70) return 'Good';
    if (score >= 50) return 'Moderate';
    return 'Critical';
  };

  const getHealthBadgeVariant = (score) => {
    if (score >= 85) return 'default';
    if (score >= 70) return 'secondary';
    if (score >= 50) return 'outline';
    return 'destructive';
  };

  // Prepare trend data for charts
  const prepareTrendData = () => {
    if (!history || history.length === 0) return [];
    
    return history.slice().reverse().map((cycle, index) => ({
      cycle: index + 1,
      health: cycle.system_health_score,
      alignment: cycle.performance_delta?.alignment || 0,
      variance: cycle.performance_delta?.variance || 0
    }));
  };

  if (loading && !status) {
    return (
      <div className="flex items-center justify-center p-12">
        <div className="text-slate-300">Loading meta-optimization system...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="meta-optimization-panel">
      {/* System Health Overview */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-white flex items-center gap-2">
                <Brain className="text-purple-500" size={24} />
                Meta-Reasoning & Self-Optimization
              </CardTitle>
              <CardDescription className="text-slate-400 mt-2">
                Autonomous system performance analysis and optimization
              </CardDescription>
            </div>
            <Button
              onClick={runOptimization}
              disabled={optimizing}
              className="bg-purple-600 hover:bg-purple-700"
              data-testid="run-optimization-btn"
            >
              {optimizing ? (
                <>
                  <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                  Optimizing...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Run Self-Optimization
                </>
              )}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {/* System Health Score */}
            <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-slate-400 text-sm">System Health</span>
                <Activity className={getHealthColor(status?.system_health_score || 75)} size={20} />
              </div>
              <div className={`text-3xl font-bold ${getHealthColor(status?.system_health_score || 75)}`}>
                {status?.system_health_score || 75}
              </div>
              <Badge variant={getHealthBadgeVariant(status?.system_health_score || 75)} className="mt-2">
                {getHealthStatus(status?.system_health_score || 75)}
              </Badge>
            </div>

            {/* Last Optimization */}
            <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-slate-400 text-sm">Last Optimization</span>
                <Clock className="text-blue-400" size={20} />
              </div>
              <div className="text-white text-sm">
                {status?.last_optimization 
                  ? new Date(status.last_optimization).toLocaleString()
                  : 'Never'
                }
              </div>
              {status?.time_since_last_seconds && (
                <div className="text-slate-500 text-xs mt-1">
                  {Math.floor(status.time_since_last_seconds / 60)} minutes ago
                </div>
              )}
            </div>

            {/* Recent Cycles */}
            <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-slate-400 text-sm">Recent Cycles</span>
                <BarChart3 className="text-purple-400" size={20} />
              </div>
              <div className="text-3xl font-bold text-white">
                {status?.recent_cycles_count || 0}
              </div>
              <div className="text-slate-500 text-xs mt-1">
                Last 24 hours
              </div>
            </div>

            {/* Optimization Interval */}
            <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-slate-400 text-sm">Auto Interval</span>
                <Settings className="text-green-400" size={20} />
              </div>
              <div className="text-3xl font-bold text-white">
                {status?.optimization_interval ? (status.optimization_interval / 3600).toFixed(0) : 1}h
              </div>
              <div className="text-slate-500 text-xs mt-1">
                Periodic optimization
              </div>
            </div>
          </div>

          {/* Metrics Summary */}
          {status?.metrics_summary && (
            <div className="mt-6">
              <h3 className="text-white font-semibold mb-4">Subsystem Metrics</h3>
              <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                {Object.entries(status.metrics_summary).map(([subsystem, metric]) => (
                  <div key={subsystem} className="bg-slate-900/30 rounded-lg p-3 border border-slate-700">
                    <div className="text-slate-400 text-xs uppercase mb-2">{subsystem}</div>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500 text-xs">Alignment</span>
                        <span className="text-white text-sm font-medium">
                          {metric.alignment?.toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500 text-xs">Variance</span>
                        <span className="text-white text-sm font-medium">
                          {metric.variance?.toFixed(3)}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500 text-xs">Stability</span>
                        <span className="text-white text-sm font-medium">
                          {(metric.stability * 100)?.toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Tabs for History and Trends */}
      <Tabs defaultValue="history" className="w-full">
        <TabsList className="grid w-full grid-cols-2 bg-slate-800/50">
          <TabsTrigger value="history">Optimization History</TabsTrigger>
          <TabsTrigger value="trends">Performance Trends</TabsTrigger>
        </TabsList>

        {/* Optimization History Tab */}
        <TabsContent value="history" className="space-y-4">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Recent Optimization Cycles</CardTitle>
              <CardDescription className="text-slate-400">
                Parameter adjustments and system impact
              </CardDescription>
            </CardHeader>
            <CardContent>
              {history.length === 0 ? (
                <div className="text-center py-12 text-slate-400">
                  No optimization cycles recorded yet. Run your first optimization to begin.
                </div>
              ) : (
                <div className="space-y-4">
                  {history.map((cycle) => (
                    <div
                      key={cycle.cycle_id}
                      className="bg-slate-900/50 rounded-lg p-4 border border-slate-700 hover:border-slate-600 transition-colors cursor-pointer"
                      onClick={() => setSelectedCycle(cycle)}
                      data-testid={`cycle-${cycle.cycle_id}`}
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <Badge variant="outline" className="text-xs">
                              {cycle.trigger}
                            </Badge>
                            {cycle.applied ? (
                              <Badge variant="default" className="text-xs bg-green-600">
                                <CheckCircle2 size={12} className="mr-1" />
                                Applied
                              </Badge>
                            ) : cycle.approval_required ? (
                              <Badge variant="destructive" className="text-xs">
                                <AlertCircle size={12} className="mr-1" />
                                Pending Approval
                              </Badge>
                            ) : (
                              <Badge variant="secondary" className="text-xs">
                                Simulated
                              </Badge>
                            )}
                          </div>
                          <div className="text-slate-400 text-sm">
                            {new Date(cycle.timestamp).toLocaleString()}
                          </div>
                        </div>
                        <div className="text-right">
                          <div className={`text-2xl font-bold ${getHealthColor(cycle.system_health_score)}`}>
                            {cycle.system_health_score}
                          </div>
                          <div className="text-slate-500 text-xs">Health Score</div>
                        </div>
                      </div>

                      <div className="text-slate-300 text-sm mb-3">
                        {cycle.reflection_summary}
                      </div>

                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <div className="text-slate-400 text-xs">
                            {cycle.adjustments_count} adjustment{cycle.adjustments_count !== 1 ? 's' : ''}
                          </div>
                          {Object.keys(cycle.performance_delta || {}).length > 0 && (
                            <div className="flex items-center gap-2">
                              {Object.entries(cycle.performance_delta).map(([metric, delta]) => (
                                <div key={metric} className="flex items-center gap-1 text-xs">
                                  {delta > 0 ? (
                                    <TrendingUp size={14} className="text-green-500" />
                                  ) : (
                                    <TrendingDown size={14} className="text-red-500" />
                                  )}
                                  <span className="text-slate-400">
                                    {metric}: {delta > 0 ? '+' : ''}{delta.toFixed(1)}%
                                  </span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                        {cycle.approval_required && !cycle.applied && (
                          <Button
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              approveCycle(cycle.cycle_id);
                            }}
                            className="bg-green-600 hover:bg-green-700"
                          >
                            Approve & Apply
                          </Button>
                        )}
                      </div>

                      {/* Show adjustments if expanded */}
                      {selectedCycle?.cycle_id === cycle.cycle_id && cycle.adjustments && (
                        <div className="mt-4 pt-4 border-t border-slate-700">
                          <h4 className="text-white text-sm font-semibold mb-3">Parameter Adjustments</h4>
                          <div className="space-y-2">
                            {cycle.adjustments.map((adj, idx) => (
                              <div key={idx} className="bg-slate-900/30 rounded p-3">
                                <div className="flex items-center justify-between mb-2">
                                  <span className="text-white font-medium">{adj.parameter_name}</span>
                                  <Badge variant={adj.is_critical ? 'destructive' : 'outline'}>
                                    {adj.is_critical ? 'Critical' : 'Non-critical'}
                                  </Badge>
                                </div>
                                <div className="grid grid-cols-3 gap-4 text-sm mb-2">
                                  <div>
                                    <span className="text-slate-500">Current:</span>
                                    <span className="text-white ml-2">{adj.current_value?.toFixed(2)}</span>
                                  </div>
                                  <div>
                                    <span className="text-slate-500">Proposed:</span>
                                    <span className="text-white ml-2">{adj.proposed_value?.toFixed(2)}</span>
                                  </div>
                                  <div>
                                    <span className="text-slate-500">Change:</span>
                                    <span className={adj.change_percent > 0 ? 'text-green-400 ml-2' : 'text-red-400 ml-2'}>
                                      {adj.change_percent > 0 ? '+' : ''}{adj.change_percent?.toFixed(1)}%
                                    </span>
                                  </div>
                                </div>
                                <div className="text-slate-400 text-xs">{adj.reason}</div>
                                <div className="mt-2">
                                  <Progress 
                                    value={adj.confidence * 100} 
                                    className="h-1"
                                  />
                                  <div className="text-slate-500 text-xs mt-1">
                                    Confidence: {(adj.confidence * 100).toFixed(0)}%
                                  </div>
                                </div>
                              </div>
                            ))}
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

        {/* Performance Trends Tab */}
        <TabsContent value="trends">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Performance Trends</CardTitle>
              <CardDescription className="text-slate-400">
                System health and optimization impact over time
              </CardDescription>
            </CardHeader>
            <CardContent>
              {history.length === 0 ? (
                <div className="text-center py-12 text-slate-400">
                  No trend data available. Run optimization cycles to see trends.
                </div>
              ) : (
                <div className="space-y-6">
                  {/* System Health Trend */}
                  <div>
                    <h3 className="text-white font-semibold mb-4">System Health Score</h3>
                    <ResponsiveContainer width="100%" height={250}>
                      <LineChart data={prepareTrendData()}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <XAxis 
                          dataKey="cycle" 
                          stroke="#94a3b8"
                          label={{ value: 'Optimization Cycle', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
                        />
                        <YAxis 
                          stroke="#94a3b8"
                          domain={[0, 100]}
                          label={{ value: 'Health Score', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                        />
                        <RechartsTooltip 
                          contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                          labelStyle={{ color: '#e2e8f0' }}
                        />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="health" 
                          stroke="#8b5cf6" 
                          strokeWidth={2}
                          name="Health Score"
                          dot={{ fill: '#8b5cf6' }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Performance Delta Trends */}
                  <div>
                    <h3 className="text-white font-semibold mb-4">Performance Changes</h3>
                    <ResponsiveContainer width="100%" height={250}>
                      <LineChart data={prepareTrendData()}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <XAxis 
                          dataKey="cycle" 
                          stroke="#94a3b8"
                          label={{ value: 'Optimization Cycle', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
                        />
                        <YAxis 
                          stroke="#94a3b8"
                          label={{ value: 'Change (%)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                        />
                        <RechartsTooltip 
                          contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                          labelStyle={{ color: '#e2e8f0' }}
                        />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="alignment" 
                          stroke="#10b981" 
                          strokeWidth={2}
                          name="Alignment Δ"
                          dot={{ fill: '#10b981' }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="variance" 
                          stroke="#ef4444" 
                          strokeWidth={2}
                          name="Variance Δ"
                          dot={{ fill: '#ef4444' }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default MetaOptimizationPanel;
