/**
 * Cognitive Synthesis Dashboard (Step 27)
 * 
 * Real-time visualization of autonomous cognitive synthesis and value preservation.
 * Displays multilayer insights, cognitive patterns, value drift monitoring, and reflection reports.
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { ScrollArea } from './ui/scroll-area';
import { Brain, TrendingUp, Shield, Activity, Sparkles, AlertTriangle, CheckCircle2, XCircle } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

export default function CognitiveSynthesisPanel() {
  const [status, setStatus] = useState(null);
  const [history, setHistory] = useState([]);
  const [values, setValues] = useState([]);
  const [patterns, setPatterns] = useState([]);
  const [insights, setInsights] = useState([]);
  const [loading, setLoading] = useState(false);
  const [synthesisLogs, setSynthesisLogs] = useState([]);
  const [selectedTab, setSelectedTab] = useState('overview');

  useEffect(() => {
    fetchStatus();
    fetchHistory();
    fetchValues();
    fetchPatterns();
    fetchInsights();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(() => {
      fetchStatus();
      fetchValues();
    }, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/llm/cognitive/status`);
      const data = await res.json();
      if (data.success) {
        setStatus(data);
      }
    } catch (error) {
      console.error('Error fetching cognitive status:', error);
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/llm/cognitive/history?limit=15`);
      const data = await res.json();
      if (data.success) {
        setHistory(data.history);
      }
    } catch (error) {
      console.error('Error fetching synthesis history:', error);
    }
  };

  const fetchValues = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/llm/cognitive/values?limit=30`);
      const data = await res.json();
      if (data.success) {
        setValues(data.current_values || []);
      }
    } catch (error) {
      console.error('Error fetching value states:', error);
    }
  };

  const fetchPatterns = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/llm/cognitive/patterns?limit=10`);
      const data = await res.json();
      if (data.success) {
        setPatterns(data.patterns);
      }
    } catch (error) {
      console.error('Error fetching cognitive patterns:', error);
    }
  };

  const fetchInsights = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/llm/cognitive/insights`);
      const data = await res.json();
      if (data.success) {
        setInsights(data.insights || []);
      }
    } catch (error) {
      console.error('Error fetching insights:', error);
    }
  };

  const runSynthesis = async () => {
    setLoading(true);
    setSynthesisLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), message: 'Starting synthesis cycle...' }]);
    
    try {
      const res = await fetch(`${BACKEND_URL}/api/llm/cognitive/synthesize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trigger: 'manual' })
      });
      
      const data = await res.json();
      
      if (data.success) {
        setSynthesisLogs(prev => [
          ...prev,
          { time: new Date().toLocaleTimeString(), message: `✓ Cycle ${data.cycle_id.substring(0, 8)} complete` },
          { time: new Date().toLocaleTimeString(), message: `Coherence: ${data.cognitive_coherence_index.toFixed(2)}, Integrity: ${data.value_integrity_score.toFixed(1)}%` },
          { time: new Date().toLocaleTimeString(), message: `Patterns: ${data.patterns_detected}, Drift: ${data.drift_status}` },
          { time: new Date().toLocaleTimeString(), message: `Reflection: ${data.reflection_summary}` }
        ]);
        
        // Refresh data
        await fetchStatus();
        await fetchHistory();
        await fetchValues();
        await fetchPatterns();
        await fetchInsights();
      } else {
        setSynthesisLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), message: '✗ Synthesis failed' }]);
      }
    } catch (error) {
      console.error('Error running synthesis:', error);
      setSynthesisLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), message: `✗ Error: ${error.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const getValueColor = (drift_direction) => {
    if (drift_direction === 'positive') return 'text-green-600';
    if (drift_direction === 'negative') return 'text-red-600';
    return 'text-gray-600';
  };

  const getDriftBadgeColor = (status) => {
    if (status === 'healthy') return 'bg-green-100 text-green-800';
    if (status === 'moderate') return 'bg-yellow-100 text-yellow-800';
    if (status === 'critical') return 'bg-red-100 text-red-800';
    return 'bg-gray-100 text-gray-800';
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="space-y-6">
      {/* Header with Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Cognitive Coherence</p>
                <p className="text-2xl font-bold" data-testid="cognitive-coherence-value">
                  {status?.cognitive_coherence_index?.toFixed(2) || '0.00'}
                </p>
              </div>
              <Brain className="h-8 w-8 text-purple-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Value Integrity</p>
                <p className="text-2xl font-bold" data-testid="value-integrity-value">
                  {status?.value_integrity_score?.toFixed(1) || '0.0'}%
                </p>
              </div>
              <Shield className="h-8 w-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Active Patterns</p>
                <p className="text-2xl font-bold">{status?.active_patterns || 0}</p>
              </div>
              <Sparkles className="h-8 w-8 text-amber-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Drift Status</p>
                <Badge className={getDriftBadgeColor(status?.drift_status || 'unknown')}>
                  {status?.drift_status || 'Unknown'}
                </Badge>
              </div>
              <Activity className="h-8 w-8 text-green-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Dashboard */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-6 w-6" />
                Cognitive Synthesis Dashboard
              </CardTitle>
              <CardDescription>
                Autonomous cognitive synthesis integrating insights from all system layers
              </CardDescription>
            </div>
            <Button 
              onClick={runSynthesis} 
              disabled={loading}
              data-testid="run-synthesis-button"
            >
              {loading ? 'Synthesizing...' : 'Run Synthesis'}
            </Button>
          </div>
        </CardHeader>

        <CardContent>
          <Tabs value={selectedTab} onValueChange={setSelectedTab}>
            <TabsList className="grid w-full grid-cols-5">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="insights">Insight Matrix</TabsTrigger>
              <TabsTrigger value="values">Value Drift</TabsTrigger>
              <TabsTrigger value="history">History</TabsTrigger>
              <TabsTrigger value="reports">Reflections</TabsTrigger>
            </TabsList>

            {/* Overview Tab */}
            <TabsContent value="overview" className="space-y-4" data-testid="overview-tab">
              <Card>
                <CardHeader>
                  <CardTitle>System Status</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Last Synthesis:</span>
                      <span className="text-sm font-medium">{formatTimestamp(status?.last_synthesis)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">System Status:</span>
                      <Badge variant="outline">{status?.system_status || 'Unknown'}</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Cognitive Coherence Index:</span>
                      <span className="text-sm font-bold text-purple-600">
                        {status?.cognitive_coherence_index?.toFixed(3) || '0.000'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Value Integrity Score:</span>
                      <span className="text-sm font-bold text-blue-600">
                        {status?.value_integrity_score?.toFixed(1) || '0.0'}%
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Recent Patterns</CardTitle>
                </CardHeader>
                <CardContent>
                  {status?.latest_patterns?.length > 0 ? (
                    <div className="space-y-2">
                      {status.latest_patterns.map((pattern, idx) => (
                        <div key={idx} className="flex items-center gap-2 p-2 bg-purple-50 rounded">
                          <Sparkles className="h-4 w-4 text-purple-600" />
                          <span className="text-sm font-medium">{pattern}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">No patterns detected yet</p>
                  )}
                </CardContent>
              </Card>

              {/* Live Synthesis Logs */}
              {synthesisLogs.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Synthesis Log</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ScrollArea className="h-48">
                      <div className="space-y-1 font-mono text-xs">
                        {synthesisLogs.map((log, idx) => (
                          <div key={idx} className="flex gap-2">
                            <span className="text-gray-500">[{log.time}]</span>
                            <span>{log.message}</span>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            {/* Insight Matrix Tab */}
            <TabsContent value="insights" className="space-y-4" data-testid="insights-tab">
              <Card>
                <CardHeader>
                  <CardTitle>Multilayer Insights</CardTitle>
                  <CardDescription>Cross-layer knowledge integration from all system components</CardDescription>
                </CardHeader>
                <CardContent>
                  {insights.length > 0 ? (
                    <div className="space-y-3">
                      {insights.map((insight, idx) => (
                        <div key={idx} className="p-3 border rounded-lg hover:bg-gray-50">
                          <div className="flex items-start justify-between mb-2">
                            <Badge variant="outline">{insight.layer_name}</Badge>
                            <span className="text-xs text-gray-500">
                              Confidence: {(insight.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                          <p className="text-sm mb-2">{insight.content}</p>
                          <div className="flex flex-wrap gap-2">
                            {Object.entries(insight.metrics || {}).map(([key, value]) => (
                              <span key={key} className="text-xs bg-gray-100 px-2 py-1 rounded">
                                {key}: {typeof value === 'number' ? value.toFixed(2) : value}
                              </span>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">No insights available. Run synthesis to collect insights.</p>
                  )}
                </CardContent>
              </Card>

              {/* Cognitive Patterns */}
              <Card>
                <CardHeader>
                  <CardTitle>Detected Cognitive Patterns</CardTitle>
                  <CardDescription>Emergent meta-patterns across system layers</CardDescription>
                </CardHeader>
                <CardContent>
                  {patterns.length > 0 ? (
                    <div className="space-y-3">
                      {patterns.map((pattern, idx) => (
                        <div key={idx} className="p-3 border rounded-lg bg-purple-50">
                          <div className="flex items-start justify-between mb-2">
                            <h4 className="font-semibold text-sm">{pattern.pattern_name}</h4>
                            <Badge className="bg-purple-100 text-purple-800">
                              Strength: {(pattern.strength * 100).toFixed(0)}%
                            </Badge>
                          </div>
                          <p className="text-sm text-gray-700 mb-2">{pattern.description}</p>
                          <div className="flex flex-wrap gap-2">
                            <span className="text-xs text-gray-600">
                              Layers: {pattern.layers_involved?.join(', ')}
                            </span>
                            <span className="text-xs text-gray-600">
                              Emerged: {pattern.emergence_count} times
                            </span>
                          </div>
                          <div className="mt-2 flex flex-wrap gap-1">
                            {pattern.impact_areas?.map((area, i) => (
                              <Badge key={i} variant="secondary" className="text-xs">
                                {area}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">No cognitive patterns detected yet</p>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Value Drift Monitor Tab */}
            <TabsContent value="values" className="space-y-4" data-testid="values-tab">
              <Card>
                <CardHeader>
                  <CardTitle>Value Preservation Monitoring</CardTitle>
                  <CardDescription>Track ethical and strategic value drift over time</CardDescription>
                </CardHeader>
                <CardContent>
                  {values.length > 0 ? (
                    <div className="space-y-4">
                      {values.map((value, idx) => (
                        <div key={idx} className="p-4 border rounded-lg">
                          <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center gap-2">
                              <h4 className="font-semibold capitalize">{value.value_name}</h4>
                              <Badge variant="outline" className="text-xs">{value.category}</Badge>
                            </div>
                            <div className={`flex items-center gap-1 ${getValueColor(value.drift_direction)}`}>
                              {value.drift_direction === 'positive' && <TrendingUp className="h-4 w-4" />}
                              {value.drift_direction === 'negative' && <AlertTriangle className="h-4 w-4" />}
                              {value.drift_direction === 'stable' && <CheckCircle2 className="h-4 w-4" />}
                              <span className="text-sm font-medium capitalize">{value.drift_direction}</span>
                            </div>
                          </div>

                          {/* Progress Bar */}
                          <div className="space-y-1">
                            <div className="flex justify-between text-xs text-gray-600">
                              <span>Current: {value.current_score?.toFixed(1)}</span>
                              <span>Target: {value.target_score?.toFixed(1)}</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div
                                className={`h-2 rounded-full ${
                                  value.drift_direction === 'negative' ? 'bg-red-500' :
                                  value.drift_direction === 'positive' ? 'bg-green-500' :
                                  'bg-blue-500'
                                }`}
                                style={{ width: `${Math.min(100, (value.current_score / value.target_score) * 100)}%` }}
                              />
                            </div>
                          </div>

                          <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                            <div>
                              <span className="text-gray-600">Drift:</span>
                              <span className={`ml-1 font-medium ${getValueColor(value.drift_direction)}`}>
                                {value.drift_amount >= 0 ? '+' : ''}{value.drift_amount?.toFixed(2)}
                              </span>
                            </div>
                            <div>
                              <span className="text-gray-600">Stability:</span>
                              <span className="ml-1 font-medium">
                                {(value.stability_index * 100).toFixed(0)}%
                              </span>
                            </div>
                            <div className="text-right text-gray-500">
                              {new Date(value.last_updated).toLocaleDateString()}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">No value data available. Run synthesis to track values.</p>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* History Tab */}
            <TabsContent value="history" className="space-y-4" data-testid="history-tab">
              <Card>
                <CardHeader>
                  <CardTitle>Synthesis Cycle History</CardTitle>
                  <CardDescription>Historical record of cognitive synthesis cycles</CardDescription>
                </CardHeader>
                <CardContent>
                  {history.length > 0 ? (
                    <div className="space-y-2">
                      <div className="grid grid-cols-7 gap-2 text-xs font-semibold text-gray-600 pb-2 border-b">
                        <div>Cycle ID</div>
                        <div>Timestamp</div>
                        <div>Trigger</div>
                        <div>Coherence</div>
                        <div>Integrity</div>
                        <div>Patterns</div>
                        <div>Drift</div>
                      </div>
                      {history.map((cycle, idx) => (
                        <div key={idx} className="grid grid-cols-7 gap-2 text-xs py-2 hover:bg-gray-50 rounded">
                          <div className="font-mono">{cycle.cycle_id?.substring(0, 8)}</div>
                          <div>{new Date(cycle.timestamp).toLocaleString()}</div>
                          <div>
                            <Badge variant="outline" className="text-xs">{cycle.trigger}</Badge>
                          </div>
                          <div className="font-semibold text-purple-600">
                            {cycle.cognitive_coherence_index?.toFixed(2)}
                          </div>
                          <div className="font-semibold text-blue-600">
                            {cycle.value_integrity_score?.toFixed(1)}%
                          </div>
                          <div>{cycle.patterns_detected}</div>
                          <div>
                            <Badge className={getDriftBadgeColor(cycle.drift_status)}>
                              {cycle.drift_status}
                            </Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">No synthesis history available</p>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Reflection Reports Tab */}
            <TabsContent value="reports" className="space-y-4" data-testid="reports-tab">
              <Card>
                <CardHeader>
                  <CardTitle>Synthesis Reflection Reports</CardTitle>
                  <CardDescription>Meta-cognitive reflections and recommendations</CardDescription>
                </CardHeader>
                <CardContent>
                  {history.length > 0 ? (
                    <ScrollArea className="h-96">
                      <div className="space-y-4">
                        {history.slice(0, 5).map((cycle, idx) => (
                          <div key={idx} className="p-4 border rounded-lg bg-gray-50">
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-xs font-mono text-gray-500">
                                Cycle {cycle.cycle_id?.substring(0, 8)}
                              </span>
                              <span className="text-xs text-gray-500">
                                {new Date(cycle.timestamp).toLocaleString()}
                              </span>
                            </div>
                            
                            <div className="mb-3">
                              <h4 className="text-sm font-semibold mb-1">Reflection Summary:</h4>
                              <p className="text-sm text-gray-700 leading-relaxed">
                                {cycle.reflection_summary || 'No reflection available'}
                              </p>
                            </div>

                            <div className="flex gap-4 text-xs mb-3">
                              <div>
                                <span className="text-gray-600">Coherence:</span>
                                <span className="ml-1 font-semibold text-purple-600">
                                  {cycle.cognitive_coherence_index?.toFixed(2)}
                                </span>
                              </div>
                              <div>
                                <span className="text-gray-600">Integrity:</span>
                                <span className="ml-1 font-semibold text-blue-600">
                                  {cycle.value_integrity_score?.toFixed(1)}%
                                </span>
                              </div>
                              <div>
                                <span className="text-gray-600">Patterns:</span>
                                <span className="ml-1 font-semibold">{cycle.patterns_detected}</span>
                              </div>
                            </div>

                            {cycle.recommendations_count > 0 && (
                              <div>
                                <h4 className="text-xs font-semibold text-gray-600 mb-1">Recommendations:</h4>
                                <Badge variant="secondary" className="text-xs">
                                  {cycle.recommendations_count} recommendation(s)
                                </Badge>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  ) : (
                    <p className="text-sm text-gray-500">No reflection reports available. Run synthesis to generate reports.</p>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
