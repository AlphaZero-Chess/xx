/**
 * Autonomous Creativity & Meta-Strategic Synthesis Panel (Step 29)
 * 
 * Features:
 * - Creative strategy generation across all game phases
 * - Meta-strategic synthesis and visualization
 * - Creativity analytics and health monitoring
 * - Ethical audit trail
 * - History management and export
 */

import React, { useState, useEffect } from 'react';
import { 
  Sparkles, Brain, TrendingUp, Shield, History, 
  Download, Play, AlertTriangle, CheckCircle, XCircle,
  Lightbulb, Target, BarChart3, FileText
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

const CreativeSynthesisPanel = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [isGenerating, setIsGenerating] = useState(false);
  const [creativityStatus, setCreativityStatus] = useState(null);
  const [strategies, setStrategies] = useState([]);
  const [metaStrategy, setMetaStrategy] = useState(null);
  const [history, setHistory] = useState([]);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch creativity status
  const fetchStatus = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/llm/creativity/status`);
      const data = await res.json();
      if (data.success) {
        setCreativityStatus(data);
      }
    } catch (err) {
      console.error('Error fetching status:', err);
    }
  };

  // Fetch history
  const fetchHistory = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/llm/creativity/history?limit=20`);
      const data = await res.json();
      if (data.success) {
        setHistory(data.strategies || []);
      }
    } catch (err) {
      console.error('Error fetching history:', err);
    }
  };

  // Fetch report
  const fetchReport = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/llm/creativity/report`);
      const data = await res.json();
      if (data.success) {
        setReport(data);
      }
    } catch (err) {
      console.error('Error fetching report:', err);
      setError('Failed to load creativity report');
    } finally {
      setLoading(false);
    }
  };

  // Generate creative strategies
  const generateStrategies = async () => {
    setIsGenerating(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/api/llm/creativity/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ count: 1, use_patterns: true })
      });
      const data = await res.json();
      
      if (data.success) {
        setStrategies(data.strategies || []);
        await fetchStatus();
        await fetchHistory();
      } else {
        setError('Failed to generate strategies');
      }
    } catch (err) {
      console.error('Error generating strategies:', err);
      setError('Error generating creative strategies');
    } finally {
      setIsGenerating(false);
    }
  };

  // Synthesize meta-strategy
  const synthesizeMeta = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/llm/creativity/synthesize-meta`, {
        method: 'POST'
      });
      const data = await res.json();
      
      if (data.success) {
        setMetaStrategy(data);
      }
    } catch (err) {
      console.error('Error synthesizing meta:', err);
    } finally {
      setLoading(false);
    }
  };

  // Initial load
  useEffect(() => {
    fetchStatus();
    fetchHistory();
    fetchReport();
    
    // Auto-refresh every 60 seconds
    const interval = setInterval(() => {
      fetchStatus();
    }, 60000);
    
    return () => clearInterval(interval);
  }, []);

  // Get health color
  const getHealthColor = (health) => {
    if (health >= 0.85) return 'text-green-600';
    if (health >= 0.70) return 'text-blue-600';
    if (health >= 0.50) return 'text-yellow-600';
    return 'text-red-600';
  };

  // Get health badge variant
  const getHealthBadge = (status) => {
    const variants = {
      'excellent': 'bg-green-100 text-green-800',
      'good': 'bg-blue-100 text-blue-800',
      'moderate': 'bg-yellow-100 text-yellow-800',
      'poor': 'bg-red-100 text-red-800'
    };
    return variants[status] || variants['moderate'];
  };

  // Export report
  const exportReport = () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `creativity_report_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
  };

  return (
    <div className="w-full space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold flex items-center gap-2">
            <Sparkles className="w-8 h-8 text-purple-600" />
            Autonomous Creativity & Meta-Strategic Synthesis
          </h2>
          <p className="text-gray-600 mt-1">
            Step 29: Novel strategy generation with ethical guardrails
          </p>
        </div>
        <Button 
          onClick={generateStrategies} 
          disabled={isGenerating}
          className="bg-purple-600 hover:bg-purple-700"
          data-testid="generate-creative-strategies-btn"
        >
          {isGenerating ? (
            <>⏳ Generating...</>
          ) : (
            <><Play className="w-4 h-4 mr-2" /> Generate Creative Strategies</>
          )}
        </Button>
      </div>

      {/* Status Cards */}
      {creativityStatus && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-600">Creativity Health</CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`text-3xl font-bold ${getHealthColor(creativityStatus.creativity_health)}`}>
                {(creativityStatus.creativity_health * 100).toFixed(0)}%
              </div>
              <Badge className={`mt-2 ${getHealthBadge(creativityStatus.health_status)}`}>
                {creativityStatus.health_status}
              </Badge>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-600">Ideas Generated</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-blue-600">
                {creativityStatus.total_ideas_generated}
              </div>
              <div className="text-sm text-gray-500 mt-1">
                ✅ {creativityStatus.ideas_approved} | ❌ {creativityStatus.ideas_rejected}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-600">Avg Novelty</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-purple-600">
                {(creativityStatus.avg_novelty * 100).toFixed(0)}%
              </div>
              <Progress value={creativityStatus.avg_novelty * 100} className="mt-2" />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-600">Ethical Alignment</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-green-600">
                {(creativityStatus.avg_ethical_alignment * 100).toFixed(0)}%
              </div>
              <Progress value={creativityStatus.avg_ethical_alignment * 100} className="mt-2" />
            </CardContent>
          </Card>
        </div>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">
            <Lightbulb className="w-4 h-4 mr-2" />
            Idea Generator
          </TabsTrigger>
          <TabsTrigger value="meta">
            <Target className="w-4 h-4 mr-2" />
            Meta-Strategy
          </TabsTrigger>
          <TabsTrigger value="analytics">
            <BarChart3 className="w-4 h-4 mr-2" />
            Analytics
          </TabsTrigger>
          <TabsTrigger value="ethical">
            <Shield className="w-4 h-4 mr-2" />
            Ethical Audit
          </TabsTrigger>
          <TabsTrigger value="history">
            <History className="w-4 h-4 mr-2" />
            History
          </TabsTrigger>
        </TabsList>

        {/* Tab 1: Idea Generator */}
        <TabsContent value="overview" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Creative Strategy Generator</CardTitle>
              <CardDescription>
                Generate novel chess strategies across opening, middlegame, and endgame phases
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {strategies.length === 0 ? (
                <div className="text-center py-12 text-gray-500">
                  <Sparkles className="w-16 h-16 mx-auto mb-4 opacity-30" />
                  <p>No strategies generated yet. Click "Generate Creative Strategies" to begin.</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {strategies.map((strategy) => (
                    <Card key={strategy.strategy_id} className="border-l-4 border-l-purple-500">
                      <CardHeader>
                        <div className="flex items-start justify-between">
                          <div>
                            <CardTitle className="flex items-center gap-2">
                              {strategy.rejected ? (
                                <XCircle className="w-5 h-5 text-red-500" />
                              ) : (
                                <CheckCircle className="w-5 h-5 text-green-500" />
                              )}
                              {strategy.strategy_name}
                            </CardTitle>
                            <div className="flex gap-2 mt-2">
                              <Badge variant="outline">{strategy.phase}</Badge>
                              <Badge className="bg-purple-100 text-purple-800">
                                {strategy.llm_provider}
                              </Badge>
                              {strategy.rejected && (
                                <Badge variant="destructive">Rejected</Badge>
                              )}
                            </div>
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <p className="text-gray-700">{strategy.description}</p>
                        
                        {strategy.rejection_reason && (
                          <Alert variant="destructive">
                            <AlertTriangle className="h-4 w-4" />
                            <AlertDescription>
                              <strong>Rejection Reason:</strong> {strategy.rejection_reason}
                            </AlertDescription>
                          </Alert>
                        )}

                        <div className="space-y-2">
                          <div className="font-semibold text-sm">Tactical Elements:</div>
                          <ul className="list-disc list-inside space-y-1 text-sm">
                            {strategy.tactical_elements.map((elem, idx) => (
                              <li key={idx} className="text-gray-600">{elem}</li>
                            ))}
                          </ul>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 pt-4 border-t">
                          <div>
                            <div className="text-xs text-gray-500">Novelty</div>
                            <div className="text-lg font-bold text-purple-600">
                              {(strategy.novelty_score * 100).toFixed(0)}%
                            </div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-500">Stability</div>
                            <div className="text-lg font-bold text-blue-600">
                              {(strategy.stability_score * 100).toFixed(0)}%
                            </div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-500">Ethical</div>
                            <div className="text-lg font-bold text-green-600">
                              {(strategy.ethical_alignment * 100).toFixed(0)}%
                            </div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-500">Educational</div>
                            <div className="text-lg font-bold text-indigo-600">
                              {(strategy.educational_value * 100).toFixed(0)}%
                            </div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-500">Risk</div>
                            <div className="text-lg font-bold text-orange-600">
                              {(strategy.risk_level * 100).toFixed(0)}%
                            </div>
                          </div>
                        </div>

                        {strategy.parent_patterns && strategy.parent_patterns.length > 0 && (
                          <div className="pt-2 border-t">
                            <div className="text-xs text-gray-500 mb-1">Built from patterns:</div>
                            <div className="flex flex-wrap gap-1">
                              {strategy.parent_patterns.map((pattern, idx) => (
                                <Badge key={idx} variant="secondary" className="text-xs">
                                  {pattern}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tab 2: Meta-Strategy Matrix */}
        <TabsContent value="meta" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="w-5 h-5" />
                Meta-Strategic Synthesis
              </CardTitle>
              <CardDescription>
                Long-term strategic vision synthesized from creative outputs
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!metaStrategy ? (
                <div className="text-center py-12">
                  <Button onClick={synthesizeMeta} disabled={loading}>
                    <Brain className="w-4 h-4 mr-2" />
                    Synthesize Meta-Strategy
                  </Button>
                  <p className="text-sm text-gray-500 mt-4">
                    Generate creative strategies first, then synthesize a meta-strategic vision
                  </p>
                </div>
              ) : (
                <div className="space-y-6">
                  <div>
                    <h3 className="text-2xl font-bold text-purple-700 mb-2">
                      {metaStrategy.theme}
                    </h3>
                    <p className="text-gray-700">{metaStrategy.description}</p>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center p-4 bg-purple-50 rounded-lg">
                      <div className="text-2xl font-bold text-purple-600">
                        {(metaStrategy.coherence_score * 100).toFixed(0)}%
                      </div>
                      <div className="text-sm text-gray-600">Coherence</div>
                    </div>
                    <div className="text-center p-4 bg-blue-50 rounded-lg">
                      <div className="text-2xl font-bold text-blue-600">
                        {(metaStrategy.adaptability_score * 100).toFixed(0)}%
                      </div>
                      <div className="text-sm text-gray-600">Adaptability</div>
                    </div>
                    <div className="text-center p-4 bg-green-50 rounded-lg">
                      <div className="text-2xl font-bold text-green-600">
                        {(metaStrategy.long_term_value * 100).toFixed(0)}%
                      </div>
                      <div className="text-sm text-gray-600">Long-term Value</div>
                    </div>
                    <div className="text-center p-4 bg-emerald-50 rounded-lg">
                      <div className="text-2xl font-bold text-emerald-600">
                        {(metaStrategy.ethical_compliance * 100).toFixed(0)}%
                      </div>
                      <div className="text-sm text-gray-600">Ethical Compliance</div>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-semibold mb-2">Recommended Contexts:</h4>
                    <ul className="list-disc list-inside space-y-1">
                      {metaStrategy.recommended_contexts.map((context, idx) => (
                        <li key={idx} className="text-gray-700">{context}</li>
                      ))}
                    </ul>
                  </div>

                  <div>
                    <h4 className="font-semibold mb-2 flex items-center gap-2">
                      <Shield className="w-4 h-4 text-orange-500" />
                      Safety Constraints:
                    </h4>
                    <ul className="space-y-1">
                      {metaStrategy.safety_constraints.map((constraint, idx) => (
                        <li key={idx} className="text-sm text-gray-600 flex items-start gap-2">
                          <AlertTriangle className="w-4 h-4 text-orange-500 mt-0.5 flex-shrink-0" />
                          {constraint}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="text-xs text-gray-500 pt-4 border-t">
                    Integrated {metaStrategy.integrated_strategies.length} creative strategies • 
                    Generated {new Date(metaStrategy.timestamp).toLocaleString()}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tab 3: Creativity Analytics */}
        <TabsContent value="analytics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Creativity Performance Analytics</CardTitle>
            </CardHeader>
            <CardContent>
              {report ? (
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {report.phase_breakdown && Object.entries(report.phase_breakdown).map(([phase, data]) => (
                      <Card key={phase} className="bg-gradient-to-br from-purple-50 to-blue-50">
                        <CardHeader>
                          <CardTitle className="text-lg capitalize">{phase}</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-2">
                            <div className="flex justify-between">
                              <span className="text-sm text-gray-600">Strategies:</span>
                              <span className="font-bold">{data.count}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-sm text-gray-600">Avg Novelty:</span>
                              <span className="font-bold text-purple-600">
                                {(data.avg_novelty * 100).toFixed(0)}%
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-sm text-gray-600">Avg Stability:</span>
                              <span className="font-bold text-blue-600">
                                {(data.avg_stability * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>

                  {report.recommendations && (
                    <div>
                      <h4 className="font-semibold mb-3">System Recommendations:</h4>
                      <div className="space-y-2">
                        {report.recommendations.map((rec, idx) => (
                          <Alert key={idx}>
                            <AlertDescription>{rec}</AlertDescription>
                          </Alert>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500">
                  <BarChart3 className="w-16 h-16 mx-auto mb-4 opacity-30" />
                  <p>Loading analytics...</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tab 4: Ethical Audit */}
        <TabsContent value="ethical" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="w-5 h-5 text-green-600" />
                Ethical Alignment Audit
              </CardTitle>
              <CardDescription>
                Verify ethical compliance and value alignment of creative outputs
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-4 bg-green-50 rounded-lg">
                    <div className="text-3xl font-bold text-green-600">
                      {creativityStatus ? (creativityStatus.avg_ethical_alignment * 100).toFixed(0) : 0}%
                    </div>
                    <div className="text-sm text-gray-600">Overall Ethical Alignment</div>
                  </div>
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <div className="text-3xl font-bold text-blue-600">
                      {creativityStatus ? creativityStatus.ideas_approved : 0}
                    </div>
                    <div className="text-sm text-gray-600">Approved Ideas</div>
                  </div>
                  <div className="p-4 bg-red-50 rounded-lg">
                    <div className="text-3xl font-bold text-red-600">
                      {creativityStatus ? creativityStatus.ideas_rejected : 0}
                    </div>
                    <div className="text-sm text-gray-600">Rejected Ideas</div>
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold mb-3">Ethical Guardrails:</h4>
                  <div className="space-y-2">
                    <div className="flex items-start gap-3 p-3 bg-green-50 rounded-lg">
                      <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
                      <div>
                        <div className="font-medium">Fair Play</div>
                        <div className="text-sm text-gray-600">
                          All strategies must not enable unfair advantages or cheating
                        </div>
                      </div>
                    </div>
                    <div className="flex items-start gap-3 p-3 bg-green-50 rounded-lg">
                      <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
                      <div>
                        <div className="font-medium">Educational Value</div>
                        <div className="text-sm text-gray-600">
                          Strategies should be explainable and provide learning value
                        </div>
                      </div>
                    </div>
                    <div className="flex items-start gap-3 p-3 bg-green-50 rounded-lg">
                      <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
                      <div>
                        <div className="font-medium">Anti-Cheating</div>
                        <div className="text-sm text-gray-600">
                          Must not facilitate misuse in competitive contexts
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {strategies.filter(s => s.rejected).length > 0 && (
                  <div>
                    <h4 className="font-semibold mb-3 text-red-600">Rejected Strategies:</h4>
                    <div className="space-y-2">
                      {strategies.filter(s => s.rejected).map((strategy) => (
                        <Alert key={strategy.strategy_id} variant="destructive">
                          <XCircle className="h-4 w-4" />
                          <AlertDescription>
                            <strong>{strategy.strategy_name}:</strong> {strategy.rejection_reason}
                          </AlertDescription>
                        </Alert>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tab 5: History & Export */}
        <TabsContent value="history" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Strategy History</CardTitle>
                  <CardDescription>Review and export past creative outputs</CardDescription>
                </div>
                <Button onClick={exportReport} variant="outline" size="sm">
                  <Download className="w-4 h-4 mr-2" />
                  Export Report
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {history.length === 0 ? (
                <div className="text-center py-12 text-gray-500">
                  <History className="w-16 h-16 mx-auto mb-4 opacity-30" />
                  <p>No history available yet</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {history.map((strategy) => (
                    <div 
                      key={strategy.strategy_id} 
                      className="p-4 border rounded-lg hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-semibold">{strategy.strategy_name}</span>
                            <Badge variant="outline" className="text-xs">{strategy.phase}</Badge>
                          </div>
                          <p className="text-sm text-gray-600 line-clamp-2">
                            {strategy.description}
                          </p>
                          <div className="flex gap-4 mt-2 text-xs text-gray-500">
                            <span>Novelty: {(strategy.novelty_score * 100).toFixed(0)}%</span>
                            <span>Stability: {(strategy.stability_score * 100).toFixed(0)}%</span>
                            <span>Ethical: {(strategy.ethical_alignment * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                        <div className="text-xs text-gray-400">
                          {new Date(strategy.timestamp).toLocaleDateString()}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default CreativeSynthesisPanel;
