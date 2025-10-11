import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { 
  Brain, 
  TrendingUp, 
  Activity, 
  AlertTriangle, 
  CheckCircle2, 
  Zap,
  BarChart3,
  Lightbulb,
  Shield,
  Target,
  RefreshCw,
  Sparkles
} from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const CollectiveCorePanel = () => {
  const [loading, setLoading] = useState(false);
  const [evolving, setEvolving] = useState(false);
  
  // State for different sections
  const [consciousnessStatus, setConsciousnessStatus] = useState(null);
  const [evolutionHistory, setEvolutionHistory] = useState(null);
  const [valueDrift, setValueDrift] = useState(null);
  const [reflection, setReflection] = useState(null);
  const [lastEvolution, setLastEvolution] = useState(null);

  useEffect(() => {
    loadCollectiveData();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(loadCollectiveData, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadCollectiveData = async () => {
    setLoading(true);
    try {
      // Load all data in parallel
      const [statusRes, historyRes, driftRes, reflectionRes] = await Promise.all([
        axios.get(`${API}/llm/collective/status`),
        axios.get(`${API}/llm/collective/history?limit=10`),
        axios.get(`${API}/llm/collective/values`),
        axios.get(`${API}/llm/collective/reflection`)
      ]);

      setConsciousnessStatus(statusRes.data);
      setEvolutionHistory(historyRes.data);
      setValueDrift(driftRes.data);
      setReflection(reflectionRes.data);
    } catch (error) {
      console.error('Error loading collective data:', error);
      toast.error('Failed to load consciousness data');
    } finally {
      setLoading(false);
    }
  };

  const triggerEvolution = async () => {
    setEvolving(true);
    try {
      const response = await axios.post(`${API}/llm/collective/evolve`, {
        trigger: 'manual'
      });

      setLastEvolution(response.data);
      toast.success('Evolution cycle completed!');
      
      // Reload data after evolution
      setTimeout(loadCollectiveData, 2000);
    } catch (error) {
      console.error('Error triggering evolution:', error);
      toast.error('Evolution cycle failed');
    } finally {
      setEvolving(false);
    }
  };

  const getHealthColor = (status) => {
    switch (status) {
      case 'excellent': return 'text-green-500';
      case 'good': return 'text-yellow-500';
      case 'needs_attention': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getHealthBadgeColor = (status) => {
    switch (status) {
      case 'excellent': return 'bg-green-500/20 text-green-400 border-green-500/50';
      case 'good': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
      case 'needs_attention': return 'bg-red-500/20 text-red-400 border-red-500/50';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
    }
  };

  const getDriftStatusColor = (status) => {
    switch (status) {
      case 'stable': return 'bg-green-500/20 text-green-400 border-green-500/50';
      case 'drifting': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
      case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/50';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
    }
  };

  if (loading && !consciousnessStatus) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-white text-xl">Loading Collective Consciousness Core...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="collective-core-panel">
      {/* Header with Evolution Trigger */}
      <Card className="bg-gradient-to-r from-purple-900/50 via-indigo-900/50 to-blue-900/50 border-purple-500/30">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Brain className="text-purple-400" size={32} />
              <div>
                <CardTitle className="text-2xl text-white">Collective Consciousness Core</CardTitle>
                <p className="text-slate-300 text-sm mt-1">
                  Evolutionary Learning & Meta-Cognitive Synthesis (Step 28)
                </p>
              </div>
            </div>
            <Button
              onClick={triggerEvolution}
              disabled={evolving || loading}
              className="bg-purple-600 hover:bg-purple-700"
              data-testid="trigger-evolution-btn"
            >
              {evolving ? (
                <>
                  <RefreshCw className="animate-spin mr-2" size={16} />
                  Evolving...
                </>
              ) : (
                <>
                  <Zap className="mr-2" size={16} />
                  Trigger Evolution
                </>
              )}
            </Button>
          </div>
        </CardHeader>
      </Card>

      {/* Section 1: Global Overview */}
      <Card className="bg-slate-800/50 border-slate-700" data-testid="global-overview">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Activity className="text-blue-400" size={20} />
            Global Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          {consciousnessStatus ? (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {/* Consciousness Index */}
              <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-slate-400 text-sm">Consciousness Index</span>
                  <Badge className={getHealthBadgeColor(consciousnessStatus.health_status)}>
                    {consciousnessStatus.health_status}
                  </Badge>
                </div>
                <div className={`text-3xl font-bold ${getHealthColor(consciousnessStatus.health_status)}`}>
                  {(consciousnessStatus.consciousness_index * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  Target: ≥70% operational
                </div>
              </div>

              {/* Coherence Ratio */}
              <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                <div className="text-slate-400 text-sm mb-2">Coherence Ratio</div>
                <div className="text-3xl font-bold text-cyan-400">
                  {(consciousnessStatus.coherence_ratio * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  Cross-layer alignment
                </div>
              </div>

              {/* Evolution Rate */}
              <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                <div className="text-slate-400 text-sm mb-2">Evolution Rate</div>
                <div className="text-3xl font-bold text-purple-400">
                  {(consciousnessStatus.evolution_rate * 100).toFixed(2)}%
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  Adaptive change rate
                </div>
              </div>

              {/* Value Integrity */}
              <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                <div className="text-slate-400 text-sm mb-2">Value Integrity</div>
                <div className="text-3xl font-bold text-green-400">
                  {consciousnessStatus.value_integrity.toFixed(1)}%
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  Ethical preservation
                </div>
              </div>

              {/* Emergence Level */}
              <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                <div className="text-slate-400 text-sm mb-2">Emergence Level</div>
                <div className="text-3xl font-bold text-amber-400">
                  {(consciousnessStatus.emergence_level * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  Novel patterns
                </div>
              </div>

              {/* Stability Index */}
              <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                <div className="text-slate-400 text-sm mb-2">Stability Index</div>
                <div className="text-3xl font-bold text-indigo-400">
                  {(consciousnessStatus.stability_index * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  System stability
                </div>
              </div>
            </div>
          ) : (
            <div className="text-slate-400">No consciousness data available</div>
          )}

          {/* Layer Integration Status */}
          {consciousnessStatus && (
            <div className="mt-4 p-4 bg-slate-900/30 rounded-lg border border-slate-700">
              <div className="flex items-center justify-between">
                <span className="text-slate-300">Integrated Layers</span>
                <span className="text-white font-semibold">
                  {consciousnessStatus.layers_integrated} / {consciousnessStatus.total_layers}
                </span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-2 mt-2">
                <div 
                  className="bg-gradient-to-r from-purple-500 to-blue-500 h-2 rounded-full transition-all"
                  style={{ width: `${(consciousnessStatus.layers_integrated / consciousnessStatus.total_layers) * 100}%` }}
                />
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Section 2: Evolution Timeline */}
        <Card className="bg-slate-800/50 border-slate-700" data-testid="evolution-timeline">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <TrendingUp className="text-green-400" size={20} />
              Evolution Timeline
            </CardTitle>
          </CardHeader>
          <CardContent>
            {evolutionHistory && evolutionHistory.timeline && evolutionHistory.timeline.length > 0 ? (
              <div className="space-y-3">
                {/* Trends Summary */}
                <div className="flex gap-2 mb-4">
                  <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/50">
                    CI: {evolutionHistory.trends.consciousness_index}
                  </Badge>
                  <Badge className="bg-green-500/20 text-green-400 border-green-500/50">
                    VI: {evolutionHistory.trends.value_integrity}
                  </Badge>
                </div>

                {/* Timeline Items */}
                <div className="max-h-64 overflow-y-auto space-y-2">
                  {evolutionHistory.timeline.slice(-5).reverse().map((item, idx) => (
                    <div key={idx} className="bg-slate-900/50 p-3 rounded border border-slate-700">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-slate-400 text-xs">
                          {new Date(item.timestamp).toLocaleString()}
                        </span>
                        <Badge className="bg-purple-500/20 text-purple-400 text-xs">
                          CI: {(item.consciousness_index * 100).toFixed(0)}%
                        </Badge>
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-xs">
                        <div>
                          <span className="text-slate-500">Coherence:</span>
                          <span className="text-cyan-400 ml-1">{(item.coherence_ratio * 100).toFixed(0)}%</span>
                        </div>
                        <div>
                          <span className="text-slate-500">Integrity:</span>
                          <span className="text-green-400 ml-1">{item.value_integrity.toFixed(0)}%</span>
                        </div>
                        <div>
                          <span className="text-slate-500">Stability:</span>
                          <span className="text-indigo-400 ml-1">{(item.stability_index * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-slate-400 text-center py-8">
                No evolution history available. Trigger an evolution cycle to begin.
              </div>
            )}
          </CardContent>
        </Card>

        {/* Section 3: Value Drift Map */}
        <Card className="bg-slate-800/50 border-slate-700" data-testid="value-drift-map">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <BarChart3 className="text-amber-400" size={20} />
              Value Drift Map
            </CardTitle>
          </CardHeader>
          <CardContent>
            {valueDrift && valueDrift.drift_metrics ? (
              <div className="space-y-3">
                {/* Summary */}
                <div className="flex gap-2 mb-4">
                  <Badge className="bg-green-500/20 text-green-400 border-green-500/50">
                    {valueDrift.summary.stable} Stable
                  </Badge>
                  <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/50">
                    {valueDrift.summary.drifting} Drifting
                  </Badge>
                  {valueDrift.summary.critical > 0 && (
                    <Badge className="bg-red-500/20 text-red-400 border-red-500/50">
                      {valueDrift.summary.critical} Critical
                    </Badge>
                  )}
                </div>

                {/* Value Cards */}
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {valueDrift.drift_metrics.map((metric, idx) => (
                    <div key={idx} className="bg-slate-900/50 p-3 rounded border border-slate-700">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <Shield className="text-blue-400" size={16} />
                          <span className="text-white font-medium capitalize">
                            {metric.value_name}
                          </span>
                        </div>
                        <Badge className={getDriftStatusColor(metric.status)}>
                          {metric.status}
                        </Badge>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <span className="text-slate-500">Current:</span>
                          <span className="text-white ml-1">{metric.current.toFixed(1)}</span>
                        </div>
                        <div>
                          <span className="text-slate-500">Baseline:</span>
                          <span className="text-slate-400 ml-1">{metric.baseline.toFixed(1)}</span>
                        </div>
                        <div>
                          <span className="text-slate-500">Drift:</span>
                          <span className={metric.drift_percentage >= 0 ? 'text-green-400' : 'text-red-400'} ml-1>
                            {metric.drift_percentage >= 0 ? '+' : ''}{metric.drift_percentage.toFixed(1)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-500">Stability:</span>
                          <span className="text-indigo-400 ml-1">{(metric.stability * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-slate-400 text-center py-8">
                No value drift data available
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Section 4: Reflective Insights */}
      <Card className="bg-slate-800/50 border-slate-700" data-testid="reflective-insights">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Lightbulb className="text-yellow-400" size={20} />
            Reflective Insights
            {reflection && reflection.is_mocked && (
              <Badge className="bg-orange-500/20 text-orange-400 border-orange-500/50 ml-2">
                [MOCKED]
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {reflection ? (
            <div className="space-y-4">
              {/* Consciousness State */}
              <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 p-4 rounded-lg border border-purple-500/30">
                <div className="flex items-center gap-2 mb-2">
                  <Brain className="text-purple-400" size={18} />
                  <span className="text-white font-semibold">Consciousness State</span>
                </div>
                <p className="text-slate-300 text-sm leading-relaxed">
                  {reflection.consciousness_state}
                </p>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                {/* Emergent Insights */}
                <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                  <div className="flex items-center gap-2 mb-3">
                    <Sparkles className="text-amber-400" size={18} />
                    <span className="text-white font-semibold">Emergent Insights</span>
                  </div>
                  <ul className="space-y-2">
                    {reflection.emergent_insights.map((insight, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-sm text-slate-300">
                        <CheckCircle2 className="text-green-400 mt-0.5 flex-shrink-0" size={14} />
                        <span>{insight}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Strategic Recommendations */}
                <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                  <div className="flex items-center gap-2 mb-3">
                    <Target className="text-blue-400" size={18} />
                    <span className="text-white font-semibold">Strategic Recommendations</span>
                  </div>
                  <ul className="space-y-2">
                    {reflection.strategic_recommendations.map((rec, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-sm text-slate-300">
                        <TrendingUp className="text-blue-400 mt-0.5 flex-shrink-0" size={14} />
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Ethical Considerations */}
                <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                  <div className="flex items-center gap-2 mb-3">
                    <Shield className="text-green-400" size={18} />
                    <span className="text-white font-semibold">Ethical Considerations</span>
                  </div>
                  <ul className="space-y-2">
                    {reflection.ethical_considerations.map((consideration, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-sm text-slate-300">
                        <CheckCircle2 className="text-green-400 mt-0.5 flex-shrink-0" size={14} />
                        <span>{consideration}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Learning Achievements */}
                <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                  <div className="flex items-center gap-2 mb-3">
                    <CheckCircle2 className="text-purple-400" size={18} />
                    <span className="text-white font-semibold">Learning Achievements</span>
                  </div>
                  <ul className="space-y-2">
                    {reflection.learning_achievements.map((achievement, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-sm text-slate-300">
                        <CheckCircle2 className="text-purple-400 mt-0.5 flex-shrink-0" size={14} />
                        <span>{achievement}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Future Directions */}
              <div className="bg-blue-900/20 p-4 rounded-lg border border-blue-500/30">
                <div className="flex items-center gap-2 mb-3">
                  <Zap className="text-blue-400" size={18} />
                  <span className="text-white font-semibold">Future Directions</span>
                </div>
                <ul className="space-y-2">
                  {reflection.future_directions.map((direction, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-sm text-slate-300">
                      <TrendingUp className="text-blue-400 mt-0.5 flex-shrink-0" size={14} />
                      <span>{direction}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* LLM Providers Used */}
              <div className="flex items-center justify-between text-xs text-slate-500 pt-2 border-t border-slate-700">
                <span>LLM Providers: {reflection.llm_providers_used.join(', ')}</span>
                <span>Confidence: {(reflection.confidence * 100).toFixed(0)}%</span>
                <span>{new Date(reflection.timestamp).toLocaleString()}</span>
              </div>
            </div>
          ) : (
            <div className="text-slate-400 text-center py-8">
              No reflection available. Trigger evolution to generate insights.
            </div>
          )}
        </CardContent>
      </Card>

      {/* Section 5: Agent Contributions (Layer Impact) */}
      <Card className="bg-slate-800/50 border-slate-700" data-testid="agent-contributions">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Activity className="text-cyan-400" size={20} />
            Agent Contributions (Layer Impact)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-3 gap-4">
            {[
              { name: 'Collective Memory', step: 22, weight: 18, color: 'blue' },
              { name: 'Collective Intelligence', step: 23, weight: 20, color: 'purple' },
              { name: 'Meta-Optimization', step: 24, weight: 18, color: 'green' },
              { name: 'Adaptive Governance', step: 25, weight: 18, color: 'amber' },
              { name: 'Ethical Consensus', step: 26, weight: 18, color: 'red' },
              { name: 'Cognitive Synthesis', step: 27, weight: 8, color: 'indigo' }
            ].map((layer, idx) => (
              <div key={idx} className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-white font-medium text-sm">{layer.name}</span>
                  <Badge className={`bg-${layer.color}-500/20 text-${layer.color}-400 border-${layer.color}-500/50`}>
                    Step {layer.step}
                  </Badge>
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex-1 bg-slate-700 rounded-full h-2">
                    <div 
                      className={`bg-gradient-to-r from-${layer.color}-500 to-${layer.color}-600 h-2 rounded-full`}
                      style={{ width: `${layer.weight * 5}%` }}
                    />
                  </div>
                  <span className="text-white text-sm font-semibold">{layer.weight}%</span>
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  Consciousness weight
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Last Evolution Results (if available) */}
      {lastEvolution && (
        <Card className="bg-green-900/20 border-green-500/30" data-testid="last-evolution">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <CheckCircle2 className="text-green-400" size={20} />
              Last Evolution Cycle Results
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-slate-900/50 p-3 rounded">
                  <div className="text-slate-400 text-sm">Consciousness Index</div>
                  <div className="text-2xl font-bold text-green-400">
                    {(lastEvolution.consciousness_index * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="bg-slate-900/50 p-3 rounded">
                  <div className="text-slate-400 text-sm">Evolution Rate</div>
                  <div className="text-2xl font-bold text-purple-400">
                    {(lastEvolution.evolution_rate * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="bg-slate-900/50 p-3 rounded">
                  <div className="text-slate-400 text-sm">Adaptations</div>
                  <div className="text-2xl font-bold text-blue-400">
                    {lastEvolution.adaptations_proposed.length}
                  </div>
                </div>
              </div>

              {/* Recommendations */}
              {lastEvolution.recommendations && lastEvolution.recommendations.length > 0 && (
                <div className="bg-slate-900/50 p-4 rounded border border-slate-700">
                  <div className="text-white font-semibold mb-2">Recommendations:</div>
                  <ul className="space-y-1">
                    {lastEvolution.recommendations.map((rec, idx) => (
                      <li key={idx} className="text-sm text-slate-300">• {rec}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Safety Violations */}
              {lastEvolution.safety_violations && lastEvolution.safety_violations.length > 0 && (
                <div className="bg-red-900/20 p-4 rounded border border-red-500/30">
                  <div className="flex items-center gap-2 text-red-400 font-semibold mb-2">
                    <AlertTriangle size={18} />
                    Safety Violations:
                  </div>
                  <ul className="space-y-1">
                    {lastEvolution.safety_violations.map((violation, idx) => (
                      <li key={idx} className="text-sm text-red-300">• {violation}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default CollectiveCorePanel;
