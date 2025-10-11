import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { AlertCircle, CheckCircle, RefreshCw, TrendingUp, Users } from 'lucide-react';

const ArbitrationPanel = () => {
  const [arbitrationHistory, setArbitrationHistory] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedArbitration, setSelectedArbitration] = useState(null);

  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

  // Fetch arbitration history
  const fetchArbitrationHistory = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${backendUrl}/api/llm/arbitration/history?limit=20`);
      const data = await response.json();
      
      if (data.success) {
        setArbitrationHistory(data.history);
        setSummary(data.summary);
      }
    } catch (error) {
      console.error('Error fetching arbitration history:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchArbitrationHistory();
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchArbitrationHistory, 30000);
    return () => clearInterval(interval);
  }, []);

  // Get outcome badge variant
  const getOutcomeBadge = (outcome) => {
    const variants = {
      'Approved': 'default',
      'Rejected': 'destructive',
      'Reassessed': 'secondary'
    };
    
    return (
      <Badge variant={variants[outcome] || 'outline'}>
        {outcome}
      </Badge>
    );
  };

  // Get confidence delta color
  const getConfidenceDeltaColor = (delta) => {
    if (delta > 0.05) return 'text-green-600';
    if (delta > 0) return 'text-green-500';
    if (delta < -0.05) return 'text-red-600';
    return 'text-gray-500';
  };

  // Format timestamp
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp).toLocaleString();
  };

  // Trigger manual arbitration
  const triggerManualArbitration = async () => {
    const task = prompt('Enter task description for arbitration:');
    if (!task) return;

    setLoading(true);
    try {
      const response = await fetch(`${backendUrl}/api/llm/arbitration/resolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task,
          force_arbitration: true,
          task_complexity: 0.7,
          task_category: 'general'
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        alert('Arbitration triggered successfully!');
        fetchArbitrationHistory();
      } else {
        alert('Failed to trigger arbitration');
      }
    } catch (error) {
      console.error('Error triggering arbitration:', error);
      alert('Error triggering arbitration');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6" data-testid="arbitration-panel">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">Meta-Agent Arbitration</h2>
          <p className="text-gray-600">Conflict resolution and consensus arbitration system</p>
        </div>
        <div className="flex gap-2">
          <Button 
            onClick={fetchArbitrationHistory} 
            variant="outline"
            disabled={loading}
            data-testid="refresh-arbitration-btn"
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button 
            onClick={triggerManualArbitration}
            disabled={loading}
            data-testid="trigger-arbitration-btn"
          >
            <AlertCircle className="h-4 w-4 mr-2" />
            Trigger Arbitration
          </Button>
        </div>
      </div>

      {/* Summary Statistics */}
      {summary && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold">{summary.total_arbitrations}</div>
              <p className="text-xs text-gray-600">Total Arbitrations</p>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-blue-600">
                {(summary.avg_confidence_before * 100).toFixed(1)}%
              </div>
              <p className="text-xs text-gray-600">Avg Confidence Before</p>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-green-600">
                {(summary.avg_confidence_after * 100).toFixed(1)}%
              </div>
              <p className="text-xs text-gray-600">Avg Confidence After</p>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="pt-6">
              <div className={`text-2xl font-bold ${summary.avg_confidence_improvement > 0 ? 'text-green-600' : 'text-gray-600'}`}>
                {summary.avg_confidence_improvement > 0 ? '+' : ''}{(summary.avg_confidence_improvement * 100).toFixed(1)}%
              </div>
              <p className="text-xs text-gray-600">Avg Improvement</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Outcome Distribution */}
      {summary && summary.outcome_distribution && Object.keys(summary.outcome_distribution).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Arbitration Outcomes</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-4">
              {Object.entries(summary.outcome_distribution).map(([outcome, count]) => (
                <div key={outcome} className="flex items-center gap-2">
                  {getOutcomeBadge(outcome)}
                  <span className="text-sm text-gray-600">{count} sessions</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Arbitration History List */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Arbitration Sessions</CardTitle>
          <CardDescription>
            Click on a session to view detailed arbitration analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading && arbitrationHistory.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-2" />
              Loading arbitration history...
            </div>
          ) : arbitrationHistory.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <AlertCircle className="h-8 w-8 mx-auto mb-2" />
              No arbitration sessions yet. Trigger one to get started.
            </div>
          ) : (
            <div className="space-y-3">
              {arbitrationHistory.map((arbitration, index) => (
                <div
                  key={arbitration.arbitration_id || index}
                  onClick={() => setSelectedArbitration(
                    selectedArbitration?.arbitration_id === arbitration.arbitration_id 
                      ? null 
                      : arbitration
                  )}
                  className="border rounded-lg p-4 hover:bg-gray-50 cursor-pointer transition-colors"
                  data-testid={`arbitration-item-${index}`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        {getOutcomeBadge(arbitration.arbitration_outcome)}
                        <span className="text-xs text-gray-500">
                          {formatTimestamp(arbitration.timestamp)}
                        </span>
                      </div>
                      <p className="text-sm font-medium text-gray-700">
                        {arbitration.trigger_reason}
                      </p>
                    </div>
                    
                    <div className="text-right">
                      <div className="flex items-center gap-2">
                        <TrendingUp className={`h-4 w-4 ${getConfidenceDeltaColor(arbitration.confidence_delta)}`} />
                        <span className={`text-sm font-bold ${getConfidenceDeltaColor(arbitration.confidence_delta)}`}>
                          {arbitration.confidence_delta > 0 ? '+' : ''}{(arbitration.confidence_delta * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {(arbitration.confidence_before * 100).toFixed(1)}% → {(arbitration.confidence_after * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-4 text-xs text-gray-500">
                    <div className="flex items-center gap-1">
                      <Users className="h-3 w-3" />
                      {arbitration.agents_involved?.length || 0} agents
                    </div>
                    <div>
                      Resolution: {arbitration.resolution_time?.toFixed(2) || 'N/A'}s
                    </div>
                  </div>

                  {/* Expanded Details */}
                  {selectedArbitration?.arbitration_id === arbitration.arbitration_id && (
                    <div className="mt-4 pt-4 border-t space-y-3">
                      {/* Meta-Agent Reasoning */}
                      <div>
                        <h4 className="text-sm font-semibold mb-1">Meta-Agent Reasoning:</h4>
                        <p className="text-sm text-gray-700 bg-blue-50 p-3 rounded">
                          {arbitration.meta_agent_reasoning}
                        </p>
                      </div>

                      {/* Revised Consensus */}
                      <div>
                        <h4 className="text-sm font-semibold mb-1">Revised Consensus:</h4>
                        <p className="text-sm text-gray-700 bg-green-50 p-3 rounded">
                          {arbitration.revised_consensus}
                        </p>
                      </div>

                      {/* Winning Rationale */}
                      <div>
                        <h4 className="text-sm font-semibold mb-1">Winning Rationale:</h4>
                        <p className="text-sm text-gray-700 bg-yellow-50 p-3 rounded">
                          {arbitration.winning_rationale}
                        </p>
                      </div>

                      {/* Divergence Map */}
                      {arbitration.divergence_map && Object.keys(arbitration.divergence_map).length > 0 && (
                        <div>
                          <h4 className="text-sm font-semibold mb-2">Agent Positions:</h4>
                          <div className="space-y-2">
                            {Object.entries(arbitration.divergence_map).map(([agent, position]) => (
                              <div key={agent} className="text-xs border-l-2 border-gray-300 pl-3">
                                <span className="font-medium">{agent}:</span> {position}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Conflict Analysis */}
                      {arbitration.conflict_analysis && (
                        <div>
                          <h4 className="text-sm font-semibold mb-2">Conflict Analysis:</h4>
                          <div className="grid grid-cols-3 gap-2 text-xs">
                            <div className="bg-gray-50 p-2 rounded">
                              <div className="font-medium">Disagreement</div>
                              <div className="text-lg font-bold text-red-600">
                                {(arbitration.conflict_analysis.disagreement_magnitude * 100).toFixed(1)}%
                              </div>
                            </div>
                            <div className="bg-gray-50 p-2 rounded">
                              <div className="font-medium">Semantic Distance</div>
                              <div className="text-lg font-bold text-orange-600">
                                {arbitration.conflict_analysis.semantic_distance?.toFixed(3) || 'N/A'}
                              </div>
                            </div>
                            <div className="bg-gray-50 p-2 rounded">
                              <div className="font-medium">Clusters</div>
                              <div className="text-lg font-bold text-blue-600">
                                {arbitration.conflict_analysis.divergence_clusters?.length || 0}
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default ArbitrationPanel;
