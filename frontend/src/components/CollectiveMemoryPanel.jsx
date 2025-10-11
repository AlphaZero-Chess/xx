import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Brain, Database, Play, Search, Filter, TrendingUp, Clock, Award, Users } from 'lucide-react';

const CollectiveMemoryPanel = () => {
  const [memoryStats, setMemoryStats] = useState(null);
  const [memorySummary, setMemorySummary] = useState(null);
  const [experiences, setExperiences] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [taskTypeFilter, setTaskTypeFilter] = useState('all');
  const [minConfidence, setMinConfidence] = useState(0.70);
  const [isReplaying, setIsReplaying] = useState(false);
  const [replayResult, setReplayResult] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [loading, setLoading] = useState(true);

  const backendUrl = process.env.REACT_APP_BACKEND_URL || '';

  // Fetch memory stats on mount
  useEffect(() => {
    fetchMemoryStats();
    fetchMemorySummary();
  }, []);

  const fetchMemoryStats = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/llm/memory/stats`);
      const data = await response.json();
      if (data.success) {
        setMemoryStats(data);
      }
      setLoading(false);
    } catch (error) {
      console.error('Error fetching memory stats:', error);
      setLoading(false);
    }
  };

  const fetchMemorySummary = async (timeframeDays = 30) => {
    try {
      const response = await fetch(`${backendUrl}/api/llm/memory/summary?timeframe_days=${timeframeDays}`);
      const data = await response.json();
      if (data.success) {
        setMemorySummary(data);
      }
    } catch (error) {
      console.error('Error fetching memory summary:', error);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    try {
      const filter = taskTypeFilter !== 'all' ? `&task_type_filter=${taskTypeFilter}` : '';
      const response = await fetch(
        `${backendUrl}/api/llm/memory/retrieve?query=${encodeURIComponent(searchQuery)}&min_confidence=${minConfidence}&limit=10${filter}`
      );
      const data = await response.json();
      if (data.success) {
        setExperiences(data.similar_experiences || []);
      }
    } catch (error) {
      console.error('Error searching experiences:', error);
    }
    setIsSearching(false);
  };

  const handleRunReplay = async () => {
    setIsReplaying(true);
    setReplayResult(null);
    try {
      const response = await fetch(`${backendUrl}/api/llm/memory/replay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          auto_select_count: 10
        })
      });
      const data = await response.json();
      if (data.success) {
        setReplayResult(data);
        // Refresh stats after replay
        fetchMemoryStats();
      }
    } catch (error) {
      console.error('Error running replay:', error);
    }
    setIsReplaying(false);
  };

  const getConfidenceBadge = (confidence) => {
    if (confidence >= 0.90) return <Badge className="bg-green-500">High ({(confidence * 100).toFixed(0)}%)</Badge>;
    if (confidence >= 0.80) return <Badge className="bg-blue-500">Good ({(confidence * 100).toFixed(0)}%)</Badge>;
    return <Badge className="bg-yellow-500">Fair ({(confidence * 100).toFixed(0)}%)</Badge>;
  };

  const getTaskTypeBadge = (taskType) => {
    const colors = {
      arbitration: 'bg-purple-500',
      consensus: 'bg-blue-500',
      coaching: 'bg-green-500',
      analytics: 'bg-orange-500'
    };
    return <Badge className={colors[taskType] || 'bg-gray-500'}>{taskType}</Badge>;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="collective-memory-panel">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold flex items-center gap-2">
            <Brain className="w-8 h-8 text-purple-500" />
            Collective Memory & Experience Replay
          </h2>
          <p className="text-gray-400 mt-1">
            Long-term learning system storing and replaying successful arbitration outcomes
          </p>
        </div>
      </div>

      {/* Memory Statistics Dashboard */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-gradient-to-br from-purple-500/10 to-purple-600/10 border-purple-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-400 flex items-center gap-2">
              <Database className="w-4 h-4" />
              Total Experiences
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-purple-400">
              {memoryStats?.total_experiences || 0}
            </div>
            <p className="text-xs text-gray-500 mt-1">
              {memoryStats?.retention_status || 'Within limit'}
            </p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-green-500/10 to-green-600/10 border-green-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-400 flex items-center gap-2">
              <Award className="w-4 h-4" />
              High Quality
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-green-400">
              {memoryStats?.high_quality_count || 0}
            </div>
            <p className="text-xs text-gray-500 mt-1">
              ≥90% confidence cases
            </p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-blue-500/10 to-blue-600/10 border-blue-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-400 flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Memory Accuracy
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-blue-400">
              {memoryStats?.memory_accuracy_percent?.toFixed(1) || 0}%
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Avg confidence: {memoryStats?.avg_confidence?.toFixed(3) || 0}
            </p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-orange-500/10 to-orange-600/10 border-orange-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-400 flex items-center gap-2">
              <Play className="w-4 h-4" />
              Replay Sessions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-orange-400">
              {memoryStats?.replay_sessions_total || 0}
            </div>
            <p className="text-xs text-gray-500 mt-1">
              {memoryStats?.experiences_since_last_replay || 0} since last replay
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Search and Filter Section */}
      <Card className="bg-gray-800/50 border-gray-700">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="w-5 h-5" />
            Search Memory
          </CardTitle>
          <CardDescription>
            Use semantic search to find similar past experiences
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-4">
            <Input
              placeholder="Describe the situation or task..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              className="flex-1"
              data-testid="memory-search-input"
            />
            <Select value={taskTypeFilter} onValueChange={setTaskTypeFilter}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Task Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="arbitration">Arbitration</SelectItem>
                <SelectItem value="consensus">Consensus</SelectItem>
                <SelectItem value="coaching">Coaching</SelectItem>
                <SelectItem value="analytics">Analytics</SelectItem>
              </SelectContent>
            </Select>
            <Button
              onClick={handleSearch}
              disabled={isSearching || !searchQuery.trim()}
              data-testid="memory-search-button"
            >
              <Search className="w-4 h-4 mr-2" />
              {isSearching ? 'Searching...' : 'Search'}
            </Button>
          </div>

          <div className="flex items-center gap-4">
            <label className="text-sm text-gray-400">Min Confidence:</label>
            <input
              type="range"
              min="0.5"
              max="1.0"
              step="0.05"
              value={minConfidence}
              onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
              className="flex-1"
            />
            <span className="text-sm font-medium">{(minConfidence * 100).toFixed(0)}%</span>
          </div>
        </CardContent>
      </Card>

      {/* Experience Results */}
      {experiences.length > 0 && (
        <Card className="bg-gray-800/50 border-gray-700">
          <CardHeader>
            <CardTitle>Similar Experiences ({experiences.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {experiences.map((exp, idx) => (
                <Card
                  key={exp.experience_id || idx}
                  className="bg-gray-700/30 border-gray-600 hover:border-purple-500/50 transition-all"
                  data-testid={`experience-card-${idx}`}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex gap-2">
                        {getTaskTypeBadge(exp.task_type)}
                        {getConfidenceBadge(exp.confidence)}
                        {exp.similarity_score && (
                          <Badge variant="outline" className="border-blue-400 text-blue-400">
                            Match: {(exp.similarity_score * 100).toFixed(0)}%
                          </Badge>
                        )}
                      </div>
                      <span className="text-xs text-gray-500">
                        {new Date(exp.timestamp).toLocaleDateString()}
                      </span>
                    </div>
                    
                    <p className="text-sm text-gray-300 mb-2">{exp.task_description}</p>
                    
                    <div className="flex items-center gap-4 text-xs text-gray-500">
                      <div className="flex items-center gap-1">
                        <Users className="w-3 h-3" />
                        {exp.agents_involved?.join(', ') || 'N/A'}
                      </div>
                      {exp.outcome?.consensus_reached !== undefined && (
                        <div>
                          Consensus: {exp.outcome.consensus_reached ? '✓' : '✗'}
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Memory Summary */}
      {memorySummary && (
        <Card className="bg-gray-800/50 border-gray-700">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="w-5 h-5" />
              Memory Summary (Last {memorySummary.timeframe_days} Days)
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Task Type Breakdown */}
              <div>
                <h4 className="font-semibold mb-2 text-gray-300">Task Type Distribution</h4>
                <div className="space-y-2">
                  {Object.entries(memorySummary.task_type_breakdown || {}).map(([type, stats]) => (
                    <div key={type} className="flex items-center justify-between text-sm">
                      <span className="capitalize">{type}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-gray-400">{stats.count} cases</span>
                        <span className="text-blue-400 font-medium">
                          {(stats.avg_confidence * 100).toFixed(0)}% avg
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Most Active Agents */}
              <div>
                <h4 className="font-semibold mb-2 text-gray-300">Most Active Agents</h4>
                <div className="space-y-2">
                  {memorySummary.most_active_agents?.slice(0, 5).map((agent, idx) => (
                    <div key={idx} className="flex items-center justify-between text-sm">
                      <span>{agent.agent}</span>
                      <Badge variant="outline">{agent.count} cases</Badge>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-gray-700">
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <p className="text-gray-400">High Quality Rate</p>
                  <p className="text-xl font-bold text-green-400">
                    {memorySummary.high_quality_rate?.toFixed(1) || 0}%
                  </p>
                </div>
                <div>
                  <p className="text-gray-400">Avg Confidence</p>
                  <p className="text-xl font-bold text-blue-400">
                    {(memorySummary.avg_confidence * 100)?.toFixed(1) || 0}%
                  </p>
                </div>
                <div>
                  <p className="text-gray-400">Last Replay</p>
                  <p className="text-xl font-bold text-purple-400">
                    {memorySummary.last_replay_date 
                      ? new Date(memorySummary.last_replay_date).toLocaleDateString()
                      : 'Never'}
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Experience Replay Section */}
      <Card className="bg-gradient-to-br from-purple-500/10 to-blue-500/10 border-purple-500/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Play className="w-5 h-5 text-purple-400" />
            Experience Replay Session
          </CardTitle>
          <CardDescription>
            Replay high-confidence cases to reinforce trust calibration and improve reasoning
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-400">
                Automatically replays every 50 new experiences or trigger manually
              </p>
              {memoryStats?.replay_recommended && (
                <p className="text-sm text-orange-400 mt-1">
                  ⚠️ Replay recommended - threshold reached
                </p>
              )}
            </div>
            <Button
              onClick={handleRunReplay}
              disabled={isReplaying}
              className="bg-purple-600 hover:bg-purple-700"
              data-testid="run-replay-button"
            >
              <Play className="w-4 h-4 mr-2" />
              {isReplaying ? 'Replaying...' : 'Run Replay Session'}
            </Button>
          </div>

          {/* Replay Result */}
          {replayResult && (
            <div className="mt-4 p-4 bg-gray-700/50 rounded-lg border border-purple-500/30" data-testid="replay-result">
              <h4 className="font-semibold text-purple-400 mb-3">
                Replay Complete - Session {replayResult.session_id.slice(0, 8)}
              </h4>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div className="text-center p-3 bg-gray-800/50 rounded">
                  <p className="text-xs text-gray-400">Experiences Replayed</p>
                  <p className="text-2xl font-bold text-purple-400">{replayResult.experiences_replayed}</p>
                </div>
                <div className="text-center p-3 bg-gray-800/50 rounded">
                  <p className="text-xs text-gray-400">Avg Confidence</p>
                  <p className="text-2xl font-bold text-blue-400">
                    {(replayResult.avg_confidence * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-center p-3 bg-gray-800/50 rounded">
                  <p className="text-xs text-gray-400">Agents Improved</p>
                  <p className="text-2xl font-bold text-green-400">
                    {Object.keys(replayResult.trust_adjustments).length}
                  </p>
                </div>
              </div>

              {/* Trust Adjustments */}
              {Object.keys(replayResult.trust_adjustments).length > 0 && (
                <div className="mb-4">
                  <h5 className="text-sm font-semibold text-gray-300 mb-2">Trust Adjustments:</h5>
                  <div className="space-y-1">
                    {Object.entries(replayResult.trust_adjustments).map(([agent, delta]) => (
                      <div key={agent} className="flex items-center justify-between text-sm">
                        <span className="text-gray-400">{agent}</span>
                        <span className="text-green-400 font-medium">+{(delta * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Performance Delta */}
              {replayResult.performance_delta && (
                <div>
                  <h5 className="text-sm font-semibold text-gray-300 mb-2">Performance Improvements:</h5>
                  <div className="space-y-1">
                    {Object.entries(replayResult.performance_delta).map(([metric, value]) => (
                      <div key={metric} className="flex items-center justify-between text-sm">
                        <span className="text-gray-400 capitalize">
                          {metric.replace(/_/g, ' ')}
                        </span>
                        <span className="text-blue-400 font-medium">
                          +{typeof value === 'number' ? value.toFixed(1) : value}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default CollectiveMemoryPanel;
