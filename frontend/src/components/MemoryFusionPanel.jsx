/**
 * Memory Fusion & Long-Term Cognitive Persistence Panel (Step 31)
 * 
 * Provides comprehensive dashboard for AlphaZero's long-term memory system:
 * - Memory Nodes View: List of fused insights with decay tracking
 * - Persistence Map: Parameter evolution visualization
 * - Experience Timeline: Chronological learning progression
 * - Memory Health Index: Balance metrics and system health
 * - Export & Oversight: Admin tools for audit, export, and reset
 */

import React, { useState, useEffect } from 'react';
import { 
  Brain, 
  Database, 
  TrendingUp, 
  Activity, 
  Download, 
  RefreshCw,
  AlertCircle,
  CheckCircle,
  BarChart3,
  Clock,
  Zap,
  Shield,
  Trash2,
  Search
} from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || import.meta.env.REACT_APP_BACKEND_URL;

const MemoryFusionPanel = () => {
  const [activeTab, setActiveTab] = useState('nodes');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Memory data
  const [memoryNodes, setMemoryNodes] = useState([]);
  const [memoryProfile, setMemoryProfile] = useState(null);
  const [memoryHealth, setMemoryHealth] = useState(null);
  const [memoryStats, setMemoryStats] = useState(null);
  const [memoryTraces, setMemoryTraces] = useState([]);
  
  // Filters and pagination
  const [activeOnly, setActiveOnly] = useState(true);
  const [sortBy, setSortBy] = useState('timestamp');
  const [searchQuery, setSearchQuery] = useState('');
  
  // Load initial data
  useEffect(() => {
    loadMemoryData();
    // Refresh every 30 seconds
    const interval = setInterval(loadMemoryData, 30000);
    return () => clearInterval(interval);
  }, [activeTab, activeOnly, sortBy]);
  
  const loadMemoryData = async () => {
    try {
      setError(null);
      
      // Load data based on active tab
      if (activeTab === 'nodes') {
        await loadMemoryNodes();
      } else if (activeTab === 'profile') {
        await loadMemoryProfile();
      } else if (activeTab === 'timeline') {
        await loadMemoryTraces();
      } else if (activeTab === 'health') {
        await loadMemoryHealth();
      }
      
      // Always load stats for overview
      await loadMemoryStats();
      
    } catch (error) {
      console.error('Error loading memory data:', error);
      setError(error.message);
    }
  };
  
  const loadMemoryNodes = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `${BACKEND_URL}/llm/memory/nodes?active_only=${activeOnly}&sort_by=${sortBy}&limit=50`
      );
      const data = await response.json();
      if (data.success) {
        setMemoryNodes(data.nodes || []);
      }
    } finally {
      setLoading(false);
    }
  };
  
  const loadMemoryProfile = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/llm/memory/profile`);
      const data = await response.json();
      if (data.success) {
        setMemoryProfile(data.profile);
      }
    } finally {
      setLoading(false);
    }
  };
  
  const loadMemoryHealth = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/llm/memory/health`);
      const data = await response.json();
      if (data.success) {
        setMemoryHealth(data.health);
      }
    } finally {
      setLoading(false);
    }
  };
  
  const loadMemoryStats = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/llm/memory/stats`);
      const data = await response.json();
      if (data.success) {
        setMemoryStats(data.statistics);
      }
    } catch (error) {
      console.error('Error loading memory stats:', error);
    }
  };
  
  const loadMemoryTraces = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/llm/memory/traces?limit=50`);
      const data = await response.json();
      if (data.success) {
        setMemoryTraces(data.traces || []);
      }
    } finally {
      setLoading(false);
    }
  };
  
  const triggerManualFusion = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Get latest reflection cycle
      const reflectionResponse = await fetch(`${BACKEND_URL}/llm/reflection/history?limit=1`);
      const reflectionData = await reflectionResponse.json();
      
      if (!reflectionData.cycles || reflectionData.cycles.length === 0) {
        alert('No reflection cycles available. Run a reflection cycle first.');
        return;
      }
      
      const latestCycle = reflectionData.cycles[0];
      
      // Trigger fusion
      const fusionResponse = await fetch(
        `${BACKEND_URL}/llm/memory/fuse?reflection_cycle_id=${latestCycle._id}&trigger=manual`,
        { method: 'POST' }
      );
      
      const fusionData = await fusionResponse.json();
      
      if (fusionData.success) {
        alert(`Memory fusion successful! Created ${fusionData.fusion_result.new_memory_nodes} new memory nodes.`);
        await loadMemoryData();
      } else {
        alert('Memory fusion failed: ' + (fusionData.error || 'Unknown error'));
      }
      
    } catch (error) {
      console.error('Error triggering fusion:', error);
      alert('Error triggering fusion: ' + error.message);
    } finally {
      setLoading(false);
    }
  };
  
  const exportMemoryData = async () => {
    try {
      // Export all memory data as JSON
      const data = {
        timestamp: new Date().toISOString(),
        nodes: memoryNodes,
        profile: memoryProfile,
        health: memoryHealth,
        stats: memoryStats
      };
      
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `memory-export-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      alert('Memory data exported successfully!');
    } catch (error) {
      console.error('Error exporting memory:', error);
      alert('Export failed: ' + error.message);
    }
  };
  
  const resetMemorySystem = async () => {
    const confirmation = prompt(
      'WARNING: This will reset the entire memory system. All memory nodes will be backed up but cleared.\n\n' +
      'Type "CONFIRM_RESET" to proceed:'
    );
    
    if (confirmation !== 'CONFIRM_RESET') {
      alert('Reset cancelled.');
      return;
    }
    
    try {
      setLoading(true);
      
      const response = await fetch(`${BACKEND_URL}/llm/memory/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          confirmation: 'CONFIRM_RESET',
          admin_override: true
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        alert(
          `Memory system reset successfully!\n\n` +
          `Backup ID: ${data.backup_id}\n` +
          `Nodes cleared: ${data.nodes_cleared}\n` +
          `Profiles cleared: ${data.profiles_cleared}`
        );
        await loadMemoryData();
      } else {
        alert('Reset failed: ' + (data.error || 'Unknown error'));
      }
      
    } catch (error) {
      console.error('Error resetting memory:', error);
      alert('Reset failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Render functions for each tab
  const renderMemoryNodesView = () => {
    const filteredNodes = memoryNodes.filter(node => 
      !searchQuery || 
      node.key_insight.toLowerCase().includes(searchQuery.toLowerCase()) ||
      node.memory_id.toLowerCase().includes(searchQuery.toLowerCase())
    );
    
    return (
      <div className="space-y-4" data-testid="memory-nodes-view">
        {/* Filters */}
        <div className="flex flex-wrap gap-4 items-center bg-gray-50 p-4 rounded-lg">
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="activeOnly"
              checked={activeOnly}
              onChange={(e) => setActiveOnly(e.target.checked)}
              className="rounded"
            />
            <label htmlFor="activeOnly" className="text-sm">Active nodes only</label>
          </div>
          
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="px-3 py-1 border rounded text-sm"
          >
            <option value="timestamp">Sort by: Date</option>
            <option value="decay_weight">Sort by: Decay Weight</option>
            <option value="usage_count">Sort by: Usage Count</option>
          </select>
          
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-2.5 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search insights..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border rounded text-sm"
              />
            </div>
          </div>
          
          <div className="text-sm text-gray-600">
            {filteredNodes.length} of {memoryNodes.length} nodes
          </div>
        </div>
        
        {/* Memory Nodes List */}
        <div className="space-y-3">
          {filteredNodes.length === 0 ? (
            <div className="text-center py-12 bg-gray-50 rounded-lg">
              <Database className="w-12 h-12 text-gray-400 mx-auto mb-3" />
              <p className="text-gray-600">No memory nodes found</p>
              <p className="text-sm text-gray-500 mt-1">
                Memory nodes are created automatically after reflection cycles
              </p>
            </div>
          ) : (
            filteredNodes.map((node, index) => (
              <div
                key={node.memory_id || index}
                className="bg-white border rounded-lg p-4 hover:shadow-md transition-shadow"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Brain className="w-5 h-5 text-purple-600" />
                    <span className="font-mono text-xs text-gray-500">
                      {node.memory_id?.substring(0, 8)}...
                    </span>
                  </div>
                  
                  <div className="flex items-center gap-3">
                    {/* Decay Weight Indicator */}
                    <div className="flex items-center gap-1">
                      <div className="relative w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div 
                          className={`absolute inset-y-0 left-0 ${
                            node.decay_weight >= 0.7 ? 'bg-green-500' :
                            node.decay_weight >= 0.4 ? 'bg-yellow-500' :
                            'bg-red-500'
                          }`}
                          style={{ width: `${(node.decay_weight || 0) * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-600">
                        {((node.decay_weight || 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                    
                    {/* Usage Count */}
                    <div className="flex items-center gap-1 text-xs text-gray-600">
                      <Zap className="w-3 h-3" />
                      <span>{node.usage_count || 0}</span>
                    </div>
                    
                    {/* Ethical Alignment */}
                    <div className="flex items-center gap-1 text-xs text-gray-600">
                      <Shield className="w-3 h-3 text-blue-600" />
                      <span>{((node.ethical_alignment || 0) * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
                
                {/* Key Insight */}
                <p className="text-sm text-gray-800 mb-3 leading-relaxed">
                  {node.key_insight}
                </p>
                
                {/* Metadata */}
                <div className="flex flex-wrap gap-4 text-xs text-gray-500">
                  <div className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    <span>{new Date(node.timestamp).toLocaleDateString()}</span>
                  </div>
                  
                  {node.context?.insight_type && (
                    <span className="px-2 py-0.5 bg-purple-100 text-purple-700 rounded">
                      {node.context.insight_type.replace('_', ' ')}
                    </span>
                  )}
                  
                  <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded">
                    Confidence: {((node.confidence_score || 0) * 100).toFixed(0)}%
                  </span>
                  
                  {Object.keys(node.parameter_delta || {}).length > 0 && (
                    <span className="px-2 py-0.5 bg-green-100 text-green-700 rounded">
                      {Object.keys(node.parameter_delta).length} param adjustments
                    </span>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    );
  };
  
  const renderPersistenceMap = () => {
    if (!memoryProfile) {
      return (
        <div className="text-center py-12">
          <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-3" />
          <p className="text-gray-600">Loading persistence profile...</p>
        </div>
      );
    }
    
    const { 
      total_active_memories, 
      memory_distribution, 
      learning_trends,
      memory_efficiency,
      learning_velocity,
      target_comparison,
      summary
    } = memoryProfile;
    
    return (
      <div className="space-y-6" data-testid="persistence-map">
        {/* Summary */}
        {summary && (
          <div className="bg-gradient-to-r from-purple-50 to-blue-50 p-4 rounded-lg border border-purple-200">
            <h3 className="font-semibold text-purple-900 mb-2 flex items-center gap-2">
              <Brain className="w-5 h-5" />
              Long-Term Cognitive Profile
            </h3>
            <p className="text-sm text-gray-700 leading-relaxed">{summary}</p>
          </div>
        )}
        
        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white border rounded-lg p-4">
            <div className="text-sm text-gray-600 mb-1">Active Memories</div>
            <div className="text-2xl font-bold text-gray-900">{total_active_memories}</div>
            <div className="text-xs text-gray-500 mt-1">Total nodes in system</div>
          </div>
          
          <div className="bg-white border rounded-lg p-4">
            <div className="text-sm text-gray-600 mb-1">Memory Efficiency</div>
            <div className="text-2xl font-bold text-gray-900">
              {((memory_efficiency || 0) * 100).toFixed(0)}%
            </div>
            <div className="text-xs text-gray-500 mt-1">Active/Total ratio</div>
          </div>
          
          <div className="bg-white border rounded-lg p-4">
            <div className="text-sm text-gray-600 mb-1">Learning Velocity</div>
            <div className="text-2xl font-bold text-gray-900">
              {(learning_velocity || 0).toFixed(3)}
            </div>
            <div className="text-xs text-gray-500 mt-1">Parameter change rate</div>
          </div>
        </div>
        
        {/* Memory Distribution */}
        {memory_distribution && (
          <div className="bg-white border rounded-lg p-4">
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              <Database className="w-5 h-5 text-purple-600" />
              Memory Distribution by Type
            </h3>
            <div className="space-y-2">
              {Object.entries(memory_distribution).map(([type, count]) => (
                <div key={type} className="flex items-center justify-between">
                  <span className="text-sm text-gray-700 capitalize">
                    {type.replace('_', ' ')}
                  </span>
                  <div className="flex items-center gap-2">
                    <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-purple-500"
                        style={{ 
                          width: `${(count / total_active_memories) * 100}%` 
                        }}
                      />
                    </div>
                    <span className="text-sm font-medium text-gray-900 w-8 text-right">
                      {count}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Learning Trends */}
        {learning_trends && (
          <div className="bg-white border rounded-lg p-4">
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-green-600" />
              Learning Trends
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <span className="text-sm text-gray-700">Creativity</span>
                <span className={`text-sm font-medium ${
                  learning_trends.creativity === 'improving' ? 'text-green-600' :
                  learning_trends.creativity === 'declining' ? 'text-red-600' :
                  'text-gray-600'
                }`}>
                  {learning_trends.creativity}
                </span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <span className="text-sm text-gray-700">Stability</span>
                <span className={`text-sm font-medium ${
                  learning_trends.stability === 'improving' ? 'text-green-600' :
                  learning_trends.stability === 'declining' ? 'text-red-600' :
                  'text-gray-600'
                }`}>
                  {learning_trends.stability}
                </span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <span className="text-sm text-gray-700">Ethical Alignment</span>
                <span className={`text-sm font-medium ${
                  learning_trends.ethical_alignment === 'improving' ? 'text-green-600' :
                  learning_trends.ethical_alignment === 'declining' ? 'text-red-600' :
                  'text-gray-600'
                }`}>
                  {learning_trends.ethical_alignment}
                </span>
              </div>
            </div>
          </div>
        )}
        
        {/* Target Comparison */}
        {target_comparison && (
          <div className="bg-white border rounded-lg p-4">
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-600" />
              Performance vs. Targets
            </h3>
            <div className="space-y-3">
              {Object.entries(target_comparison).map(([metric, data]) => (
                <div key={metric} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">{data.status}</span>
                    <span className="text-sm text-gray-700 capitalize">
                      {metric.replace(/_/g, ' ')}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium text-gray-900">
                      {typeof data.current === 'number' ? 
                        data.current.toFixed(2) : data.current}
                    </div>
                    <div className="text-xs text-gray-500">
                      Target: {typeof data.target === 'number' ? 
                        data.target.toFixed(2) : data.target}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };
  
  const renderExperienceTimeline = () => {
    return (
      <div className="space-y-4" data-testid="experience-timeline">
        <div className="bg-white border rounded-lg p-4">
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            <Clock className="w-5 h-5 text-purple-600" />
            Memory Traces & Events
          </h3>
          
          {memoryTraces.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <p>No memory traces recorded yet</p>
            </div>
          ) : (
            <div className="space-y-3">
              {memoryTraces.slice(0, 20).map((trace, index) => (
                <div 
                  key={trace.trace_id || index}
                  className="flex items-start gap-3 p-3 bg-gray-50 rounded hover:bg-gray-100 transition-colors"
                >
                  <div className="flex-shrink-0 mt-1">
                    {trace.trace_type === 'fusion_cycle' ? (
                      <Brain className="w-5 h-5 text-purple-600" />
                    ) : (
                      <Search className="w-5 h-5 text-blue-600" />
                    )}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-sm font-medium text-gray-900">
                        {trace.trace_type === 'fusion_cycle' ? 'Memory Fusion' : 'Memory Retrieval'}
                      </span>
                      <span className="text-xs text-gray-500">
                        {new Date(trace.timestamp).toLocaleString()}
                      </span>
                    </div>
                    
                    {trace.trace_type === 'fusion_cycle' && (
                      <div className="text-sm text-gray-600">
                        Created {trace.new_memory_nodes} nodes in {trace.fusion_time_seconds?.toFixed(2)}s
                      </div>
                    )}
                    
                    {trace.trace_type === 'retrieval' && trace.retrieval_query && (
                      <div className="text-sm text-gray-600">
                        Query: "{trace.retrieval_query.substring(0, 80)}..."
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  };
  
  const renderMemoryHealth = () => {
    if (!memoryHealth) {
      return (
        <div className="text-center py-12">
          <Activity className="w-12 h-12 text-gray-400 mx-auto mb-3" />
          <p className="text-gray-600">Loading health metrics...</p>
        </div>
      );
    }
    
    const { status, emoji, health_score, metrics, recommendations } = memoryHealth;
    
    return (
      <div className="space-y-6" data-testid="memory-health-index">
        {/* Overall Health Status */}
        <div className={`p-6 rounded-lg border-2 ${
          status === 'excellent' || status === 'good' ? 'bg-green-50 border-green-300' :
          status === 'moderate' ? 'bg-yellow-50 border-yellow-300' :
          'bg-red-50 border-red-300'
        }`}>
          <div className="flex items-center gap-3 mb-2">
            <span className="text-4xl">{emoji}</span>
            <div>
              <h3 className="text-lg font-bold capitalize">{status}</h3>
              <p className="text-sm text-gray-600">Overall Health Score: {(health_score * 100).toFixed(0)}%</p>
            </div>
          </div>
        </div>
        
        {/* Health Metrics Grid */}
        {metrics && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-white border rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">Memory Retention Index</span>
                <span className="text-lg font-bold">{(metrics.memory_retention_index * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-blue-500"
                  style={{ width: `${metrics.memory_retention_index * 100}%` }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">Target: ≥90%</p>
            </div>
            
            <div className="bg-white border rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">Fusion Efficiency</span>
                <span className="text-lg font-bold">{(metrics.fusion_efficiency * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-purple-500"
                  style={{ width: `${metrics.fusion_efficiency * 100}%` }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">Target: ≥85%</p>
            </div>
            
            <div className="bg-white border rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">Ethical Continuity</span>
                <span className="text-lg font-bold">{(metrics.ethical_continuity * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-green-500"
                  style={{ width: `${metrics.ethical_continuity * 100}%` }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">Target: ≥92%</p>
            </div>
            
            <div className="bg-white border rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">Retrieval Latency</span>
                <span className="text-lg font-bold">{metrics.retrieval_latency.toFixed(2)}s</span>
              </div>
              <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className={`h-full ${
                    metrics.retrieval_latency <= 2.0 ? 'bg-green-500' : 'bg-yellow-500'
                  }`}
                  style={{ width: `${Math.min(100, (2.0 / metrics.retrieval_latency) * 100)}%` }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">Target: ≤2.0s</p>
            </div>
          </div>
        )}
        
        {/* Additional Stats */}
        {metrics && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white border rounded-lg p-3 text-center">
              <div className="text-2xl font-bold text-gray-900">{metrics.total_nodes}</div>
              <div className="text-xs text-gray-500">Total Nodes</div>
            </div>
            <div className="bg-white border rounded-lg p-3 text-center">
              <div className="text-2xl font-bold text-green-600">{metrics.active_nodes}</div>
              <div className="text-xs text-gray-500">Active Nodes</div>
            </div>
            <div className="bg-white border rounded-lg p-3 text-center">
              <div className="text-2xl font-bold text-gray-400">{metrics.decayed_nodes}</div>
              <div className="text-xs text-gray-500">Decayed Nodes</div>
            </div>
            <div className="bg-white border rounded-lg p-3 text-center">
              <div className="text-2xl font-bold text-purple-600">
                {(metrics.memory_diversity * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-gray-500">Memory Diversity</div>
            </div>
          </div>
        )}
        
        {/* Recommendations */}
        {recommendations && recommendations.length > 0 && (
          <div className="bg-white border rounded-lg p-4">
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-blue-600" />
              Health Recommendations
            </h3>
            <ul className="space-y-2">
              {recommendations.map((rec, index) => (
                <li key={index} className="flex items-start gap-2 text-sm text-gray-700">
                  <span className="text-blue-600 mt-0.5">•</span>
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  };
  
  const renderExportOversight = () => {
    return (
      <div className="space-y-6" data-testid="export-oversight">
        {/* Statistics Overview */}
        {memoryStats && (
          <div className="bg-white border rounded-lg p-4">
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              <Database className="w-5 h-5 text-purple-600" />
              System Statistics
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div>
                <div className="text-sm text-gray-600">Total Memory Nodes</div>
                <div className="text-2xl font-bold text-gray-900">{memoryStats.total_memory_nodes}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600">Active Nodes</div>
                <div className="text-2xl font-bold text-green-600">{memoryStats.active_memory_nodes}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600">Decayed Nodes</div>
                <div className="text-2xl font-bold text-gray-400">{memoryStats.decayed_memory_nodes}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600">Total Traces</div>
                <div className="text-2xl font-bold text-blue-600">{memoryStats.total_traces}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600">Decay Lambda (λ)</div>
                <div className="text-2xl font-bold text-purple-600">{memoryStats.decay_lambda}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600">Retention Window</div>
                <div className="text-2xl font-bold text-indigo-600">{memoryStats.retention_window_games} games</div>
              </div>
            </div>
            
            {memoryStats.latest_fusion && (
              <div className="mt-4 pt-4 border-t">
                <div className="text-sm text-gray-600">Last Fusion Cycle</div>
                <div className="text-sm font-medium text-gray-900">
                  {new Date(memoryStats.latest_fusion).toLocaleString()}
                </div>
              </div>
            )}
          </div>
        )}
        
        {/* Admin Actions */}
        <div className="bg-white border rounded-lg p-4">
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-600" />
            Admin Actions
          </h3>
          
          <div className="space-y-3">
            <button
              onClick={triggerManualFusion}
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
              Trigger Manual Fusion Cycle
            </button>
            
            <button
              onClick={exportMemoryData}
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Download className="w-5 h-5" />
              Export Memory Data
            </button>
            
            <button
              onClick={resetMemorySystem}
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Trash2 className="w-5 h-5" />
              Reset Memory System
            </button>
          </div>
          
          <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded">
            <div className="flex items-start gap-2">
              <AlertCircle className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-yellow-800">
                <p className="font-medium mb-1">Important Notes:</p>
                <ul className="space-y-1 text-xs">
                  <li>• Memory fusion automatically triggers after reflection cycles</li>
                  <li>• Memory nodes use exponential decay (λ = {memoryStats?.decay_lambda || 0.05})</li>
                  <li>• System operates in advisory mode only - no automatic gameplay changes</li>
                  <li>• All resets create backups before clearing data</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };
  
  // Main render
  return (
    <div className="w-full bg-white rounded-lg shadow-lg" data-testid="memory-fusion-panel">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-6 rounded-t-lg">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold flex items-center gap-2">
              <Brain className="w-8 h-8" />
              Memory Fusion & Long-Term Persistence
            </h2>
            <p className="text-purple-100 mt-1">
              Step 31: Emergent cognitive memory architecture with cross-step recall
            </p>
          </div>
          
          {memoryStats && (
            <div className="text-right">
              <div className="text-3xl font-bold">{memoryStats.active_memory_nodes}</div>
              <div className="text-sm text-purple-100">Active Memories</div>
            </div>
          )}
        </div>
      </div>
      
      {/* Error Display */}
      {error && (
        <div className="mx-6 mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-2">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <div className="font-medium text-red-900">Error</div>
            <div className="text-sm text-red-700">{error}</div>
          </div>
        </div>
      )}
      
      {/* Tab Navigation */}
      <div className="border-b px-6">
        <nav className="flex space-x-8">
          <button
            onClick={() => setActiveTab('nodes')}
            className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'nodes'
                ? 'border-purple-600 text-purple-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4" />
              Memory Nodes
            </div>
          </button>
          
          <button
            onClick={() => setActiveTab('profile')}
            className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'profile'
                ? 'border-purple-600 text-purple-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              Persistence Map
            </div>
          </button>
          
          <button
            onClick={() => setActiveTab('timeline')}
            className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'timeline'
                ? 'border-purple-600 text-purple-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4" />
              Experience Timeline
            </div>
          </button>
          
          <button
            onClick={() => setActiveTab('health')}
            className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'health'
                ? 'border-purple-600 text-purple-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Memory Health
            </div>
          </button>
          
          <button
            onClick={() => setActiveTab('export')}
            className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'export'
                ? 'border-purple-600 text-purple-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center gap-2">
              <Download className="w-4 h-4" />
              Export & Oversight
            </div>
          </button>
        </nav>
      </div>
      
      {/* Tab Content */}
      <div className="p-6">
        {loading && !memoryNodes.length && !memoryProfile && !memoryHealth ? (
          <div className="text-center py-12">
            <RefreshCw className="w-8 h-8 text-purple-600 animate-spin mx-auto mb-3" />
            <p className="text-gray-600">Loading memory data...</p>
          </div>
        ) : (
          <>
            {activeTab === 'nodes' && renderMemoryNodesView()}
            {activeTab === 'profile' && renderPersistenceMap()}
            {activeTab === 'timeline' && renderExperienceTimeline()}
            {activeTab === 'health' && renderMemoryHealth()}
            {activeTab === 'export' && renderExportOversight()}
          </>
        )}
      </div>
    </div>
  );
};

export default MemoryFusionPanel;
