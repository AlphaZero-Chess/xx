import React, { useState, useEffect } from 'react';
import { Activity, TrendingUp, Shield, AlertCircle, CheckCircle, Zap, BarChart3, Network, FileText } from 'lucide-react';

const CohesionCorePanel = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [cohesionStatus, setCohesionStatus] = useState(null);
  const [cohesionHistory, setCohesionHistory] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [healthRecords, setHealthRecords] = useState([]);
  const [latestReport, setLatestReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [triggering, setTriggering] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

  // Fetch cohesion data
  const fetchCohesionData = async () => {
    try {
      // Get status
      const statusRes = await fetch(`${BACKEND_URL}/api/llm/cohesion/status`);
      const statusData = await statusRes.json();
      if (statusData.success) {
        setCohesionStatus(statusData);
      }

      // Get history
      const historyRes = await fetch(`${BACKEND_URL}/api/llm/cohesion/history?limit=10`);
      const historyData = await historyRes.json();
      if (historyData.success) {
        setCohesionHistory(historyData.cycles || []);
      }

      // Get metrics
      const metricsRes = await fetch(`${BACKEND_URL}/api/llm/cohesion/metrics?limit=20`);
      const metricsData = await metricsRes.json();
      if (metricsData.success) {
        setMetrics(metricsData.metrics || []);
      }

      // Get health records
      const healthRes = await fetch(`${BACKEND_URL}/api/llm/cohesion/health?limit=20`);
      const healthData = await healthRes.json();
      if (healthData.success) {
        setHealthRecords(healthData.health_records || []);
      }

      // Get latest report
      const reportRes = await fetch(`${BACKEND_URL}/api/llm/cohesion/report`);
      const reportData = await reportRes.json();
      if (reportData.success && reportData.report) {
        setLatestReport(reportData.report);
      }

      setLoading(false);
    } catch (error) {
      console.error('Error fetching cohesion data:', error);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCohesionData();
    
    // Auto-refresh every 30 seconds if enabled
    let interval;
    if (autoRefresh) {
      interval = setInterval(fetchCohesionData, 30000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  // Trigger cohesion cycle
  const triggerCohesion = async () => {
    setTriggering(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/llm/cohesion/trigger?trigger=manual`, {
        method: 'POST'
      });
      const data = await res.json();
      if (data.success) {
        alert(`Cohesion cycle triggered successfully!\nAlignment: ${data.metrics.alignment_score}\nHealth: ${data.metrics.system_health_index}`);
        fetchCohesionData();
      } else {
        alert('Failed to trigger cohesion cycle');
      }
    } catch (error) {
      console.error('Error triggering cohesion:', error);
      alert('Error triggering cohesion cycle');
    } finally {
      setTriggering(false);
    }
  };

  // Helper to format timestamp
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    try {
      return new Date(timestamp).toLocaleString();
    } catch {
      return timestamp;
    }
  };

  // Helper to get status icon and color
  const getStatusIndicator = (status, value, target) => {
    if (status === '✅' || value >= target) {
      return { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-100' };
    } else {
      return { icon: AlertCircle, color: 'text-amber-500', bg: 'bg-amber-100' };
    }
  };

  // Helper to get health color
  const getHealthColor = (health) => {
    if (health === 'excellent') return 'text-green-600 bg-green-100';
    if (health === 'good') return 'text-blue-600 bg-blue-100';
    if (health === 'moderate') return 'text-amber-600 bg-amber-100';
    return 'text-red-600 bg-red-100';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-indigo-600 rounded-xl shadow-lg p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center space-x-3">
              <Network className="w-10 h-10" />
              <div>
                <h2 className="text-3xl font-bold">Cohesion Core</h2>
                <p className="text-purple-100">Systemic Unification Dashboard</p>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-4 py-2 rounded-lg font-medium transition ${
                autoRefresh 
                  ? 'bg-white text-indigo-600 hover:bg-purple-50' 
                  : 'bg-purple-700 text-white hover:bg-purple-800'
              }`}
            >
              {autoRefresh ? 'Auto-Refresh ON' : 'Auto-Refresh OFF'}
            </button>
            <button
              onClick={triggerCohesion}
              disabled={triggering}
              className="px-6 py-2 bg-white text-indigo-600 rounded-lg font-medium hover:bg-purple-50 transition disabled:opacity-50 flex items-center space-x-2"
            >
              <Zap className="w-5 h-5" />
              <span>{triggering ? 'Triggering...' : 'Trigger Cohesion'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="bg-white rounded-xl shadow-lg overflow-hidden">
        <div className="flex border-b border-gray-200">
          {[
            { id: 'overview', label: 'System Overview', icon: Activity },
            { id: 'harmony', label: 'Parameter Harmony', icon: TrendingUp },
            { id: 'graph', label: 'Cross-System Graph', icon: Network },
            { id: 'ethics', label: 'Ethical Integrity', icon: Shield },
            { id: 'history', label: 'History & Reports', icon: FileText }
          ].map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 flex items-center justify-center space-x-2 px-6 py-4 font-medium transition ${
                  activeTab === tab.id
                    ? 'bg-indigo-50 text-indigo-600 border-b-2 border-indigo-600'
                    : 'text-gray-600 hover:bg-gray-50'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Tab Content */}
        <div className="p-6">
          {/* Tab 1: System Overview */}
          {activeTab === 'overview' && (
            <div className="space-y-6">
              <h3 className="text-2xl font-bold text-gray-800 flex items-center space-x-2">
                <Activity className="w-7 h-7 text-indigo-600" />
                <span>System Overview</span>
              </h3>

              {/* Status Cards */}
              {cohesionStatus && cohesionStatus.target_comparison && (
                <div className="grid grid-cols-3 gap-6">
                  {/* Alignment Score */}
                  <div className="bg-gradient-to-br from-blue-50 to-indigo-100 rounded-xl p-6 border border-blue-200">
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-sm font-semibold text-gray-600">Alignment Score</span>
                      <span className={getStatusIndicator('', cohesionStatus.alignment_score, cohesionStatus.target_comparison.alignment.target).color}>
                        {cohesionStatus.target_comparison.alignment.status}
                      </span>
                    </div>
                    <div className="flex items-baseline space-x-2">
                      <span className="text-4xl font-bold text-gray-800">
                        {cohesionStatus.alignment_score?.toFixed(2) || 'N/A'}
                      </span>
                      <span className="text-sm text-gray-500">/ {cohesionStatus.target_comparison.alignment.target}</span>
                    </div>
                    <div className="mt-2">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-indigo-600 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${(cohesionStatus.alignment_score || 0) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>

                  {/* System Health */}
                  <div className="bg-gradient-to-br from-green-50 to-emerald-100 rounded-xl p-6 border border-green-200">
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-sm font-semibold text-gray-600">System Health</span>
                      <span className={getStatusIndicator('', cohesionStatus.system_health_index, cohesionStatus.target_comparison.health.target).color}>
                        {cohesionStatus.target_comparison.health.status}
                      </span>
                    </div>
                    <div className="flex items-baseline space-x-2">
                      <span className="text-4xl font-bold text-gray-800">
                        {cohesionStatus.system_health_index?.toFixed(2) || 'N/A'}
                      </span>
                      <span className="text-sm text-gray-500">/ {cohesionStatus.target_comparison.health.target}</span>
                    </div>
                    <div className="mt-2">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-green-600 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${(cohesionStatus.system_health_index || 0) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>

                  {/* Ethical Continuity */}
                  <div className="bg-gradient-to-br from-purple-50 to-pink-100 rounded-xl p-6 border border-purple-200">
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-sm font-semibold text-gray-600">Ethical Continuity</span>
                      <span className={getStatusIndicator('', cohesionStatus.ethical_continuity, cohesionStatus.target_comparison.ethics.target).color}>
                        {cohesionStatus.target_comparison.ethics.status}
                      </span>
                    </div>
                    <div className="flex items-baseline space-x-2">
                      <span className="text-4xl font-bold text-gray-800">
                        {cohesionStatus.ethical_continuity?.toFixed(2) || 'N/A'}
                      </span>
                      <span className="text-sm text-gray-500">/ {cohesionStatus.target_comparison.ethics.target}</span>
                    </div>
                    <div className="mt-2">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-purple-600 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${(cohesionStatus.ethical_continuity || 0) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Cohesion Health Status */}
              {cohesionStatus && (
                <div className={`rounded-xl p-6 border ${getHealthColor(cohesionStatus.cohesion_health || 'unknown')}`}>
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="text-lg font-bold mb-1">Overall Cohesion Status</h4>
                      <p className="text-sm opacity-80">
                        Last cycle: {formatTimestamp(cohesionStatus.last_cycle)}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="text-3xl font-bold uppercase">{cohesionStatus.cohesion_health || 'Unknown'}</div>
                      <div className="text-sm mt-1">
                        {cohesionStatus.drift_detected && (
                          <span className="flex items-center space-x-1">
                            <AlertCircle className="w-4 h-4" />
                            <span>Drift Detected</span>
                          </span>
                        )}
                        {cohesionStatus.auto_healing_active && (
                          <span className="flex items-center space-x-1">
                            <Zap className="w-4 h-4" />
                            <span>Auto-Healing Active</span>
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Module States */}
              {latestReport && latestReport.module_states && (
                <div className="space-y-4">
                  <h4 className="text-lg font-bold text-gray-800">Module States</h4>
                  <div className="grid grid-cols-3 gap-4">
                    {Object.entries(latestReport.module_states).map(([moduleName, state]) => (
                      <div key={moduleName} className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition">
                        <div className="flex items-center justify-between mb-2">
                          <h5 className="font-bold text-gray-700 capitalize">{moduleName}</h5>
                          <span className={`px-2 py-1 rounded text-xs font-semibold ${
                            state.status === 'operational' ? 'bg-green-100 text-green-700' : 
                            state.status === 'degraded' ? 'bg-amber-100 text-amber-700' : 
                            'bg-gray-100 text-gray-700'
                          }`}>
                            {state.status}
                          </span>
                        </div>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">Health:</span>
                            <span className="font-semibold">{(state.health_score * 100).toFixed(0)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Ethics:</span>
                            <span className="font-semibold">{(state.ethical_alignment * 100).toFixed(0)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Activity:</span>
                            <span className="font-semibold">{(state.activity_level * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Tab 2: Parameter Harmony */}
          {activeTab === 'harmony' && (
            <div className="space-y-6">
              <h3 className="text-2xl font-bold text-gray-800 flex items-center space-x-2">
                <TrendingUp className="w-7 h-7 text-indigo-600" />
                <span>Parameter Harmony</span>
              </h3>

              {latestReport && latestReport.parameter_comparison && (
                <div className="space-y-6">
                  {/* Parameter Alignment Table */}
                  <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
                    <table className="w-full">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Parameter</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Min</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Max</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Avg</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Delta</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Aligned</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {Object.entries(latestReport.parameter_comparison.parameter_alignment || {}).map(([param, data]) => (
                          <tr key={param} className="hover:bg-gray-50">
                            <td className="px-6 py-4 font-medium text-gray-900">{param}</td>
                            <td className="px-6 py-4 text-gray-700">{data.min}</td>
                            <td className="px-6 py-4 text-gray-700">{data.max}</td>
                            <td className="px-6 py-4 text-gray-700">{data.avg}</td>
                            <td className="px-6 py-4">
                              <span className={`font-semibold ${data.delta > 0.10 ? 'text-red-600' : 'text-green-600'}`}>
                                {data.delta}
                              </span>
                            </td>
                            <td className="px-6 py-4">
                              {data.aligned ? (
                                <CheckCircle className="w-5 h-5 text-green-600" />
                              ) : (
                                <AlertCircle className="w-5 h-5 text-amber-600" />
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Parameter by Module */}
                  <div className="grid grid-cols-3 gap-4">
                    {Object.entries(latestReport.parameter_comparison.modules || {}).map(([moduleName, moduleData]) => (
                      <div key={moduleName} className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                        <h5 className="font-bold text-gray-800 capitalize mb-3">{moduleName}</h5>
                        <div className="space-y-2 text-sm">
                          {Object.entries(moduleData.parameters || {}).map(([param, value]) => (
                            <div key={param} className="flex justify-between">
                              <span className="text-gray-600">{param}:</span>
                              <span className="font-semibold">{typeof value === 'number' ? value.toFixed(3) : value}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Metrics Trend */}
              {metrics.length > 0 && (
                <div className="bg-white border border-gray-200 rounded-lg p-6">
                  <h4 className="font-bold text-gray-800 mb-4">Parameter Harmony Trend</h4>
                  <div className="space-y-2">
                    {metrics.slice(0, 10).map((metric, idx) => (
                      <div key={idx} className="flex items-center space-x-3">
                        <span className="text-xs text-gray-500 w-32">{formatTimestamp(metric.timestamp)}</span>
                        <div className="flex-1 bg-gray-100 rounded-full h-6 overflow-hidden">
                          <div 
                            className="bg-gradient-to-r from-indigo-500 to-purple-500 h-full flex items-center justify-end pr-2"
                            style={{ width: `${(metric.parameter_harmony_score || 0) * 100}%` }}
                          >
                            <span className="text-xs font-semibold text-white">{(metric.parameter_harmony_score || 0).toFixed(2)}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Tab 3: Cross-System Graph */}
          {activeTab === 'graph' && (
            <div className="space-y-6">
              <h3 className="text-2xl font-bold text-gray-800 flex items-center space-x-2">
                <Network className="w-7 h-7 text-indigo-600" />
                <span>Cross-System Communication</span>
              </h3>

              {/* Visual representation of module connections */}
              <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl p-8 border border-indigo-200">
                <div className="flex items-center justify-center space-x-12">
                  {/* Creativity Node */}
                  <div className="text-center">
                    <div className="w-32 h-32 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold text-lg shadow-lg hover:scale-110 transition">
                      Creativity
                      <br />
                      <span className="text-sm">Step 29</span>
                    </div>
                    {latestReport && latestReport.module_states?.creativity && (
                      <div className="mt-3 text-sm">
                        <div>Health: {(latestReport.module_states.creativity.health_score * 100).toFixed(0)}%</div>
                        <div>Ethics: {(latestReport.module_states.creativity.ethical_alignment * 100).toFixed(0)}%</div>
                      </div>
                    )}
                  </div>

                  {/* Arrows */}
                  <div className="flex flex-col items-center space-y-4">
                    <div className="text-2xl">→</div>
                    <div className="text-xs text-gray-600">Strategies</div>
                  </div>

                  {/* Reflection Node */}
                  <div className="text-center">
                    <div className="w-32 h-32 bg-green-500 rounded-full flex items-center justify-center text-white font-bold text-lg shadow-lg hover:scale-110 transition">
                      Reflection
                      <br />
                      <span className="text-sm">Step 30</span>
                    </div>
                    {latestReport && latestReport.module_states?.reflection && (
                      <div className="mt-3 text-sm">
                        <div>Health: {(latestReport.module_states.reflection.health_score * 100).toFixed(0)}%</div>
                        <div>Ethics: {(latestReport.module_states.reflection.ethical_alignment * 100).toFixed(0)}%</div>
                      </div>
                    )}
                  </div>

                  {/* Arrows */}
                  <div className="flex flex-col items-center space-y-4">
                    <div className="text-2xl">→</div>
                    <div className="text-xs text-gray-600">Insights</div>
                  </div>

                  {/* Memory Node */}
                  <div className="text-center">
                    <div className="w-32 h-32 bg-purple-500 rounded-full flex items-center justify-center text-white font-bold text-lg shadow-lg hover:scale-110 transition">
                      Memory
                      <br />
                      <span className="text-sm">Step 31</span>
                    </div>
                    {latestReport && latestReport.module_states?.memory && (
                      <div className="mt-3 text-sm">
                        <div>Health: {(latestReport.module_states.memory.health_score * 100).toFixed(0)}%</div>
                        <div>Ethics: {(latestReport.module_states.memory.ethical_alignment * 100).toFixed(0)}%</div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Cohesion Core in center */}
                <div className="flex justify-center mt-8">
                  <div className="text-center">
                    <div className="text-sm text-gray-600 mb-2">↓ Cohesion Synchronization ↓</div>
                    <div className="w-40 h-40 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-full flex items-center justify-center text-white font-bold text-xl shadow-2xl">
                      Cohesion
                      <br />
                      Core
                      <br />
                      <span className="text-sm">Step 32</span>
                    </div>
                    {cohesionStatus && (
                      <div className="mt-3 text-sm font-semibold">
                        <div>Alignment: {cohesionStatus.alignment_score?.toFixed(2)}</div>
                        <div>Health: {cohesionStatus.system_health_index?.toFixed(2)}</div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Communication Metrics */}
              {metrics.length > 0 && (
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-white border border-gray-200 rounded-lg p-6">
                    <h5 className="font-bold text-gray-800 mb-4">Creativity ↔ Reflection Delta</h5>
                    <div className="space-y-2">
                      {metrics.slice(0, 5).map((metric, idx) => (
                        <div key={idx} className="flex justify-between items-center text-sm">
                          <span className="text-gray-600">{formatTimestamp(metric.timestamp).split(',')[1]}</span>
                          <span className={`font-semibold ${
                            metric.creativity_reflection_delta > 0.10 ? 'text-red-600' : 'text-green-600'
                          }`}>
                            Δ {metric.creativity_reflection_delta?.toFixed(3)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="bg-white border border-gray-200 rounded-lg p-6">
                    <h5 className="font-bold text-gray-800 mb-4">Memory ↔ Ethics Delta</h5>
                    <div className="space-y-2">
                      {metrics.slice(0, 5).map((metric, idx) => (
                        <div key={idx} className="flex justify-between items-center text-sm">
                          <span className="text-gray-600">{formatTimestamp(metric.timestamp).split(',')[1]}</span>
                          <span className={`font-semibold ${
                            metric.memory_ethics_delta > 0.08 ? 'text-red-600' : 'text-green-600'
                          }`}>
                            Δ {metric.memory_ethics_delta?.toFixed(3)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Tab 4: Ethical Integrity Monitor */}
          {activeTab === 'ethics' && (
            <div className="space-y-6">
              <h3 className="text-2xl font-bold text-gray-800 flex items-center space-x-2">
                <Shield className="w-7 h-7 text-indigo-600" />
                <span>Ethical Integrity Monitor</span>
              </h3>

              {/* Ethical Continuity Chart */}
              {metrics.length > 0 && (
                <div className="bg-white border border-gray-200 rounded-lg p-6">
                  <h4 className="font-bold text-gray-800 mb-4">Ethical Continuity Timeline</h4>
                  <div className="space-y-3">
                    {metrics.slice(0, 10).map((metric, idx) => (
                      <div key={idx}>
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-xs text-gray-600">{formatTimestamp(metric.timestamp)}</span>
                          <span className={`text-sm font-semibold ${
                            metric.ethical_continuity >= 0.92 ? 'text-green-600' : 
                            metric.ethical_continuity >= 0.85 ? 'text-blue-600' : 'text-amber-600'
                          }`}>
                            {metric.ethical_continuity?.toFixed(2)}
                          </span>
                        </div>
                        <div className="w-full bg-gray-100 rounded-full h-4 overflow-hidden">
                          <div 
                            className={`h-full flex items-center justify-end pr-2 ${
                              metric.ethical_continuity >= 0.92 ? 'bg-green-500' : 
                              metric.ethical_continuity >= 0.85 ? 'bg-blue-500' : 'bg-amber-500'
                            }`}
                            style={{ width: `${(metric.ethical_continuity || 0) * 100}%` }}
                          >
                            <span className="text-xs font-semibold text-white">
                              {((metric.ethical_continuity || 0) * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Module Ethical Scores */}
              {metrics.length > 0 && (
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                    <h5 className="font-bold text-gray-800 mb-3">Creativity Ethics</h5>
                    <div className="text-4xl font-bold text-blue-600 mb-2">
                      {metrics[0]?.creativity_health?.toFixed(2) || 'N/A'}
                    </div>
                    <div className="text-sm text-gray-600">Health Score</div>
                  </div>

                  <div className="bg-green-50 border border-green-200 rounded-lg p-6">
                    <h5 className="font-bold text-gray-800 mb-3">Reflection Ethics</h5>
                    <div className="text-4xl font-bold text-green-600 mb-2">
                      {metrics[0]?.reflection_health?.toFixed(2) || 'N/A'}
                    </div>
                    <div className="text-sm text-gray-600">Health Score</div>
                  </div>

                  <div className="bg-purple-50 border border-purple-200 rounded-lg p-6">
                    <h5 className="font-bold text-gray-800 mb-3">Memory Ethics</h5>
                    <div className="text-4xl font-bold text-purple-600 mb-2">
                      {metrics[0]?.memory_health?.toFixed(2) || 'N/A'}
                    </div>
                    <div className="text-sm text-gray-600">Health Score</div>
                  </div>
                </div>
              )}

              {/* Ethical Compliance Summary */}
              {latestReport && (
                <div className="bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200 rounded-lg p-6">
                  <h4 className="font-bold text-gray-800 mb-4">Ethical Compliance Summary</h4>
                  <div className="prose max-w-none">
                    <p className="text-gray-700 leading-relaxed">
                      {latestReport.health_analysis || 'No analysis available'}
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Tab 5: History & Reports */}
          {activeTab === 'history' && (
            <div className="space-y-6">
              <h3 className="text-2xl font-bold text-gray-800 flex items-center space-x-2">
                <FileText className="w-7 h-7 text-indigo-600" />
                <span>History & Reports</span>
              </h3>

              {/* Cohesion History */}
              <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
                <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
                  <h4 className="font-bold text-gray-800">Cohesion Cycle History</h4>
                </div>
                <div className="divide-y divide-gray-200">
                  {cohesionHistory.length > 0 ? (
                    cohesionHistory.map((cycle, idx) => (
                      <div key={idx} className="px-6 py-4 hover:bg-gray-50 transition">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-3">
                            <span className="text-sm font-mono text-gray-600">{cycle.cycle_id?.slice(0, 8)}</span>
                            <span className={`px-2 py-1 rounded text-xs font-semibold ${
                              getHealthColor(cycle.metrics?.cohesion_health)
                            }`}>
                              {cycle.metrics?.cohesion_health}
                            </span>
                          </div>
                          <span className="text-sm text-gray-500">{formatTimestamp(cycle.timestamp)}</span>
                        </div>
                        <div className="grid grid-cols-4 gap-4 text-sm">
                          <div>
                            <span className="text-gray-600">Alignment:</span>
                            <span className="font-semibold ml-2">{cycle.metrics?.alignment_score?.toFixed(2)}</span>
                          </div>
                          <div>
                            <span className="text-gray-600">Health:</span>
                            <span className="font-semibold ml-2">{cycle.metrics?.system_health_index?.toFixed(2)}</span>
                          </div>
                          <div>
                            <span className="text-gray-600">Ethics:</span>
                            <span className="font-semibold ml-2">{cycle.metrics?.ethical_continuity?.toFixed(2)}</span>
                          </div>
                          <div>
                            <span className="text-gray-600">Latency:</span>
                            <span className="font-semibold ml-2">{cycle.metrics?.synchronization_latency?.toFixed(2)}s</span>
                          </div>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="px-6 py-8 text-center text-gray-500">
                      No cohesion cycles executed yet. Trigger a manual cycle to start.
                    </div>
                  )}
                </div>
              </div>

              {/* Latest Report */}
              {latestReport && (
                <div className="bg-white border border-gray-200 rounded-lg p-6">
                  <h4 className="font-bold text-gray-800 mb-4">Latest Cohesion Report</h4>
                  
                  {/* Report Header */}
                  <div className="grid grid-cols-2 gap-4 mb-6 text-sm">
                    <div>
                      <span className="text-gray-600">Report ID:</span>
                      <span className="ml-2 font-mono">{latestReport.report_id}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Cycle ID:</span>
                      <span className="ml-2 font-mono">{latestReport.cycle_id}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Timestamp:</span>
                      <span className="ml-2">{formatTimestamp(latestReport.timestamp)}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Cohesion Health:</span>
                      <span className={`ml-2 px-2 py-1 rounded text-xs font-semibold ${
                        getHealthColor(latestReport.metrics?.cohesion_health)
                      }`}>
                        {latestReport.metrics?.cohesion_health}
                      </span>
                    </div>
                  </div>

                  {/* Recommendations */}
                  {latestReport.recommendations && latestReport.recommendations.length > 0 && (
                    <div className="mb-6">
                      <h5 className="font-bold text-gray-800 mb-3">Recommendations</h5>
                      <ul className="space-y-2">
                        {latestReport.recommendations.map((rec, idx) => (
                          <li key={idx} className="flex items-start space-x-2 text-sm">
                            <span className="text-indigo-600 mt-1">•</span>
                            <span className="text-gray-700">{rec}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Actions Log */}
                  {latestReport.actions_log && latestReport.actions_log.length > 0 && (
                    <div>
                      <h5 className="font-bold text-gray-800 mb-3">Actions Log</h5>
                      <div className="space-y-2">
                        {latestReport.actions_log.map((action, idx) => (
                          <div key={idx} className="flex items-center justify-between bg-gray-50 rounded-lg px-4 py-2 text-sm">
                            <div>
                              <span className="font-semibold text-gray-700">{action.action_type}</span>
                              {action.description && (
                                <span className="ml-2 text-gray-600">- {action.description}</span>
                              )}
                              {action.parameter && (
                                <span className="ml-2 text-gray-600">
                                  ({action.parameter}: {action.adjustment || action.correction})
                                </span>
                              )}
                            </div>
                            <span className={`px-2 py-1 rounded text-xs font-semibold ${
                              action.status === 'stable' ? 'bg-green-100 text-green-700' :
                              action.status === 'detected' ? 'bg-amber-100 text-amber-700' :
                              'bg-blue-100 text-blue-700'
                            }`}>
                              {action.status}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CohesionCorePanel;
