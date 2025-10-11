/**
 * Step 30: Emergent Self-Reflection & Continuous Learning Loop Panel
 * 
 * A comprehensive dashboard for the AI's self-reflective learning system.
 * Displays reflection cycles, learning trajectory, game insights, ethical adjustments,
 * and historical data.
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Brain, 
  TrendingUp, 
  Target, 
  Activity, 
  Settings, 
  History,
  AlertCircle,
  CheckCircle,
  BarChart3,
  Lightbulb,
  Clock,
  Award
} from 'lucide-react';

const API_BASE = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

const ReflectionLoopPanel = () => {
  const [activeTab, setActiveTab] = useState('self-review');
  const [reflectionStatus, setReflectionStatus] = useState(null);
  const [reflectionHistory, setReflectionHistory] = useState([]);
  const [learningParams, setLearningParams] = useState(null);
  const [reflectionMetrics, setReflectionMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [triggeringReflection, setTriggeringReflection] = useState(false);
  const [paramUpdating, setParamUpdating] = useState(false);

  // Fetch data on mount
  useEffect(() => {
    fetchAllData();
    // Refresh every 30 seconds
    const interval = setInterval(fetchAllData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchAllData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        fetchReflectionStatus(),
        fetchReflectionHistory(),
        fetchLearningParameters(),
        fetchReflectionMetrics()
      ]);
    } catch (error) {
      console.error('Error fetching reflection data:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchReflectionStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/llm/reflection/status`);
      if (response.data.success) {
        setReflectionStatus(response.data);
      }
    } catch (error) {
      console.error('Error fetching reflection status:', error);
    }
  };

  const fetchReflectionHistory = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/llm/reflection/history?limit=20`);
      if (response.data.success) {
        setReflectionHistory(response.data.cycles || []);
      }
    } catch (error) {
      console.error('Error fetching reflection history:', error);
    }
  };

  const fetchLearningParameters = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/llm/reflection/parameters`);
      if (response.data.success) {
        setLearningParams(response.data.parameters);
      }
    } catch (error) {
      console.error('Error fetching learning parameters:', error);
    }
  };

  const fetchReflectionMetrics = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/llm/reflection/metrics`);
      if (response.data.success) {
        setReflectionMetrics(response.data);
      }
    } catch (error) {
      console.error('Error fetching reflection metrics:', error);
    }
  };

  const triggerReflection = async () => {
    setTriggeringReflection(true);
    try {
      const response = await axios.post(`${API_BASE}/api/llm/reflection/trigger`, {
        trigger: 'manual'
      });
      
      if (response.data.success) {
        alert('Reflection cycle triggered successfully!');
        await fetchAllData();
      }
    } catch (error) {
      console.error('Error triggering reflection:', error);
      alert('Failed to trigger reflection cycle');
    } finally {
      setTriggeringReflection(false);
    }
  };

  const updateParameters = async (updates) => {
    setParamUpdating(true);
    try {
      const response = await axios.post(`${API_BASE}/api/llm/reflection/parameters`, updates);
      
      if (response.data.success) {
        alert('Learning parameters updated successfully!');
        await fetchLearningParameters();
      }
    } catch (error) {
      console.error('Error updating parameters:', error);
      alert('Failed to update parameters');
    } finally {
      setParamUpdating(false);
    }
  };

  const getStatusColor = (status) => {
    const colors = {
      excellent: 'text-green-600',
      good: 'text-blue-600',
      needs_attention: 'text-yellow-600',
      critical: 'text-red-600',
      operational: 'text-green-600',
      initializing: 'text-gray-500'
    };
    return colors[status] || 'text-gray-500';
  };

  const getStatusIcon = (status) => {
    if (status === 'excellent' || status === 'operational') {
      return <CheckCircle className="w-5 h-5 text-green-600" />;
    }
    return <AlertCircle className="w-5 h-5 text-yellow-600" />;
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3">
              <Brain className="w-8 h-8 text-purple-600" />
              <h1 className="text-3xl font-bold text-gray-800">
                Step 30: Self-Reflection & Continuous Learning
              </h1>
            </div>
            <p className="text-gray-600 mt-2">
              Autonomous self-critique, learning from experience, and adaptive parameter tuning
            </p>
          </div>
          <button
            onClick={triggerReflection}
            disabled={triggeringReflection || loading}
            className={`px-6 py-3 rounded-lg font-semibold flex items-center gap-2
              ${triggeringReflection || loading
                ? 'bg-gray-300 cursor-not-allowed'
                : 'bg-purple-600 hover:bg-purple-700 text-white'
              }`}
          >
            <Activity className="w-5 h-5" />
            {triggeringReflection ? 'Reflecting...' : 'Trigger Reflection'}
          </button>
        </div>

        {/* Status Summary */}
        {reflectionStatus && (
          <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="text-sm text-purple-700 font-medium">Performance Score</span>
                <Award className="w-5 h-5 text-purple-600" />
              </div>
              <p className="text-2xl font-bold text-purple-900 mt-2">
                {reflectionStatus.performance_score?.toFixed(1) || '0.0'}%
              </p>
            </div>

            <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="text-sm text-blue-700 font-medium">Learning Health</span>
                <Activity className="w-5 h-5 text-blue-600" />
              </div>
              <p className="text-2xl font-bold text-blue-900 mt-2">
                {reflectionStatus.learning_health?.toFixed(2) || '0.00'}
              </p>
            </div>

            <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="text-sm text-green-700 font-medium">Ethical Status</span>
                {getStatusIcon(reflectionStatus.ethical_status)}
              </div>
              <p className={`text-lg font-bold mt-2 ${getStatusColor(reflectionStatus.ethical_status)}`}>
                {reflectionStatus.ethical_status?.replace('_', ' ').toUpperCase() || 'UNKNOWN'}
              </p>
            </div>

            <div className="bg-gradient-to-br from-gray-50 to-gray-100 p-4 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-700 font-medium">Games Analyzed</span>
                <BarChart3 className="w-5 h-5 text-gray-600" />
              </div>
              <p className="text-2xl font-bold text-gray-900 mt-2">
                {reflectionStatus.games_analyzed || 0}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Tab Navigation */}
      <div className="bg-white rounded-lg shadow-md p-2">
        <div className="flex flex-wrap gap-2">
          {[
            { id: 'self-review', label: 'Self-Review', icon: Brain },
            { id: 'trajectory', label: 'Learning Trajectory', icon: TrendingUp },
            { id: 'game-insights', label: 'Game Insights', icon: Lightbulb },
            { id: 'ethical-adjustments', label: 'Ethical Adjustments', icon: Target },
            { id: 'history', label: 'History & Export', icon: History }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors
                ${activeTab === tab.id
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="bg-white rounded-lg shadow-md p-6">
        {activeTab === 'self-review' && <SelfReviewTab reflectionHistory={reflectionHistory} loading={loading} />}
        {activeTab === 'trajectory' && <LearningTrajectoryTab reflectionMetrics={reflectionMetrics} reflectionHistory={reflectionHistory} loading={loading} />}
        {activeTab === 'game-insights' && <GameInsightsTab reflectionHistory={reflectionHistory} loading={loading} />}
        {activeTab === 'ethical-adjustments' && (
          <EthicalAdjustmentsTab 
            learningParams={learningParams} 
            updateParameters={updateParameters}
            paramUpdating={paramUpdating}
            loading={loading}
          />
        )}
        {activeTab === 'history' && <HistoryTab reflectionHistory={reflectionHistory} loading={loading} />}
      </div>
    </div>
  );
};

// Tab Components

const SelfReviewTab = ({ reflectionHistory, loading }) => {
  if (loading) {
    return <div className="text-center py-8 text-gray-500">Loading reflection data...</div>;
  }

  const latestCycle = reflectionHistory[0];

  if (!latestCycle) {
    return (
      <div className="text-center py-12">
        <Brain className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <p className="text-gray-600">No reflection cycles yet. Trigger your first reflection!</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Latest Reflection Cycle</h2>
        <div className="bg-gradient-to-r from-purple-50 to-blue-50 p-6 rounded-lg border-l-4 border-purple-600">
          <div className="flex items-start gap-4">
            <Clock className="w-6 h-6 text-purple-600 mt-1" />
            <div className="flex-1">
              <p className="text-sm text-gray-600">
                {new Date(latestCycle.timestamp).toLocaleString()}
              </p>
              <p className="text-gray-800 mt-2 leading-relaxed">
                {latestCycle.insights_summary || 'Reflection summary not available'}
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
          <h3 className="font-semibold text-blue-900 mb-2">Performance Score</h3>
          <p className="text-3xl font-bold text-blue-600">
            {latestCycle.overall_performance_score?.toFixed(1) || '0.0'}%
          </p>
          <p className="text-sm text-blue-700 mt-1">Overall gameplay quality</p>
        </div>

        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
          <h3 className="font-semibold text-green-900 mb-2">Learning Health</h3>
          <p className="text-3xl font-bold text-green-600">
            {latestCycle.learning_health_index?.toFixed(2) || '0.00'}
          </p>
          <p className="text-sm text-green-700 mt-1">Insight generation quality</p>
        </div>

        <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
          <h3 className="font-semibold text-purple-900 mb-2">Strategies Evaluated</h3>
          <p className="text-3xl font-bold text-purple-600">
            {latestCycle.strategies_evaluated || 0}
          </p>
          <p className="text-sm text-purple-700 mt-1">Creative strategies analyzed</p>
        </div>
      </div>

      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-3">Recommendations</h3>
        <div className="space-y-2">
          {latestCycle.recommendations?.map((rec, idx) => (
            <div key={idx} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
              <Lightbulb className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
              <p className="text-gray-700 text-sm">{rec}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const LearningTrajectoryTab = ({ reflectionMetrics, reflectionHistory, loading }) => {
  if (loading) {
    return <div className="text-center py-8 text-gray-500">Loading trajectory data...</div>;
  }

  if (!reflectionMetrics) {
    return (
      <div className="text-center py-12 text-gray-600">
        No metrics available. Complete reflection cycles to track learning trajectory.
      </div>
    );
  }

  const perfMetrics = reflectionMetrics.performance_metrics || {};

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Learning Trajectory Over Time</h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-lg">
          <h3 className="text-sm font-semibold text-blue-900 mb-2">Average Performance</h3>
          <p className="text-3xl font-bold text-blue-600">
            {perfMetrics.avg_performance_score?.toFixed(1) || '0.0'}%
          </p>
          <div className="mt-2 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-blue-600" />
            <span className="text-sm text-blue-700">
              {perfMetrics.performance_change >= 0 ? '+' : ''}
              {perfMetrics.performance_change?.toFixed(1) || '0.0'}% change
            </span>
          </div>
        </div>

        <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-lg">
          <h3 className="text-sm font-semibold text-green-900 mb-2">Learning Health</h3>
          <p className="text-3xl font-bold text-green-600">
            {perfMetrics.avg_learning_health?.toFixed(2) || '0.00'}
          </p>
          <p className="text-sm text-green-700 mt-2">Insight generation capacity</p>
        </div>

        <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-lg">
          <h3 className="text-sm font-semibold text-purple-900 mb-2">Total Cycles</h3>
          <p className="text-3xl font-bold text-purple-600">
            {perfMetrics.total_cycles || 0}
          </p>
          <p className="text-sm text-purple-700 mt-2">Reflection cycles completed</p>
        </div>
      </div>

      {/* Recent Cycles Timeline */}
      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Recent Reflection Cycles</h3>
        <div className="space-y-3">
          {reflectionHistory.slice(0, 5).map((cycle, idx) => (
            <div 
              key={idx} 
              className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-purple-100 flex items-center justify-center">
                    <span className="text-purple-600 font-bold">{idx + 1}</span>
                  </div>
                  <div>
                    <p className="font-semibold text-gray-800">
                      Cycle {cycle.cycle_id?.substring(0, 8)}
                    </p>
                    <p className="text-sm text-gray-600">
                      {new Date(cycle.timestamp).toLocaleDateString()}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-xl font-bold text-purple-600">
                    {cycle.overall_performance_score?.toFixed(1) || '0.0'}%
                  </p>
                  <p className="text-sm text-gray-600">Performance</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const GameInsightsTab = ({ reflectionHistory, loading }) => {
  if (loading) {
    return <div className="text-center py-8 text-gray-500">Loading game insights...</div>;
  }

  const latestCycle = reflectionHistory[0];

  if (!latestCycle || !latestCycle.game_reflections || latestCycle.game_reflections.length === 0) {
    return (
      <div className="text-center py-12 text-gray-600">
        No game insights available yet. Play games to generate insights!
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Game-by-Game Reflection</h2>

      <div className="space-y-4">
        {latestCycle.game_reflections.map((reflection, idx) => (
          <div key={idx} className="border border-gray-200 rounded-lg p-5 hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="font-semibold text-gray-800">Game {reflection.game_id}</h3>
                <p className="text-sm text-gray-600">
                  Outcome: <span className={`font-semibold ${
                    reflection.game_outcome === 'win' ? 'text-green-600' :
                    reflection.game_outcome === 'loss' ? 'text-red-600' :
                    'text-gray-600'
                  }`}>
                    {reflection.game_outcome?.toUpperCase()}
                  </span>
                </p>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-blue-600">
                  {(reflection.decision_quality_score * 100).toFixed(0)}%
                </p>
                <p className="text-sm text-gray-600">Decision Quality</p>
              </div>
            </div>

            <div className="space-y-3">
              <div>
                <h4 className="text-sm font-semibold text-gray-700 mb-2">💡 Learning Insights</h4>
                <ul className="space-y-1">
                  {reflection.learning_insights?.map((insight, i) => (
                    <li key={i} className="text-sm text-gray-600 pl-4 border-l-2 border-blue-300">
                      {insight}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="text-sm font-semibold text-green-700 mb-2">✅ Strengths</h4>
                  <ul className="space-y-1">
                    {reflection.strengths_identified?.map((strength, i) => (
                      <li key={i} className="text-sm text-gray-600">• {strength}</li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-yellow-700 mb-2">⚠️ Weaknesses</h4>
                  <ul className="space-y-1">
                    {reflection.weaknesses_identified?.map((weakness, i) => (
                      <li key={i} className="text-sm text-gray-600">• {weakness}</li>
                    ))}
                  </ul>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-semibold text-purple-700 mb-2">🎯 Improvement Actions</h4>
                <ul className="space-y-1">
                  {reflection.improvement_actions?.map((action, i) => (
                    <li key={i} className="text-sm text-gray-600">→ {action}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const EthicalAdjustmentsTab = ({ learningParams, updateParameters, paramUpdating, loading }) => {
  const [editedParams, setEditedParams] = useState({});

  useEffect(() => {
    if (learningParams) {
      setEditedParams(learningParams);
    }
  }, [learningParams]);

  if (loading || !learningParams) {
    return <div className="text-center py-8 text-gray-500">Loading parameters...</div>;
  }

  const handleUpdate = () => {
    updateParameters(editedParams);
  };

  const ParameterSlider = ({ label, paramKey, min, max, step, description }) => (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-semibold text-gray-700">{label}</label>
        <span className="text-sm font-mono text-purple-600">
          {editedParams[paramKey]?.toFixed(2) || '0.00'}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={editedParams[paramKey] || 0}
        onChange={(e) => setEditedParams({...editedParams, [paramKey]: parseFloat(e.target.value)})}
        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
      />
      <p className="text-xs text-gray-500">{description}</p>
    </div>
  );

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Learning Parameters</h2>
        <p className="text-gray-600">
          Adjust how the AI learns and adapts. Changes affect future reflection cycles.
        </p>
      </div>

      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-yellow-900">Advisory Mode</p>
            <p className="text-sm text-yellow-800 mt-1">
              All parameter changes operate in advisory mode. The system will suggest adjustments
              but requires human approval before applying to live games.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <ParameterSlider
          label="Novelty Weight"
          paramKey="novelty_weight"
          min={0.40}
          max={0.90}
          step={0.05}
          description="How much to favor creative/novel strategies (higher = more experimental)"
        />

        <ParameterSlider
          label="Stability Weight"
          paramKey="stability_weight"
          min={0.40}
          max={0.90}
          step={0.05}
          description="How much to favor stable/reliable strategies (higher = more conservative)"
        />

        <ParameterSlider
          label="Ethical Threshold"
          paramKey="ethical_threshold"
          min={0.65}
          max={0.95}
          step={0.05}
          description="Minimum ethical compliance required (higher = stricter standards)"
        />

        <ParameterSlider
          label="Creativity Bias"
          paramKey="creativity_bias"
          min={0.30}
          max={0.80}
          step={0.05}
          description="Balance between novel and stable approaches"
        />

        <ParameterSlider
          label="Risk Tolerance"
          paramKey="risk_tolerance"
          min={0.20}
          max={0.80}
          step={0.05}
          description="Willingness to try experimental strategies (higher = more risk)"
        />

        <div className="space-y-2">
          <label className="text-sm font-semibold text-gray-700">Reflection Depth (games)</label>
          <input
            type="number"
            min={1}
            max={10}
            value={editedParams.reflection_depth || 3}
            onChange={(e) => setEditedParams({...editedParams, reflection_depth: parseInt(e.target.value)})}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
          <p className="text-xs text-gray-500">Number of recent games to analyze per cycle</p>
        </div>
      </div>

      <div className="flex items-center justify-between pt-4">
        <p className="text-sm text-gray-600">
          Last updated: {new Date(learningParams.timestamp).toLocaleString()}
        </p>
        <button
          onClick={handleUpdate}
          disabled={paramUpdating}
          className={`px-6 py-3 rounded-lg font-semibold flex items-center gap-2
            ${paramUpdating
              ? 'bg-gray-300 cursor-not-allowed'
              : 'bg-purple-600 hover:bg-purple-700 text-white'
            }`}
        >
          <Settings className="w-5 h-5" />
          {paramUpdating ? 'Updating...' : 'Update Parameters'}
        </button>
      </div>
    </div>
  );
};

const HistoryTab = ({ reflectionHistory, loading }) => {
  if (loading) {
    return <div className="text-center py-8 text-gray-500">Loading history...</div>;
  }

  const exportData = () => {
    const dataStr = JSON.stringify(reflectionHistory, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `reflection-history-${new Date().toISOString()}.json`;
    link.click();
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">Reflection Cycle History</h2>
          <p className="text-gray-600 mt-1">Complete chronicle of all self-reflection cycles</p>
        </div>
        <button
          onClick={exportData}
          disabled={reflectionHistory.length === 0}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 flex items-center gap-2"
        >
          <History className="w-5 h-5" />
          Export Data
        </button>
      </div>

      {reflectionHistory.length === 0 ? (
        <div className="text-center py-12">
          <History className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">No reflection history available yet</p>
        </div>
      ) : (
        <div className="space-y-3">
          {reflectionHistory.map((cycle, idx) => (
            <div key={idx} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="text-sm font-semibold text-purple-600">
                      #{reflectionHistory.length - idx}
                    </span>
                    <span className="text-sm text-gray-600">
                      {new Date(cycle.timestamp).toLocaleString()}
                    </span>
                    <span className={`px-2 py-1 rounded-full text-xs font-semibold
                      ${cycle.trigger === 'post_game' ? 'bg-blue-100 text-blue-700' :
                        cycle.trigger === 'manual' ? 'bg-purple-100 text-purple-700' :
                        'bg-gray-100 text-gray-700'
                      }`}>
                      {cycle.trigger}
                    </span>
                  </div>
                  <p className="text-sm text-gray-700">
                    {cycle.insights_summary?.substring(0, 150)}...
                  </p>
                </div>
                <div className="text-right ml-4">
                  <p className="text-xl font-bold text-purple-600">
                    {cycle.overall_performance_score?.toFixed(1) || '0.0'}%
                  </p>
                  <p className="text-xs text-gray-600">Performance</p>
                </div>
              </div>
              <div className="mt-3 flex items-center gap-4 text-xs text-gray-600">
                <span>🎮 {cycle.games_analyzed || 0} games</span>
                <span>🎨 {cycle.strategies_evaluated || 0} strategies</span>
                <span className={`font-semibold ${
                  cycle.ethical_alignment_status === 'excellent' ? 'text-green-600' :
                  cycle.ethical_alignment_status === 'good' ? 'text-blue-600' :
                  'text-yellow-600'
                }`}>
                  ✓ {cycle.ethical_alignment_status}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ReflectionLoopPanel;
