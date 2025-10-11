import React, { useState, useEffect } from 'react';
import { Brain, Link, Target, TrendingUp, CheckCircle, AlertCircle, RefreshCw, Search, Filter } from 'lucide-react';
import axios from 'axios';

const StrategicInsightFusionPanel = () => {
  const [insights, setInsights] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [dataSources, setDataSources] = useState({});

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || '';

  const fetchInsights = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${BACKEND_URL}/api/llm/insight-fusion`);
      if (response.data.success) {
        setInsights(response.data.insights || []);
        setDataSources(response.data.data_sources || {});
      }
    } catch (error) {
      console.error('Error fetching insights:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchInsights();
  }, []);

  const getAlignmentColor = (status) => {
    switch (status) {
      case 'aligned': return 'text-green-400';
      case 'deviation_detected': return 'text-yellow-400';
      case 're_evaluation_needed': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getAlignmentIcon = (status) => {
    switch (status) {
      case 'aligned': return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'deviation_detected': return <AlertCircle className="w-5 h-5 text-yellow-400" />;
      case 're_evaluation_needed': return <AlertCircle className="w-5 h-5 text-red-400" />;
      default: return <AlertCircle className="w-5 h-5 text-gray-400" />;
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'bg-green-500';
    if (confidence >= 0.6) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const filteredInsights = insights.filter(insight => {
    const matchesSearch = insight.reason_summary.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         insight.suggested_action.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterType === 'all' || insight.alignment_status === filterType;
    return matchesSearch && matchesFilter;
  });

  return (
    <div className="bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 rounded-lg p-6 mb-6 shadow-xl border border-purple-500/30">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-purple-500/20 rounded-lg">
            <Brain className="w-8 h-8 text-purple-400" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white">Strategic Insight Fusion</h2>
            <p className="text-gray-400 text-sm">LLM-powered reasoning for all system decisions</p>
          </div>
        </div>
        <button
          onClick={fetchInsights}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-800 text-white rounded-lg transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh Insights
        </button>
      </div>

      {/* Data Sources Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
          <div className="text-gray-400 text-sm mb-1">Auto-Tuning Decisions</div>
          <div className="text-2xl font-bold text-white">{dataSources.auto_tuning_decisions || 0}</div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
          <div className="text-gray-400 text-sm mb-1">Distilled Knowledge</div>
          <div className="text-2xl font-bold text-white">{dataSources.distilled_knowledge || 0}</div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
          <div className="text-gray-400 text-sm mb-1">Training Metrics</div>
          <div className="text-2xl font-bold text-white">{dataSources.training_metrics || 0}</div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
          <div className="text-gray-400 text-sm mb-1">Evaluations</div>
          <div className="text-2xl font-bold text-white">{dataSources.evaluations || 0}</div>
        </div>
      </div>

      {/* Search and Filter */}
      <div className="flex flex-col md:flex-row gap-4 mb-6">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search insights..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter className="w-5 h-5 text-gray-400" />
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
          >
            <option value="all">All Status</option>
            <option value="aligned">✅ Aligned</option>
            <option value="deviation_detected">⚠️ Deviation Detected</option>
            <option value="re_evaluation_needed">🔄 Re-evaluation Needed</option>
          </select>
        </div>
      </div>

      {/* Insights Cards */}
      {loading ? (
        <div className="text-center py-12">
          <RefreshCw className="w-12 h-12 text-purple-400 animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Generating strategic insights...</p>
        </div>
      ) : filteredInsights.length === 0 ? (
        <div className="text-center py-12 bg-slate-800/30 rounded-lg border border-slate-700">
          <Brain className="w-16 h-16 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400 text-lg mb-2">No Insights Available</p>
          <p className="text-gray-500 text-sm">
            {searchTerm || filterType !== 'all' 
              ? 'Try adjusting your search or filter criteria'
              : 'Generate auto-tuning decisions to see strategic insights'}
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {filteredInsights.map((insight, index) => (
            <div
              key={insight.reasoning_id || index}
              className="bg-gradient-to-r from-slate-800/80 to-slate-800/40 rounded-lg p-6 border border-slate-700 hover:border-purple-500/50 transition-all"
            >
              {/* Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-start gap-3 flex-1">
                  {getAlignmentIcon(insight.alignment_status)}
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-white mb-1">
                      {insight.reason_summary}
                    </h3>
                    <div className="flex items-center gap-4 text-sm text-gray-400">
                      <span>{new Date(insight.timestamp).toLocaleString()}</span>
                      <span className={`px-2 py-1 rounded ${getAlignmentColor(insight.alignment_status)} bg-opacity-20`}>
                        {insight.alignment_status.replace(/_/g, ' ')}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-400">Confidence</span>
                  <div className="w-20 bg-gray-700 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${getConfidenceColor(insight.confidence)}`}
                      style={{ width: `${insight.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-semibold text-white">{(insight.confidence * 100).toFixed(0)}%</span>
                </div>
              </div>

              {/* Evidence Sources */}
              {insight.evidence_sources && insight.evidence_sources.length > 0 && (
                <div className="mb-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Link className="w-4 h-4 text-blue-400" />
                    <span className="text-sm font-semibold text-gray-300">Evidence Sources:</span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {insight.evidence_sources.map((source, idx) => (
                      <span
                        key={idx}
                        className="px-3 py-1 bg-blue-500/20 text-blue-300 rounded-full text-xs border border-blue-500/30"
                      >
                        {source.source}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Reasoning Steps */}
              {insight.reasoning_steps && insight.reasoning_steps.length > 0 && (
                <div className="mb-4">
                  <div className="flex items-center gap-2 mb-3">
                    <TrendingUp className="w-4 h-4 text-purple-400" />
                    <span className="text-sm font-semibold text-gray-300">Reasoning Chain:</span>
                  </div>
                  <div className="space-y-2">
                    {insight.reasoning_steps.map((step, idx) => (
                      <div
                        key={idx}
                        className="flex gap-3 items-start bg-slate-900/50 rounded-lg p-3 border border-slate-700/50"
                      >
                        <div className="flex-shrink-0 w-6 h-6 bg-purple-500/20 rounded-full flex items-center justify-center">
                          <span className="text-xs font-bold text-purple-400">{idx + 1}</span>
                        </div>
                        <p className="text-sm text-gray-300 leading-relaxed">{step}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Suggested Action */}
              <div className="flex items-start gap-3 bg-green-500/10 rounded-lg p-4 border border-green-500/30">
                <Target className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <span className="text-sm font-semibold text-green-300 block mb-1">Suggested Action:</span>
                  <p className="text-sm text-gray-300">{insight.suggested_action}</p>
                </div>
              </div>

              {/* Impact Prediction */}
              {insight.impact_prediction && (
                <div className="mt-3 text-xs text-gray-400 italic">
                  Expected Impact: {insight.impact_prediction}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Export Button */}
      {filteredInsights.length > 0 && (
        <div className="mt-6 text-center">
          <button
            onClick={() => {
              const dataStr = JSON.stringify(filteredInsights, null, 2);
              const dataBlob = new Blob([dataStr], { type: 'application/json' });
              const url = URL.createObjectURL(dataBlob);
              const link = document.createElement('a');
              link.href = url;
              link.download = `strategic-insights-${Date.now()}.json`;
              link.click();
            }}
            className="px-6 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
          >
            Export Insights as JSON
          </button>
        </div>
      )}
    </div>
  );
};

export default StrategicInsightFusionPanel;
