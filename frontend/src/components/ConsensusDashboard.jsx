import React, { useState, useEffect } from 'react';
import { Activity, TrendingUp, TrendingDown, AlertCircle, RefreshCw, BarChart3, Users, Brain } from 'lucide-react';
import ArbitrationPanel from './ArbitrationPanel';
import DynamicTrustMonitor from './DynamicTrustMonitor';
import CollectiveMemoryPanel from './CollectiveMemoryPanel';

const ConsensusDashboard = () => {
  const [trustScores, setTrustScores] = useState(null);
  const [consensusHistory, setConsensusHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [recalibrating, setRecalibrating] = useState(false);
  const [showHistoryModal, setShowHistoryModal] = useState(false);
  const [activeTab, setActiveTab] = useState('consensus'); // 'consensus', 'arbitration', 'threshold', 'memory'

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

  useEffect(() => {
    fetchTrustScores();
    fetchConsensusHistory();
  }, []);

  const fetchTrustScores = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${BACKEND_URL}/api/llm/consensus/trust-scores`);
      const data = await response.json();
      if (data.success) {
        setTrustScores(data);
      }
    } catch (error) {
      console.error('Error fetching trust scores:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchConsensusHistory = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/llm/consensus/history?limit=20`);
      const data = await response.json();
      if (data.success) {
        setConsensusHistory(data.history);
      }
    } catch (error) {
      console.error('Error fetching consensus history:', error);
    }
  };

  const handleRecalibrate = async () => {
    try {
      setRecalibrating(true);
      const response = await fetch(`${BACKEND_URL}/api/llm/consensus/recalibrate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const data = await response.json();
      
      if (data.success) {
        await fetchTrustScores();
        await fetchConsensusHistory();
        alert(`✓ Recalibration complete! ${data.profiles_updated} profiles updated.`);
      }
    } catch (error) {
      console.error('Error recalibrating:', error);
      alert('✗ Recalibration failed');
    } finally {
      setRecalibrating(false);
    }
  };

  const getTrustColor = (score) => {
    if (score >= 0.85) return 'bg-green-500';
    if (score >= 0.75) return 'bg-blue-500';
    if (score >= 0.65) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getTrustLabel = (score) => {
    if (score >= 0.85) return 'Excellent';
    if (score >= 0.75) return 'Good';
    if (score >= 0.65) return 'Fair';
    return 'Needs Improvement';
  };

  if (loading && !trustScores) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">Loading consensus data...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg shadow-md p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold flex items-center gap-2">
              <Activity size={28} />
              Cognitive Consensus & Arbitration System
            </h2>
            <p className="text-purple-100 mt-1">Trust-Calibrated Multi-Agent Reasoning</p>
          </div>
          <button
            onClick={handleRecalibrate}
            disabled={recalibrating}
            className="bg-white text-purple-600 px-4 py-2 rounded-lg font-semibold hover:bg-purple-50 transition-colors disabled:opacity-50 flex items-center gap-2"
          >
            <RefreshCw size={18} className={recalibrating ? 'animate-spin' : ''} />
            {recalibrating ? 'Recalibrating...' : 'Recalibrate Trust'}
          </button>
        </div>
        
        {trustScores && (
          <div className="grid grid-cols-3 gap-4 mt-4">
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-purple-200 text-sm">Average Trust</div>
              <div className="text-2xl font-bold">{(trustScores.summary.average_trust * 100).toFixed(1)}%</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-purple-200 text-sm">Total Agents</div>
              <div className="text-2xl font-bold">{trustScores.summary.total_agents}</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-purple-200 text-sm">Highest Trust</div>
              <div className="text-lg font-semibold truncate">{trustScores.summary.highest_trust_agent || 'N/A'}</div>
            </div>
          </div>
        )}
      </div>

      {/* Tab Navigation */}
      <div className="bg-white rounded-lg shadow-md">
        <div className="flex border-b">
          <button
            onClick={() => setActiveTab('consensus')}
            className={`px-6 py-3 font-semibold transition-colors ${
              activeTab === 'consensus'
                ? 'border-b-2 border-purple-600 text-purple-600'
                : 'text-gray-600 hover:text-gray-800'
            }`}
            data-testid="consensus-tab"
          >
            <div className="flex items-center gap-2">
              <Activity size={18} />
              Consensus Dashboard
            </div>
          </button>
          <button
            onClick={() => setActiveTab('arbitration')}
            className={`px-6 py-3 font-semibold transition-colors ${
              activeTab === 'arbitration'
                ? 'border-b-2 border-purple-600 text-purple-600'
                : 'text-gray-600 hover:text-gray-800'
            }`}
            data-testid="arbitration-tab"
          >
            <div className="flex items-center gap-2">
              <AlertCircle size={18} />
              Meta-Arbitration
            </div>
          </button>
          <button
            onClick={() => setActiveTab('threshold')}
            className={`px-6 py-3 font-semibold transition-colors ${
              activeTab === 'threshold'
                ? 'border-b-2 border-purple-600 text-purple-600'
                : 'text-gray-600 hover:text-gray-800'
            }`}
            data-testid="threshold-tab"
          >
            <div className="flex items-center gap-2">
              <BarChart3 size={18} />
              Dynamic Thresholds
            </div>
          </button>
          <button
            onClick={() => setActiveTab('memory')}
            className={`px-6 py-3 font-semibold transition-colors ${
              activeTab === 'memory'
                ? 'border-b-2 border-purple-600 text-purple-600'
                : 'text-gray-600 hover:text-gray-800'
            }`}
            data-testid="memory-tab"
          >
            <div className="flex items-center gap-2">
              <Brain size={18} />
              Memory & Replay
            </div>
          </button>
        </div>

        {/* Tab Content */}
        <div className="p-6">
          {activeTab === 'consensus' && (
            <div className="space-y-6">
              {/* Trust Meters for Each Agent */}
              <div>
                <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                  <Users size={24} />
                  Agent Trust Profiles
                </h3>
        
        <div className="space-y-4">
          {trustScores && trustScores.trust_profiles.map((profile) => (
            <div key={profile.agent_name} className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <h4 className="font-semibold text-gray-800">{profile.agent_name}</h4>
                  <p className="text-sm text-gray-500">
                    {getTrustLabel(profile.trust_score)} • {profile.total_decisions} decisions
                  </p>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-gray-800">
                    {(profile.trust_score * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500">Trust Score</div>
                </div>
              </div>
              
              {/* Trust Meter Bar */}
              <div className="w-full bg-gray-200 rounded-full h-3 mb-3">
                <div
                  className={`${getTrustColor(profile.trust_score)} h-3 rounded-full transition-all duration-500`}
                  style={{ width: `${profile.trust_score * 100}%` }}
                />
              </div>
              
              {/* Performance Metrics Grid */}
              <div className="grid grid-cols-4 gap-3 text-sm">
                <div className="bg-gray-50 rounded p-2">
                  <div className="text-gray-500 text-xs">Accuracy</div>
                  <div className="font-semibold text-gray-800">
                    {profile.accurate_decisions}/{profile.total_decisions}
                  </div>
                </div>
                <div className="bg-gray-50 rounded p-2">
                  <div className="text-gray-500 text-xs">Confidence</div>
                  <div className="font-semibold text-gray-800">
                    {(profile.avg_confidence * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="bg-gray-50 rounded p-2">
                  <div className="text-gray-500 text-xs">Stability</div>
                  <div className="font-semibold text-gray-800">
                    {(profile.confidence_stability * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="bg-gray-50 rounded p-2">
                  <div className="text-gray-500 text-xs">Agreement</div>
                  <div className="font-semibold text-gray-800">
                    {(profile.agreement_rate * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Consensus History Summary */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-gray-800 flex items-center gap-2">
            <BarChart3 size={24} />
            Recent Consensus History
          </h3>
          <button
            onClick={() => setShowHistoryModal(true)}
            className="text-blue-600 hover:text-blue-700 text-sm font-semibold"
          >
            View All →
          </button>
        </div>

        {consensusHistory.length > 0 ? (
          <div className="space-y-2">
            {consensusHistory.slice(0, 5).map((entry, idx) => (
              <div
                key={entry.consensus_id || idx}
                className="flex items-center justify-between border-l-4 border-gray-300 pl-4 py-2 hover:bg-gray-50"
              >
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    {entry.consensus_reached ? (
                      <span className="text-green-600 text-sm font-semibold">✓ Consensus</span>
                    ) : (
                      <span className="text-red-600 text-sm font-semibold">✗ No Consensus</span>
                    )}
                    <span className="text-gray-600 text-sm">
                      {entry.weighted_confidence ? `${(entry.weighted_confidence * 100).toFixed(1)}% confidence` : 'N/A'}
                    </span>
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Stability: <span className="font-semibold">{entry.stability_index || 'N/A'}</span>
                    {entry.timestamp && (
                      <> • {new Date(entry.timestamp).toLocaleString()}</>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center text-gray-500 py-8">
            No consensus history available yet
          </div>
        )}
              </div>
            </div>
          )}

          {/* Arbitration Tab */}
          {activeTab === 'arbitration' && (
            <ArbitrationPanel />
          )}

          {/* Dynamic Threshold Tab */}
          {activeTab === 'threshold' && (
            <DynamicTrustMonitor />
          )}

          {/* Collective Memory & Replay Tab */}
          {activeTab === 'memory' && (
            <CollectiveMemoryPanel />
          )}
        </div>
      </div>

      {/* Consensus History Modal */}
      {showHistoryModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[80vh] overflow-hidden flex flex-col">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h3 className="text-xl font-bold text-gray-800">Consensus History</h3>
                <button
                  onClick={() => setShowHistoryModal(false)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  ✕
                </button>
              </div>
            </div>
            
            <div className="flex-1 overflow-y-auto p-6">
              <div className="space-y-4">
                {consensusHistory.map((entry, idx) => (
                  <div
                    key={entry.consensus_id || idx}
                    className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          {entry.consensus_reached ? (
                            <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs font-semibold">
                              Consensus Reached
                            </span>
                          ) : (
                            <span className="bg-red-100 text-red-800 px-2 py-1 rounded text-xs font-semibold">
                              No Consensus
                            </span>
                          )}
                          <span className="text-sm text-gray-600">
                            {entry.weighted_confidence ? `${(entry.weighted_confidence * 100).toFixed(1)}% confidence` : 'N/A'}
                          </span>
                        </div>
                        <div className="text-xs text-gray-500">
                          Stability: <span className="font-semibold">{entry.stability_index || 'N/A'}</span>
                          {entry.timestamp && (
                            <> • {new Date(entry.timestamp).toLocaleString()}</>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Agent Influences */}
                    {entry.agent_influences && (
                      <div className="mt-3">
                        <div className="text-sm font-semibold text-gray-700 mb-2">Agent Influences:</div>
                        <div className="grid grid-cols-2 gap-2">
                          {Object.entries(entry.agent_influences).map(([agent, influence]) => (
                            <div key={agent} className="flex items-center justify-between text-sm">
                              <span className="text-gray-600">{agent}</span>
                              <span className="font-semibold text-gray-800">{influence}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Reasoning Summary */}
                    {entry.reasoning_summary && (
                      <div className="mt-3 text-xs text-gray-600 bg-gray-50 rounded p-2">
                        {entry.reasoning_summary}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ConsensusDashboard;
