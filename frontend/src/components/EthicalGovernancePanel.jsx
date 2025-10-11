import React, { useState, useEffect } from 'react';
import { Shield, AlertTriangle, CheckCircle, Clock, TrendingUp, FileText, Eye, ThumbsUp, ThumbsDown, Activity } from 'lucide-react';

const EthicalGovernancePanel = () => {
  const [activeTab, setActiveTab] = useState('live-compliance');
  const [ethicsStatus, setEthicsStatus] = useState(null);
  const [violations, setViolations] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [thresholds, setThresholds] = useState([]);
  const [pendingApprovals, setPendingApprovals] = useState([]);
  const [ethicsReport, setEthicsReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [showApprovalModal, setShowApprovalModal] = useState(false);
  const [selectedRequest, setSelectedRequest] = useState(null);

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

  // Fetch all ethics data
  const fetchEthicsData = async () => {
    try {
      // Get status
      const statusRes = await fetch(`${BACKEND_URL}/api/llm/ethics/status`);
      const statusData = await statusRes.json();
      if (statusData.success) {
        setEthicsStatus(statusData);
      }

      // Get violations
      const violationsRes = await fetch(`${BACKEND_URL}/api/llm/ethics/violations?limit=50`);
      const violationsData = await violationsRes.json();
      if (violationsData.success) {
        setViolations(violationsData.violations || []);
      }

      // Get metrics
      const metricsRes = await fetch(`${BACKEND_URL}/api/llm/ethics/metrics?days=7`);
      const metricsData = await metricsRes.json();
      if (metricsData.success) {
        setMetrics(metricsData.metrics || []);
      }

      // Get thresholds
      const thresholdsRes = await fetch(`${BACKEND_URL}/api/llm/ethics/thresholds?context=general`);
      const thresholdsData = await thresholdsRes.json();
      if (thresholdsData.success) {
        setThresholds(thresholdsData.thresholds || []);
      }

      // Get pending approvals
      const approvalsRes = await fetch(`${BACKEND_URL}/api/llm/ethics/approvals`);
      const approvalsData = await approvalsRes.json();
      if (approvalsData.success) {
        setPendingApprovals(approvalsData.pending_approvals || []);
      }

      // Get latest report
      const reportRes = await fetch(`${BACKEND_URL}/api/llm/ethics/report`);
      const reportData = await reportRes.json();
      if (reportData.success && reportData.report) {
        setEthicsReport(reportData.report);
      }

      setLoading(false);
    } catch (error) {
      console.error('Error fetching ethics data:', error);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEthicsData();
    
    // Auto-refresh every 30 seconds
    let interval;
    if (autoRefresh) {
      interval = setInterval(fetchEthicsData, 30000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  // Trigger ethics scan
  const triggerEthicsScan = async () => {
    setScanning(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/llm/ethics/trigger?context=general`, {
        method: 'POST'
      });
      const data = await res.json();
      if (data.success) {
        alert(`Ethics scan completed!\nCompliance Index: ${data.metrics.compliance_index.toFixed(3)}\nViolations Flagged: ${data.violations_flagged}`);
        fetchEthicsData();
      } else {
        alert('Failed to trigger ethics scan');
      }
    } catch (error) {
      console.error('Error triggering ethics scan:', error);
      alert('Error triggering ethics scan');
    } finally {
      setScanning(false);
    }
  };

  // Approve/Reject parameter change
  const handleApproval = async (requestId, approved, notes) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/llm/ethics/approve?request_id=${requestId}&approved=${approved}&approved_by=Admin&notes=${encodeURIComponent(notes || '')}`, {
        method: 'POST'
      });
      const data = await res.json();
      if (data.success) {
        alert(`Request ${approved ? 'approved' : 'rejected'} successfully`);
        setShowApprovalModal(false);
        setSelectedRequest(null);
        fetchEthicsData();
      } else {
        alert('Failed to process approval');
      }
    } catch (error) {
      console.error('Error processing approval:', error);
      alert('Error processing approval');
    }
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    try {
      return new Date(timestamp).toLocaleString();
    } catch {
      return timestamp;
    }
  };

  const getStatusColor = (status) => {
    const colors = {
      'excellent': 'text-green-600 bg-green-100',
      'good': 'text-blue-600 bg-blue-100',
      'needs_attention': 'text-yellow-600 bg-yellow-100',
      'critical': 'text-red-600 bg-red-100'
    };
    return colors[status] || 'text-gray-600 bg-gray-100';
  };

  const getSeverityColor = (severity) => {
    const colors = {
      'critical': 'text-red-600 bg-red-100',
      'high': 'text-orange-600 bg-orange-100',
      'medium': 'text-yellow-600 bg-yellow-100',
      'low': 'text-blue-600 bg-blue-100'
    };
    return colors[severity] || 'text-gray-600 bg-gray-100';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Activity className="w-8 h-8 animate-spin text-indigo-600 mx-auto mb-2" />
          <p className="text-gray-600">Loading Ethical Governance Layer 2.0...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-lg p-6 text-white">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Shield className="w-8 h-8" />
            <div>
              <h2 className="text-2xl font-bold">Ethical Governance Layer 2.0</h2>
              <p className="text-indigo-100 text-sm">Continuous ethical oversight & compliance monitoring</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                autoRefresh ? 'bg-white text-indigo-600' : 'bg-indigo-700 text-white'
              }`}
            >
              {autoRefresh ? 'Auto-Refresh On' : 'Auto-Refresh Off'}
            </button>
            <button
              onClick={triggerEthicsScan}
              disabled={scanning}
              className="bg-white text-indigo-600 px-4 py-2 rounded-lg font-medium hover:bg-indigo-50 transition-colors disabled:opacity-50"
            >
              {scanning ? 'Scanning...' : 'Trigger Ethics Scan'}
            </button>
          </div>
        </div>

        {/* Quick Status */}
        {ethicsStatus && (
          <div className="grid grid-cols-4 gap-4 mt-4">
            <div className="bg-white/10 rounded-lg p-3">
              <p className="text-indigo-100 text-xs">Compliance Index</p>
              <p className="text-2xl font-bold">
                {ethicsStatus.compliance_index ? ethicsStatus.compliance_index.toFixed(3) : 'N/A'}
              </p>
              <p className="text-xs text-indigo-100">Target: ≥0.950</p>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <p className="text-indigo-100 text-xs">Ethical Continuity</p>
              <p className="text-2xl font-bold">
                {ethicsStatus.ethical_continuity ? ethicsStatus.ethical_continuity.toFixed(3) : 'N/A'}
              </p>
              <p className="text-xs text-indigo-100">Target: ≥0.930</p>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <p className="text-indigo-100 text-xs">Violations (24h)</p>
              <p className="text-2xl font-bold">{ethicsStatus.violations_flagged_24h || 0}</p>
              <p className="text-xs text-indigo-100">Auto-flagged</p>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <p className="text-indigo-100 text-xs">Pending Approvals</p>
              <p className="text-2xl font-bold">{ethicsStatus.pending_approvals || 0}</p>
              <p className="text-xs text-indigo-100">Require review</p>
            </div>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-lg shadow">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-4 px-6" aria-label="Tabs">
            {[
              { id: 'live-compliance', name: 'Live Compliance', icon: Activity },
              { id: 'violations', name: 'Violation Log', icon: AlertTriangle },
              { id: 'thresholds', name: 'Adaptive Thresholds', icon: TrendingUp },
              { id: 'trends', name: 'Governance Trends', icon: TrendingUp },
              { id: 'human-review', name: 'Human Review', icon: Eye }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                <span>{tab.name}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="p-6">
          {/* Live Compliance Tab */}
          {activeTab === 'live-compliance' && (
            <div className="space-y-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Real-Time Compliance Metrics</h3>
              
              {ethicsStatus && (
                <>
                  {/* Overall Status */}
                  <div className={`p-4 rounded-lg ${getStatusColor(ethicsStatus.status)}`}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <CheckCircle className="w-6 h-6" />
                        <div>
                          <p className="font-semibold">System Status: {ethicsStatus.status?.toUpperCase()}</p>
                          <p className="text-sm">Last scan: {formatTimestamp(ethicsStatus.last_scan)}</p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Target Comparison */}
                  {ethicsStatus.target_comparison && (
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-gray-50 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <p className="text-sm font-medium text-gray-700">Compliance Index</p>
                          <span className="text-2xl">{ethicsStatus.target_comparison.compliance?.status}</span>
                        </div>
                        <p className="text-2xl font-bold text-gray-900">
                          {ethicsStatus.target_comparison.compliance?.current?.toFixed(3) || 'N/A'}
                        </p>
                        <p className="text-xs text-gray-500">Target: {ethicsStatus.target_comparison.compliance?.target?.toFixed(3)}</p>
                      </div>

                      <div className="bg-gray-50 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <p className="text-sm font-medium text-gray-700">Ethical Continuity</p>
                          <span className="text-2xl">{ethicsStatus.target_comparison.continuity?.status}</span>
                        </div>
                        <p className="text-2xl font-bold text-gray-900">
                          {ethicsStatus.target_comparison.continuity?.current?.toFixed(3) || 'N/A'}
                        </p>
                        <p className="text-xs text-gray-500">Target: {ethicsStatus.target_comparison.continuity?.target?.toFixed(3)}</p>
                      </div>

                      <div className="bg-gray-50 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <p className="text-sm font-medium text-gray-700">False Positive Rate</p>
                          <span className="text-2xl">{ethicsStatus.target_comparison.false_positive_rate?.status}</span>
                        </div>
                        <p className="text-2xl font-bold text-gray-900">
                          {ethicsStatus.target_comparison.false_positive_rate?.current ? 
                            (ethicsStatus.target_comparison.false_positive_rate.current * 100).toFixed(1) + '%' : 'N/A'}
                        </p>
                        <p className="text-xs text-gray-500">Target: ≤{(ethicsStatus.target_comparison.false_positive_rate?.target * 100)?.toFixed(1)}%</p>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* Violation Log Tab */}
          {activeTab === 'violations' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Flagged Violations & Anomalies</h3>
              
              {violations.length === 0 ? (
                <div className="text-center py-12 bg-gray-50 rounded-lg">
                  <CheckCircle className="w-12 h-12 text-green-600 mx-auto mb-3" />
                  <p className="text-gray-600 font-medium">No violations flagged</p>
                  <p className="text-sm text-gray-500">System operating within ethical guidelines</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {violations.map((violation) => (
                    <div key={violation.violation_id} className="bg-white border border-gray-200 rounded-lg p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center space-x-3 mb-2">
                            <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(violation.severity)}`}>
                              {violation.severity?.toUpperCase()}
                            </span>
                            <span className="text-sm text-gray-500">{violation.module}</span>
                            <span className="text-xs text-gray-400">{formatTimestamp(violation.timestamp)}</span>
                          </div>
                          <p className="font-medium text-gray-900 mb-1">{violation.violation_type}</p>
                          <p className="text-sm text-gray-600 mb-2">{violation.description}</p>
                          <p className="text-sm text-indigo-600">
                            <strong>Recommended:</strong> {violation.recommended_action}
                          </p>
                        </div>
                        <div className={`ml-4 px-3 py-1 rounded text-xs font-medium ${
                          violation.resolution_status === 'resolved' ? 'bg-green-100 text-green-700' :
                          violation.resolution_status === 'pending' ? 'bg-yellow-100 text-yellow-700' :
                          'bg-gray-100 text-gray-700'
                        }`}>
                          {violation.resolution_status}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Adaptive Thresholds Tab */}
          {activeTab === 'thresholds' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Dynamic Ethical Thresholds</h3>
              
              {thresholds.length === 0 ? (
                <div className="text-center py-12 bg-gray-50 rounded-lg">
                  <p className="text-gray-600">No threshold data available</p>
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-4">
                  {thresholds.map((threshold, idx) => (
                    <div key={idx} className="bg-gray-50 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <p className="font-medium text-gray-900">
                          {threshold.module} - {threshold.parameter}
                        </p>
                        <span className="text-xs text-gray-500">{threshold.context}</span>
                      </div>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600">Base:</span>
                          <span className="font-medium">{threshold.base_threshold?.toFixed(3)}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600">Current:</span>
                          <span className="font-bold text-indigo-600">{threshold.current_threshold?.toFixed(3)}</span>
                        </div>
                        <p className="text-xs text-gray-500 mt-2">{threshold.reason}</p>
                        <p className="text-xs text-gray-400">Updated: {formatTimestamp(threshold.last_adjusted)}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Governance Trends Tab */}
          {activeTab === 'trends' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Historical Compliance Trends</h3>
              
              {metrics.length === 0 ? (
                <div className="text-center py-12 bg-gray-50 rounded-lg">
                  <p className="text-gray-600">No historical data available yet</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-sm font-medium text-gray-700 mb-3">Recent Compliance Scores</p>
                    <div className="space-y-2">
                      {metrics.slice(0, 10).map((metric, idx) => (
                        <div key={idx} className="flex items-center justify-between text-sm">
                          <span className="text-gray-600">{formatTimestamp(metric.timestamp)}</span>
                          <div className="flex items-center space-x-4">
                            <span className={`font-medium ${
                              metric.compliance_index >= 0.95 ? 'text-green-600' :
                              metric.compliance_index >= 0.85 ? 'text-blue-600' :
                              'text-yellow-600'
                            }`}>
                              {metric.compliance_index?.toFixed(3)}
                            </span>
                            <span className={`px-2 py-1 rounded text-xs ${getStatusColor(metric.status)}`}>
                              {metric.status}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Human Review Tab */}
          {activeTab === 'human-review' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Parameter Change Approvals</h3>
              
              {pendingApprovals.length === 0 ? (
                <div className="text-center py-12 bg-gray-50 rounded-lg">
                  <CheckCircle className="w-12 h-12 text-green-600 mx-auto mb-3" />
                  <p className="text-gray-600 font-medium">No pending approvals</p>
                  <p className="text-sm text-gray-500">All parameter changes have been reviewed</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {pendingApprovals.map((request) => (
                    <div key={request.request_id} className="bg-white border-2 border-indigo-200 rounded-lg p-4">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex-1">
                          <div className="flex items-center space-x-3 mb-2">
                            <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(request.severity)}`}>
                              {request.severity?.toUpperCase()}
                            </span>
                            <span className="text-sm font-medium text-gray-700">{request.module}</span>
                            <span className="text-xs text-gray-400">{formatTimestamp(request.timestamp)}</span>
                          </div>
                          <p className="font-semibold text-gray-900 mb-2">
                            {request.parameter}: {request.current_value?.toFixed(3)} → {request.proposed_value?.toFixed(3)}
                          </p>
                          <p className="text-sm text-gray-600 mb-2">
                            <strong>Change:</strong> {(request.delta > 0 ? '+' : '')}{request.delta?.toFixed(3)} 
                            ({(request.delta / request.current_value * 100).toFixed(1)}%)
                          </p>
                          <p className="text-sm text-gray-700 mb-2"><strong>Reason:</strong> {request.reason}</p>
                          <p className="text-sm text-indigo-600"><strong>Impact:</strong> {request.impact_analysis}</p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-3 pt-3 border-t border-gray-200">
                        <button
                          onClick={() => {
                            setSelectedRequest(request);
                            setShowApprovalModal(true);
                          }}
                          className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                        >
                          <ThumbsUp className="w-4 h-4" />
                          <span>Approve</span>
                        </button>
                        <button
                          onClick={() => handleApproval(request.request_id, false, 'Rejected by admin')}
                          className="flex items-center space-x-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                        >
                          <ThumbsDown className="w-4 h-4" />
                          <span>Reject</span>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Approval Modal */}
      {showApprovalModal && selectedRequest && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <h3 className="text-lg font-bold mb-4">Approve Parameter Change</h3>
            <p className="text-sm text-gray-600 mb-4">
              Approving this will allow: <strong>{selectedRequest.parameter}</strong> to change from{' '}
              {selectedRequest.current_value?.toFixed(3)} to {selectedRequest.proposed_value?.toFixed(3)}
            </p>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">Approval Notes (Optional)</label>
              <textarea
                id="approval-notes"
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                placeholder="Add any notes about this approval..."
              />
            </div>
            <div className="flex items-center justify-end space-x-3">
              <button
                onClick={() => {
                  setShowApprovalModal(false);
                  setSelectedRequest(null);
                }}
                className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  const notes = document.getElementById('approval-notes').value;
                  handleApproval(selectedRequest.request_id, true, notes);
                }}
                className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
              >
                Confirm Approval
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EthicalGovernancePanel;
