import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart } from 'recharts';
import { Brain, TrendingUp, Activity, RefreshCw, BookOpen, Target, AlertCircle, CheckCircle } from 'lucide-react';
import axios from 'axios';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const KnowledgeDistillationPanel = () => {
  const [knowledgeBase, setKnowledgeBase] = useState([]);
  const [auditReport, setAuditReport] = useState(null);
  const [distillationLoading, setDistillationLoading] = useState(false);
  const [auditLoading, setAuditLoading] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadKnowledgeBase();
    loadAuditReport();
  }, []);

  const loadKnowledgeBase = async () => {
    try {
      const response = await axios.get(`${API}/llm/knowledge-base?limit=20`);
      if (response.data.success) {
        setKnowledgeBase(response.data.entries || []);
      }
    } catch (err) {
      console.error('Error loading knowledge base:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadAuditReport = async () => {
    try {
      const response = await axios.get(`${API}/llm/audit`);
      if (response.data.success) {
        setAuditReport(response.data.audit_report);
      }
    } catch (err) {
      console.error('Error loading audit report:', err);
    }
  };

  const runDistillation = async () => {
    setDistillationLoading(true);
    try {
      const response = await axios.post(`${API}/llm/distill`);
      if (response.data.success) {
        toast.success(`Distilled ${response.data.distilled_count} knowledge entries!`);
        await loadKnowledgeBase();
        await loadAuditReport();
      } else {
        toast.error(response.data.message || 'Distillation failed');
      }
    } catch (err) {
      console.error('Error running distillation:', err);
      toast.error('Failed to run distillation. Please try again.');
    } finally {
      setDistillationLoading(false);
    }
  };

  const refreshAudit = async () => {
    setAuditLoading(true);
    try {
      await loadAuditReport();
      toast.success('Audit report refreshed!');
    } catch (err) {
      toast.error('Failed to refresh audit report');
    } finally {
      setAuditLoading(false);
    }
  };

  // Prepare chart data for accuracy trend
  const getAccuracyTrendData = () => {
    if (!auditReport || !auditReport.accuracy_trend) return [];
    return auditReport.accuracy_trend.map((item, idx) => ({
      point: `P${idx + 1}`,
      accuracy: item.accuracy,
      timestamp: item.timestamp
    }));
  };

  // Prepare chart data for latency trend
  const getLatencyTrendData = () => {
    if (!auditReport || !auditReport.latency_trend) return [];
    return auditReport.latency_trend.map((item, idx) => ({
      point: `P${idx + 1}`,
      latency: item.latency,
      timestamp: item.timestamp
    }));
  };

  // Prepare optimization cycles comparison data
  const getOptimizationComparisonData = () => {
    if (!auditReport || !auditReport.optimization_cycles) return [];
    return auditReport.optimization_cycles.map((cycle, idx) => ({
      cycle: `Cycle ${idx + 1}`,
      changes: cycle.changes.length,
      timestamp: cycle.timestamp,
      trigger: cycle.trigger
    }));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 animate-spin text-blue-500" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header Section */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Brain className="w-7 h-7 text-purple-500" />
            Knowledge Distillation & Audit
          </h2>
          <p className="text-sm text-gray-600 mt-1">
            Self-improving LLM system with distilled strategic insights and performance tracking
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={runDistillation}
            disabled={distillationLoading}
            className="bg-purple-600 hover:bg-purple-700"
          >
            {distillationLoading ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                Distilling...
              </>
            ) : (
              <>
                <Brain className="w-4 h-4 mr-2" />
                Run Distillation
              </>
            )}
          </Button>
          <Button
            onClick={refreshAudit}
            disabled={auditLoading}
            variant="outline"
          >
            {auditLoading ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                Loading...
              </>
            ) : (
              <>
                <Activity className="w-4 h-4 mr-2" />
                View Audit
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Performance Summary Card */}
      {auditReport && auditReport.summary && (
        <Card className="bg-gradient-to-br from-blue-50 to-purple-50 border-blue-200">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="w-5 h-5 text-blue-600" />
              Performance Summary
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-white p-4 rounded-lg shadow-sm">
                <div className="text-sm text-gray-600">Avg Accuracy</div>
                <div className="text-2xl font-bold text-blue-600">
                  {auditReport.summary.avg_accuracy || 0}/5
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {auditReport.summary.total_feedback || 0} feedback entries
                </div>
              </div>
              
              <div className="bg-white p-4 rounded-lg shadow-sm">
                <div className="text-sm text-gray-600">Avg Latency</div>
                <div className="text-2xl font-bold text-green-600">
                  {auditReport.summary.avg_latency || 0}s
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {auditReport.summary.total_requests || 0} requests
                </div>
              </div>
              
              <div className="bg-white p-4 rounded-lg shadow-sm">
                <div className="text-sm text-gray-600">Success Rate</div>
                <div className="text-2xl font-bold text-purple-600">
                  {auditReport.summary.success_rate || 0}%
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {auditReport.summary.fallback_count || 0} fallbacks
                </div>
              </div>
              
              <div className="bg-white p-4 rounded-lg shadow-sm">
                <div className="text-sm text-gray-600">Distilled Entries</div>
                <div className="text-2xl font-bold text-orange-600">
                  {auditReport.distillation_coverage?.distilled_entries || 0}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {auditReport.distillation_coverage?.coverage_percentage || 0}% coverage
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Performance Audit Trends */}
      {auditReport && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Accuracy Trend Chart */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-blue-500" />
                Accuracy Trend
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={getAccuracyTrendData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="point" />
                  <YAxis domain={[0, 5]} />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="accuracy" 
                    stroke="#3b82f6" 
                    strokeWidth={2}
                    name="Accuracy Score"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Latency Trend Chart */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-green-500" />
                Response Latency Trend
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={getLatencyTrendData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="point" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="latency" 
                    stroke="#10b981" 
                    strokeWidth={2}
                    name="Latency (seconds)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Optimization Cycles Comparison */}
      {auditReport && auditReport.optimization_cycles && auditReport.optimization_cycles.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="w-5 h-5 text-purple-500" />
              Optimization Cycles Comparison
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={getOptimizationComparisonData()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="cycle" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="changes" fill="#8b5cf6" name="Configuration Changes" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Distilled Knowledge Base */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-orange-500" />
            Distilled Strategic Knowledge
            <Badge variant="secondary" className="ml-2">
              {knowledgeBase.length} entries
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {knowledgeBase.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <Brain className="w-12 h-12 mx-auto mb-3 text-gray-300" />
              <p>No distilled knowledge available yet.</p>
              <p className="text-sm">Click "Run Distillation" to extract strategic insights from high-rated feedback.</p>
            </div>
          ) : (
            <div className="space-y-4">
              {knowledgeBase.map((entry, idx) => (
                <div 
                  key={entry.distillation_id || idx}
                  className="p-4 bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg border border-purple-200"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="bg-white">
                        {entry.operation_type || 'general'}
                      </Badge>
                      <span className="text-xs text-gray-500">
                        Confidence: {(entry.confidence_score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <span className="text-xs text-gray-400">
                      {entry.source_count || 0} sources
                    </span>
                  </div>
                  
                  <h4 className="font-semibold text-purple-900 mb-1 flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-purple-600" />
                    {entry.pattern}
                  </h4>
                  
                  <p className="text-sm text-gray-700 mb-2">
                    <span className="font-medium">Insight:</span> {entry.insight}
                  </p>
                  
                  <p className="text-sm text-blue-700 flex items-start gap-1">
                    <Target className="w-4 h-4 mt-0.5 flex-shrink-0" />
                    <span><span className="font-medium">Recommendation:</span> {entry.recommendation}</span>
                  </p>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Audit Recommendations */}
      {auditReport && auditReport.recommendations && auditReport.recommendations.length > 0 && (
        <Card className="border-orange-200 bg-orange-50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-orange-600" />
              Performance Recommendations
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {auditReport.recommendations.map((rec, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm">
                  <span className="text-orange-600 mt-0.5">•</span>
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default KnowledgeDistillationPanel;
