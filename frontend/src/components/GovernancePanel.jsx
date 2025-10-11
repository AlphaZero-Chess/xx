import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Progress } from './ui/progress';
import { 
  Shield, 
  Target, 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  Download,
  RefreshCw,
  Play,
  Eye,
  Edit
} from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const GovernancePanel = () => {
  const [goals, setGoals] = useState([]);
  const [report, setReport] = useState(null);
  const [rules, setRules] = useState([]);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [selectedGoal, setSelectedGoal] = useState(null);

  useEffect(() => {
    loadGoals();
    loadGovernanceReport();
    loadGovernanceRules();
  }, []);

  const loadGoals = async () => {
    try {
      const response = await axios.get(`${API}/llm/goals/status?limit=50`);
      if (response.data.success) {
        setGoals(response.data.goals || []);
      }
    } catch (error) {
      console.error('Error loading goals:', error);
    }
  };

  const loadGovernanceReport = async () => {
    try {
      const response = await axios.get(`${API}/llm/governance/report`);
      if (response.data.success) {
        setReport(response.data.report);
      }
    } catch (error) {
      console.error('Error loading governance report:', error);
    }
  };

  const loadGovernanceRules = async () => {
    try {
      const response = await axios.get(`${API}/llm/governance/rules`);
      if (response.data.success) {
        setRules(response.data.rules || []);
      }
    } catch (error) {
      console.error('Error loading governance rules:', error);
    }
  };

  const generateGoals = async () => {
    try {
      setGenerating(true);
      toast.info('Generating adaptive goals...');
      
      const response = await axios.post(`${API}/llm/goals/generate`, {
        lookback_hours: 24,
        include_performance: true,
        include_alignment: true,
        include_stability: true
      });
      
      if (response.data.success) {
        toast.success(`Generated ${response.data.goals_generated} new goals`);
        await loadGoals();
        await loadGovernanceReport();
      }
    } catch (error) {
      console.error('Error generating goals:', error);
      toast.error('Failed to generate goals');
    } finally {
      setGenerating(false);
    }
  };

  const approveGoal = async (goalId, approved) => {
    try {
      const response = await axios.post(`${API}/llm/goals/approve`, {
        goal_id: goalId,
        approved: approved,
        approver: 'user',
        notes: approved ? 'Approved via UI' : 'Rejected via UI'
      });
      
      if (response.data.success) {
        toast.success(`Goal ${approved ? 'approved' : 'rejected'} successfully`);
        await loadGoals();
        await loadGovernanceReport();
      }
    } catch (error) {
      console.error('Error approving goal:', error);
      toast.error('Failed to update goal status');
    }
  };

  const exportReport = () => {
    if (!report) {
      toast.error('No report data available');
      return;
    }

    const dataStr = JSON.stringify(report, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `governance_report_${new Date().toISOString()}.json`;
    link.click();
    URL.revokeObjectURL(url);
    toast.success('Report exported successfully');
  };

  const refreshAll = async () => {
    setLoading(true);
    await Promise.all([loadGoals(), loadGovernanceReport(), loadGovernanceRules()]);
    setLoading(false);
    toast.success('Data refreshed');
  };

  const getAlignmentColor = (score) => {
    if (score >= 0.85) return 'text-green-500';
    if (score >= 0.70) return 'text-yellow-500';
    if (score >= 0.50) return 'text-orange-500';
    return 'text-red-500';
  };

  const getAlignmentBadge = (score) => {
    if (score >= 0.85) return <Badge className="bg-green-500">High</Badge>;
    if (score >= 0.70) return <Badge className="bg-yellow-500">Moderate</Badge>;
    if (score >= 0.50) return <Badge className="bg-orange-500">Low</Badge>;
    return <Badge className="bg-red-500">Critical</Badge>;
  };

  const getStatusBadge = (status) => {
    const statusColors = {
      proposed: 'bg-blue-500',
      approved: 'bg-green-500',
      rejected: 'bg-red-500',
      active: 'bg-purple-500',
      completed: 'bg-gray-500'
    };
    return <Badge className={statusColors[status] || 'bg-gray-500'}>{status}</Badge>;
  };

  return (
    <div className="space-y-6" data-testid="governance-panel">
      {/* Header */}
      <Card className="bg-gradient-to-r from-purple-900/50 to-indigo-900/50 border-purple-500/30">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Shield className="text-purple-400" size={32} />
              <div>
                <CardTitle className="text-2xl text-white">Governance & Ethics Dashboard</CardTitle>
                <p className="text-slate-300 text-sm mt-1">
                  Adaptive Goal Formation with Ethical Constraint Enforcement
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              <Button
                onClick={generateGoals}
                disabled={generating}
                className="bg-purple-600 hover:bg-purple-700"
                data-testid="generate-goals-btn"
              >
                {generating ? (
                  <>
                    <RefreshCw className="mr-2 animate-spin" size={16} />
                    Generating...
                  </>
                ) : (
                  <>
                    <Play className="mr-2" size={16} />
                    Generate Goals
                  </>
                )}
              </Button>
              <Button
                onClick={exportReport}
                variant="outline"
                className="border-purple-500/50"
                data-testid="export-report-btn"
              >
                <Download className="mr-2" size={16} />
                Export Report
              </Button>
              <Button
                onClick={refreshAll}
                variant="outline"
                className="border-purple-500/50"
                disabled={loading}
                data-testid="refresh-btn"
              >
                <RefreshCw className={loading ? 'animate-spin' : ''} size={16} />
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* System Health Overview */}
      {report && report.system_health && (
        <Card className="bg-slate-800/50 border-slate-700/50">
          <CardContent className="pt-6">
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-400">
                  {report.system_health.health_score}%
                </div>
                <div className="text-sm text-slate-400 mt-1">Health Score</div>
                <Badge className={
                  report.system_health.status === 'excellent' ? 'bg-green-500 mt-2' :
                  report.system_health.status === 'good' ? 'bg-blue-500 mt-2' :
                  report.system_health.status === 'needs_attention' ? 'bg-yellow-500 mt-2' :
                  'bg-red-500 mt-2'
                }>
                  {report.system_health.status}
                </Badge>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-400">
                  {report.recent_period_stats?.goals_executed || 0}
                </div>
                <div className="text-sm text-slate-400 mt-1">Goals Executed</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-green-400">
                  {((report.alignment_metrics?.overall_alignment || 0) * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-slate-400 mt-1">Alignment</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-orange-400">
                  {report.violations?.total_violations || 0}
                </div>
                <div className="text-sm text-slate-400 mt-1">Violations</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Tabs */}
      <Tabs defaultValue="active-goals" className="w-full">
        <TabsList className="grid w-full grid-cols-4 bg-slate-800/50">
          <TabsTrigger value="active-goals" data-testid="active-goals-tab">
            <Target className="mr-2" size={16} />
            Active Goals
          </TabsTrigger>
          <TabsTrigger value="ethical-analysis" data-testid="ethical-analysis-tab">
            <Shield className="mr-2" size={16} />
            Ethical Analysis
          </TabsTrigger>
          <TabsTrigger value="governance-rules" data-testid="governance-rules-tab">
            <Edit className="mr-2" size={16} />
            Governance Rules
          </TabsTrigger>
          <TabsTrigger value="history" data-testid="history-tab">
            <Eye className="mr-2" size={16} />
            History Log
          </TabsTrigger>
        </TabsList>

        {/* Tab 1: Active Goals */}
        <TabsContent value="active-goals" className="space-y-4" data-testid="active-goals-content">
          {goals.length === 0 ? (
            <Card className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="py-12 text-center">
                <Target className="mx-auto text-slate-600 mb-4" size={48} />
                <p className="text-slate-400 mb-4">No adaptive goals generated yet</p>
                <Button onClick={generateGoals} className="bg-purple-600 hover:bg-purple-700">
                  Generate First Goals
                </Button>
              </CardContent>
            </Card>
          ) : (
            goals.map((goal) => (
              <Card 
                key={goal.goal_id} 
                className="bg-slate-800/50 border-slate-700/50 hover:border-purple-500/50 transition-all"
                data-testid={`goal-card-${goal.goal_id}`}
              >
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        {getStatusBadge(goal.status)}
                        {goal.is_critical && (
                          <Badge className="bg-red-500">
                            <AlertTriangle size={12} className="mr-1" />
                            Critical
                          </Badge>
                        )}
                        {goal.auto_apply && (
                          <Badge className="bg-blue-500">Auto-Apply</Badge>
                        )}
                      </div>
                      <CardTitle className="text-lg text-white mb-2">
                        {goal.description}
                      </CardTitle>
                      <p className="text-sm text-slate-400">{goal.rationale}</p>
                    </div>
                    <div className="flex gap-2">
                      {goal.status === 'proposed' && (
                        <>
                          <Button
                            size="sm"
                            onClick={() => approveGoal(goal.goal_id, true)}
                            className="bg-green-600 hover:bg-green-700"
                            data-testid={`approve-btn-${goal.goal_id}`}
                          >
                            <CheckCircle size={14} className="mr-1" />
                            Approve
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => approveGoal(goal.goal_id, false)}
                            className="border-red-500/50 text-red-400 hover:bg-red-500/20"
                            data-testid={`reject-btn-${goal.goal_id}`}
                          >
                            <XCircle size={14} className="mr-1" />
                            Reject
                          </Button>
                        </>
                      )}
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4 mb-4">
                    <div>
                      <div className="text-xs text-slate-500 mb-1">Strategic Alignment</div>
                      <div className="flex items-center gap-2">
                        <Progress 
                          value={goal.strategic_alignment * 100} 
                          className="h-2 flex-1"
                        />
                        <span className={`text-sm font-medium ${getAlignmentColor(goal.strategic_alignment)}`}>
                          {(goal.strategic_alignment * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 mb-1">Ethical Alignment</div>
                      <div className="flex items-center gap-2">
                        <Progress 
                          value={goal.ethical_alignment * 100} 
                          className="h-2 flex-1"
                        />
                        <span className={`text-sm font-medium ${getAlignmentColor(goal.ethical_alignment)}`}>
                          {(goal.ethical_alignment * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 mb-1">Confidence</div>
                      <div className="flex items-center gap-2">
                        <Progress 
                          value={goal.confidence * 100} 
                          className="h-2 flex-1"
                        />
                        <span className="text-sm font-medium text-blue-400">
                          {(goal.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>

                  {goal.expected_outcomes && goal.expected_outcomes.length > 0 && (
                    <div className="mb-3">
                      <div className="text-xs text-slate-500 mb-2">Expected Outcomes:</div>
                      <ul className="list-disc list-inside text-sm text-slate-300 space-y-1">
                        {goal.expected_outcomes.map((outcome, idx) => (
                          <li key={idx}>{outcome}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {goal.risks && goal.risks.length > 0 && (
                    <div>
                      <div className="text-xs text-slate-500 mb-2">Risks:</div>
                      <ul className="list-disc list-inside text-sm text-orange-300 space-y-1">
                        {goal.risks.map((risk, idx) => (
                          <li key={idx}>{risk}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {goal.evaluation && (
                    <Alert className="mt-3 bg-slate-700/50 border-slate-600">
                      <Shield className="h-4 w-4" />
                      <AlertDescription className="text-sm text-slate-300">
                        <strong>Governance Status:</strong> {goal.evaluation.approval_status}
                        {goal.evaluation.violations && goal.evaluation.violations.length > 0 && (
                          <span className="text-orange-400 ml-2">
                            ({goal.evaluation.violations.length} violation{goal.evaluation.violations.length > 1 ? 's' : ''})
                          </span>
                        )}
                      </AlertDescription>
                    </Alert>
                  )}
                </CardContent>
              </Card>
            ))
          )}
        </TabsContent>

        {/* Tab 2: Ethical Analysis */}
        <TabsContent value="ethical-analysis" className="space-y-4" data-testid="ethical-analysis-content">
          {report && report.alignment_metrics && (
            <div className="grid grid-cols-2 gap-4">
              <Card className="bg-slate-800/50 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-white">Alignment Metrics</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-slate-400">Overall Alignment</span>
                      <span className="text-sm font-medium text-purple-400">
                        {(report.alignment_metrics.overall_alignment * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress value={report.alignment_metrics.overall_alignment * 100} className="h-2" />
                  </div>
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-slate-400">Transparency</span>
                      <span className="text-sm font-medium text-blue-400">
                        {(report.alignment_metrics.transparency_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress value={report.alignment_metrics.transparency_score * 100} className="h-2" />
                  </div>
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-slate-400">Fairness</span>
                      <span className="text-sm font-medium text-green-400">
                        {(report.alignment_metrics.fairness_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress value={report.alignment_metrics.fairness_score * 100} className="h-2" />
                  </div>
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-slate-400">Safety</span>
                      <span className="text-sm font-medium text-orange-400">
                        {(report.alignment_metrics.safety_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress value={report.alignment_metrics.safety_score * 100} className="h-2" />
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-slate-800/50 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-white">Goal Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  {report.goal_distribution && Object.keys(report.goal_distribution).length > 0 ? (
                    <div className="space-y-3">
                      {Object.entries(report.goal_distribution).map(([type, count]) => (
                        <div key={type}>
                          <div className="flex justify-between mb-1">
                            <span className="text-sm text-slate-400 capitalize">
                              {type.replace('_', ' ')}
                            </span>
                            <span className="text-sm font-medium text-white">{count}</span>
                          </div>
                          <Progress 
                            value={(count / Object.values(report.goal_distribution).reduce((a, b) => a + b, 0)) * 100} 
                            className="h-2" 
                          />
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-slate-400 text-center py-8">No goal data available</p>
                  )}
                </CardContent>
              </Card>

              <Card className="bg-slate-800/50 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-white">Approval Status</CardTitle>
                </CardHeader>
                <CardContent>
                  {report.approval_distribution && Object.keys(report.approval_distribution).length > 0 ? (
                    <div className="space-y-3">
                      {Object.entries(report.approval_distribution).map(([status, count]) => (
                        <div key={status} className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            {status === 'approved' && <CheckCircle size={16} className="text-green-500" />}
                            {status === 'rejected' && <XCircle size={16} className="text-red-500" />}
                            {status === 'requires_review' && <AlertTriangle size={16} className="text-yellow-500" />}
                            <span className="text-sm text-slate-300 capitalize">{status.replace('_', ' ')}</span>
                          </div>
                          <Badge className="bg-slate-700">{count}</Badge>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-slate-400 text-center py-8">No approval data available</p>
                  )}
                </CardContent>
              </Card>

              <Card className="bg-slate-800/50 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-white">Violations Summary</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-slate-400">Total Violations</span>
                      <span className="text-2xl font-bold text-orange-400">
                        {report.violations?.total_violations || 0}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-slate-400">Violation Rate</span>
                      <span className="text-lg font-medium text-orange-400">
                        {((report.violations?.violation_rate || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    {(report.violations?.total_violations || 0) > 0 && (
                      <Alert className="bg-orange-500/10 border-orange-500/50">
                        <AlertTriangle className="h-4 w-4 text-orange-500" />
                        <AlertDescription className="text-sm text-orange-300">
                          Ethics violations detected. Review goals and adjust governance rules.
                        </AlertDescription>
                      </Alert>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        {/* Tab 3: Governance Rules */}
        <TabsContent value="governance-rules" className="space-y-4" data-testid="governance-rules-content">
          {rules.map((rule) => (
            <Card 
              key={rule.rule_id} 
              className="bg-slate-800/50 border-slate-700/50"
              data-testid={`rule-card-${rule.rule_id}`}
            >
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <Badge className={
                        rule.constraint_type === 'transparency' ? 'bg-blue-500' :
                        rule.constraint_type === 'fairness' ? 'bg-green-500' :
                        'bg-orange-500'
                      }>
                        {rule.constraint_type}
                      </Badge>
                      {rule.enabled && <Badge className="bg-green-600">Active</Badge>}
                    </div>
                    <CardTitle className="text-lg text-white">{rule.name}</CardTitle>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-slate-400 mb-3">{rule.description}</p>
                <div className="flex items-center gap-4">
                  <div>
                    <span className="text-xs text-slate-500">Threshold: </span>
                    <span className="text-sm font-medium text-purple-400">
                      {rule.constraint_type === 'safety' && rule.threshold <= 0.10 ? '±' : ''}
                      {(rule.threshold * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-xs text-slate-500">Rule ID: </span>
                    <span className="text-xs text-slate-400">{rule.rule_id.slice(0, 16)}...</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
          
          {rules.length === 0 && (
            <Card className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="py-12 text-center">
                <Shield className="mx-auto text-slate-600 mb-4" size={48} />
                <p className="text-slate-400">No governance rules configured</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Tab 4: History Log */}
        <TabsContent value="history" className="space-y-4" data-testid="history-content">
          {report && report.recent_period_stats && (
            <Card className="bg-slate-800/50 border-slate-700/50">
              <CardHeader>
                <CardTitle className="text-white">Recent Governance Activity</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center py-2 border-b border-slate-700">
                    <span className="text-slate-400">Logs Analyzed</span>
                    <span className="text-white font-medium">{report.recent_period_stats.logs_analyzed || 0}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-700">
                    <span className="text-slate-400">Goals Executed</span>
                    <span className="text-white font-medium">{report.recent_period_stats.goals_executed || 0}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-700">
                    <span className="text-slate-400">Auto-Applied Goals</span>
                    <span className="text-white font-medium">{report.recent_period_stats.goals_auto_applied || 0}</span>
                  </div>
                  <div className="flex justify-between items-center py-2">
                    <span className="text-slate-400">Execution Rate</span>
                    <span className="text-white font-medium">
                      {((report.recent_period_stats.execution_rate || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          <Alert className="bg-blue-500/10 border-blue-500/50">
            <Eye className="h-4 w-4 text-blue-500" />
            <AlertDescription className="text-sm text-blue-300">
              Complete governance audit trail is maintained in the database. Export the report for detailed history.
            </AlertDescription>
          </Alert>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default GovernancePanel;
