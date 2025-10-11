import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Progress } from './ui/progress';
import { 
  Users,
  Vote, 
  Network,
  TrendingUp,
  History,
  Download,
  RefreshCw,
  Play,
  Eye,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Brain,
  Scale,
  Shield
} from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const EthicalConsensusPanel = () => {
  const [consensusStatus, setConsensusStatus] = useState(null);
  const [consensusHistory, setConsensusHistory] = useState([]);
  const [report, setReport] = useState(null);
  const [refinements, setRefinements] = useState([]);
  const [selectedConsensus, setSelectedConsensus] = useState(null);
  const [agentDetails, setAgentDetails] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('deliberation');

  useEffect(() => {
    loadConsensusData();
    // Auto-refresh every 30 seconds
    const interval = setInterval(loadConsensusData, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadConsensusData = async () => {
    try {
      await Promise.all([
        loadConsensusStatus(),
        loadConsensusHistory(),
        loadRefinements(),
        loadReport()
      ]);
    } catch (error) {
      console.error('Error loading consensus data:', error);
    }
  };

  const loadConsensusStatus = async () => {
    try {
      const response = await axios.get(`${API}/llm/ethics/status?limit=20`);
      if (response.data.success) {
        setConsensusStatus(response.data);
      }
    } catch (error) {
      console.error('Error loading consensus status:', error);
    }
  };

  const loadConsensusHistory = async () => {
    try {
      const response = await axios.get(`${API}/llm/ethics/history?limit=50`);
      if (response.data.success) {
        setConsensusHistory(response.data.timeline || []);
      }
    } catch (error) {
      console.error('Error loading consensus history:', error);
    }
  };

  const loadRefinements = async () => {
    try {
      // Get refinements from MongoDB via a new endpoint or existing one
      // For now, we'll get them from the report
    } catch (error) {
      console.error('Error loading refinements:', error);
    }
  };

  const loadReport = async () => {
    try {
      const response = await axios.get(`${API}/llm/ethics/report?lookback_hours=72`);
      if (response.data.success) {
        setReport(response.data.report);
        if (response.data.report.rule_refinement) {
          setRefinements(response.data.report.rule_refinement.recent_changes || []);
        }
      }
    } catch (error) {
      console.error('Error loading report:', error);
    }
  };

  const triggerConsensus = async (goalId) => {
    try {
      setLoading(true);
      toast.info('Initiating multi-agent ethical deliberation...');
      
      const response = await axios.post(`${API}/llm/ethics/consensus`, {
        goal_id: goalId,
        trigger_type: 'manual',
        include_refinement: true
      });
      
      if (response.data.success) {
        toast.success(`Consensus reached: ${response.data.final_decision}`);
        await loadConsensusData();
        
        // Load agent details for the new consensus
        if (response.data.consensus_id) {
          await loadAgentDetails(response.data.consensus_id);
        }
      }
    } catch (error) {
      console.error('Error triggering consensus:', error);
      toast.error('Failed to complete consensus deliberation');
    } finally {
      setLoading(false);
    }
  };

  const loadAgentDetails = async (consensusId) => {
    try {
      const response = await axios.get(`${API}/llm/ethics/agent-details/${consensusId}`);
      if (response.data.success) {
        setAgentDetails(response.data);
        setSelectedConsensus(consensusId);
      }
    } catch (error) {
      console.error('Error loading agent details:', error);
      toast.error('Failed to load agent details');
    }
  };

  const applyRefinement = async (refinementId) => {
    try {
      const response = await axios.post(`${API}/llm/ethics/rules/update?refinement_id=${refinementId}&approved=true`);
      
      if (response.data.success) {
        toast.success('Rule refinement applied successfully');
        await loadReport();
      }
    } catch (error) {
      console.error('Error applying refinement:', error);
      toast.error('Failed to apply refinement');
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
    link.download = `ethics_consensus_report_${new Date().toISOString()}.json`;
    link.click();
    URL.revokeObjectURL(url);
    toast.success('Report exported successfully');
  };

  const getAlignmentColor = (score) => {
    if (score >= 0.80) return 'text-green-500';
    if (score >= 0.60) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getAlignmentBadge = (score) => {
    if (score >= 0.80) return <Badge className="bg-green-500">High Alignment</Badge>;
    if (score >= 0.60) return <Badge className="bg-yellow-500">Moderate</Badge>;
    return <Badge className="bg-red-500">Low Alignment</Badge>;
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'excellent':
        return <CheckCircle className="text-green-500" size={20} />;
      case 'good':
        return <CheckCircle className="text-blue-500" size={20} />;
      case 'needs_attention':
        return <AlertTriangle className="text-yellow-500" size={20} />;
      case 'critical':
        return <XCircle className="text-red-500" size={20} />;
      default:
        return <AlertTriangle className="text-gray-500" size={20} />;
    }
  };

  const getDecisionIcon = (decision) => {
    switch (decision) {
      case 'approved':
        return <CheckCircle className="text-green-500" size={18} />;
      case 'rejected':
        return <XCircle className="text-red-500" size={18} />;
      case 'requires_review':
        return <AlertTriangle className="text-yellow-500" size={18} />;
      default:
        return <AlertTriangle className="text-gray-500" size={18} />;
    }
  };

  const getProviderIcon = (provider) => {
    switch (provider) {
      case 'openai':
        return <Brain className="text-blue-500" size={16} />;
      case 'anthropic':
        return <Shield className="text-purple-500" size={16} />;
      case 'gemini':
        return <Scale className="text-amber-500" size={16} />;
      default:
        return <Users size={16} />;
    }
  };

  return (
    <div className="space-y-6" data-testid="ethical-consensus-panel">
      {/* Header with Real-time Status */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-500/20 rounded-lg">
                <Network className="text-purple-400" size={24} />
              </div>
              <div>
                <CardTitle className="text-white">Ethical Consensus System</CardTitle>
                <p className="text-slate-400 text-sm">Multi-Agent Deliberation & Rule Refinement</p>
              </div>
            </div>
            <div className="flex gap-2">
              <Button 
                variant="outline" 
                size="sm" 
                onClick={loadConsensusData}
                disabled={loading}
                data-testid="refresh-consensus-btn"
              >
                <RefreshCw className={`mr-2 ${loading ? 'animate-spin' : ''}`} size={16} />
                Refresh
              </Button>
              <Button 
                variant="outline" 
                size="sm" 
                onClick={exportReport}
                data-testid="export-report-btn"
              >
                <Download className="mr-2" size={16} />
                Export Report
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Status Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm">System Status</p>
                <div className="flex items-center gap-2 mt-2">
                  {consensusStatus && getStatusIcon(consensusStatus.status)}
                  <span className="text-white font-semibold capitalize">
                    {consensusStatus?.status || 'Loading...'}
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="pt-6">
            <div>
              <p className="text-slate-400 text-sm">Ethical Alignment Index (EAI)</p>
              <div className={`text-2xl font-bold mt-2 ${getAlignmentColor(consensusStatus?.overall_eai || 0)}`}>
                {(consensusStatus?.overall_eai || 0).toFixed(3)}
              </div>
              <Progress 
                value={(consensusStatus?.overall_eai || 0) * 100} 
                className="mt-2 h-2"
              />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="pt-6">
            <div>
              <p className="text-slate-400 text-sm">Agreement Variance (σ)</p>
              <div className="text-2xl font-bold text-white mt-2">
                {(consensusStatus?.avg_variance || 0).toFixed(3)}
              </div>
              <p className="text-slate-400 text-xs mt-1">
                {consensusStatus?.avg_variance < 0.15 ? 'Stable' : consensusStatus?.avg_variance < 0.25 ? 'Moderate' : 'High Variance'}
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="pt-6">
            <div>
              <p className="text-slate-400 text-sm">Total Consensuses</p>
              <div className="text-2xl font-bold text-white mt-2">
                {consensusStatus?.total_consensuses || 0}
              </div>
              <p className="text-slate-400 text-xs mt-1">
                Conflict Rate: {(consensusStatus?.conflict_rate || 0).toFixed(2)}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4 bg-slate-800/50">
          <TabsTrigger value="deliberation" data-testid="deliberation-tab">
            <Vote className="mr-2" size={16} />
            Deliberation
          </TabsTrigger>
          <TabsTrigger value="consensus-map" data-testid="consensus-map-tab">
            <Network className="mr-2" size={16} />
            Consensus Map
          </TabsTrigger>
          <TabsTrigger value="rule-refinement" data-testid="rule-refinement-tab">
            <TrendingUp className="mr-2" size={16} />
            Rule Refinement
          </TabsTrigger>
          <TabsTrigger value="history" data-testid="history-tab">
            <History className="mr-2" size={16} />
            History & Reports
          </TabsTrigger>
        </TabsList>

        {/* Deliberation View */}
        <TabsContent value="deliberation" className="space-y-4">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Agent Opinions & Voting</CardTitle>
              <p className="text-slate-400 text-sm">
                Multi-agent deliberation results with confidence scores and reasoning
              </p>
            </CardHeader>
            <CardContent>
              {agentDetails ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="text-white font-semibold">
                        {agentDetails.goal_description?.substring(0, 100)}...
                      </h3>
                      <div className="flex items-center gap-2 mt-2">
                        {getDecisionIcon(agentDetails.final_decision)}
                        <span className="text-white capitalize font-medium">
                          Final Decision: {agentDetails.final_decision}
                        </span>
                      </div>
                    </div>
                    <Badge className="bg-purple-500">
                      {agentDetails.agents_participated} Agents Participated
                    </Badge>
                  </div>

                  <div className="space-y-3">
                    {agentDetails.agent_opinions?.map((opinion, idx) => (
                      <Card key={idx} className="bg-slate-700/50 border-slate-600">
                        <CardContent className="pt-4">
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center gap-2 mb-2">
                                {getProviderIcon(opinion.provider)}
                                <span className="text-white font-medium">{opinion.agent_name}</span>
                                <Badge variant="outline" className="text-xs">
                                  {opinion.model}
                                </Badge>
                              </div>
                              
                              <div className="flex items-center gap-4 mb-3">
                                <div className="flex items-center gap-2">
                                  {opinion.vote === 'approve' && <CheckCircle className="text-green-500" size={16} />}
                                  {opinion.vote === 'reject' && <XCircle className="text-red-500" size={16} />}
                                  {opinion.vote === 'abstain' && <AlertTriangle className="text-gray-500" size={16} />}
                                  <span className={`font-semibold capitalize ${
                                    opinion.vote === 'approve' ? 'text-green-400' :
                                    opinion.vote === 'reject' ? 'text-red-400' : 'text-gray-400'
                                  }`}>
                                    {opinion.vote}
                                  </span>
                                </div>
                                
                                <div className="text-sm">
                                  <span className="text-slate-400">Alignment:</span>{' '}
                                  <span className={`font-semibold ${getAlignmentColor(opinion.alignment_score)}`}>
                                    {(opinion.alignment_score * 100).toFixed(0)}%
                                  </span>
                                </div>
                                
                                <div className="text-sm">
                                  <span className="text-slate-400">Confidence:</span>{' '}
                                  <span className="text-white font-semibold">
                                    {(opinion.confidence * 100).toFixed(0)}%
                                  </span>
                                </div>
                              </div>
                              
                              <p className="text-slate-300 text-sm mb-2">
                                <strong>Opinion:</strong> {opinion.opinion}
                              </p>
                              
                              <details className="text-slate-400 text-sm">
                                <summary className="cursor-pointer hover:text-slate-300">
                                  View detailed reasoning
                                </summary>
                                <p className="mt-2 pl-4 border-l-2 border-slate-600">
                                  {opinion.reasoning}
                                </p>
                              </details>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              ) : (
                <Alert className="bg-slate-700/50 border-slate-600">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription className="text-slate-300">
                    No deliberation data selected. Click on a consensus from the history to view agent opinions.
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Consensus Map */}
        <TabsContent value="consensus-map" className="space-y-4">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Consensus Alignment Network</CardTitle>
              <p className="text-slate-400 text-sm">
                Visual representation of agent agreement and provider alignment
              </p>
            </CardHeader>
            <CardContent>
              {consensusStatus && report ? (
                <div className="space-y-6">
                  {/* Provider Alignment Chart */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {Object.entries(report.provider_analysis?.avg_alignment_by_provider || {}).map(([provider, score]) => (
                      <Card key={provider} className="bg-slate-700/50 border-slate-600">
                        <CardContent className="pt-6">
                          <div className="flex items-center gap-3 mb-3">
                            {getProviderIcon(provider)}
                            <span className="text-white font-semibold capitalize">{provider}</span>
                          </div>
                          <div className={`text-2xl font-bold ${getAlignmentColor(score)}`}>
                            {(score * 100).toFixed(1)}%
                          </div>
                          <Progress value={score * 100} className="mt-2 h-2" />
                        </CardContent>
                      </Card>
                    ))}
                  </div>

                  {/* Decision Distribution */}
                  <Card className="bg-slate-700/50 border-slate-600">
                    <CardHeader>
                      <CardTitle className="text-white text-lg">Decision Distribution</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {Object.entries(consensusStatus.decision_distribution || {}).map(([decision, count]) => (
                          <div key={decision} className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              {getDecisionIcon(decision)}
                              <span className="text-white capitalize">{decision.replace('_', ' ')}</span>
                            </div>
                            <div className="flex items-center gap-3">
                              <Progress 
                                value={(count / consensusStatus.total_consensuses) * 100} 
                                className="w-32 h-2"
                              />
                              <span className="text-white font-semibold min-w-[3rem] text-right">
                                {count} ({((count / consensusStatus.total_consensuses) * 100).toFixed(0)}%)
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  {/* System Health */}
                  <Card className="bg-slate-700/50 border-slate-600">
                    <CardHeader>
                      <CardTitle className="text-white text-lg">System Health Assessment</CardTitle>
                    </CardHeader>
                    <CardContent>
                      {report.system_health && (
                        <div className="space-y-4">
                          <div className="flex items-center justify-between">
                            <span className="text-slate-400">Health Score:</span>
                            <span className={`text-2xl font-bold ${
                              report.system_health.health_score >= 85 ? 'text-green-500' :
                              report.system_health.health_score >= 70 ? 'text-blue-500' :
                              report.system_health.health_score >= 55 ? 'text-yellow-500' : 'text-red-500'
                            }`}>
                              {report.system_health.health_score}/100
                            </span>
                          </div>
                          <Progress value={report.system_health.health_score} className="h-3" />
                          <p className="text-slate-300 text-sm">
                            {report.system_health.description}
                          </p>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>
              ) : (
                <Alert className="bg-slate-700/50 border-slate-600">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription className="text-slate-300">
                    Loading consensus map data...
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Rule Refinement */}
        <TabsContent value="rule-refinement" className="space-y-4">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Governance Rule Refinements</CardTitle>
              <p className="text-slate-400 text-sm">
                Adaptive adjustments to governance weights based on consensus patterns
              </p>
            </CardHeader>
            <CardContent>
              {refinements && refinements.length > 0 ? (
                <div className="space-y-3">
                  {refinements.map((refinement, idx) => (
                    <Card key={idx} className="bg-slate-700/50 border-slate-600">
                      <CardContent className="pt-4">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <h4 className="text-white font-semibold mb-2">{refinement.rule_name}</h4>
                            <p className="text-slate-300 text-sm mb-3">{refinement.reason}</p>
                            
                            <div className="flex items-center gap-4 text-sm">
                              <div>
                                <span className="text-slate-400">Weight Change:</span>{' '}
                                <span className={`font-semibold ${
                                  refinement.weight_delta < 0 ? 'text-red-400' : 'text-green-400'
                                }`}>
                                  {refinement.weight_delta > 0 ? '+' : ''}{(refinement.weight_delta * 100).toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : (
                <Alert className="bg-slate-700/50 border-slate-600">
                  <AlertDescription className="text-slate-300">
                    No recent rule refinements. The system will propose adjustments based on consensus patterns.
                  </AlertDescription>
                </Alert>
              )}

              {report?.rule_refinement && (
                <Card className="bg-slate-700/50 border-slate-600 mt-4">
                  <CardContent className="pt-6">
                    <div className="grid grid-cols-3 gap-4 text-center">
                      <div>
                        <p className="text-slate-400 text-sm">Total Refinements</p>
                        <p className="text-white text-2xl font-bold mt-1">
                          {report.rule_refinement.total_refinements}
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-400 text-sm">Pending Approval</p>
                        <p className="text-yellow-400 text-2xl font-bold mt-1">
                          {report.rule_refinement.pending_approval}
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-400 text-sm">Auto-Applied</p>
                        <p className="text-green-400 text-2xl font-bold mt-1">
                          {report.rule_refinement.auto_applied}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* History & Reports */}
        <TabsContent value="history" className="space-y-4">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Consensus Timeline</CardTitle>
              <p className="text-slate-400 text-sm">
                Historical record of all ethical deliberations and outcomes
              </p>
            </CardHeader>
            <CardContent>
              {consensusHistory.length > 0 ? (
                <div className="space-y-3 max-h-[600px] overflow-y-auto">
                  {consensusHistory.map((consensus) => (
                    <Card 
                      key={consensus.consensus_id} 
                      className="bg-slate-700/50 border-slate-600 cursor-pointer hover:bg-slate-700/70 transition-colors"
                      onClick={() => loadAgentDetails(consensus.consensus_id)}
                      data-testid={`consensus-item-${consensus.consensus_id}`}
                    >
                      <CardContent className="pt-4">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-2">
                              {getDecisionIcon(consensus.final_decision)}
                              <span className="text-white font-semibold capitalize">
                                {consensus.final_decision.replace('_', ' ')}
                              </span>
                              {getAlignmentBadge(consensus.agreement_score)}
                            </div>
                            
                            <p className="text-slate-300 text-sm mb-2">
                              {consensus.goal_description}
                            </p>
                            
                            <div className="flex items-center gap-4 text-xs text-slate-400">
                              <span>EAI: {consensus.agreement_score?.toFixed(3)}</span>
                              <span>σ: {consensus.agreement_variance?.toFixed(3)}</span>
                              <span>Agents: {consensus.agents_participated}/5</span>
                              {consensus.conflicts_detected?.length > 0 && (
                                <Badge variant="outline" className="text-xs text-yellow-400 border-yellow-400">
                                  {consensus.conflicts_detected.length} Conflicts
                                </Badge>
                              )}
                            </div>
                            
                            <p className="text-slate-400 text-xs mt-2">
                              {new Date(consensus.timestamp).toLocaleString()}
                            </p>
                          </div>
                          
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              loadAgentDetails(consensus.consensus_id);
                              setActiveTab('deliberation');
                            }}
                          >
                            <Eye size={16} />
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : (
                <Alert className="bg-slate-700/50 border-slate-600">
                  <AlertDescription className="text-slate-300">
                    No consensus history available yet. Trigger ethical deliberation to begin.
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Recommendations */}
          {report?.recommendations && report.recommendations.length > 0 && (
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">System Recommendations</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {report.recommendations.map((rec, idx) => (
                    <Alert key={idx} className="bg-slate-700/50 border-slate-600">
                      <AlertDescription className="text-slate-300">
                        {rec}
                      </AlertDescription>
                    </Alert>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default EthicalConsensusPanel;
