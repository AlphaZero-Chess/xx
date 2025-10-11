import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
  RefreshCw,
  Download,
  AlertTriangle,
  CheckCircle2,
  Clock,
  BarChart3,
  Zap
} from 'lucide-react';

const CognitiveResonancePanel = () => {
  const [resonanceStatus, setResonanceStatus] = useState(null);
  const [resonanceMetrics, setResonanceMetrics] = useState([]);
  const [stabilityForecast, setStabilityForecast] = useState(null);
  const [balanceResult, setBalanceResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

  // Fetch resonance status on mount and every 30 seconds
  useEffect(() => {
    fetchResonanceStatus();
    const interval = setInterval(fetchResonanceStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  // Fetch historical metrics
  useEffect(() => {
    fetchResonanceMetrics();
  }, []);

  const fetchResonanceStatus = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/llm/resonance/status`);
      const data = await response.json();
      setResonanceStatus(data);
    } catch (error) {
      console.error('Error fetching resonance status:', error);
    }
  };

  const fetchResonanceMetrics = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/llm/resonance/metrics?days=7`);
      const data = await response.json();
      if (data.success) {
        setResonanceMetrics(data.metrics);
      }
    } catch (error) {
      console.error('Error fetching resonance metrics:', error);
    }
  };

  const handleAnalyzeResonance = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${backendUrl}/api/llm/resonance/analyze`, {
        method: 'POST'
      });
      const data = await response.json();
      if (data.success) {
        await fetchResonanceStatus();
        await fetchResonanceMetrics();
      }
    } catch (error) {
      console.error('Error analyzing resonance:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRecalibrate = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${backendUrl}/api/llm/resonance/recalibrate?force=true`, {
        method: 'POST'
      });
      const data = await response.json();
      if (data.success) {
        setBalanceResult(data.balance_result);
        await fetchResonanceStatus();
      }
    } catch (error) {
      console.error('Error recalibrating:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFetchForecast = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${backendUrl}/api/llm/resonance/forecast?horizon_hours=24`);
      const data = await response.json();
      if (data.success) {
        setStabilityForecast(data.forecast);
      }
    } catch (error) {
      console.error('Error fetching forecast:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadReport = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/llm/resonance/report`);
      const data = await response.json();
      if (data.success) {
        const blob = new Blob([JSON.stringify(data.report, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `resonance-report-${new Date().toISOString()}.json`;
        a.click();
      }
    } catch (error) {
      console.error('Error downloading report:', error);
    }
  };

  const getHealthBadge = (health) => {
    const variants = {
      excellent: 'bg-green-100 text-green-800',
      good: 'bg-blue-100 text-blue-800',
      moderate: 'bg-yellow-100 text-yellow-800',
      needs_attention: 'bg-red-100 text-red-800'
    };
    return variants[health] || variants.moderate;
  };

  const getStatusIcon = (status) => {
    if (status === '✅') return <CheckCircle2 className="h-4 w-4 text-green-600" />;
    return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
  };

  const getTrendIcon = (trend) => {
    if (trend === 'improving') return <TrendingUp className="h-4 w-4 text-green-600" />;
    if (trend === 'declining') return <TrendingDown className="h-4 w-4 text-red-600" />;
    return <Minus className="h-4 w-4 text-gray-600" />;
  };

  const getRiskBadge = (risk) => {
    const variants = {
      low: 'bg-green-100 text-green-800',
      medium: 'bg-yellow-100 text-yellow-800',
      high: 'bg-orange-100 text-orange-800',
      critical: 'bg-red-100 text-red-800'
    };
    return variants[risk] || variants.low;
  };

  return (
    <div className="space-y-6" data-testid="cognitive-resonance-panel">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle data-testid="resonance-panel-title">Cognitive Resonance Framework</CardTitle>
              <CardDescription>
                Long-term system stability and cross-module synchronization (Step 34)
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button
                onClick={handleAnalyzeResonance}
                disabled={loading}
                size="sm"
                variant="outline"
                data-testid="analyze-resonance-button"
              >
                <Activity className="h-4 w-4 mr-2" />
                {loading ? 'Analyzing...' : 'Analyze'}
              </Button>
              <Button
                onClick={handleDownloadReport}
                size="sm"
                variant="outline"
                data-testid="download-report-button"
              >
                <Download className="h-4 w-4 mr-2" />
                Report
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview" data-testid="tab-overview">Resonance Overview</TabsTrigger>
          <TabsTrigger value="stability" data-testid="tab-stability">Temporal Stability</TabsTrigger>
          <TabsTrigger value="feedback" data-testid="tab-feedback">Feedback Regulation</TabsTrigger>
          <TabsTrigger value="entropy" data-testid="tab-entropy">Entropy Control</TabsTrigger>
          <TabsTrigger value="historical" data-testid="tab-historical">Historical Resonance</TabsTrigger>
        </TabsList>

        {/* Tab 1: Resonance Overview */}
        <TabsContent value="overview" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Live Alignment Graphs</CardTitle>
              <CardDescription>Real-time cross-module alignment and resonance indices</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {resonanceStatus?.system_status === 'operational' ? (
                <>
                  {/* Core Metrics */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="space-y-2" data-testid="resonance-index-metric">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Resonance Index</span>
                        {getStatusIcon(resonanceStatus.target_comparison?.resonance?.status)}
                      </div>
                      <div className="text-2xl font-bold">
                        {resonanceStatus.resonance_index?.toFixed(3) || 'N/A'}
                      </div>
                      <Progress
                        value={(resonanceStatus.resonance_index || 0) * 100}
                        className="h-2"
                      />
                      <p className="text-xs text-muted-foreground">
                        Target: ≥ {resonanceStatus.target_comparison?.resonance?.target}
                      </p>
                    </div>

                    <div className="space-y-2" data-testid="temporal-stability-metric">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Temporal Stability</span>
                        {getStatusIcon(resonanceStatus.target_comparison?.stability?.status)}
                      </div>
                      <div className="text-2xl font-bold">
                        {resonanceStatus.temporal_stability?.toFixed(3) || 'N/A'}
                      </div>
                      <Progress
                        value={(resonanceStatus.temporal_stability || 0) * 100}
                        className="h-2"
                      />
                      <p className="text-xs text-muted-foreground">
                        Target: ≥ {resonanceStatus.target_comparison?.stability?.target}
                      </p>
                    </div>

                    <div className="space-y-2" data-testid="feedback-equilibrium-metric">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Feedback Equilibrium</span>
                        {getStatusIcon(resonanceStatus.target_comparison?.equilibrium?.status)}
                      </div>
                      <div className="text-2xl font-bold">
                        {resonanceStatus.feedback_equilibrium?.toFixed(3) || 'N/A'}
                      </div>
                      <Progress
                        value={(resonanceStatus.feedback_equilibrium || 0.5) * 100}
                        className="h-2"
                      />
                      <p className="text-xs text-muted-foreground">
                        Target: {resonanceStatus.target_comparison?.equilibrium?.target}
                      </p>
                    </div>

                    <div className="space-y-2" data-testid="entropy-balance-metric">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Entropy Balance</span>
                        {getStatusIcon(resonanceStatus.target_comparison?.entropy?.status)}
                      </div>
                      <div className="text-2xl font-bold">
                        {resonanceStatus.entropy_balance?.toFixed(3) || 'N/A'}
                      </div>
                      <Progress
                        value={(resonanceStatus.entropy_balance || 0.5) * 100}
                        className="h-2"
                      />
                      <p className="text-xs text-muted-foreground">
                        Target: {resonanceStatus.target_comparison?.entropy?.target}
                      </p>
                    </div>
                  </div>

                  {/* Health Status */}
                  <div className="flex items-center gap-4">
                    <Badge className={getHealthBadge(resonanceStatus.resonance_health)}>
                      {resonanceStatus.resonance_health?.toUpperCase() || 'UNKNOWN'}
                    </Badge>
                    <span className="text-sm text-muted-foreground">
                      Last Analysis: {resonanceStatus.last_analysis ? new Date(resonanceStatus.last_analysis).toLocaleString() : 'Never'}
                    </span>
                  </div>

                  {/* Warnings */}
                  {resonanceStatus.stability_warnings && resonanceStatus.stability_warnings.length > 0 && (
                    <Alert>
                      <AlertTriangle className="h-4 w-4" />
                      <AlertDescription>
                        <div className="font-medium mb-2">Stability Warnings:</div>
                        <ul className="list-disc list-inside space-y-1">
                          {resonanceStatus.stability_warnings.map((warning, idx) => (
                            <li key={idx} className="text-sm">{warning}</li>
                          ))}
                        </ul>
                      </AlertDescription>
                    </Alert>
                  )}
                </>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>{resonanceStatus?.message || 'Resonance system initializing...'}</p>
                  <Button onClick={handleAnalyzeResonance} className="mt-4" disabled={loading}>
                    Run Initial Analysis
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tab 2: Temporal Stability */}
        <TabsContent value="stability" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Drift Forecasts & Trend Lines</CardTitle>
                  <CardDescription>Predicted stability trends and parameter drift analysis</CardDescription>
                </div>
                <Button onClick={handleFetchForecast} disabled={loading} size="sm" variant="outline">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Update Forecast
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              {stabilityForecast ? (
                <>
                  {/* Forecast Summary */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <Clock className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm font-medium">Forecast Horizon</span>
                      </div>
                      <div className="text-2xl font-bold">
                        {stabilityForecast.forecast_horizon_hours}h
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Confidence: {(stabilityForecast.confidence_score * 100).toFixed(0)}%
                      </p>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <TrendingUp className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm font-medium">Predicted Resonance</span>
                      </div>
                      <div className="text-2xl font-bold">
                        {stabilityForecast.predicted_resonance_index?.toFixed(3)}
                      </div>
                      <div className="flex items-center gap-2">
                        {getTrendIcon(stabilityForecast.resonance_trend)}
                        <span className="text-xs text-muted-foreground capitalize">
                          {stabilityForecast.resonance_trend}
                        </span>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <BarChart3 className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm font-medium">Predicted Stability</span>
                      </div>
                      <div className="text-2xl font-bold">
                        {stabilityForecast.predicted_temporal_stability?.toFixed(3)}
                      </div>
                      <div className="flex items-center gap-2">
                        {getTrendIcon(stabilityForecast.stability_trend)}
                        <span className="text-xs text-muted-foreground capitalize">
                          {stabilityForecast.stability_trend}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Risk Assessment */}
                  <div>
                    <h4 className="font-medium mb-3">Risk Assessment</h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      <div className="flex items-center justify-between p-3 border rounded-lg">
                        <span className="text-sm">Drift Risk</span>
                        <Badge className={getRiskBadge(stabilityForecast.drift_risk_level)}>
                          {stabilityForecast.drift_risk_level?.toUpperCase()}
                        </Badge>
                      </div>
                      <div className="flex items-center justify-between p-3 border rounded-lg">
                        <span className="text-sm">Oscillation Risk</span>
                        <Badge className={getRiskBadge(stabilityForecast.oscillation_risk_level)}>
                          {stabilityForecast.oscillation_risk_level?.toUpperCase()}
                        </Badge>
                      </div>
                      <div className="flex items-center justify-between p-3 border rounded-lg">
                        <span className="text-sm">Stagnation Risk</span>
                        <Badge className={getRiskBadge(stabilityForecast.stagnation_risk_level)}>
                          {stabilityForecast.stagnation_risk_level?.toUpperCase()}
                        </Badge>
                      </div>
                    </div>
                  </div>

                  {/* Recommended Interventions */}
                  {stabilityForecast.recommended_interventions && stabilityForecast.recommended_interventions.length > 0 && (
                    <div>
                      <h4 className="font-medium mb-3">Recommended Interventions</h4>
                      <div className="space-y-2">
                        {stabilityForecast.recommended_interventions.map((intervention, idx) => (
                          <div key={idx} className="p-3 border rounded-lg text-sm">
                            {intervention}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Clock className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No forecast available</p>
                  <Button onClick={handleFetchForecast} className="mt-4" disabled={loading}>
                    Generate Forecast
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tab 3: Feedback Regulation */}
        <TabsContent value="feedback" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Adaptive Feedback Weights</CardTitle>
                  <CardDescription>Real-time feedback weight adjustments (view-only)</CardDescription>
                </div>
                <Button onClick={handleRecalibrate} disabled={loading} size="sm" variant="outline">
                  <Zap className="h-4 w-4 mr-2" />
                  Recalibrate
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              {balanceResult ? (
                <>
                  {/* Current Parameters */}
                  {balanceResult.current_parameters && (
                    <div>
                      <h4 className="font-medium mb-3">Current Parameters</h4>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="p-4 border rounded-lg">
                          <div className="text-sm text-muted-foreground mb-1">Novelty Weight</div>
                          <div className="text-2xl font-bold">
                            {balanceResult.current_parameters.novelty_weight?.toFixed(3)}
                          </div>
                          <Progress 
                            value={(balanceResult.current_parameters.novelty_weight || 0) * 100}
                            className="h-2 mt-2"
                          />
                        </div>
                        <div className="p-4 border rounded-lg">
                          <div className="text-sm text-muted-foreground mb-1">Stability Weight</div>
                          <div className="text-2xl font-bold">
                            {balanceResult.current_parameters.stability_weight?.toFixed(3)}
                          </div>
                          <Progress 
                            value={(balanceResult.current_parameters.stability_weight || 0) * 100}
                            className="h-2 mt-2"
                          />
                        </div>
                        <div className="p-4 border rounded-lg">
                          <div className="text-sm text-muted-foreground mb-1">Ethical Threshold</div>
                          <div className="text-2xl font-bold">
                            {balanceResult.current_parameters.ethical_threshold?.toFixed(3)}
                          </div>
                          <Progress 
                            value={(balanceResult.current_parameters.ethical_threshold || 0) * 100}
                            className="h-2 mt-2"
                          />
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Recommended Adjustments */}
                  {balanceResult.recommended_adjustments && Object.keys(balanceResult.recommended_adjustments).length > 0 ? (
                    <div>
                      <h4 className="font-medium mb-3">Recommended Adjustments</h4>
                      <Alert>
                        <AlertDescription>
                          <div className="space-y-2">
                            {Object.entries(balanceResult.recommended_adjustments).map(([param, value]) => (
                              <div key={param} className="flex items-center justify-between">
                                <span className="text-sm font-medium capitalize">
                                  {param.replace('_', ' ')}
                                </span>
                                <span className="text-sm">
                                  → {value.toFixed(3)}
                                </span>
                              </div>
                            ))}
                          </div>
                          <p className="text-xs mt-3 text-muted-foreground">
                            {balanceResult.advisory_note}
                          </p>
                        </AlertDescription>
                      </Alert>
                    </div>
                  ) : (
                    <Alert>
                      <CheckCircle2 className="h-4 w-4" />
                      <AlertDescription>
                        System feedback weights are balanced. No adjustments needed.
                      </AlertDescription>
                    </Alert>
                  )}

                  {/* Balance Metrics */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 border rounded-lg">
                      <div className="text-sm text-muted-foreground mb-1">Current Equilibrium</div>
                      <div className="text-2xl font-bold">
                        {balanceResult.current_equilibrium?.toFixed(3)}
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        Target: {balanceResult.target_equilibrium?.toFixed(2)}
                      </p>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <div className="text-sm text-muted-foreground mb-1">Current Entropy</div>
                      <div className="text-2xl font-bold">
                        {balanceResult.current_entropy?.toFixed(3)}
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        Target: {balanceResult.target_entropy_range?.join(' - ')}
                      </p>
                    </div>
                  </div>
                </>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Zap className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No calibration data available</p>
                  <Button onClick={handleRecalibrate} className="mt-4" disabled={loading}>
                    Run Calibration
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tab 4: Entropy Control */}
        <TabsContent value="entropy" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Novelty vs Stability Index</CardTitle>
              <CardDescription>Entropy balance visualization and control metrics</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {resonanceStatus?.system_status === 'operational' && (
                <>
                  {/* Entropy Visualization */}
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Entropy Balance</span>
                      <Badge className={
                        resonanceStatus.entropy_balance >= 0.45 && resonanceStatus.entropy_balance <= 0.55
                          ? 'bg-green-100 text-green-800'
                          : 'bg-yellow-100 text-yellow-800'
                      }>
                        {resonanceStatus.entropy_balance >= 0.45 && resonanceStatus.entropy_balance <= 0.55 
                          ? 'BALANCED' 
                          : resonanceStatus.entropy_balance < 0.45 ? 'LOW ENTROPY' : 'HIGH ENTROPY'}
                      </Badge>
                    </div>
                    
                    <div className="relative h-8 bg-gradient-to-r from-blue-200 via-green-200 to-orange-200 rounded-lg">
                      <div 
                        className="absolute top-0 bottom-0 w-1 bg-black"
                        style={{ 
                          left: `${(resonanceStatus.entropy_balance || 0.5) * 100}%`,
                          transform: 'translateX(-50%)'
                        }}
                      />
                      <div className="absolute inset-0 flex items-center justify-between px-2 text-xs">
                        <span>Stability</span>
                        <span>Balanced</span>
                        <span>Novelty</span>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-2 text-center text-xs text-muted-foreground">
                      <div>0.0 - 0.45<br/>(Stagnation Risk)</div>
                      <div>0.45 - 0.55<br/>(Optimal)</div>
                      <div>0.55 - 1.0<br/>(Instability Risk)</div>
                    </div>
                  </div>

                  {/* Entropy Metrics */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 border rounded-lg">
                      <div className="text-sm text-muted-foreground mb-2">Current Entropy</div>
                      <div className="text-3xl font-bold">
                        {resonanceStatus.entropy_balance?.toFixed(3)}
                      </div>
                      <div className="mt-2 flex items-center gap-2 text-sm">
                        {resonanceStatus.entropy_balance < 0.45 && (
                          <>
                            <AlertTriangle className="h-4 w-4 text-yellow-600" />
                            <span className="text-muted-foreground">Below target range</span>
                          </>
                        )}
                        {resonanceStatus.entropy_balance > 0.55 && (
                          <>
                            <AlertTriangle className="h-4 w-4 text-orange-600" />
                            <span className="text-muted-foreground">Above target range</span>
                          </>
                        )}
                        {resonanceStatus.entropy_balance >= 0.45 && resonanceStatus.entropy_balance <= 0.55 && (
                          <>
                            <CheckCircle2 className="h-4 w-4 text-green-600" />
                            <span className="text-muted-foreground">Within target range</span>
                          </>
                        )}
                      </div>
                    </div>

                    <div className="p-4 border rounded-lg">
                      <div className="text-sm text-muted-foreground mb-2">Target Range</div>
                      <div className="text-3xl font-bold">
                        0.45 - 0.55
                      </div>
                      <p className="text-sm text-muted-foreground mt-2">
                        Optimal exploration/exploitation balance
                      </p>
                    </div>
                  </div>

                  {/* Recommendations */}
                  {(resonanceStatus.entropy_balance < 0.45 || resonanceStatus.entropy_balance > 0.55) && (
                    <Alert>
                      <AlertTriangle className="h-4 w-4" />
                      <AlertDescription>
                        <div className="font-medium mb-2">Entropy Imbalance Detected</div>
                        {resonanceStatus.entropy_balance < 0.45 ? (
                          <p className="text-sm">
                            System entropy is low. Consider increasing novelty exploration to prevent stagnation.
                            Recommended: Boost creative strategy generation and parameter variation.
                          </p>
                        ) : (
                          <p className="text-sm">
                            System entropy is high. Consider increasing stability focus to prevent instability.
                            Recommended: Strengthen parameter constraints and reduce exploration rate.
                          </p>
                        )}
                      </AlertDescription>
                    </Alert>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tab 5: Historical Resonance */}
        <TabsContent value="historical" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Timeline & Export Tools</CardTitle>
              <CardDescription>Historical resonance data and trend analysis</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {resonanceMetrics.length > 0 ? (
                <>
                  {/* Timeline Summary */}
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="p-4 border rounded-lg">
                      <div className="text-sm text-muted-foreground mb-1">Total Analyses</div>
                      <div className="text-2xl font-bold">{resonanceMetrics.length}</div>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <div className="text-sm text-muted-foreground mb-1">Avg Resonance</div>
                      <div className="text-2xl font-bold">
                        {(resonanceMetrics.reduce((sum, m) => sum + (m.resonance_index || 0), 0) / resonanceMetrics.length).toFixed(3)}
                      </div>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <div className="text-sm text-muted-foreground mb-1">Avg Stability</div>
                      <div className="text-2xl font-bold">
                        {(resonanceMetrics.reduce((sum, m) => sum + (m.temporal_stability || 0), 0) / resonanceMetrics.length).toFixed(3)}
                      </div>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <div className="text-sm text-muted-foreground mb-1">Health Status</div>
                      <div className="text-sm font-bold">
                        {resonanceMetrics[0]?.resonance_health?.toUpperCase() || 'UNKNOWN'}
                      </div>
                    </div>
                  </div>

                  {/* Recent Metrics */}
                  <div>
                    <h4 className="font-medium mb-3">Recent Analyses (Last 10)</h4>
                    <div className="space-y-2">
                      {resonanceMetrics.slice(0, 10).map((metric, idx) => (
                        <div key={idx} className="p-3 border rounded-lg flex items-center justify-between">
                          <div>
                            <div className="text-sm font-medium">
                              {new Date(metric.timestamp).toLocaleString()}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              Resonance: {metric.resonance_index?.toFixed(3)} | 
                              Stability: {metric.temporal_stability?.toFixed(3)} | 
                              Entropy: {metric.entropy_balance?.toFixed(3)}
                            </div>
                          </div>
                          <Badge className={getHealthBadge(metric.resonance_health)}>
                            {metric.resonance_health}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Export Options */}
                  <div className="flex gap-2">
                    <Button onClick={handleDownloadReport} variant="outline" size="sm">
                      <Download className="h-4 w-4 mr-2" />
                      Download Full Report
                    </Button>
                  </div>
                </>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No historical data available</p>
                  <p className="text-sm mt-2">Run resonance analyses to build historical data</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default CognitiveResonancePanel;
