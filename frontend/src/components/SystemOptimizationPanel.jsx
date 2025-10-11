import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { 
  Activity, Cpu, Database, Zap, FileText, PlayCircle, 
  TrendingUp, TrendingDown, AlertCircle, CheckCircle, 
  Clock, BarChart3, Settings, Download, RefreshCw,
  HardDrive, Gauge, Target
} from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || '';

const SystemOptimizationPanel = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [status, setStatus] = useState(null);
  const [report, setReport] = useState(null);
  const [logs, setLogs] = useState([]);
  const [metricsHistory, setMetricsHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [optimizing, setOptimizing] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);

  // Fetch current optimization status
  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/llm/optimize/status`);
      const data = await response.json();
      if (data.success) {
        setStatus(data);
        setLastUpdate(new Date());
      }
    } catch (error) {
      console.error('Error fetching optimization status:', error);
    }
  }, []);

  // Fetch optimization report
  const fetchReport = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`${BACKEND_URL}/api/llm/optimize/report`);
      const data = await response.json();
      if (data.success) {
        setReport(data);
      }
    } catch (error) {
      console.error('Error fetching optimization report:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch optimization logs
  const fetchLogs = useCallback(async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/llm/optimize/logs?limit=20`);
      const data = await response.json();
      if (data.success) {
        setLogs(data.logs);
      }
    } catch (error) {
      console.error('Error fetching optimization logs:', error);
    }
  }, []);

  // Fetch metrics history
  const fetchMetricsHistory = useCallback(async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/llm/optimize/metrics/history?limit=50`);
      const data = await response.json();
      if (data.success) {
        setMetricsHistory(data.metrics);
      }
    } catch (error) {
      console.error('Error fetching metrics history:', error);
    }
  }, []);

  // Run full optimization cycle
  const runOptimization = async () => {
    try {
      setOptimizing(true);
      const response = await fetch(`${BACKEND_URL}/api/llm/optimize/run`, {
        method: 'POST'
      });
      const data = await response.json();
      
      if (data.success) {
        // Refresh all data after optimization
        await Promise.all([
          fetchStatus(),
          fetchReport(),
          fetchLogs(),
          fetchMetricsHistory()
        ]);
      }
    } catch (error) {
      console.error('Error running optimization:', error);
    } finally {
      setOptimizing(false);
    }
  };

  // Auto-refresh effect
  useEffect(() => {
    fetchStatus();
    fetchReport();
    fetchLogs();
    fetchMetricsHistory();

    if (autoRefresh) {
      const interval = setInterval(() => {
        fetchStatus();
      }, 5000); // Refresh status every 5 seconds

      return () => clearInterval(interval);
    }
  }, [autoRefresh, fetchStatus, fetchReport, fetchLogs, fetchMetricsHistory]);

  // Helper function to get health badge
  const getHealthBadge = (health) => {
    const badges = {
      'excellent': <Badge className="bg-green-500" data-testid="health-badge-excellent">Excellent</Badge>,
      'good': <Badge className="bg-blue-500" data-testid="health-badge-good">Good</Badge>,
      'fair': <Badge className="bg-yellow-500" data-testid="health-badge-fair">Fair</Badge>,
      'needs_optimization': <Badge className="bg-orange-500" data-testid="health-badge-needs-optimization">Needs Optimization</Badge>,
      'needs_attention': <Badge className="bg-red-500" data-testid="health-badge-needs-attention">Needs Attention</Badge>
    };
    return badges[health] || <Badge data-testid="health-badge-unknown">Unknown</Badge>;
  };

  // Helper function to format metric value
  const formatMetric = (value, unit = '', decimals = 2) => {
    if (typeof value === 'number') {
      return `${value.toFixed(decimals)}${unit}`;
    }
    return value || 'N/A';
  };

  // Helper function to get trend icon
  const getTrendIcon = (trend) => {
    if (trend === 'improving') return <TrendingUp className="w-4 h-4 text-green-500" />;
    if (trend === 'degrading') return <TrendingDown className="w-4 h-4 text-red-500" />;
    return <Activity className="w-4 h-4 text-blue-500" />;
  };

  return (
    <div className="space-y-6" data-testid="system-optimization-panel">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight" data-testid="panel-title">
            System Optimization & Performance
          </h2>
          <p className="text-muted-foreground" data-testid="panel-description">
            Unified optimization framework for cognitive subsystems
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant={autoRefresh ? "default" : "outline"}
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
            data-testid="auto-refresh-toggle"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
            Auto Refresh
          </Button>
          <Button
            onClick={runOptimization}
            disabled={optimizing || status?.optimization_active}
            data-testid="run-optimization-button"
          >
            <PlayCircle className="w-4 h-4 mr-2" />
            {optimizing ? 'Optimizing...' : 'Run Optimization'}
          </Button>
        </div>
      </div>

      {/* Status Bar */}
      {status && (
        <Card data-testid="status-bar">
          <CardContent className="pt-6">
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">System Health</p>
                <div className="text-lg font-semibold">
                  {getHealthBadge(status.system_health)}
                </div>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">CPU Usage</p>
                <p className="text-lg font-semibold" data-testid="cpu-usage">
                  {formatMetric(status.current_cpu, '%', 1)}
                </p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Memory</p>
                <p className="text-lg font-semibold" data-testid="memory-usage">
                  {formatMetric(status.current_memory_percent, '%', 1)}
                </p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Avg Latency</p>
                <p className="text-lg font-semibold" data-testid="avg-latency">
                  {formatMetric(status.avg_latency, 's')}
                </p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Optimizations</p>
                <p className="text-lg font-semibold" data-testid="total-optimizations">
                  {status.total_optimizations || 0}
                </p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Last Update</p>
                <p className="text-sm font-medium" data-testid="last-update">
                  {lastUpdate ? lastUpdate.toLocaleTimeString() : 'N/A'}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} data-testid="optimization-tabs">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview" data-testid="tab-overview">
            <BarChart3 className="w-4 h-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="resources" data-testid="tab-resources">
            <Cpu className="w-4 h-4 mr-2" />
            Resources
          </TabsTrigger>
          <TabsTrigger value="llm" data-testid="tab-llm">
            <Zap className="w-4 h-4 mr-2" />
            LLM Scaling
          </TabsTrigger>
          <TabsTrigger value="database" data-testid="tab-database">
            <Database className="w-4 h-4 mr-2" />
            Database
          </TabsTrigger>
          <TabsTrigger value="logs" data-testid="tab-logs">
            <FileText className="w-4 h-4 mr-2" />
            Logs
          </TabsTrigger>
        </TabsList>

        {/* Tab 1: Performance Overview */}
        <TabsContent value="overview" className="space-y-4" data-testid="overview-content">
          {report ? (
            <>
              {/* Overall Health Summary */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="w-5 h-5" />
                    Performance Overview
                  </CardTitle>
                  <CardDescription>Report #{report.report_number} - {new Date(report.timestamp).toLocaleString()}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Health Status */}
                  <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
                    <div className="space-y-1">
                      <p className="text-sm text-muted-foreground">Overall System Health</p>
                      <div className="flex items-center gap-2">
                        {getHealthBadge(report.overall_health)}
                        {report.overall_health === 'excellent' && (
                          <CheckCircle className="w-5 h-5 text-green-500" />
                        )}
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-muted-foreground">Critical Issues</p>
                      <p className="text-2xl font-bold" data-testid="critical-issues-count">
                        {report.critical_issues?.length || 0}
                      </p>
                    </div>
                  </div>

                  {/* Critical Issues */}
                  {report.critical_issues && report.critical_issues.length > 0 && (
                    <div className="space-y-2">
                      <h4 className="font-semibold flex items-center gap-2">
                        <AlertCircle className="w-4 h-4 text-orange-500" />
                        Critical Issues
                      </h4>
                      <div className="space-y-2">
                        {report.critical_issues.map((issue, idx) => (
                          <div key={idx} className="flex items-start gap-2 p-3 bg-orange-50 border border-orange-200 rounded" data-testid={`critical-issue-${idx}`}>
                            <AlertCircle className="w-4 h-4 text-orange-500 mt-0.5" />
                            <p className="text-sm">{issue}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Key Metrics */}
                  <div className="space-y-2">
                    <h4 className="font-semibold">Key Performance Metrics</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {report.current_metrics && Object.entries(report.current_metrics).map(([key, value]) => {
                        const target = report.target_metrics?.[key];
                        const delta = report.metrics_delta?.[key];
                        const isGood = delta >= 0;
                        
                        return (
                          <div key={key} className="p-4 border rounded-lg" data-testid={`metric-${key}`}>
                            <div className="flex items-center justify-between mb-2">
                              <p className="text-sm font-medium capitalize">
                                {key.replace(/_/g, ' ')}
                              </p>
                              {isGood ? (
                                <CheckCircle className="w-4 h-4 text-green-500" />
                              ) : (
                                <AlertCircle className="w-4 h-4 text-orange-500" />
                              )}
                            </div>
                            <div className="space-y-1">
                              <p className="text-2xl font-bold">{formatMetric(value)}</p>
                              <div className="flex items-center gap-2 text-sm">
                                <span className="text-muted-foreground">
                                  Target: {formatMetric(target)}
                                </span>
                                {delta !== undefined && (
                                  <Badge variant={isGood ? "success" : "warning"} className="text-xs">
                                    {delta >= 0 ? '+' : ''}{formatMetric(delta, '%', 1)}
                                  </Badge>
                                )}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Trends */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 border rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        {getTrendIcon(report.latency_trend)}
                        <p className="font-medium">Latency Trend</p>
                      </div>
                      <p className="text-sm text-muted-foreground capitalize">{report.latency_trend}</p>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        {getTrendIcon(report.resource_trend)}
                        <p className="font-medium">Resource Trend</p>
                      </div>
                      <p className="text-sm text-muted-foreground capitalize">{report.resource_trend}</p>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        {getTrendIcon(report.efficiency_trend)}
                        <p className="font-medium">Efficiency Trend</p>
                      </div>
                      <p className="text-sm text-muted-foreground capitalize">{report.efficiency_trend}</p>
                    </div>
                  </div>

                  {/* Top Recommendations */}
                  {report.recommendations && report.recommendations.length > 0 && (
                    <div className="space-y-2">
                      <h4 className="font-semibold flex items-center gap-2">
                        <Target className="w-4 h-4 text-blue-500" />
                        Top Optimization Recommendations
                      </h4>
                      <div className="space-y-2">
                        {report.recommendations.slice(0, 5).map((rec, idx) => (
                          <div key={idx} className="flex items-start gap-2 p-3 bg-blue-50 border border-blue-200 rounded" data-testid={`recommendation-${idx}`}>
                            <CheckCircle className="w-4 h-4 text-blue-500 mt-0.5" />
                            <p className="text-sm">{rec}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Module Performance */}
              <Card>
                <CardHeader>
                  <CardTitle>Module-Specific Performance</CardTitle>
                  <CardDescription>Performance breakdown by cognitive module</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {report.module_performance && Object.entries(report.module_performance).map(([module, perf]) => (
                      <div key={module} className="p-4 border rounded-lg" data-testid={`module-${module}`}>
                        <div className="flex items-center justify-between mb-2">
                          <p className="font-medium capitalize">{module}</p>
                          <Badge variant={perf.status === 'healthy' ? 'success' : 'warning'}>
                            {perf.status}
                          </Badge>
                        </div>
                        <div className="space-y-1">
                          <p className="text-sm text-muted-foreground">Avg Latency</p>
                          <p className="text-xl font-bold">{formatMetric(perf.avg_latency, 's')}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card>
              <CardContent className="pt-6 text-center">
                <p className="text-muted-foreground">Loading optimization report...</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Tab 2: Resource Balancer */}
        <TabsContent value="resources" className="space-y-4" data-testid="resources-content">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cpu className="w-5 h-5" />
                CPU/GPU Resource Balancer
              </CardTitle>
              <CardDescription>Real-time resource utilization and balancing</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {status && report && (
                <>
                  {/* Resource Utilization Gauges */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <p className="font-medium flex items-center gap-2">
                          <Cpu className="w-4 h-4" />
                          CPU Utilization
                        </p>
                        <span className="text-2xl font-bold">{formatMetric(status.current_cpu, '%', 1)}</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-4">
                        <div
                          className={`h-4 rounded-full transition-all ${
                            status.current_cpu > 85 ? 'bg-red-500' : status.current_cpu > 70 ? 'bg-yellow-500' : 'bg-green-500'
                          }`}
                          style={{ width: `${Math.min(status.current_cpu, 100)}%` }}
                        />
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Target: {'<'} 70% for optimal performance
                      </p>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <p className="font-medium flex items-center gap-2">
                          <Gauge className="w-4 h-4" />
                          GPU Utilization (Simulated)
                        </p>
                        <span className="text-2xl font-bold">65%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-4">
                        <div className="h-4 rounded-full bg-blue-500 transition-all" style={{ width: '65%' }} />
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Target: 60-80% for optimal throughput
                      </p>
                    </div>
                  </div>

                  {/* CPU/GPU Balance Score */}
                  <div className="p-4 bg-muted rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <p className="font-medium">CPU/GPU Balance Score</p>
                      <span className="text-3xl font-bold">
                        {report.current_metrics?.cpu_gpu_balance 
                          ? formatMetric(report.current_metrics.cpu_gpu_balance * 100, '%', 0)
                          : 'N/A'}
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-4">
                      <div
                        className={`h-4 rounded-full transition-all ${
                          (report.current_metrics?.cpu_gpu_balance || 0) >= 0.90 ? 'bg-green-500' : 
                          (report.current_metrics?.cpu_gpu_balance || 0) >= 0.80 ? 'bg-yellow-500' : 'bg-orange-500'
                        }`}
                        style={{ width: `${(report.current_metrics?.cpu_gpu_balance || 0) * 100}%` }}
                      />
                    </div>
                    <p className="text-sm text-muted-foreground mt-2">
                      Target: ≥ 90% for excellent balance
                    </p>
                  </div>

                  {/* Memory Usage */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <p className="font-medium flex items-center gap-2">
                        <HardDrive className="w-4 h-4" />
                        Memory Usage
                      </p>
                      <span className="text-2xl font-bold">{formatMetric(status.current_memory_percent, '%', 1)}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-4">
                      <div
                        className={`h-4 rounded-full transition-all ${
                          status.current_memory_percent > 85 ? 'bg-red-500' : status.current_memory_percent > 75 ? 'bg-yellow-500' : 'bg-green-500'
                        }`}
                        style={{ width: `${Math.min(status.current_memory_percent, 100)}%` }}
                      />
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Target: {'<'} 75% for optimal performance
                    </p>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tab 3: LLM Scaling */}
        <TabsContent value="llm" className="space-y-4" data-testid="llm-content">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="w-5 h-5" />
                LLM Inference Scaling
              </CardTitle>
              <CardDescription>Adaptive inference depth and load balancing</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {status && report && (
                <>
                  {/* LLM Stats */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 border rounded-lg">
                      <p className="text-sm text-muted-foreground mb-1">Total Inferences</p>
                      <p className="text-3xl font-bold" data-testid="llm-inferences">{status.llm_inferences || 0}</p>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <p className="text-sm text-muted-foreground mb-1">Avg Latency</p>
                      <p className="text-3xl font-bold">{formatMetric(report.current_metrics?.llm_scaling_responsiveness, 's')}</p>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <p className="text-sm text-muted-foreground mb-1">Depth Adjustments</p>
                      <p className="text-3xl font-bold">0</p>
                    </div>
                  </div>

                  {/* Provider Distribution */}
                  <div className="space-y-2">
                    <h4 className="font-semibold">Provider Distribution</h4>
                    <div className="space-y-3">
                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-sm">
                          <span>Claude 3.5 Sonnet (Primary)</span>
                          <span className="font-medium">50%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div className="h-2 rounded-full bg-purple-500" style={{ width: '50%' }} />
                        </div>
                      </div>
                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-sm">
                          <span>GPT-4o-mini (Secondary)</span>
                          <span className="font-medium">30%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div className="h-2 rounded-full bg-green-500" style={{ width: '30%' }} />
                        </div>
                      </div>
                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-sm">
                          <span>Gemini 2.0 Flash (Fallback)</span>
                          <span className="font-medium">20%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div className="h-2 rounded-full bg-blue-500" style={{ width: '20%' }} />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Scaling Responsiveness */}
                  <div className="p-4 bg-muted rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <p className="font-medium">Scaling Responsiveness</p>
                      <span className="text-3xl font-bold">
                        {formatMetric(report.current_metrics?.llm_scaling_responsiveness, 's')}
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-4">
                      <div
                        className={`h-4 rounded-full transition-all ${
                          (report.current_metrics?.llm_scaling_responsiveness || 0) <= 0.5 ? 'bg-green-500' : 
                          (report.current_metrics?.llm_scaling_responsiveness || 0) <= 1.0 ? 'bg-yellow-500' : 'bg-orange-500'
                        }`}
                        style={{ width: `${Math.min((report.current_metrics?.llm_scaling_responsiveness || 0) / 2 * 100, 100)}%` }}
                      />
                    </div>
                    <p className="text-sm text-muted-foreground mt-2">
                      Target: ≤ 0.5s for optimal responsiveness
                    </p>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tab 4: Database Efficiency */}
        <TabsContent value="database" className="space-y-4" data-testid="database-content">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="w-5 h-5" />
                Database I/O Efficiency
              </CardTitle>
              <CardDescription>MongoDB operations and optimization</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {status && report && (
                <>
                  {/* Database Operations */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 border rounded-lg">
                      <p className="text-sm text-muted-foreground mb-1">Total Operations</p>
                      <p className="text-3xl font-bold" data-testid="db-operations">{status.db_operations || 0}</p>
                      <p className="text-xs text-muted-foreground mt-1">Read + Write combined</p>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <p className="text-sm text-muted-foreground mb-1">I/O Efficiency</p>
                      <p className="text-3xl font-bold">
                        {formatMetric((report.current_metrics?.db_io_efficiency || 0) * 100, '%', 0)}
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">Target: ≥ 92%</p>
                    </div>
                  </div>

                  {/* I/O Efficiency Score */}
                  <div className="p-4 bg-muted rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <p className="font-medium">Database I/O Efficiency Score</p>
                      <span className="text-3xl font-bold">
                        {formatMetric((report.current_metrics?.db_io_efficiency || 0) * 100, '%', 0)}
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-4">
                      <div
                        className={`h-4 rounded-full transition-all ${
                          (report.current_metrics?.db_io_efficiency || 0) >= 0.92 ? 'bg-green-500' : 
                          (report.current_metrics?.db_io_efficiency || 0) >= 0.85 ? 'bg-yellow-500' : 'bg-orange-500'
                        }`}
                        style={{ width: `${(report.current_metrics?.db_io_efficiency || 0) * 100}%` }}
                      />
                    </div>
                    <p className="text-sm text-muted-foreground mt-2">
                      Target: ≥ 92% for excellent efficiency
                    </p>
                  </div>

                  {/* Collections Status */}
                  <div className="space-y-2">
                    <h4 className="font-semibold">Active Collections</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {[
                        'llm_optimization_metrics',
                        'llm_optimization_logs',
                        'llm_memory_nodes',
                        'llm_creative_strategies',
                        'llm_cohesion_sessions',
                        'llm_reflection_logs'
                      ].map((collection) => (
                        <div key={collection} className="flex items-center justify-between p-3 border rounded">
                          <span className="text-sm font-mono">{collection}</span>
                          <CheckCircle className="w-4 h-4 text-green-500" />
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tab 5: Optimization Logs */}
        <TabsContent value="logs" className="space-y-4" data-testid="logs-content">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="w-5 h-5" />
                Optimization Logs & Reports
              </CardTitle>
              <CardDescription>Historical optimization actions and reports</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Export Report Button */}
                <div className="flex justify-end">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      if (report) {
                        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `optimization_report_${report.report_number}.json`;
                        a.click();
                      }
                    }}
                    data-testid="export-report-button"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Export Current Report
                  </Button>
                </div>

                {/* Logs Display */}
                <div className="space-y-2">
                  {logs.length > 0 ? (
                    logs.map((log, idx) => (
                      <div
                        key={idx}
                        className="p-4 border rounded-lg space-y-2"
                        data-testid={`log-entry-${idx}`}
                      >
                        <div className="flex items-center justify-between">
                          <Badge variant="outline">{log.type}</Badge>
                          <span className="text-sm text-muted-foreground flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {new Date(log.timestamp).toLocaleString()}
                          </span>
                        </div>
                        {log.type === 'report' && log.data && (
                          <div className="space-y-1">
                            <p className="font-medium">Report #{log.data.report_number}</p>
                            <p className="text-sm">Health: {log.data.overall_health}</p>
                            <p className="text-sm text-muted-foreground">
                              {log.data.critical_issues?.length || 0} critical issues, 
                              {' '}{log.data.recommendations?.length || 0} recommendations
                            </p>
                          </div>
                        )}
                        {log.type === 'optimization_cycle' && log.data && (
                          <div className="space-y-1">
                            <p className="font-medium">Optimization Cycle</p>
                            <p className="text-sm">Duration: {log.data.cycle_duration?.toFixed(2)}s</p>
                            <p className="text-sm">Status: {log.data.overall_status}</p>
                          </div>
                        )}
                      </div>
                    ))
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      No optimization logs available yet
                    </div>
                  )}
                </div>

                {/* Optimization Roadmap */}
                {report && report.optimization_roadmap && report.optimization_roadmap.length > 0 && (
                  <div className="mt-6 space-y-2">
                    <h4 className="font-semibold">Optimization Roadmap</h4>
                    <div className="space-y-2">
                      {report.optimization_roadmap.map((item, idx) => (
                        <div key={idx} className="flex items-start gap-2 p-3 bg-blue-50 border border-blue-200 rounded">
                          <span className="flex-shrink-0 w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-bold">
                            {idx + 1}
                          </span>
                          <p className="text-sm">{item}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default SystemOptimizationPanel;
