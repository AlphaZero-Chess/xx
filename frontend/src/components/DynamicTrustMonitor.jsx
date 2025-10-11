import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { RefreshCw, TrendingUp, Settings, Activity } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const DynamicTrustMonitor = () => {
  const [thresholdStatus, setThresholdStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [autoAdjustEnabled, setAutoAdjustEnabled] = useState(true);

  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

  // Fetch threshold status
  const fetchThresholdStatus = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${backendUrl}/api/llm/threshold/status`);
      const data = await response.json();
      
      if (data.success) {
        setThresholdStatus(data);
        setAutoAdjustEnabled(data.current_status?.auto_adjust_enabled ?? true);
      }
    } catch (error) {
      console.error('Error fetching threshold status:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchThresholdStatus();
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchThresholdStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  // Prepare chart data
  const prepareChartData = () => {
    if (!thresholdStatus?.threshold_trend) return [];
    
    return thresholdStatus.threshold_trend
      .slice()
      .reverse()
      .map((item, index) => ({
        index: index + 1,
        threshold: (item.threshold * 100).toFixed(1),
        trust_variance: (item.trust_variance * 100).toFixed(1),
        category: item.task_category || 'general',
        timestamp: item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : `T${index + 1}`
      }));
  };

  // Get threshold color based on value
  const getThresholdColor = (threshold) => {
    if (threshold >= 0.92) return 'text-red-600';
    if (threshold >= 0.88) return 'text-orange-500';
    if (threshold >= 0.85) return 'text-yellow-600';
    return 'text-green-600';
  };

  // Get variance level badge
  const getVarianceBadge = (variance) => {
    if (variance > 0.12) return <Badge variant="destructive">High Variance</Badge>;
    if (variance > 0.08) return <Badge variant="secondary">Medium Variance</Badge>;
    return <Badge variant="default">Low Variance</Badge>;
  };

  const chartData = prepareChartData();

  return (
    <div className="space-y-6" data-testid="dynamic-trust-monitor">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">Dynamic Trust Threshold Monitor</h2>
          <p className="text-gray-600">Real-time adaptive confidence threshold tracking</p>
        </div>
        <div className="flex gap-2 items-center">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">Auto-Adjust:</span>
            <Badge variant={autoAdjustEnabled ? "default" : "outline"}>
              {autoAdjustEnabled ? 'Enabled' : 'Disabled'}
            </Badge>
          </div>
          <Button 
            onClick={fetchThresholdStatus} 
            variant="outline"
            disabled={loading}
            data-testid="refresh-threshold-btn"
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Current Status Overview */}
      {thresholdStatus?.current_status && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="pt-6">
              <div className={`text-2xl font-bold ${getThresholdColor(thresholdStatus.current_status.global_threshold)}`}>
                {(thresholdStatus.current_status.global_threshold * 100).toFixed(1)}%
              </div>
              <p className="text-xs text-gray-600">Global Threshold</p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-blue-600">
                {(thresholdStatus.current_status.trust_variance * 100).toFixed(1)}%
              </div>
              <p className="text-xs text-gray-600">Trust Variance</p>
              <div className="mt-1">
                {getVarianceBadge(thresholdStatus.current_status.trust_variance)}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-purple-600">
                {(thresholdStatus.current_status.avg_complexity * 100).toFixed(0)}%
              </div>
              <p className="text-xs text-gray-600">Avg Complexity</p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="text-sm font-mono">
                <span className="text-gray-500">Range: </span>
                <span className="font-bold">
                  {(thresholdStatus.current_status.threshold_range.min * 100).toFixed(0)}% - {(thresholdStatus.current_status.threshold_range.max * 100).toFixed(0)}%
                </span>
              </div>
              <p className="text-xs text-gray-600 mt-1">Configured: {thresholdStatus.current_status.threshold_range.configured_range}</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Threshold Trend Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Confidence Threshold Over Time
          </CardTitle>
          <CardDescription>
            Real-time adaptive threshold adjustments based on trust variance and task complexity
          </CardDescription>
        </CardHeader>
        <CardContent>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  tick={{ fontSize: 11 }}
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis 
                  label={{ value: 'Threshold (%)', angle: -90, position: 'insideLeft' }}
                  domain={[75, 100]}
                />
                <Tooltip 
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="bg-white p-3 border rounded shadow-lg">
                          <p className="text-sm font-semibold">{payload[0].payload.timestamp}</p>
                          <p className="text-sm text-blue-600">
                            Threshold: {payload[0].value}%
                          </p>
                          <p className="text-sm text-orange-500">
                            Trust Variance: {payload[1]?.value || 'N/A'}%
                          </p>
                          <p className="text-xs text-gray-500 mt-1">
                            Category: {payload[0].payload.category}
                          </p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="threshold" 
                  stroke="#3b82f6" 
                  strokeWidth={2}
                  name="Threshold (%)"
                  dot={{ r: 4 }}
                />
                <Line 
                  type="monotone" 
                  dataKey="trust_variance" 
                  stroke="#f59e0b" 
                  strokeWidth={2}
                  name="Trust Variance (%)"
                  dot={{ r: 3 }}
                  strokeDasharray="5 5"
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center py-12 text-gray-500">
              <Activity className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No threshold data available yet</p>
              <p className="text-sm">Data will appear as arbitration sessions occur</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Threshold by Task Category */}
      {thresholdStatus?.thresholds_by_category && Object.keys(thresholdStatus.thresholds_by_category).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Threshold Values per Task Type
            </CardTitle>
            <CardDescription>
              Dynamic thresholds adjusted based on task complexity and agent trust variance
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 border-b">
                  <tr>
                    <th className="text-left p-3 font-semibold">Task Category</th>
                    <th className="text-center p-3 font-semibold">Current Threshold</th>
                    <th className="text-center p-3 font-semibold">Trust Variance</th>
                    <th className="text-center p-3 font-semibold">Complexity</th>
                    <th className="text-left p-3 font-semibold">Adjustment Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(thresholdStatus.thresholds_by_category).map(([category, threshold]) => (
                    <tr key={category} className="border-b hover:bg-gray-50">
                      <td className="p-3">
                        <Badge variant="outline" className="capitalize">
                          {category}
                        </Badge>
                      </td>
                      <td className="p-3 text-center">
                        <span className={`font-bold text-lg ${getThresholdColor(threshold.current_threshold)}`}>
                          {(threshold.current_threshold * 100).toFixed(1)}%
                        </span>
                        <div className="text-xs text-gray-500">
                          (base: {(threshold.base_threshold * 100).toFixed(0)}%)
                        </div>
                      </td>
                      <td className="p-3 text-center">
                        <span className="font-medium">
                          {(threshold.trust_variance * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="p-3 text-center">
                        <div className="flex items-center justify-center gap-2">
                          <div className="w-24 bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-purple-600 h-2 rounded-full"
                              style={{ width: `${threshold.complexity_rating * 100}%` }}
                            />
                          </div>
                          <span className="text-xs">
                            {(threshold.complexity_rating * 100).toFixed(0)}%
                          </span>
                        </div>
                      </td>
                      <td className="p-3 text-xs text-gray-600">
                        {threshold.adjustment_reason}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Statistics Summary */}
      {thresholdStatus?.statistics && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Threshold Calculation Statistics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
              <div>
                <div className="text-gray-600 mb-1">Total Calculations</div>
                <div className="text-xl font-bold">
                  {thresholdStatus.statistics.total_threshold_calculations}
                </div>
              </div>
              <div>
                <div className="text-gray-600 mb-1">Avg Threshold</div>
                <div className="text-xl font-bold text-blue-600">
                  {(thresholdStatus.statistics.avg_threshold * 100).toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-gray-600 mb-1">Avg Trust Variance</div>
                <div className="text-xl font-bold text-orange-600">
                  {(thresholdStatus.statistics.avg_trust_variance * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Explanation Tooltip */}
      <Card className="bg-blue-50 border-blue-200">
        <CardContent className="pt-6">
          <div className="flex gap-3">
            <div className="flex-shrink-0">
              <div className="h-10 w-10 bg-blue-100 rounded-full flex items-center justify-center">
                <TrendingUp className="h-5 w-5 text-blue-600" />
              </div>
            </div>
            <div className="text-sm">
              <p className="font-semibold text-blue-900 mb-1">How Dynamic Thresholds Work</p>
              <p className="text-blue-800">
                The system automatically adjusts confidence thresholds (80-95%) based on:
              </p>
              <ul className="list-disc list-inside mt-2 text-blue-700 space-y-1">
                <li><strong>Trust Variance:</strong> High variance between agent trust scores lowers the threshold (requires stronger consensus)</li>
                <li><strong>Task Complexity:</strong> Complex tasks (0.7-1.0) tighten thresholds; simple tasks (0.0-0.3) relax them</li>
                <li><strong>Formula:</strong> <code className="bg-blue-100 px-1 rounded">threshold = base × (1 - variance) × complexity_factor</code></li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default DynamicTrustMonitor;
