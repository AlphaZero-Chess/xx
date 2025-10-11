import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { 
  Activity,
  Clock,
  TrendingUp,
  TrendingDown,
  Minus,
  AlertCircle,
  CheckCircle2,
  Power,
  PowerOff,
  RefreshCw,
  Zap,
  Settings
} from 'lucide-react';
import { toast } from 'sonner';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const AdaptiveForecastPanel = () => {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [forecastData, setForecastData] = useState(null);

  // Load status on mount
  useEffect(() => {
    loadStatus();
    
    // Set up auto-refresh if enabled
    let interval;
    if (autoRefresh && status?.enabled) {
      interval = setInterval(() => {
        loadStatus();
      }, 60000); // Refresh every minute
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, status?.enabled]);

  const loadStatus = async () => {
    try {
      const response = await axios.get(`${API}/strategy/auto-tune/status`);
      if (response.data.success) {
        setStatus(response.data);
        setLastUpdate(new Date().toLocaleTimeString());
      }
    } catch (error) {
      console.error('Error loading auto-tuning status:', error);
    }
  };

  const toggleAutoTuning = async (enable) => {
    try {
      setLoading(true);
      const response = await axios.post(`${API}/strategy/auto-tune/toggle?enable=${enable}`);
      
      if (response.data.success) {
        setStatus(response.data.status);
        toast.success(response.data.message);
        await loadStatus();
      }
    } catch (error) {
      console.error('Error toggling auto-tuning:', error);
      toast.error('Failed to toggle auto-tuning');
    } finally {
      setLoading(false);
    }
  };

  const triggerManualForecast = async () => {
    try {
      setLoading(true);
      toast.info('Running real-time forecast...');
      
      const response = await axios.post(`${API}/llm/auto-forecast`);
      
      if (response.data.success) {
        setForecastData(response.data);
        
        if (response.data.tuning_applied) {
          toast.success('Auto-tuning applied successfully!');
        } else {
          toast.info('No tuning needed - system is stable');
        }
        
        await loadStatus();
      }
    } catch (error) {
      console.error('Error triggering forecast:', error);
      toast.error('Failed to run forecast');
    } finally {
      setLoading(false);
    }
  };

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'improving':
        return <TrendingUp className="text-green-500" size={18} />;
      case 'declining':
        return <TrendingDown className="text-red-500" size={18} />;
      case 'alert':
        return <AlertCircle className="text-red-500" size={18} />;
      default:
        return <Minus className="text-yellow-500" size={18} />;
    }
  };

  const getTrendColor = (trend) => {
    switch (trend) {
      case 'improving':
        return 'bg-green-500/20 text-green-400 border-green-500';
      case 'declining':
        return 'bg-red-500/20 text-red-400 border-red-500';
      case 'alert':
        return 'bg-red-600/20 text-red-300 border-red-600';
      default:
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Card */}
      <Card className="bg-gradient-to-r from-indigo-900/50 to-purple-900/50 border-indigo-700/50">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-indigo-500/20 rounded-lg">
                <Activity className="text-indigo-400" size={24} />
              </div>
              <div>
                <CardTitle className="text-white text-xl flex items-center gap-2">
                  Real-Time Adaptive Forecasting
                  {status?.enabled && (
                    <Badge className="bg-green-600 text-white">
                      <Power className="mr-1 h-3 w-3" />
                      Active
                    </Badge>
                  )}
                </CardTitle>
                <p className="text-slate-300 text-sm mt-1">
                  Continuous monitoring with automatic strategy optimization
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Button
                onClick={triggerManualForecast}
                disabled={loading}
                variant="outline"
                className="border-indigo-500 text-indigo-300 hover:bg-indigo-900/30"
                data-testid="trigger-forecast-btn"
              >
                {loading ? (
                  <>
                    <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                    Running...
                  </>
                ) : (
                  <>
                    <Zap className="mr-2 h-4 w-4" />
                    Run Now
                  </>
                )}
              </Button>

              <Button
                onClick={() => toggleAutoTuning(!status?.enabled)}
                disabled={loading}
                className={status?.enabled ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}
                data-testid="toggle-auto-tune-btn"
              >
                {status?.enabled ? (
                  <>
                    <PowerOff className="mr-2 h-4 w-4" />
                    Disable
                  </>
                ) : (
                  <>
                    <Power className="mr-2 h-4 w-4" />
                    Enable
                  </>
                )}
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Status Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* System Status */}
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-400">System Status</span>
              {status?.enabled ? (
                <CheckCircle2 className="text-green-500" size={20} />
              ) : (
                <PowerOff className="text-slate-500" size={20} />
              )}
            </div>
            <div className="text-xl font-bold text-white">
              {status?.enabled ? 'Active' : 'Paused'}
            </div>
            {lastUpdate && (
              <div className="text-xs text-slate-500 mt-1">
                Updated: {lastUpdate}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Next Forecast */}
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-400">Next Forecast</span>
              <Clock className="text-blue-400" size={20} />
            </div>
            <div className="text-xl font-bold text-white">
              {status?.next_run ? new Date(status.next_run).toLocaleTimeString() : 'N/A'}
            </div>
            <div className="text-xs text-slate-500 mt-1">
              Interval: {status?.interval_hours || 1}h
            </div>
          </CardContent>
        </Card>

        {/* Total Tunings */}
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-400">Total Tunings</span>
              <Settings className="text-purple-400" size={20} />
            </div>
            <div className="text-xl font-bold text-white">
              {status?.total_tunings || 0}
            </div>
            <div className="text-xs text-slate-500 mt-1">
              Threshold: {status?.deviation_threshold || 5}%
            </div>
          </CardContent>
        </Card>

        {/* Last Run */}
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-400">Last Run</span>
              <RefreshCw className="text-green-400" size={20} />
            </div>
            <div className="text-sm font-bold text-white">
              {status?.last_run ? new Date(status.last_run).toLocaleString() : 'Not run yet'}
            </div>
            <div className="text-xs text-slate-500 mt-1">
              Auto-refresh: {autoRefresh ? 'On' : 'Off'}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Latest Forecast Update */}
      {forecastData && (
        <Card className="bg-slate-800/30 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white text-lg flex items-center gap-2">
              Latest Forecast Update
              {forecastData.tuning_applied && (
                <Badge className="bg-purple-600">
                  <Zap className="mr-1 h-3 w-3" />
                  Auto-Tuned
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Forecast Metrics */}
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-slate-400">Forecast Metrics</h3>
                {forecastData.forecast_update && (
                  <div className="bg-slate-900/50 rounded-lg p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-300 capitalize">
                        {forecastData.forecast_update.metric_name}
                      </span>
                      {getTrendIcon(forecastData.forecast_update.trend)}
                    </div>
                    
                    <div className="flex items-baseline gap-2">
                      <span className="text-2xl font-bold text-white">
                        {forecastData.forecast_update.current_value}
                      </span>
                      <span className="text-slate-400">→</span>
                      <span className="text-xl font-semibold text-indigo-400">
                        {forecastData.forecast_update.forecasted_value}
                      </span>
                    </div>

                    <div className="flex items-center justify-between text-sm">
                      <Badge className={`border ${getTrendColor(forecastData.forecast_update.trend)}`}>
                        {forecastData.forecast_update.deviation_percent >= 0 ? '+' : ''}
                        {forecastData.forecast_update.deviation_percent.toFixed(1)}% deviation
                      </Badge>
                      <span className="text-slate-400">
                        Confidence: {(forecastData.forecast_update.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Tuning Decision */}
              {forecastData.tuning_decision && (
                <div className="space-y-3">
                  <h3 className="text-sm font-semibold text-slate-400">Tuning Applied</h3>
                  <div className="bg-purple-900/20 rounded-lg p-4 space-y-3 border border-purple-700/50">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-purple-300">
                        {forecastData.tuning_decision.trigger_reason.replace(/_/g, ' ')}
                      </span>
                      <Badge className="bg-purple-600 text-white">
                        {(forecastData.tuning_decision.confidence_score * 100).toFixed(0)}%
                      </Badge>
                    </div>

                    <div className="text-sm text-slate-300">
                      {forecastData.tuning_decision.reasoning}
                    </div>

                    {Object.keys(forecastData.tuning_decision.parameters_adjusted).length > 0 && (
                      <div className="space-y-2">
                        <div className="text-xs text-slate-400">Parameters Adjusted:</div>
                        {Object.entries(forecastData.tuning_decision.parameters_adjusted).map(([param, values]) => (
                          <div key={param} className="flex items-center justify-between text-sm bg-slate-800/50 rounded px-2 py-1">
                            <span className="text-slate-300 capitalize">
                              {param.replace(/_/g, ' ')}
                            </span>
                            <span className="text-indigo-400 font-mono">
                              {values.old.toFixed(4)} → {values.new.toFixed(4)}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}

                    <div className="text-xs text-green-400 flex items-center gap-1">
                      <CheckCircle2 size={14} />
                      Expected: {forecastData.tuning_decision.expected_impact.replace(/_/g, ' ')}
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="mt-4 text-xs text-slate-500">
              Generated: {new Date(forecastData.timestamp).toLocaleString()}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Last Decision Summary */}
      {status?.last_decision && !forecastData && (
        <Card className="bg-slate-800/30 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white text-lg">Last Auto-Tuning Decision</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Trigger:</span>
                <span className="text-white capitalize">
                  {status.last_decision.trigger_reason.replace(/_/g, ' ')}
                </span>
              </div>
              
              <div className="text-sm text-slate-300">
                {status.last_decision.reasoning}
              </div>

              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-400">Confidence:</span>
                <Badge className="bg-purple-600">
                  {(status.last_decision.confidence_score * 100).toFixed(0)}%
                </Badge>
              </div>

              <div className="text-xs text-slate-500">
                {new Date(status.last_decision.timestamp).toLocaleString()}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Instructions Card */}
      {!status?.enabled && (
        <Card className="bg-blue-900/20 border-blue-700/50">
          <CardContent className="p-6">
            <div className="flex items-start gap-3">
              <AlertCircle className="text-blue-400 flex-shrink-0 mt-1" size={20} />
              <div>
                <h3 className="text-blue-300 font-semibold mb-2">Enable Real-Time Adaptive Forecasting</h3>
                <p className="text-slate-300 text-sm mb-3">
                  When enabled, the system will automatically monitor model performance every hour and 
                  apply strategic adjustments when deviations exceed {status?.deviation_threshold || 5}%.
                </p>
                <ul className="text-sm text-slate-400 space-y-1 list-disc list-inside">
                  <li>Continuous performance monitoring</li>
                  <li>Automatic parameter optimization</li>
                  <li>Learning rate and MCTS depth adjustment</li>
                  <li>Prompt depth and temperature tuning</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default AdaptiveForecastPanel;
