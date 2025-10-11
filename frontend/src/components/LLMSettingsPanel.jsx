import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Loader2, Settings, Zap, TrendingUp, Brain, RefreshCw, CheckCircle2, AlertCircle } from 'lucide-react';
import axios from 'axios';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const LLMSettingsPanel = () => {
  const [config, setConfig] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [optimizationStatus, setOptimizationStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [localConfig, setLocalConfig] = useState({
    response_mode: 'balanced',
    prompt_depth: 5,
    adaptive_enabled: true,
    max_response_time: 10.0,
    fallback_mode: 'fast'
  });

  useEffect(() => {
    loadLLMSettings();
  }, []);

  const loadLLMSettings = async () => {
    setLoading(true);
    try {
      const [configRes, metricsRes, statusRes] = await Promise.all([
        axios.get(`${API}/llm/tune`),
        axios.get(`${API}/llm/performance-metrics?limit=50`),
        axios.get(`${API}/llm/optimization-status`)
      ]);

      setConfig(configRes.data.config);
      setLocalConfig(configRes.data.config);
      setMetrics(metricsRes.data);
      setOptimizationStatus(statusRes.data);
    } catch (error) {
      console.error('Error loading LLM settings:', error);
      toast.error('Failed to load LLM settings');
    } finally {
      setLoading(false);
    }
  };

  const saveConfig = async () => {
    setSaving(true);
    try {
      const response = await axios.post(`${API}/llm/tune`, localConfig);
      
      if (response.data.success) {
        setConfig(response.data.config);
        toast.success('LLM settings updated successfully!');
        
        // Reload metrics and status
        const [metricsRes, statusRes] = await Promise.all([
          axios.get(`${API}/llm/performance-metrics?limit=50`),
          axios.get(`${API}/llm/optimization-status`)
        ]);
        setMetrics(metricsRes.data);
        setOptimizationStatus(statusRes.data);
      }
    } catch (error) {
      console.error('Error saving LLM settings:', error);
      toast.error('Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const resetToDefaults = () => {
    setLocalConfig({
      response_mode: 'balanced',
      prompt_depth: 5,
      adaptive_enabled: true,
      max_response_time: 10.0,
      fallback_mode: 'fast'
    });
  };

  const getModeInfo = (mode) => {
    switch (mode) {
      case 'fast':
        return {
          label: 'Fast',
          description: 'Quick responses (1-2 sentences)',
          color: 'bg-green-500',
          icon: <Zap className="h-4 w-4" />
        };
      case 'insightful':
        return {
          label: 'Insightful',
          description: 'Detailed analysis with examples',
          color: 'bg-purple-500',
          icon: <Brain className="h-4 w-4" />
        };
      default:
        return {
          label: 'Balanced',
          description: 'Moderate detail (2-4 sentences)',
          color: 'bg-blue-500',
          icon: <TrendingUp className="h-4 w-4" />
        };
    }
  };

  const hasChanges = JSON.stringify(config) !== JSON.stringify(localConfig);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12" data-testid="llm-settings-loading">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500 mx-auto mb-4"></div>
          <p className="text-slate-400">Loading LLM settings...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="llm-settings-panel">
      {/* Configuration Card */}
      <Card className="bg-gradient-to-br from-indigo-900/50 via-blue-900/50 to-cyan-900/50 border-indigo-500" data-testid="llm-config-card">
        <CardHeader>
          <CardTitle className="flex items-center justify-between text-white">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-indigo-500 to-cyan-500 rounded-lg">
                <Settings className="h-6 w-6 text-white" />
              </div>
              <span>LLM Configuration</span>
            </div>
            <Badge className="bg-gradient-to-r from-indigo-500 to-cyan-500 text-white border-0">
              Global Settings
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Response Mode */}
          <div className="space-y-3">
            <label className="text-sm font-medium text-white">Response Mode</label>
            <div className="grid grid-cols-3 gap-3">
              {['fast', 'balanced', 'insightful'].map((mode) => {
                const modeInfo = getModeInfo(mode);
                const isSelected = localConfig.response_mode === mode;
                return (
                  <button
                    key={mode}
                    onClick={() => setLocalConfig({ ...localConfig, response_mode: mode })}
                    className={`p-4 rounded-lg border-2 transition-all ${
                      isSelected
                        ? `${modeInfo.color} border-white text-white`
                        : 'bg-slate-800 border-slate-600 text-slate-300 hover:border-slate-400'
                    }`}
                    data-testid={`mode-${mode}-btn`}
                  >
                    <div className="flex items-center justify-center mb-2">
                      {modeInfo.icon}
                    </div>
                    <div className="font-semibold text-sm mb-1">{modeInfo.label}</div>
                    <div className="text-xs opacity-80">{modeInfo.description}</div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Prompt Depth Slider */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-white">Prompt Depth</label>
              <Badge variant="outline" className="text-white border-white">
                Level {localConfig.prompt_depth}
              </Badge>
            </div>
            <input
              type="range"
              min="1"
              max="10"
              step="1"
              value={localConfig.prompt_depth}
              onChange={(e) => setLocalConfig({ ...localConfig, prompt_depth: parseInt(e.target.value) })}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
              data-testid="prompt-depth-slider"
            />
            <div className="flex justify-between text-xs text-slate-400">
              <span>Minimal (1)</span>
              <span>Moderate (5)</span>
              <span>Detailed (10)</span>
            </div>
          </div>

          {/* Adaptive Mode Toggle */}
          <div className="flex items-center justify-between p-4 bg-slate-800/50 rounded-lg">
            <div>
              <div className="text-sm font-medium text-white flex items-center gap-2">
                <Brain className="h-4 w-4" />
                Adaptive Mode
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Automatically optimize for speed when responses are slow
              </p>
            </div>
            <button
              onClick={() => setLocalConfig({ ...localConfig, adaptive_enabled: !localConfig.adaptive_enabled })}
              className={`relative inline-flex h-8 w-14 items-center rounded-full transition-colors ${
                localConfig.adaptive_enabled ? 'bg-indigo-500' : 'bg-slate-600'
              }`}
              data-testid="adaptive-toggle"
            >
              <span
                className={`inline-block h-6 w-6 transform rounded-full bg-white transition-transform ${
                  localConfig.adaptive_enabled ? 'translate-x-7' : 'translate-x-1'
                }`}
              />
            </button>
          </div>

          {/* Max Response Time */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-white">Max Response Time</label>
              <Badge variant="outline" className="text-white border-white">
                {localConfig.max_response_time}s
              </Badge>
            </div>
            <input
              type="range"
              min="5"
              max="30"
              step="1"
              value={localConfig.max_response_time}
              onChange={(e) => setLocalConfig({ ...localConfig, max_response_time: parseFloat(e.target.value) })}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
              data-testid="max-time-slider"
            />
            <div className="flex justify-between text-xs text-slate-400">
              <span>5s (Fast)</span>
              <span>30s (Patient)</span>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3 pt-4 border-t border-slate-700">
            <Button
              onClick={saveConfig}
              disabled={!hasChanges || saving}
              className="flex-1 bg-gradient-to-r from-indigo-600 to-cyan-600 hover:from-indigo-700 hover:to-cyan-700 text-white"
              data-testid="save-config-btn"
            >
              {saving ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <CheckCircle2 className="mr-2 h-4 w-4" />
                  Save Changes
                </>
              )}
            </Button>
            <Button
              onClick={resetToDefaults}
              variant="outline"
              className="border-slate-600 text-slate-300 hover:bg-slate-800"
              data-testid="reset-config-btn"
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              Reset
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Performance Metrics Card */}
      {metrics && (
        <Card className="bg-slate-800 border-slate-700" data-testid="performance-metrics-card">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Performance Metrics
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-slate-900 p-4 rounded-lg">
                <div className="text-xs text-slate-400 mb-1">Total Requests</div>
                <div className="text-2xl font-bold text-white">
                  {metrics.summary?.total_requests || 0}
                </div>
              </div>
              <div className="bg-slate-900 p-4 rounded-lg">
                <div className="text-xs text-slate-400 mb-1">Avg Response Time</div>
                <div className="text-2xl font-bold text-cyan-400">
                  {metrics.summary?.avg_response_time || 0}s
                </div>
              </div>
              <div className="bg-slate-900 p-4 rounded-lg">
                <div className="text-xs text-slate-400 mb-1">Success Rate</div>
                <div className="text-2xl font-bold text-green-400">
                  {metrics.summary?.success_rate || 0}%
                </div>
              </div>
              <div className="bg-slate-900 p-4 rounded-lg">
                <div className="text-xs text-slate-400 mb-1">Fallback Count</div>
                <div className="text-2xl font-bold text-orange-400">
                  {metrics.summary?.fallback_count || 0}
                </div>
              </div>
            </div>

            {/* Mode Distribution */}
            {metrics.summary?.mode_distribution && Object.keys(metrics.summary.mode_distribution).length > 0 && (
              <div className="mt-4">
                <div className="text-sm font-medium text-slate-300 mb-2">Mode Distribution</div>
                <div className="flex gap-2">
                  {Object.entries(metrics.summary.mode_distribution).map(([mode, count]) => {
                    const modeInfo = getModeInfo(mode);
                    return (
                      <Badge key={mode} className={`${modeInfo.color} text-white`}>
                        {modeInfo.label}: {count}
                      </Badge>
                    );
                  })}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Optimization Status Card */}
      {optimizationStatus && (
        <Card className="bg-slate-800 border-slate-700" data-testid="optimization-status-card">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Brain className="h-5 w-5" />
              Optimization Status
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Status Badge */}
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-300">Adaptive Tuning</span>
              <Badge className={optimizationStatus.adaptive_active ? 'bg-green-600' : 'bg-gray-600'}>
                {optimizationStatus.adaptive_active ? 'Active' : 'Inactive'}
              </Badge>
            </div>

            {/* Current Stats */}
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-slate-900 p-3 rounded">
                <div className="text-xs text-slate-400">Current Mode</div>
                <div className="text-sm font-semibold text-white capitalize">
                  {optimizationStatus.current_config?.response_mode || 'N/A'}
                </div>
              </div>
              <div className="bg-slate-900 p-3 rounded">
                <div className="text-xs text-slate-400">Model Used</div>
                <div className="text-sm font-semibold text-white">GPT-4o-mini</div>
              </div>
            </div>

            {/* Recommendations */}
            {optimizationStatus.recommendations && optimizationStatus.recommendations.length > 0 && (
              <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
                <div className="flex items-start gap-2 mb-2">
                  <AlertCircle className="h-4 w-4 text-blue-400 mt-0.5 flex-shrink-0" />
                  <div className="text-sm font-medium text-blue-300">Recommendations</div>
                </div>
                <ul className="space-y-1 ml-6">
                  {optimizationStatus.recommendations.map((rec, idx) => (
                    <li key={idx} className="text-xs text-slate-300 list-disc">
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <Button
              onClick={loadLLMSettings}
              variant="outline"
              className="w-full border-slate-600 text-slate-300 hover:bg-slate-700"
              data-testid="refresh-metrics-btn"
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh Metrics
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default LLMSettingsPanel;
