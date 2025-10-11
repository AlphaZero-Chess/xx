import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { 
  History,
  TrendingUp,
  TrendingDown,
  Settings,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Filter,
  Download
} from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const StrategyAdjustmentLog = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState('all'); // 'all', 'high_confidence', 'recent'

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API}/strategy/auto-tune/history?limit=50`);
      
      if (response.data.success) {
        setHistory(response.data.history);
      }
    } catch (error) {
      console.error('Error loading tuning history:', error);
      toast.error('Failed to load tuning history');
    } finally {
      setLoading(false);
    }
  };

  const exportHistory = () => {
    const data = JSON.stringify(history, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `strategy-tuning-history-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success('History exported successfully');
  };

  const getFilteredHistory = () => {
    switch (filter) {
      case 'high_confidence':
        return history.filter(h => h.confidence_score >= 0.8);
      case 'recent':
        const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
        return history.filter(h => new Date(h.timestamp) > oneDayAgo);
      default:
        return history;
    }
  };

  const getTriggerIcon = (trigger) => {
    if (trigger.includes('decline') || trigger.includes('spike')) {
      return <TrendingDown className="text-red-400" size={16} />;
    }
    return <Settings className="text-blue-400" size={16} />;
  };

  const getConfidenceBadge = (confidence) => {
    if (confidence >= 0.8) {
      return <Badge className="bg-green-600 text-white">High ({(confidence * 100).toFixed(0)}%)</Badge>;
    } else if (confidence >= 0.6) {
      return <Badge className="bg-yellow-600 text-white">Medium ({(confidence * 100).toFixed(0)}%)</Badge>;
    } else {
      return <Badge className="bg-orange-600 text-white">Low ({(confidence * 100).toFixed(0)}%)</Badge>;
    }
  };

  const filteredHistory = getFilteredHistory();

  return (
    <Card className="bg-slate-800/30 border-slate-700">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-white text-xl flex items-center gap-2">
            <History size={24} className="text-indigo-400" />
            Strategy Adjustment Log
            <Badge variant="outline" className="text-slate-300">
              {filteredHistory.length} entries
            </Badge>
          </CardTitle>

          <div className="flex items-center gap-2">
            <div className="flex gap-1 bg-slate-700/50 rounded-lg p-1">
              <Button
                variant={filter === 'all' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setFilter('all')}
                className={filter === 'all' ? 'bg-indigo-600' : ''}
              >
                All
              </Button>
              <Button
                variant={filter === 'high_confidence' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setFilter('high_confidence')}
                className={filter === 'high_confidence' ? 'bg-indigo-600' : ''}
              >
                High Confidence
              </Button>
              <Button
                variant={filter === 'recent' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setFilter('recent')}
                className={filter === 'recent' ? 'bg-indigo-600' : ''}
              >
                Last 24h
              </Button>
            </div>

            <Button
              onClick={exportHistory}
              variant="outline"
              size="sm"
              className="border-slate-600 text-slate-300"
              disabled={history.length === 0}
            >
              <Download className="mr-2 h-4 w-4" />
              Export
            </Button>

            <Button
              onClick={loadHistory}
              variant="outline"
              size="sm"
              className="border-slate-600 text-slate-300"
              disabled={loading}
            >
              <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        {loading && history.length === 0 ? (
          <div className="text-center py-12">
            <RefreshCw className="mx-auto text-slate-600 animate-spin mb-4" size={32} />
            <p className="text-slate-400">Loading tuning history...</p>
          </div>
        ) : filteredHistory.length === 0 ? (
          <div className="text-center py-12">
            <History className="mx-auto text-slate-600 mb-4" size={48} />
            <h3 className="text-lg font-semibold text-slate-400 mb-2">
              No Tuning Events Yet
            </h3>
            <p className="text-slate-500 text-sm">
              Enable auto-tuning to start monitoring and optimizing your system
            </p>
          </div>
        ) : (
          <div className="space-y-3 max-h-[600px] overflow-y-auto">
            {filteredHistory.map((entry, idx) => (
              <div 
                key={entry.tuning_id || idx}
                className="bg-slate-900/50 rounded-lg p-4 hover:bg-slate-900/70 transition-colors border border-slate-700/50"
              >
                {/* Header */}
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-2">
                    {getTriggerIcon(entry.trigger_reason)}
                    <div>
                      <h3 className="text-white font-medium capitalize">
                        {entry.trigger_reason.replace(/_/g, ' ')}
                      </h3>
                      <p className="text-xs text-slate-500">
                        {new Date(entry.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex flex-col items-end gap-1">
                    {getConfidenceBadge(entry.confidence_score)}
                    <Badge variant="outline" className="text-xs text-slate-400">
                      {entry.deviation_magnitude.toFixed(1)}% deviation
                    </Badge>
                  </div>
                </div>

                {/* Reasoning */}
                <div className="mb-3 text-sm text-slate-300 bg-slate-800/50 rounded p-3">
                  {entry.reasoning}
                </div>

                {/* Parameters Adjusted */}
                {entry.parameters_adjusted && Object.keys(entry.parameters_adjusted).length > 0 && (
                  <div className="space-y-2">
                    <div className="text-xs font-semibold text-slate-400 flex items-center gap-1">
                      <Settings size={12} />
                      Parameters Adjusted:
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                      {Object.entries(entry.parameters_adjusted).map(([param, values]) => (
                        <div 
                          key={param}
                          className="bg-indigo-900/20 rounded px-3 py-2 border border-indigo-700/30"
                        >
                          <div className="text-xs text-indigo-300 font-medium mb-1 capitalize">
                            {param.replace(/_/g, ' ')}
                          </div>
                          <div className="flex items-center gap-2 text-sm font-mono">
                            <span className="text-slate-400">{values.old.toFixed(4)}</span>
                            <span className="text-slate-500">→</span>
                            <span className="text-indigo-400 font-semibold">{values.new.toFixed(4)}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Expected Impact */}
                <div className="mt-3 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="text-green-400" size={14} />
                    <span className="text-xs text-green-400">
                      Expected: {entry.expected_impact.replace(/_/g, ' ')}
                    </span>
                  </div>
                  
                  {/* Explain This Decision Button */}
                  <Button
                    variant="outline"
                    size="sm"
                    className="text-xs text-purple-400 border-purple-500/30 hover:bg-purple-500/20"
                    onClick={async () => {
                      try {
                        toast.info('Generating reasoning chain...');
                        const response = await axios.post(`${API}/llm/explain-decision/${entry.tuning_id}`);
                        if (response.data.success) {
                          toast.success('Decision explanation generated! Check Strategic Insight Fusion panel below.');
                        }
                      } catch (error) {
                        console.error('Error explaining decision:', error);
                        toast.error('Failed to generate explanation');
                      }
                    }}
                  >
                    💡 Explain This Decision
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Summary Stats */}
        {filteredHistory.length > 0 && (
          <div className="mt-6 pt-4 border-t border-slate-700">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold text-white">{filteredHistory.length}</div>
                <div className="text-xs text-slate-400">Total Adjustments</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-white">
                  {filteredHistory.filter(h => h.confidence_score >= 0.8).length}
                </div>
                <div className="text-xs text-slate-400">High Confidence</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-white">
                  {(filteredHistory.reduce((sum, h) => sum + h.deviation_magnitude, 0) / filteredHistory.length).toFixed(1)}%
                </div>
                <div className="text-xs text-slate-400">Avg Deviation</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-white">
                  {new Set(filteredHistory.flatMap(h => Object.keys(h.parameters_adjusted || {}))).size}
                </div>
                <div className="text-xs text-slate-400">Unique Parameters</div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default StrategyAdjustmentLog;
