import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { TrendingUp, TrendingDown, Sparkles, CheckCircle, AlertCircle, Settings, RefreshCw } from 'lucide-react';
import axios from 'axios';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

/**
 * AutoEvaluationCard - Shows LLM auto-evaluation metrics and optimization controls
 * Displays aggregate accuracy, clarity trends, and recommended parameter changes
 */
const AutoEvaluationCard = () => {
  const [feedbackSummary, setFeedbackSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [optimizing, setOptimizing] = useState(false);
  const [lastOptimization, setLastOptimization] = useState(null);

  useEffect(() => {
    loadFeedbackSummary();
  }, []);

  const loadFeedbackSummary = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/llm/feedback-summary?limit=50`);
      if (response.data.success) {
        setFeedbackSummary(response.data);
      }
    } catch (err) {
      console.error('Error loading feedback summary:', err);
    } finally {
      setLoading(false);
    }
  };

  const applyAutoOptimization = async () => {
    setOptimizing(true);
    try {
      const response = await axios.post(`${API}/llm/auto-optimize`);
      
      if (response.data.success) {
        setLastOptimization(response.data);
        
        if (response.data.config_changed) {
          toast.success('LLM Configuration Optimized!', {
            description: `Applied ${response.data.recommendations.length} recommendations`
          });
        } else {
          toast.info('No Changes Needed', {
            description: 'Current configuration is already optimal'
          });
        }
        
        // Reload feedback summary
        await loadFeedbackSummary();
      }
    } catch (err) {
      console.error('Error applying optimization:', err);
      toast.error('Failed to apply optimization');
    } finally {
      setOptimizing(false);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBadgeVariant = (score) => {
    if (score >= 80) return 'default';
    if (score >= 60) return 'secondary';
    return 'destructive';
  };

  const getTrendIcon = (score) => {
    if (score >= 80) return <TrendingUp className="h-4 w-4 text-green-600" />;
    if (score >= 60) return <AlertCircle className="h-4 w-4 text-yellow-600" />;
    return <TrendingDown className="h-4 w-4 text-red-600" />;
  };

  if (loading && !feedbackSummary) {
    return (
      <Card className="border-purple-200 bg-gradient-to-br from-purple-50 to-white" data-testid="auto-evaluation-card-loading">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-purple-600" />
            LLM Auto-Evaluation
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-4 border-purple-200 border-t-purple-600" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!feedbackSummary || feedbackSummary.total_feedback === 0) {
    return (
      <Card className="border-purple-200 bg-gradient-to-br from-purple-50 to-white" data-testid="auto-evaluation-card-empty">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-purple-600" />
            LLM Auto-Evaluation
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6">
            <Sparkles className="h-12 w-12 mx-auto text-purple-300 mb-3" />
            <p className="text-sm text-gray-600 mb-2">No feedback data yet</p>
            <p className="text-xs text-gray-500">
              Rate LLM responses in the Coaching panel to enable auto-evaluation
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const scores = feedbackSummary.aggregate_scores || {};
  const optimizationHistory = feedbackSummary.optimization_history || [];
  const latestOptimization = optimizationHistory[0];

  return (
    <Card className="border-purple-200 bg-gradient-to-br from-purple-50 to-white" data-testid="auto-evaluation-card">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-purple-600" />
            LLM Auto-Evaluation
          </CardTitle>
          <Button
            onClick={loadFeedbackSummary}
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0"
            data-testid="refresh-feedback-button"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Aggregate Scores */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-white rounded-lg p-3 border border-purple-100">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-medium text-gray-600">Overall Score</span>
              {getTrendIcon(scores.overall_score)}
            </div>
            <div className={`text-2xl font-bold ${getScoreColor(scores.overall_score)}`}>
              {scores.overall_score?.toFixed(1) || 0}%
            </div>
          </div>
          
          <div className="bg-white rounded-lg p-3 border border-purple-100">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-medium text-gray-600">Accuracy</span>
              {getTrendIcon(scores.accuracy_score)}
            </div>
            <div className={`text-2xl font-bold ${getScoreColor(scores.accuracy_score)}`}>
              {scores.accuracy_score?.toFixed(1) || 0}%
            </div>
          </div>
          
          <div className="bg-white rounded-lg p-3 border border-purple-100">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-medium text-gray-600">Clarity</span>
              {getTrendIcon(scores.clarity_score)}
            </div>
            <div className={`text-2xl font-bold ${getScoreColor(scores.clarity_score)}`}>
              {scores.clarity_score?.toFixed(1) || 0}%
            </div>
          </div>
          
          <div className="bg-white rounded-lg p-3 border border-purple-100">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-medium text-gray-600">Usefulness</span>
              {getTrendIcon(scores.usefulness_score)}
            </div>
            <div className={`text-2xl font-bold ${getScoreColor(scores.usefulness_score)}`}>
              {scores.usefulness_score?.toFixed(1) || 0}%
            </div>
          </div>
        </div>

        {/* Feedback Stats */}
        <div className="bg-white rounded-lg p-3 border border-purple-100">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Total Feedback Collected</span>
            <Badge variant="secondary">{feedbackSummary.total_feedback}</Badge>
          </div>
        </div>

        {/* Latest Optimization */}
        {latestOptimization && (
          <div className="bg-white rounded-lg p-3 border border-purple-100">
            <div className="flex items-center gap-2 mb-2">
              <Settings className="h-4 w-4 text-purple-600" />
              <span className="text-sm font-medium text-gray-700">Last Optimization</span>
            </div>
            <div className="text-xs text-gray-600 mb-2">
              {new Date(latestOptimization.timestamp).toLocaleString()}
            </div>
            {latestOptimization.recommendations && latestOptimization.recommendations.length > 0 && (
              <ul className="space-y-1">
                {latestOptimization.recommendations.map((rec, idx) => (
                  <li key={idx} className="text-xs text-gray-600 flex items-start gap-1">
                    <CheckCircle className="h-3 w-3 text-green-500 mt-0.5 flex-shrink-0" />
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {/* Recommendations from last optimization result */}
        {lastOptimization && lastOptimization.recommendations && (
          <div className="bg-amber-50 rounded-lg p-3 border border-amber-200">
            <div className="flex items-center gap-2 mb-2">
              <AlertCircle className="h-4 w-4 text-amber-600" />
              <span className="text-sm font-medium text-amber-900">Current Recommendations</span>
            </div>
            <ul className="space-y-1">
              {lastOptimization.recommendations.slice(0, 3).map((rec, idx) => (
                <li key={idx} className="text-xs text-amber-800">
                  • {rec}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Auto-Optimize Button */}
        <Button
          onClick={applyAutoOptimization}
          disabled={optimizing || feedbackSummary.total_feedback < 5}
          className="w-full bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800"
          data-testid="apply-auto-optimization-button"
        >
          {optimizing ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2" />
              Optimizing...
            </>
          ) : (
            <>
              <Sparkles className="mr-2 h-4 w-4" />
              Apply Suggested Tuning
            </>
          )}
        </Button>

        {feedbackSummary.total_feedback < 5 && (
          <p className="text-xs text-center text-gray-500">
            Collect at least 5 feedbacks to enable auto-optimization
          </p>
        )}
      </CardContent>
    </Card>
  );
};

export default AutoEvaluationCard;
