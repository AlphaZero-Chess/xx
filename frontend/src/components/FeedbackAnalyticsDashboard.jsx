import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { 
  MessageSquare, TrendingUp, AlertTriangle, Star, 
  Users, Activity, RefreshCw, BarChart3 
} from 'lucide-react';
import { toast } from 'sonner';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const FeedbackAnalyticsDashboard = () => {
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAnalytics();
  }, []);

  const fetchAnalytics = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/feedback/analytics?limit=100`);
      if (response.data.success) {
        setAnalytics(response.data);
      }
    } catch (error) {
      console.error('Error fetching analytics:', error);
      toast.error('Failed to load feedback analytics');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-12">
        <div className="text-slate-400">Loading analytics...</div>
      </div>
    );
  }

  if (!analytics || analytics.total_feedback === 0) {
    return (
      <Card className="bg-slate-800 border-slate-700">
        <CardContent className="p-12 text-center">
          <MessageSquare size={48} className="mx-auto text-slate-600 mb-4" />
          <h3 className="text-xl font-semibold text-white mb-2">No Feedback Yet</h3>
          <p className="text-slate-400">
            User feedback will appear here once submitted through the app.
          </p>
        </CardContent>
      </Card>
    );
  }

  const { overview, distributions, telemetry, recent_feedback } = analytics;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-white mb-1">Feedback Analytics</h2>
          <p className="text-slate-400">User feedback and sentiment analysis for v1.0</p>
        </div>
        <Button
          onClick={fetchAnalytics}
          className="bg-blue-600 hover:bg-blue-700"
        >
          <RefreshCw size={16} className="mr-2" />
          Refresh
        </Button>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-slate-800 border-slate-700">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-2">
              <MessageSquare className="text-blue-400" size={24} />
              <span className="text-2xl font-bold text-white">{overview.total_feedback}</span>
            </div>
            <p className="text-sm text-slate-400">Total Feedback</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800 border-slate-700">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-2">
              <Star className="text-yellow-400" size={24} />
              <span className="text-2xl font-bold text-white">{overview.average_rating.toFixed(1)}/5</span>
            </div>
            <p className="text-sm text-slate-400">Average Rating</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800 border-slate-700">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-2">
              <TrendingUp className="text-green-400" size={24} />
              <span className="text-2xl font-bold text-white">{overview.sentiment.positive}</span>
            </div>
            <p className="text-sm text-slate-400">Positive Feedback</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800 border-slate-700">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-2">
              <Activity className="text-purple-400" size={24} />
              <span className="text-2xl font-bold text-white">{telemetry.total_events}</span>
            </div>
            <p className="text-sm text-slate-400">Telemetry Events</p>
          </CardContent>
        </Card>
      </div>

      {/* Sentiment Distribution */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <BarChart3 className="text-blue-400" />
            Sentiment Distribution
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Positive */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-green-400">Positive (4-5 stars)</span>
                <span className="text-white">{overview.sentiment.positive}</span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-3">
                <div
                  className="bg-green-500 h-3 rounded-full transition-all"
                  style={{
                    width: `${(overview.sentiment.positive / overview.total_feedback) * 100}%`
                  }}
                />
              </div>
            </div>

            {/* Neutral */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-yellow-400">Neutral (3 stars)</span>
                <span className="text-white">{overview.sentiment.neutral}</span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-3">
                <div
                  className="bg-yellow-500 h-3 rounded-full transition-all"
                  style={{
                    width: `${(overview.sentiment.neutral / overview.total_feedback) * 100}%`
                  }}
                />
              </div>
            </div>

            {/* Negative */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-red-400">Negative (1-2 stars)</span>
                <span className="text-white">{overview.sentiment.negative}</span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-3">
                <div
                  className="bg-red-500 h-3 rounded-full transition-all"
                  style={{
                    width: `${(overview.sentiment.negative / overview.total_feedback) * 100}%`
                  }}
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Feedback by Type & Category */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white">Feedback by Type</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(distributions.by_type).map(([type, count]) => (
                <div key={type} className="flex justify-between items-center">
                  <span className="text-slate-300 capitalize">{type}</span>
                  <span className="text-white font-semibold">{count}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white">Feedback by Category</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(distributions.by_category).map(([category, count]) => (
                <div key={category} className="flex justify-between items-center">
                  <span className="text-slate-300 capitalize">{category}</span>
                  <span className="text-white font-semibold">{count}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Feedback */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white">Recent Feedback</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {recent_feedback.map((feedback) => (
              <div
                key={feedback.feedback_id}
                className="p-4 bg-slate-700/50 rounded-lg border border-slate-600"
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      feedback.priority === 'high' ? 'bg-red-600 text-white' :
                      feedback.priority === 'medium' ? 'bg-yellow-600 text-white' :
                      'bg-blue-600 text-white'
                    }`}>
                      {feedback.type}
                    </span>
                    <span className="text-slate-400 text-xs">{feedback.category}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    {[...Array(5)].map((_, i) => (
                      <Star
                        key={i}
                        size={14}
                        className={i < feedback.rating ? 'fill-yellow-400 text-yellow-400' : 'text-slate-600'}
                      />
                    ))}
                  </div>
                </div>
                <p className="text-slate-300 text-sm mb-2">{feedback.message}</p>
                <p className="text-xs text-slate-500">
                  {new Date(feedback.timestamp).toLocaleString()}
                </p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Top Telemetry Events */}
      {telemetry.top_events && telemetry.top_events.length > 0 && (
        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white">Top Usage Events</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {telemetry.top_events.map((event) => (
                <div key={event.event_type} className="flex justify-between items-center">
                  <span className="text-slate-300">{event.event_type}</span>
                  <span className="text-white font-semibold">{event.count}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default FeedbackAnalyticsDashboard;
