import React, { useState } from 'react';
import axios from 'axios';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  Activity,
  Target,
  AlertTriangle,
  CheckCircle,
  Clock,
  BarChart3,
  Zap
} from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ForecastPanel = () => {
  const [loading, setLoading] = useState(false);
  const [forecastData, setForecastData] = useState(null);
  const [selectedTimeframe, setSelectedTimeframe] = useState('7');

  const generateForecast = async () => {
    try {
      setLoading(true);
      toast.info('Generating predictive forecast...');
      
      const response = await axios.post(`${API}/llm/forecast`, {
        timeframes: [7, 30, 90],
        include_narrative: true
      });
      
      if (response.data.success) {
        setForecastData(response.data);
        toast.success('Forecast generated successfully!');
      }
    } catch (error) {
      console.error('Error generating forecast:', error);
      toast.error('Failed to generate forecast');
    } finally {
      setLoading(false);
    }
  };

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'improving':
        return <TrendingUp className="text-green-500" size={20} />;
      case 'declining':
        return <TrendingDown className="text-red-500" size={20} />;
      default:
        return <Minus className="text-yellow-500" size={20} />;
    }
  };

  const getTrendColor = (trend) => {
    switch (trend) {
      case 'improving':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'declining':
        return 'text-red-600 bg-red-50 border-red-200';
      default:
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    }
  };

  const renderMetricCard = (metricName, metricData, icon) => {
    if (!metricData) {
      return (
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="p-4">
            <div className="text-slate-400 text-sm">No data available</div>
          </CardContent>
        </Card>
      );
    }

    const changeSign = metricData.change_percent >= 0 ? '+' : '';
    const isPositive = metricName === 'latency' 
      ? metricData.change_percent < 0  // For latency, decrease is good
      : metricData.change_percent > 0; // For others, increase is good

    return (
      <Card className="bg-slate-800/50 border-slate-700 hover:border-purple-500 transition-all">
        <CardContent className="p-4">
          <div className="flex items-start justify-between mb-3">
            <div className="flex items-center gap-2">
              {icon}
              <h3 className="text-sm font-medium text-slate-300 capitalize">
                {metricName.replace('_', ' ')}
              </h3>
            </div>
            {getTrendIcon(metricData.trend_direction)}
          </div>

          <div className="space-y-2">
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold text-white">
                {metricData.predicted_value}
                {metricName === 'latency' ? 's' : '%'}
              </span>
              <span className="text-sm text-slate-400">
                from {metricData.current_value}{metricName === 'latency' ? 's' : '%'}
              </span>
            </div>

            <div className="flex items-center justify-between">
              <span className={`text-sm font-medium px-2 py-1 rounded border ${getTrendColor(metricData.trend_direction)}`}>
                {changeSign}{metricData.change_percent.toFixed(1)}% change
              </span>
              
              <div className="flex items-center gap-1">
                <div className="w-16 bg-slate-700 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full ${
                      metricData.confidence > 0.7 ? 'bg-green-500' : 
                      metricData.confidence > 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${metricData.confidence * 100}%` }}
                  />
                </div>
                <span className="text-xs text-slate-400">
                  {(metricData.confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            <div className="text-xs text-slate-500 capitalize">
              Trend: {metricData.trend_direction}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  const renderConfidenceGauge = (confidence) => {
    const percentage = (confidence * 100).toFixed(1);
    const color = confidence > 0.7 ? 'text-green-400' : confidence > 0.5 ? 'text-yellow-400' : 'text-red-400';
    const bgColor = confidence > 0.7 ? 'bg-green-500' : confidence > 0.5 ? 'bg-yellow-500' : 'bg-red-500';

    return (
      <Card className="bg-gradient-to-br from-slate-800 to-slate-900 border-slate-700">
        <CardContent className="p-6">
          <div className="text-center space-y-4">
            <h3 className="text-sm font-medium text-slate-400">Overall Forecast Confidence</h3>
            
            <div className="relative inline-flex items-center justify-center">
              <svg className="w-32 h-32 transform -rotate-90">
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  stroke="currentColor"
                  strokeWidth="8"
                  fill="none"
                  className="text-slate-700"
                />
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  stroke="currentColor"
                  strokeWidth="8"
                  fill="none"
                  strokeDasharray={`${2 * Math.PI * 56}`}
                  strokeDashoffset={`${2 * Math.PI * 56 * (1 - confidence)}`}
                  className={bgColor}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className={`text-3xl font-bold ${color}`}>
                  {percentage}%
                </span>
              </div>
            </div>

            <p className="text-xs text-slate-400">
              Based on {forecastData?.data_sufficiency?.training_data_points || 0} training points,{' '}
              {forecastData?.data_sufficiency?.evaluation_data_points || 0} evaluations
            </p>
          </div>
        </CardContent>
      </Card>
    );
  };

  const currentForecasts = forecastData?.forecasts?.[`${selectedTimeframe}_days`];

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="bg-gradient-to-r from-purple-900/50 to-blue-900/50 border-purple-700/50">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-500/20 rounded-lg">
                <Activity className="text-purple-400" size={24} />
              </div>
              <div>
                <CardTitle className="text-white text-xl">
                  Predictive Trend Analysis & Forecasting
                </CardTitle>
                <p className="text-slate-300 text-sm mt-1">
                  AI-powered predictions using linear regression analysis
                </p>
              </div>
            </div>
            
            <Button
              onClick={generateForecast}
              disabled={loading}
              className="bg-purple-600 hover:bg-purple-700 text-white"
              data-testid="generate-forecast-btn"
            >
              {loading ? (
                <>
                  <Clock className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Zap className="mr-2 h-4 w-4" />
                  Generate Forecast
                </>
              )}
            </Button>
          </div>
        </CardHeader>
      </Card>

      {!forecastData && (
        <Card className="bg-slate-800/30 border-slate-700">
          <CardContent className="p-12">
            <div className="text-center space-y-4">
              <BarChart3 className="mx-auto text-slate-600" size={64} />
              <h3 className="text-xl font-semibold text-slate-400">
                No Forecast Data Available
              </h3>
              <p className="text-slate-500 max-w-md mx-auto">
                Click "Generate Forecast" to analyze historical trends and predict future model performance
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {forecastData && (
        <>
          {/* Timeframe Selector */}
          <Tabs value={selectedTimeframe} onValueChange={setSelectedTimeframe} className="w-full">
            <TabsList className="grid w-full grid-cols-3 bg-slate-800/50">
              <TabsTrigger value="7" data-testid="forecast-7-days">
                7 Days
              </TabsTrigger>
              <TabsTrigger value="30" data-testid="forecast-30-days">
                30 Days
              </TabsTrigger>
              <TabsTrigger value="90" data-testid="forecast-90-days">
                90 Days
              </TabsTrigger>
            </TabsList>
          </Tabs>

          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Metric Cards */}
            <div className="lg:col-span-3 space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {renderMetricCard(
                  'accuracy',
                  currentForecasts?.accuracy,
                  <Target className="text-blue-400" size={20} />
                )}
                {renderMetricCard(
                  'win_rate',
                  currentForecasts?.win_rate,
                  <TrendingUp className="text-green-400" size={20} />
                )}
                {renderMetricCard(
                  'latency',
                  currentForecasts?.latency,
                  <Clock className="text-orange-400" size={20} />
                )}
              </div>

              {/* Strategic Recommendations */}
              {forecastData.strategic_recommendations && forecastData.strategic_recommendations.length > 0 && (
                <Card className="bg-gradient-to-br from-orange-900/20 to-red-900/20 border-orange-700/50">
                  <CardHeader>
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="text-orange-400" size={20} />
                      <CardTitle className="text-white text-lg">
                        Strategic Recommendations
                      </CardTitle>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2">
                      {forecastData.strategic_recommendations.map((rec, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-slate-300">
                          <CheckCircle className="text-orange-400 flex-shrink-0 mt-0.5" size={16} />
                          <span className="text-sm">{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}

              {/* LLM Forecast Narrative */}
              {forecastData.forecast_narrative && (
                <Card className="bg-slate-800/30 border-slate-700">
                  <CardHeader>
                    <CardTitle className="text-white text-lg">
                      AI Forecast Analysis
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="prose prose-invert prose-sm max-w-none">
                      <div className="text-slate-300 whitespace-pre-wrap leading-relaxed">
                        {forecastData.forecast_narrative}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>

            {/* Confidence Gauge */}
            <div className="lg:col-span-1">
              {renderConfidenceGauge(forecastData.overall_confidence)}

              {/* Data Sufficiency Info */}
              <Card className="bg-slate-800/50 border-slate-700 mt-4">
                <CardHeader>
                  <CardTitle className="text-white text-sm">Data Sources</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Training Metrics</span>
                      <span className="text-white font-medium">
                        {forecastData.data_sufficiency?.training_data_points || 0}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Evaluations</span>
                      <span className="text-white font-medium">
                        {forecastData.data_sufficiency?.evaluation_data_points || 0}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Performance Data</span>
                      <span className="text-white font-medium">
                        {forecastData.data_sufficiency?.performance_data_points || 0}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Knowledge Base</span>
                      <span className="text-white font-medium">
                        {forecastData.data_sufficiency?.distilled_knowledge_entries || 0}
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default ForecastPanel;
