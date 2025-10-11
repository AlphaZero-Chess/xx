import React, { useState, useEffect, useRef } from 'react';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { Loader2, Clock, CheckCircle2 } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const TrainingProgress = ({ active, onComplete }) => {
  const [progress, setProgress] = useState({
    step: 0,
    total_steps: 0,
    percent: 0,
    eta_seconds: 0,
    message: 'Initializing...'
  });
  const eventSourceRef = useRef(null);

  useEffect(() => {
    if (active) {
      // Connect to SSE endpoint
      const eventSource = new EventSource(`${BACKEND_URL}/api/training/progress/stream`);
      eventSourceRef.current = eventSource;

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.done) {
            // Training complete
            eventSource.close();
            if (onComplete) {
              onComplete();
            }
          } else {
            // Update progress
            setProgress({
              step: data.step || 0,
              total_steps: data.total_steps || 0,
              percent: data.percent || 0,
              eta_seconds: data.eta_seconds || 0,
              message: data.message || 'Processing...'
            });
          }
        } catch (error) {
          console.error('Error parsing progress data:', error);
        }
      };

      eventSource.onerror = (error) => {
        console.error('SSE Error:', error);
        eventSource.close();
      };

      return () => {
        if (eventSource.readyState !== EventSource.CLOSED) {
          eventSource.close();
        }
      };
    } else {
      // Not active, close any existing connection
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    }
  }, [active, onComplete]);

  const formatTime = (seconds) => {
    if (seconds <= 0) return 'Calculating...';
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  if (!active) {
    return null;
  }

  return (
    <div className="p-4 bg-slate-700 rounded-lg space-y-3" data-testid="training-progress">
      {/* Progress Header */}
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          {progress.percent < 100 ? (
            <Loader2 className="h-4 w-4 animate-spin text-purple-400" />
          ) : (
            <CheckCircle2 className="h-4 w-4 text-green-400" />
          )}
          <span className="text-sm font-medium text-slate-300">Training Progress</span>
        </div>
        <Badge variant="outline" className="bg-purple-500/20 text-purple-300 border-purple-500/30">
          {progress.percent}%
        </Badge>
      </div>

      {/* Progress Bar */}
      <div className="space-y-2">
        <Progress 
          value={progress.percent} 
          className="h-3 bg-slate-600"
          data-testid="progress-bar"
        />
        
        {/* Step Counter and ETA */}
        <div className="flex justify-between items-center text-xs">
          <span className="text-slate-400">
            Step {progress.step} of {progress.total_steps}
          </span>
          
          {progress.eta_seconds > 0 && progress.percent < 100 && (
            <div className="flex items-center gap-1 text-slate-400">
              <Clock className="h-3 w-3" />
              <span>ETA: {formatTime(progress.eta_seconds)}</span>
            </div>
          )}
        </div>
      </div>

      {/* Status Message */}
      <div className="text-xs text-slate-400 italic" data-testid="progress-message">
        {progress.message}
      </div>
    </div>
  );
};

export default TrainingProgress;
