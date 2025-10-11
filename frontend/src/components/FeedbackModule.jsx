import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Star, Send, Sparkles, CheckCircle2 } from 'lucide-react';
import { toast } from 'sonner';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

/**
 * FeedbackModule - Reusable component for collecting user feedback on LLM outputs
 * 
 * @param {string} sessionId - Unique identifier for the session
 * @param {string} operationType - Type of operation: "coaching", "analytics", or "general"
 * @param {number} llmConfidence - Optional LLM confidence score (0-1)
 * @param {number} responseTime - Optional response time in seconds
 * @param {boolean} compact - Whether to show compact version
 * @param {function} onFeedbackSubmitted - Callback after successful submission
 */
const FeedbackModule = ({ 
  sessionId, 
  operationType = "general", 
  llmConfidence = null,
  responseTime = null,
  compact = false,
  onFeedbackSubmitted = null
}) => {
  const [accuracyScore, setAccuracyScore] = useState(0);
  const [usefulnessScore, setUsefulnessScore] = useState(0);
  const [clarityScore, setClarityScore] = useState(0);
  const [comment, setComment] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [showForm, setShowForm] = useState(!compact);

  const resetForm = () => {
    setAccuracyScore(0);
    setUsefulnessScore(0);
    setClarityScore(0);
    setComment('');
    setSubmitted(false);
  };

  const handleSubmit = async () => {
    if (accuracyScore === 0 || usefulnessScore === 0 || clarityScore === 0) {
      toast.error('Please rate all aspects before submitting');
      return;
    }

    setSubmitting(true);

    try {
      const response = await axios.post(`${API}/llm/evaluate`, {
        session_id: sessionId,
        operation_type: operationType,
        accuracy_score: accuracyScore,
        usefulness: usefulnessScore,
        clarity: clarityScore,
        response_time: responseTime,
        comment: comment.trim() || null,
        llm_confidence: llmConfidence
      });

      if (response.data.success) {
        setSubmitted(true);
        toast.success('Thank you for your feedback!', {
          description: `Overall score: ${response.data.evaluation.overall_score.toFixed(1)}%`
        });

        // Show auto-optimize message if triggered
        if (response.data.auto_optimize_triggered) {
          toast.info('Auto-optimization triggered!', {
            description: response.data.auto_optimize_message
          });
        }

        // Call callback if provided
        if (onFeedbackSubmitted) {
          onFeedbackSubmitted(response.data);
        }

        // Reset form after 2 seconds
        setTimeout(() => {
          resetForm();
          if (compact) setShowForm(false);
        }, 2000);
      }
    } catch (err) {
      console.error('Error submitting feedback:', err);
      toast.error('Failed to submit feedback. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  const renderStarRating = (score, setScore, label) => {
    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-gray-700">{label}</label>
          <span className="text-xs text-gray-500">{score > 0 ? `${score}/5` : 'Not rated'}</span>
        </div>
        <div className="flex gap-1">
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              onClick={() => setScore(star)}
              className="transition-all hover:scale-110"
              disabled={submitted}
              data-testid={`${label.toLowerCase().replace(' ', '-')}-star-${star}`}
            >
              <Star
                size={24}
                className={`${
                  star <= score
                    ? 'fill-amber-400 text-amber-400'
                    : 'text-gray-300'
                } transition-colors`}
              />
            </button>
          ))}
        </div>
      </div>
    );
  };

  if (compact && !showForm) {
    return (
      <Button
        onClick={() => setShowForm(true)}
        variant="outline"
        size="sm"
        className="gap-2"
        data-testid="feedback-open-button"
      >
        <Sparkles size={16} />
        Rate this response
      </Button>
    );
  }

  return (
    <Card className="border-blue-100 bg-gradient-to-br from-blue-50 to-white" data-testid="feedback-module">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Sparkles size={18} className="text-blue-500" />
          Rate This Response
          {llmConfidence && (
            <Badge variant="outline" className="ml-auto text-xs">
              AI Confidence: {(llmConfidence * 100).toFixed(0)}%
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {!submitted ? (
          <>
            {renderStarRating(accuracyScore, setAccuracyScore, 'Accuracy')}
            {renderStarRating(usefulnessScore, setUsefulnessScore, 'Usefulness')}
            {renderStarRating(clarityScore, setClarityScore, 'Clarity')}
            
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700">
                Additional Comments (Optional)
              </label>
              <Textarea
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                placeholder="What made this response helpful or unhelpful?"
                className="resize-none"
                rows={3}
                disabled={submitted}
                data-testid="feedback-comment-input"
              />
            </div>

            <div className="flex gap-2">
              <Button
                onClick={handleSubmit}
                disabled={submitting || accuracyScore === 0 || usefulnessScore === 0 || clarityScore === 0}
                className="flex-1 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700"
                data-testid="feedback-submit-button"
              >
                {submitting ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2" />
                    Submitting...
                  </>
                ) : (
                  <>
                    <Send size={16} className="mr-2" />
                    Submit Feedback
                  </>
                )}
              </Button>
              {compact && (
                <Button
                  onClick={() => setShowForm(false)}
                  variant="outline"
                  data-testid="feedback-cancel-button"
                >
                  Cancel
                </Button>
              )}
            </div>
          </>
        ) : (
          <div className="text-center py-6" data-testid="feedback-success-message">
            <CheckCircle2 size={48} className="mx-auto text-green-500 mb-3" />
            <p className="text-lg font-semibold text-gray-800">Thank you!</p>
            <p className="text-sm text-gray-600 mt-1">Your feedback helps improve the AI coach</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default FeedbackModule;
