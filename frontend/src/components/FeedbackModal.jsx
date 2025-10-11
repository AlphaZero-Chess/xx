import React, { useState } from 'react';
import { X, Send, Star, MessageSquare } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { toast } from 'sonner';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const FeedbackModal = ({ onClose }) => {
  const [feedbackType, setFeedbackType] = useState('general');
  const [category, setCategory] = useState('ui');
  const [rating, setRating] = useState(5);
  const [message, setMessage] = useState('');
  const [email, setEmail] = useState('');
  const [allowContact, setAllowContact] = useState(false);
  const [telemetryOptIn, setTelemetryOptIn] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!message.trim()) {
      toast.error('Please provide feedback message');
      return;
    }

    setSubmitting(true);
    
    try {
      const platform = window.electronAPI ? 'desktop' : 'web';
      
      const response = await axios.post(`${API}/feedback/submit`, {
        feedback_type: feedbackType,
        category,
        rating,
        message: message.trim(),
        email: email.trim() || null,
        allow_contact: allowContact,
        telemetry_opt_in: telemetryOptIn,
        app_version: '1.0.0',
        platform
      });

      if (response.data.success) {
        toast.success('Thank you for your feedback!');
        onClose();
      } else {
        toast.error('Failed to submit feedback');
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
      toast.error('Failed to submit feedback');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <Card 
        className="w-full max-w-2xl bg-slate-800 border-slate-700 max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <CardHeader className="flex flex-row items-center justify-between pb-4 border-b border-slate-700">
          <CardTitle className="text-2xl font-bold text-white flex items-center gap-2">
            <MessageSquare className="text-blue-400" />
            Send Feedback
          </CardTitle>
          <Button
            onClick={onClose}
            variant="ghost"
            size="sm"
            className="text-slate-400 hover:text-white"
          >
            <X size={20} />
          </Button>
        </CardHeader>

        <CardContent className="pt-6">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Feedback Type */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Feedback Type
              </label>
              <div className="grid grid-cols-2 gap-2">
                {['bug', 'feature', 'performance', 'general'].map((type) => (
                  <button
                    key={type}
                    type="button"
                    onClick={() => setFeedbackType(type)}
                    className={`px-4 py-2 rounded-lg border transition-colors capitalize ${
                      feedbackType === type
                        ? 'bg-blue-600 border-blue-500 text-white'
                        : 'bg-slate-700 border-slate-600 text-slate-300 hover:bg-slate-600'
                    }`}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>

            {/* Category */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Category
              </label>
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="ui">User Interface</option>
                <option value="ai">AI Behavior</option>
                <option value="performance">Performance</option>
                <option value="other">Other</option>
              </select>
            </div>

            {/* Rating */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Overall Rating
              </label>
              <div className="flex gap-2">
                {[1, 2, 3, 4, 5].map((star) => (
                  <button
                    key={star}
                    type="button"
                    onClick={() => setRating(star)}
                    className="transition-transform hover:scale-110"
                  >
                    <Star
                      size={32}
                      className={star <= rating ? 'fill-yellow-400 text-yellow-400' : 'text-slate-600'}
                    />
                  </button>
                ))}
                <span className="ml-2 text-slate-400 self-center">{rating}/5</span>
              </div>
            </div>

            {/* Message */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Your Feedback *
              </label>
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Tell us what you think..."
                rows={6}
                required
                className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
              />
              <p className="text-xs text-slate-500 mt-1">{message.length} characters</p>
            </div>

            {/* Contact Information */}
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <input
                  type="checkbox"
                  id="allowContact"
                  checked={allowContact}
                  onChange={(e) => setAllowContact(e.target.checked)}
                  className="mt-1"
                />
                <label htmlFor="allowContact" className="text-sm text-slate-300 cursor-pointer">
                  Allow the team to contact me about this feedback
                </label>
              </div>

              {allowContact && (
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="your.email@example.com"
                  className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              )}
            </div>

            {/* Telemetry Opt-in */}
            <div className="border-t border-slate-700 pt-4">
              <div className="flex items-start gap-3">
                <input
                  type="checkbox"
                  id="telemetryOptIn"
                  checked={telemetryOptIn}
                  onChange={(e) => setTelemetryOptIn(e.target.checked)}
                  className="mt-1"
                />
                <label htmlFor="telemetryOptIn" className="text-sm text-slate-300 cursor-pointer">
                  <span className="font-medium">Share anonymous usage data</span>
                  <p className="text-xs text-slate-500 mt-1">
                    Help us improve by sharing anonymous performance and usage metrics. 
                    No personal information is collected.
                  </p>
                </label>
              </div>
            </div>

            {/* Submit Button */}
            <div className="flex justify-end gap-3 pt-4 border-t border-slate-700">
              <Button
                type="button"
                onClick={onClose}
                variant="outline"
                className="border-slate-600 text-slate-300 hover:bg-slate-700"
              >
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={submitting || !message.trim()}
                className="bg-blue-600 hover:bg-blue-700 text-white"
              >
                {submitting ? (
                  'Submitting...'
                ) : (
                  <>
                    <Send size={16} className="mr-2" />
                    Send Feedback
                  </>
                )}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
};

export default FeedbackModal;
