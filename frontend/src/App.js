import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Card, CardHeader, CardTitle, CardContent } from './components/ui/card';
import { Toaster } from './components/ui/sonner';
import { toast } from 'sonner';
import { CheckCircle2, XCircle, AlertCircle, Settings as SettingsIcon, MessageSquare } from 'lucide-react';
import { Button } from './components/ui/button';
import ChessBoard from './components/ChessBoard';
import TrainingPanel from './components/TrainingPanel';
import AnalyticsPanel from './components/AnalyticsPanel';
import ModelManagement from './components/ModelManagement';
import GameTab from './components/GameTab';
import CollectiveIntelligencePanel from './components/CollectiveIntelligencePanel';
import GovernancePanel from './components/GovernancePanel';
import EthicalConsensusPanel from './components/EthicalConsensusPanel';
import CognitiveSynthesisPanel from './components/CognitiveSynthesisPanel';
import CollectiveCorePanel from './components/CollectiveCorePanel';
import CreativeSynthesisPanel from './components/CreativeSynthesisPanel';
import ReflectionLoopPanel from './components/ReflectionLoopPanel';
import MemoryFusionPanel from './components/MemoryFusionPanel';
import MemoryArchivePanel from './components/MemoryArchivePanel';
import CohesionCorePanel from './components/CohesionCorePanel';
import EthicalGovernancePanel from './components/EthicalGovernancePanel';
import CognitiveResonancePanel from './components/CognitiveResonancePanel';
import SystemOptimizationPanel from './components/SystemOptimizationPanel';
import DiagnosticsPanel from './components/DiagnosticsPanel';
import SetupWizard from './components/SetupWizard';
import SettingsPanel from './components/SettingsPanel';
import FeedbackModal from './components/FeedbackModal';
import FeedbackAnalyticsDashboard from './components/FeedbackAnalyticsDashboard';
import UpdateNotification from './components/UpdateNotification';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [backendStatus, setBackendStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showSetup, setShowSetup] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);
  const [isFirstLaunch, setIsFirstLaunch] = useState(false);

  useEffect(() => {
    checkFirstLaunch();
    checkBackendConnection();
    
    // Listen for settings menu from Electron
    if (window.electronAPI) {
      window.electronAPI.onOpenSettings(() => {
        setShowSettings(true);
      });
    }
  }, []);

  const checkFirstLaunch = async () => {
    try {
      if (window.electronAPI) {
        const firstLaunch = await window.electronAPI.checkFirstLaunch();
        setIsFirstLaunch(firstLaunch);
        setShowSetup(firstLaunch);
      }
    } catch (error) {
      console.error('Error checking first launch:', error);
    }
  };

  const checkBackendConnection = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API}/`);
      setBackendStatus({
        connected: true,
        message: response.data.message,
        status: response.data.status
      });
      toast.success('Connected to backend!');
    } catch (error) {
      console.error('Error connecting to backend:', error);
      setBackendStatus({
        connected: false,
        error: error.message
      });
      toast.error('Unable to connect to backend');
    } finally {
      setLoading(false);
    }
  };

  // Show setup wizard on first launch
  if (showSetup) {
    return (
      <SetupWizard 
        onComplete={() => {
          setShowSetup(false);
          setIsFirstLaunch(false);
          checkBackendConnection();
        }} 
      />
    );
  }

  // Show settings panel
  if (showSettings) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        <Toaster richColors position="top-right" />
        <SettingsPanel onClose={() => setShowSettings(false)} />
      </div>
    );
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="text-white text-2xl">Loading AlphaZero Chess...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-4" data-testid="app-container">
      <Toaster richColors position="top-right" />
      
      <div className="max-w-7xl mx-auto">
        <header className="text-center mb-8 pt-8">
          <div className="flex items-center justify-between mb-4">
            <div className="flex-1"></div>
            <div className="flex-1 text-center">
              <h1 className="text-5xl font-bold text-white mb-2 tracking-tight" data-testid="app-title">
                AlphaZero Chess
              </h1>
              <p className="text-slate-300 text-lg">
                Self-Learning Chess AI with Deep Reinforcement Learning
              </p>
            </div>
            <div className="flex-1 flex justify-end gap-2">
              <Button
                onClick={() => setShowFeedback(true)}
                variant="ghost"
                className="text-slate-400 hover:text-white"
                title="Send Feedback"
                data-testid="feedback-button"
              >
                <MessageSquare size={24} />
              </Button>
              <Button
                onClick={() => setShowSettings(true)}
                variant="ghost"
                className="text-slate-400 hover:text-white"
                title="Settings"
              >
                <SettingsIcon size={24} />
              </Button>
            </div>
          </div>
          
          {/* Backend Status Indicator */}
          <div className="mt-4 flex items-center justify-center gap-2">
            {backendStatus?.connected ? (
              <>
                <CheckCircle2 className="text-green-500" size={20} />
                <span className="text-green-400">Backend Connected</span>
              </>
            ) : (
              <>
                <XCircle className="text-red-500" size={20} />
                <span className="text-red-400">Backend Disconnected</span>
              </>
            )}
          </div>
        </header>

        <Tabs defaultValue="game" className="w-full" data-testid="main-tabs">
          <TabsList className="grid w-full grid-cols-12 mb-6 bg-slate-800/50">
            <TabsTrigger value="game" data-testid="game-tab">Game</TabsTrigger>
            <TabsTrigger value="training" data-testid="training-tab">Training</TabsTrigger>
            <TabsTrigger value="analytics" data-testid="analytics-tab">Analytics</TabsTrigger>
            <TabsTrigger value="memory-archive" data-testid="memory-archive-tab">Memory Archive</TabsTrigger>
            <TabsTrigger value="feedback-dashboard" data-testid="feedback-dashboard-tab">Feedback</TabsTrigger>
            <TabsTrigger value="diagnostics" data-testid="diagnostics-tab">Diagnostics</TabsTrigger>
            <TabsTrigger value="collective" data-testid="collective-tab">Collective</TabsTrigger>
            <TabsTrigger value="governance" data-testid="governance-tab">Governance</TabsTrigger>
            <TabsTrigger value="ethics" data-testid="ethics-tab">Ethics</TabsTrigger>
            <TabsTrigger value="synthesis" data-testid="synthesis-tab">Synthesis</TabsTrigger>
            <TabsTrigger value="creativity" data-testid="creativity-tab">Creativity</TabsTrigger>
            <TabsTrigger value="optimization" data-testid="optimization-tab">Optimization</TabsTrigger>
          </TabsList>

          <TabsContent value="game" data-testid="game-content">
            <GameTab />
          </TabsContent>

          <TabsContent value="training" data-testid="training-content">
            <TrainingPanel />
          </TabsContent>

          <TabsContent value="analytics" data-testid="analytics-content">
            <AnalyticsPanel />
          </TabsContent>

          <TabsContent value="memory-archive" data-testid="memory-archive-content">
            <MemoryArchivePanel />
          </TabsContent>

          <TabsContent value="feedback-dashboard" data-testid="feedback-dashboard-content">
            <FeedbackAnalyticsDashboard />
          </TabsContent>

          <TabsContent value="diagnostics" data-testid="diagnostics-content">
            <DiagnosticsPanel />
          </TabsContent>

          <TabsContent value="collective" data-testid="collective-content">
            <CollectiveIntelligencePanel />
          </TabsContent>

          <TabsContent value="governance" data-testid="governance-content">
            <GovernancePanel />
          </TabsContent>

          <TabsContent value="ethics" data-testid="ethics-content">
            <EthicalConsensusPanel />
          </TabsContent>

          <TabsContent value="synthesis" data-testid="synthesis-content">
            <CognitiveSynthesisPanel />
          </TabsContent>

          <TabsContent value="consciousness" data-testid="consciousness-content">
            <CollectiveCorePanel />
          </TabsContent>

          <TabsContent value="creativity" data-testid="creativity-content">
            <CreativeSynthesisPanel />
          </TabsContent>

          <TabsContent value="reflection" data-testid="reflection-content">
            <ReflectionLoopPanel />
          </TabsContent>

          <TabsContent value="memory" data-testid="memory-content">
            <MemoryFusionPanel />
          </TabsContent>

          <TabsContent value="cohesion" data-testid="cohesion-content">
            <CohesionCorePanel />
          </TabsContent>

          <TabsContent value="ethical-governance" data-testid="ethical-governance-content">
            <EthicalGovernancePanel />
          </TabsContent>

          <TabsContent value="resonance" data-testid="resonance-content">
            <CognitiveResonancePanel />
          </TabsContent>

          <TabsContent value="optimization" data-testid="optimization-content">
            <SystemOptimizationPanel />
          </TabsContent>

          <TabsContent value="models" data-testid="models-content">
            <ModelManagement />
          </TabsContent>
        </Tabs>
      </div>

      {/* Feedback Modal */}
      {showFeedback && <FeedbackModal onClose={() => setShowFeedback(false)} />}
      
      {/* Update Notification */}
      <UpdateNotification />
    </div>
  );
}

export default App;
