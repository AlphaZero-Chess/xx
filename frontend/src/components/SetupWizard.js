import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription, CardFooter } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Alert, AlertDescription } from './ui/alert';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { CheckCircle2, Database, Key, Settings, Zap } from 'lucide-react';

const SetupWizard = ({ onComplete }) => {
  const [step, setStep] = useState(1);
  const [config, setConfig] = useState({
    mongoUrl: 'mongodb://localhost:27017',
    llmProvider: 'claude-3-5-sonnet',
    llmKey: '',
    offlineMode: false,
    ethicsWatchdogEnabled: true
  });
  const [error, setError] = useState('');
  const [testing, setTesting] = useState(false);

  const handleInputChange = (field, value) => {
    setConfig(prev => ({ ...prev, [field]: value }));
    setError('');
  };

  const testMongoConnection = async () => {
    setTesting(true);
    try {
      // Simple validation
      if (!config.mongoUrl.startsWith('mongodb://') && !config.mongoUrl.startsWith('mongodb+srv://')) {
        throw new Error('Invalid MongoDB URL format');
      }
      setError('');
      return true;
    } catch (err) {
      setError(err.message);
      return false;
    } finally {
      setTesting(false);
    }
  };

  const handleNext = async () => {
    if (step === 1) {
      const success = await testMongoConnection();
      if (!success) return;
    }
    
    if (step === 2) {
      if (!config.offlineMode && !config.llmKey) {
        setError('Please enter an LLM key or enable offline mode');
        return;
      }
    }
    
    if (step === 3) {
      // Save configuration
      try {
        if (window.electronAPI) {
          await window.electronAPI.saveSettings({
            firstLaunch: false,
            mongo_url: config.mongoUrl,
            llmProvider: config.llmProvider,
            llm_key: config.llmKey,
            offlineMode: config.offlineMode,
            ethicsWatchdogEnabled: config.ethicsWatchdogEnabled,
            backend_port: 8001,
            autoUpdate: true
          });
          await window.electronAPI.completeFirstLaunch();
        }
        onComplete();
      } catch (err) {
        setError('Failed to save configuration: ' + err.message);
        return;
      }
    }
    
    setStep(step + 1);
  };

  const handleBack = () => {
    setStep(step - 1);
    setError('');
  };

  const renderStep = () => {
    switch (step) {
      case 1:
        return (
          <div className="space-y-4">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 bg-blue-500/20 rounded-lg">
                <Database className="text-blue-400" size={24} />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-white">Database Configuration</h3>
                <p className="text-slate-400">Connect to your MongoDB instance</p>
              </div>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="mongoUrl" className="text-slate-300">MongoDB Connection URL</Label>
              <Input
                id="mongoUrl"
                type="text"
                value={config.mongoUrl}
                onChange={(e) => handleInputChange('mongoUrl', e.target.value)}
                placeholder="mongodb://localhost:27017"
                className="bg-slate-800 border-slate-700 text-white"
              />
              <p className="text-sm text-slate-400">
                Enter your MongoDB connection string. Default: mongodb://localhost:27017
              </p>
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
              <h4 className="text-sm font-semibold text-white mb-2">Connection Examples:</h4>
              <ul className="text-sm text-slate-400 space-y-1">
                <li>• Local: mongodb://localhost:27017</li>
                <li>• Atlas: mongodb+srv://user:pass@cluster.mongodb.net</li>
                <li>• Custom: mongodb://host:port/database</li>
              </ul>
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-4">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 bg-purple-500/20 rounded-lg">
                <Key className="text-purple-400" size={24} />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-white">LLM Integration</h3>
                <p className="text-slate-400">Configure AI coaching features</p>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="llmProvider" className="text-slate-300">LLM Provider</Label>
              <Select value={config.llmProvider} onValueChange={(value) => handleInputChange('llmProvider', value)}>
                <SelectTrigger className="bg-slate-800 border-slate-700 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="claude-3-5-sonnet">Claude 3.5 Sonnet (Primary)</SelectItem>
                  <SelectItem value="gpt-4o-mini">GPT-4o-mini (Secondary)</SelectItem>
                  <SelectItem value="gemini-2.0-flash">Gemini 2.0 Flash (Fallback)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="llmKey" className="text-slate-300">Emergent LLM Key</Label>
              <Input
                id="llmKey"
                type="password"
                value={config.llmKey}
                onChange={(e) => handleInputChange('llmKey', e.target.value)}
                placeholder="Enter your Emergent LLM key"
                className="bg-slate-800 border-slate-700 text-white"
                disabled={config.offlineMode}
              />
              <p className="text-sm text-slate-400">
                Your key will be encrypted locally using AES-256 encryption
              </p>
            </div>

            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="offlineMode"
                checked={config.offlineMode}
                onChange={(e) => handleInputChange('offlineMode', e.target.checked)}
                className="rounded border-slate-700"
              />
              <Label htmlFor="offlineMode" className="text-slate-300 cursor-pointer">
                Enable Offline Mode (Skip LLM features)
              </Label>
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <div className="bg-blue-500/10 p-4 rounded-lg border border-blue-500/30">
              <h4 className="text-sm font-semibold text-blue-400 mb-2">About Emergent LLM Key:</h4>
              <p className="text-sm text-slate-300">
                The Emergent LLM key provides unified access to multiple AI providers (Claude, GPT, Gemini) 
                through a single key. You can obtain one from your Emergent dashboard.
              </p>
            </div>
          </div>
        );

      case 3:
        return (
          <div className="space-y-4">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 bg-green-500/20 rounded-lg">
                <Settings className="text-green-400" size={24} />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-white">Additional Settings</h3>
                <p className="text-slate-400">Configure advanced features</p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-slate-800 rounded-lg border border-slate-700">
                <div>
                  <h4 className="text-white font-medium">Ethical Governance 2.0 Watchdog</h4>
                  <p className="text-sm text-slate-400">Monitor AI decisions for ethical compliance</p>
                </div>
                <input
                  type="checkbox"
                  checked={config.ethicsWatchdogEnabled}
                  onChange={(e) => handleInputChange('ethicsWatchdogEnabled', e.target.checked)}
                  className="rounded border-slate-700"
                />
              </div>

              <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
                <h4 className="text-sm font-semibold text-white mb-3">Configuration Summary:</h4>
                <dl className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Database:</dt>
                    <dd className="text-white font-mono text-xs">{config.mongoUrl}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-slate-400">LLM Provider:</dt>
                    <dd className="text-white">{config.llmProvider}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Offline Mode:</dt>
                    <dd className="text-white">{config.offlineMode ? 'Enabled' : 'Disabled'}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Ethics Watchdog:</dt>
                    <dd className="text-white">{config.ethicsWatchdogEnabled ? 'Enabled' : 'Disabled'}</dd>
                  </div>
                </dl>
              </div>
            </div>
          </div>
        );

      case 4:
        return (
          <div className="space-y-4 text-center">
            <div className="flex justify-center mb-6">
              <div className="p-4 bg-green-500/20 rounded-full">
                <CheckCircle2 className="text-green-400" size={48} />
              </div>
            </div>
            
            <h3 className="text-2xl font-semibold text-white">Setup Complete!</h3>
            <p className="text-slate-400">
              AlphaZero Chess AI is now configured and ready to use.
            </p>

            <div className="bg-slate-800/50 p-6 rounded-lg border border-slate-700 text-left mt-6">
              <h4 className="text-white font-semibold mb-4 flex items-center gap-2">
                <Zap className="text-yellow-400" size={20} />
                Quick Start Guide:
              </h4>
              <ul className="space-y-2 text-sm text-slate-300">
                <li>1. Navigate to the <strong>Game</strong> tab to play against the AI</li>
                <li>2. Use the <strong>Training</strong> tab to improve the AI model</li>
                <li>3. View insights in the <strong>Analytics</strong> dashboard</li>
                <li>4. Explore advanced features in the specialty tabs</li>
                <li>5. Access settings from the <strong>File</strong> menu anytime</li>
              </ul>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-4">
      <Card className="w-full max-w-2xl bg-slate-900/90 border-slate-700">
        <CardHeader>
          <CardTitle className="text-2xl text-white">Welcome to AlphaZero Chess AI</CardTitle>
          <CardDescription className="text-slate-400">
            Step {step} of 4 - Let's configure your application
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          {/* Progress indicator */}
          <div className="flex items-center gap-2 mb-8">
            {[1, 2, 3, 4].map((i) => (
              <div
                key={i}
                className={`flex-1 h-2 rounded-full transition-colors ${
                  i <= step ? 'bg-blue-500' : 'bg-slate-700'
                }`}
              />
            ))}
          </div>

          {renderStep()}
        </CardContent>

        <CardFooter className="flex justify-between">
          <Button
            onClick={handleBack}
            variant="outline"
            disabled={step === 1 || step === 4}
            className="border-slate-700 text-slate-300"
          >
            Back
          </Button>
          <Button
            onClick={handleNext}
            disabled={testing}
            className="bg-blue-600 hover:bg-blue-700"
          >
            {step === 4 ? 'Get Started' : step === 3 ? 'Complete Setup' : 'Next'}
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
};

export default SetupWizard;
