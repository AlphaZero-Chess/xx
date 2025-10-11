import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Alert, AlertDescription } from './ui/alert';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Switch } from './ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Settings, Database, Key, Shield, Info, Save } from 'lucide-react';
import { toast } from 'sonner';

const SettingsPanel = ({ onClose }) => {
  const [settings, setSettings] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      if (window.electronAPI) {
        const data = await window.electronAPI.getSettings();
        setSettings(data);
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
      toast.error('Failed to load settings');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      if (window.electronAPI) {
        await window.electronAPI.saveSettings(settings);
        toast.success('Settings saved successfully');
        
        // Restart backend if needed
        if (window.confirm('Backend needs to restart to apply changes. Restart now?')) {
          await window.electronAPI.restartBackend();
          toast.success('Backend restarted');
        }
      }
    } catch (error) {
      console.error('Failed to save settings:', error);
      toast.error('Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const handleToggleOfflineMode = async (enabled) => {
    try {
      if (window.electronAPI) {
        await window.electronAPI.toggleOfflineMode(enabled);
        setSettings(prev => ({ ...prev, offlineMode: enabled }));
        toast.success(`Offline mode ${enabled ? 'enabled' : 'disabled'}`);
      }
    } catch (error) {
      console.error('Failed to toggle offline mode:', error);
      toast.error('Failed to toggle offline mode');
    }
  };

  if (loading || !settings) {
    return (
      <div className="p-8 text-center text-slate-300">
        Loading settings...
      </div>
    );
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Settings className="text-blue-400" size={32} />
          <h2 className="text-2xl font-bold text-white">Application Settings</h2>
        </div>
        {onClose && (
          <Button onClick={onClose} variant="ghost" className="text-slate-300">
            Close
          </Button>
        )}
      </div>

      <Tabs defaultValue="database" className="w-full">
        <TabsList className="grid w-full grid-cols-4 bg-slate-800">
          <TabsTrigger value="database">Database</TabsTrigger>
          <TabsTrigger value="llm">LLM</TabsTrigger>
          <TabsTrigger value="features">Features</TabsTrigger>
          <TabsTrigger value="about">About</TabsTrigger>
        </TabsList>

        <TabsContent value="database" className="space-y-4 mt-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Database className="text-blue-400" size={20} />
                Database Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="mongoUrl" className="text-slate-300">MongoDB Connection URL</Label>
                <Input
                  id="mongoUrl"
                  type="text"
                  value={settings.mongo_url || ''}
                  onChange={(e) => setSettings({ ...settings, mongo_url: e.target.value })}
                  className="bg-slate-900 border-slate-700 text-white font-mono text-sm"
                />
                <p className="text-xs text-slate-400">
                  Requires backend restart to apply changes
                </p>
              </div>

              <Alert>
                <AlertDescription className="text-sm text-slate-300">
                  <strong>Note:</strong> Connection string is encrypted locally for security.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="llm" className="space-y-4 mt-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Key className="text-purple-400" size={20} />
                LLM Integration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="llmProvider" className="text-slate-300">LLM Provider</Label>
                <Select 
                  value={settings.llmProvider || 'claude-3-5-sonnet'} 
                  onValueChange={(value) => setSettings({ ...settings, llmProvider: value })}
                >
                  <SelectTrigger className="bg-slate-900 border-slate-700 text-white">
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
                  value={settings.llm_key || ''}
                  onChange={(e) => setSettings({ ...settings, llm_key: e.target.value })}
                  placeholder="Enter your Emergent LLM key"
                  className="bg-slate-900 border-slate-700 text-white"
                  disabled={settings.offlineMode}
                />
                <p className="text-xs text-slate-400">
                  Encrypted using AES-256. Never stored in plain text.
                </p>
              </div>

              <div className="flex items-center justify-between p-4 bg-slate-900 rounded-lg">
                <div>
                  <Label htmlFor="offlineMode" className="text-slate-300 font-medium">Offline Mode</Label>
                  <p className="text-sm text-slate-400">Disable all LLM features</p>
                </div>
                <Switch
                  id="offlineMode"
                  checked={settings.offlineMode || false}
                  onCheckedChange={handleToggleOfflineMode}
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="features" className="space-y-4 mt-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Shield className="text-green-400" size={20} />
                Advanced Features
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-slate-900 rounded-lg">
                <div>
                  <Label htmlFor="ethicsWatchdog" className="text-slate-300 font-medium">
                    Ethical Governance 2.0 Watchdog
                  </Label>
                  <p className="text-sm text-slate-400">Monitor AI decisions for ethical compliance</p>
                </div>
                <Switch
                  id="ethicsWatchdog"
                  checked={settings.ethicsWatchdogEnabled || false}
                  onCheckedChange={(checked) => setSettings({ ...settings, ethicsWatchdogEnabled: checked })}
                />
              </div>

              <div className="flex items-center justify-between p-4 bg-slate-900 rounded-lg">
                <div>
                  <Label htmlFor="autoUpdate" className="text-slate-300 font-medium">
                    Auto Updates
                  </Label>
                  <p className="text-sm text-slate-400">Automatically check for application updates</p>
                </div>
                <Switch
                  id="autoUpdate"
                  checked={settings.autoUpdate || false}
                  onCheckedChange={(checked) => setSettings({ ...settings, autoUpdate: checked })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="backendPort" className="text-slate-300">Backend Port</Label>
                <Input
                  id="backendPort"
                  type="number"
                  value={settings.backend_port || 8001}
                  onChange={(e) => setSettings({ ...settings, backend_port: parseInt(e.target.value) })}
                  className="bg-slate-900 border-slate-700 text-white"
                  min="1024"
                  max="65535"
                />
                <p className="text-xs text-slate-400">
                  Default: 8001 (Requires restart)
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="about" className="space-y-4 mt-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Info className="text-blue-400" size={20} />
                About AlphaZero Chess AI
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div className="flex justify-between items-center py-2 border-b border-slate-700">
                  <span className="text-slate-400">Version</span>
                  <span className="text-white font-semibold">1.0.0 (Build #37)</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-slate-700">
                  <span className="text-slate-400">Build Date</span>
                  <span className="text-white">2025</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-slate-700">
                  <span className="text-slate-400">Platform</span>
                  <span className="text-white">Electron</span>
                </div>
              </div>

              <div className="bg-slate-900 p-4 rounded-lg mt-6">
                <h4 className="text-white font-semibold mb-2">Features</h4>
                <ul className="text-sm text-slate-300 space-y-1">
                  <li>✓ Self-Learning Chess AI with AlphaZero</li>
                  <li>✓ Deep Reinforcement Learning</li>
                  <li>✓ LLM-Powered Coaching & Analysis</li>
                  <li>✓ Autonomous Creativity System</li>
                  <li>✓ Ethical Governance 2.0</li>
                  <li>✓ Cognitive Resonance & Memory Fusion</li>
                  <li>✓ CASV-1 Validation Framework</li>
                  <li>✓ Cross-Platform Support</li>
                </ul>
              </div>

              <p className="text-xs text-slate-400 text-center mt-4">
                © 2025 Emergent AI. All rights reserved.
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <div className="flex justify-end gap-3 mt-6">
        <Button
          onClick={handleSave}
          disabled={saving}
          className="bg-blue-600 hover:bg-blue-700 flex items-center gap-2"
        >
          <Save size={16} />
          {saving ? 'Saving...' : 'Save Settings'}
        </Button>
      </div>
    </div>
  );
};

export default SettingsPanel;
