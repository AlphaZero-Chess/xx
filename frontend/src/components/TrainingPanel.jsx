import React, { useState, useEffect, useRef } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Loader2, Play, Square, Download, Upload, Trash2, Database } from 'lucide-react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const TrainingPanel = () => {
  // Training state
  const [trainingActive, setTrainingActive] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState({
    num_games: 5,
    num_epochs: 10,
    batch_size: 32,
    num_simulations: 400,
    learning_rate: 0.001
  });
  const [trainingStatus, setTrainingStatus] = useState(null);
  
  // Evaluation state
  const [evaluationActive, setEvaluationActive] = useState(false);
  const [evaluations, setEvaluations] = useState([]);
  const [evaluationStatus, setEvaluationStatus] = useState(null);
  const [activeModel, setActiveModel] = useState(null);
  
  // Device info
  const [deviceInfo, setDeviceInfo] = useState(null);
  
  // Self-play state
  const [selfPlayActive, setSelfPlayActive] = useState(false);
  const [selfPlayConfig, setSelfPlayConfig] = useState({
    num_games: 10,
    num_simulations: 800
  });
  const [selfPlayStatus, setSelfPlayStatus] = useState(null);
  const [sessions, setSessions] = useState([]);
  
  // Model state
  const [models, setModels] = useState([]);
  const fileInputRef = useRef(null);

  useEffect(() => {
    loadModels();
    loadSessions();
    loadActiveModel();
    loadEvaluations();
    loadDeviceInfo();
    const interval = setInterval(() => {
      checkTrainingStatus();
      checkSelfPlayStatus();
      checkEvaluationStatus();
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const loadDeviceInfo = async () => {
    try {
      const response = await axios.get(`${API}/system/device`);
      setDeviceInfo(response.data);
    } catch (error) {
      console.error('Error loading device info:', error);
    }
  };

  const checkTrainingStatus = async () => {
    try {
      const response = await axios.get(`${API}/training/status`);
      setTrainingStatus(response.data);
      setTrainingActive(response.data.active);
    } catch (error) {
      console.error('Error checking training status:', error);
    }
  };

  const checkSelfPlayStatus = async () => {
    try {
      const response = await axios.get(`${API}/self-play/status`);
      setSelfPlayStatus(response.data);
      setSelfPlayActive(response.data.active);
      
      // Reload sessions when self-play completes
      if (!response.data.active && selfPlayActive) {
        loadSessions();
      }
    } catch (error) {
      console.error('Error checking self-play status:', error);
    }
  };

  const loadSessions = async () => {
    try {
      const response = await axios.get(`${API}/self-play/sessions?limit=20`);
      setSessions(response.data.sessions || []);
    } catch (error) {
      console.error('Error loading sessions:', error);
    }
  };

  const loadModels = async () => {
    try {
      const response = await axios.get(`${API}/model/list`);
      // Extract just the model names from the model objects
      const modelNames = response.data.models.map(m => m.name);
      setModels(modelNames);
    } catch (error) {
      console.error('Error loading models:', error);
    }
  };

  const loadActiveModel = async () => {
    try {
      const response = await axios.get(`${API}/model/current`);
      setActiveModel(response.data.active_model);
    } catch (error) {
      console.error('Error loading active model:', error);
    }
  };

  const loadEvaluations = async () => {
    try {
      const response = await axios.get(`${API}/evaluation/history?limit=10`);
      setEvaluations(response.data.evaluations || []);
    } catch (error) {
      console.error('Error loading evaluations:', error);
    }
  };

  const checkEvaluationStatus = async () => {
    try {
      const response = await axios.get(`${API}/evaluation/status`);
      setEvaluationStatus(response.data);
      setEvaluationActive(response.data.active);
      
      // Reload evaluations when complete
      if (!response.data.active && evaluationActive) {
        loadEvaluations();
        loadActiveModel();
      }
    } catch (error) {
      console.error('Error checking evaluation status:', error);
    }
  };

  const stopEvaluation = async () => {
    try {
      await axios.post(`${API}/evaluation/stop`);
      alert('Evaluation cancellation requested');
    } catch (error) {
      console.error('Error stopping evaluation:', error);
      alert('Failed to stop evaluation: ' + (error.response?.data?.detail || error.message));
    }
  };

  const startTraining = async () => {
    try {
      await axios.post(`${API}/training/start`, trainingConfig);
      setTrainingActive(true);
    } catch (error) {
      console.error('Error starting training:', error);
      alert('Failed to start training: ' + error.response?.data?.detail);
    }
  };

  const runEvaluation = async (challengerName, championName) => {
    try {
      await axios.post(`${API}/evaluation/run`, {
        challenger_name: challengerName,
        champion_name: championName,
        num_games: 20,
        num_simulations: 400
      });
      setEvaluationActive(true);
    } catch (error) {
      console.error('Error starting evaluation:', error);
      alert('Failed to start evaluation: ' + (error.response?.data?.detail || error.message));
    }
  };

  const activateModel = async (modelName) => {
    try {
      await axios.post(`${API}/model/activate/${modelName}`);
      alert(`Model ${modelName} is now active!`);
      loadActiveModel();
    } catch (error) {
      console.error('Error activating model:', error);
      alert('Failed to activate model: ' + (error.response?.data?.detail || error.message));
    }
  };

  const stopTraining = async () => {
    try {
      await axios.post(`${API}/training/stop`);
      setTrainingActive(false);
    } catch (error) {
      console.error('Error stopping training:', error);
    }
  };

  const startSelfPlay = async () => {
    try {
      await axios.post(`${API}/self-play/run`, selfPlayConfig);
      setSelfPlayActive(true);
    } catch (error) {
      console.error('Error starting self-play:', error);
      alert('Failed to start self-play: ' + (error.response?.data?.detail || error.message));
    }
  };

  const exportSession = async (sessionId, format) => {
    try {
      if (format === 'json') {
        const response = await axios.get(`${API}/self-play/export/${sessionId}?format=json`);
        
        // Download as JSON file
        const dataStr = JSON.stringify(response.data, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = window.URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `selfplay_${sessionId}.json`;
        document.body.appendChild(link);
        link.click();
        link.parentNode.removeChild(link);
        window.URL.revokeObjectURL(url);
      } else {
        // CSV download
        const response = await axios.get(`${API}/self-play/export/${sessionId}?format=csv`, {
          responseType: 'blob'
        });
        
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.download = `selfplay_${sessionId}.csv`;
        document.body.appendChild(link);
        link.click();
        link.parentNode.removeChild(link);
        window.URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Error exporting session:', error);
      alert('Failed to export dataset');
    }
  };

  const saveModel = async () => {
    try {
      const name = prompt('Enter model name:') || 'model_' + Date.now();
      await axios.post(`${API}/model/save?name=${name}`);
      alert('Model saved successfully!');
      loadModels();
    } catch (error) {
      console.error('Error saving model:', error);
      alert('Failed to save model');
    }
  };

  const loadModel = async (name) => {
    try {
      await axios.post(`${API}/model/load?name=${name}`);
      alert(`Model ${name} loaded successfully!`);
    } catch (error) {
      console.error('Error loading model:', error);
      alert('Failed to load model');
    }
  };

  const exportModel = async (name) => {
    try {
      const response = await axios.get(`${API}/model/export/${name}`, {
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${name}.pt`);
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error exporting model:', error);
      alert('Failed to export model');
    }
  };

  const importModel = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const name = prompt('Enter name for imported model:', file.name.replace('.pt', ''));
      if (!name) return;

      await axios.post(`${API}/model/import?name=${name}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      alert(`Model ${name} imported and loaded successfully!`);
      loadModels();
    } catch (error) {
      console.error('Error importing model:', error);
      alert('Failed to import model: ' + (error.response?.data?.detail || error.message));
    }
    
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const deleteModel = async (name) => {
    if (!window.confirm(`Are you sure you want to delete model "${name}"?`)) {
      return;
    }

    try {
      await axios.delete(`${API}/model/delete/${name}`);
      alert(`Model ${name} deleted successfully!`);
      loadModels();
    } catch (error) {
      console.error('Error deleting model:', error);
      alert('Failed to delete model');
    }
  };

  return (
    <div className="space-y-4">
      {/* Device Info Card */}
      {deviceInfo && (
        <Card className="shadow-xl bg-slate-800 border-slate-700" data-testid="device-info">
          <CardHeader className="bg-gradient-to-r from-blue-900 to-indigo-900">
            <CardTitle className="flex items-center justify-between text-white">
              <span className="text-sm">System Information</span>
              <Badge variant={deviceInfo.is_gpu ? "default" : "secondary"} className={deviceInfo.is_gpu ? "bg-green-600" : "bg-gray-600"}>
                {deviceInfo.is_gpu ? "GPU" : "CPU"}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-4">
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-400">Device:</span>
                <span className="text-slate-200 font-medium">{deviceInfo.device_name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Type:</span>
                <span className="text-slate-200 font-medium">{deviceInfo.device_type.toUpperCase()}</span>
              </div>
              {deviceInfo.is_gpu && deviceInfo.cuda_version && (
                <>
                  <div className="flex justify-between">
                    <span className="text-slate-400">CUDA:</span>
                    <span className="text-slate-200 font-medium">{deviceInfo.cuda_version}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Memory:</span>
                    <span className="text-slate-200 font-medium">{deviceInfo.gpu_memory_allocated}</span>
                  </div>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      )}
      
      {/* Self-Play Data Collection Section */}
      <Card className="shadow-xl bg-slate-800 border-slate-700" data-testid="selfplay-panel">
        <CardHeader className="bg-gradient-to-r from-emerald-900 to-teal-900">
          <CardTitle className="flex items-center justify-between text-white">
            <div className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              <span>Self-Play Data Collection</span>
            </div>
            {selfPlayActive && (
              <Badge variant="default" className="animate-pulse bg-emerald-500" data-testid="selfplay-active-badge">
                Collecting Data
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 pt-6">
          <div className="space-y-3">
            <div>
              <label className="text-sm font-medium text-slate-300">Number of Games</label>
              <input
                type="number"
                value={selfPlayConfig.num_games}
                onChange={(e) => setSelfPlayConfig({...selfPlayConfig, num_games: parseInt(e.target.value)})}
                className="w-full px-3 py-2 border border-slate-600 rounded-lg mt-1 bg-slate-700 text-white"
                min="1"
                max="100"
                disabled={selfPlayActive}
                data-testid="selfplay-num-games-input"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-slate-300">MCTS Simulations per Move</label>
              <input
                type="number"
                value={selfPlayConfig.num_simulations}
                onChange={(e) => setSelfPlayConfig({...selfPlayConfig, num_simulations: parseInt(e.target.value)})}
                className="w-full px-3 py-2 border border-slate-600 rounded-lg mt-1 bg-slate-700 text-white"
                min="100"
                max="1600"
                step="100"
                disabled={selfPlayActive}
                data-testid="selfplay-simulations-input"
              />
            </div>
          </div>

          <Button
            onClick={startSelfPlay}
            disabled={selfPlayActive}
            className="w-full bg-emerald-600 hover:bg-emerald-700"
            data-testid="run-selfplay-btn"
          >
            {selfPlayActive ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating Games...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Run Self-Play
              </>
            )}
          </Button>

          {selfPlayStatus && (
            <div className="p-4 bg-slate-700 rounded-lg text-slate-300" data-testid="selfplay-status">
              <div className="text-sm space-y-1">
                <div className="flex justify-between">
                  <span>Total Sessions:</span>
                  <span className="font-semibold">{selfPlayStatus.total_sessions}</span>
                </div>
                <div className="flex justify-between">
                  <span>Total Positions:</span>
                  <span className="font-semibold">{selfPlayStatus.total_positions}</span>
                </div>
              </div>
            </div>
          )}

          {sessions.length > 0 && (
            <div className="mt-4" data-testid="session-list">
              <div className="text-sm font-medium mb-2 text-slate-300">Generated Datasets</div>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {sessions.map((session) => (
                  <div key={session.session_id} className="p-3 bg-slate-700 rounded-lg">
                    <div className="text-xs text-slate-400 mb-2">
                      {new Date(session.timestamp).toLocaleString()}
                    </div>
                    <div className="flex justify-between text-sm text-slate-300 mb-2">
                      <span>{session.num_games} games</span>
                      <span>{session.num_positions} positions</span>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        onClick={() => exportSession(session.session_id, 'json')}
                        variant="outline"
                        size="sm"
                        className="flex-1 text-xs border-slate-600 text-slate-300 hover:bg-slate-600"
                        data-testid={`export-json-${session.session_id}`}
                      >
                        <Download className="mr-1 h-3 w-3" />
                        JSON
                      </Button>
                      <Button
                        onClick={() => exportSession(session.session_id, 'csv')}
                        variant="outline"
                        size="sm"
                        className="flex-1 text-xs border-slate-600 text-slate-300 hover:bg-slate-600"
                        data-testid={`export-csv-${session.session_id}`}
                      >
                        <Download className="mr-1 h-3 w-3" />
                        CSV
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Training System Section */}
      <Card className="shadow-xl bg-slate-800 border-slate-700" data-testid="training-panel">
        <CardHeader className="bg-gradient-to-r from-purple-900 to-purple-800">
          <CardTitle className="flex items-center justify-between text-white">
            <span>Training System</span>
            {trainingActive && (
              <Badge variant="default" className="animate-pulse bg-purple-500" data-testid="training-active-badge">
                Training Active
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 pt-6">
          <div className="space-y-3">
            <div>
              <label className="text-sm font-medium text-slate-300">Self-Play Games</label>
              <input
                type="number"
                value={trainingConfig.num_games}
                onChange={(e) => setTrainingConfig({...trainingConfig, num_games: parseInt(e.target.value)})}
                className="w-full px-3 py-2 border border-slate-600 rounded-lg mt-1 bg-slate-700 text-white"
                min="1"
                max="100"
                disabled={trainingActive}
                data-testid="num-games-input"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-slate-300">Training Epochs</label>
              <input
                type="number"
                value={trainingConfig.num_epochs}
                onChange={(e) => setTrainingConfig({...trainingConfig, num_epochs: parseInt(e.target.value)})}
                className="w-full px-3 py-2 border border-slate-600 rounded-lg mt-1 bg-slate-700 text-white"
                min="1"
                max="50"
                disabled={trainingActive}
                data-testid="num-epochs-input"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-slate-300">MCTS Simulations (Training)</label>
              <input
                type="number"
                value={trainingConfig.num_simulations}
                onChange={(e) => setTrainingConfig({...trainingConfig, num_simulations: parseInt(e.target.value)})}
                className="w-full px-3 py-2 border border-slate-600 rounded-lg mt-1 bg-slate-700 text-white"
                min="100"
                max="1600"
                step="100"
                disabled={trainingActive}
                data-testid="training-sims-input"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-slate-300">Learning Rate</label>
              <input
                type="number"
                value={trainingConfig.learning_rate}
                onChange={(e) => setTrainingConfig({...trainingConfig, learning_rate: parseFloat(e.target.value)})}
                className="w-full px-3 py-2 border border-slate-600 rounded-lg mt-1 bg-slate-700 text-white"
                min="0.0001"
                max="0.1"
                step="0.0001"
                disabled={trainingActive}
                data-testid="learning-rate-input"
              />
            </div>
          </div>

          <div className="flex gap-2">
            <Button
              onClick={startTraining}
              disabled={trainingActive}
              className="flex-1 bg-purple-600 hover:bg-purple-700"
              data-testid="start-training-btn"
            >
              {trainingActive ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Training...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Start Training
                </>
              )}
            </Button>

            <Button
              onClick={stopTraining}
              disabled={!trainingActive}
              variant="destructive"
              className="flex-1"
              data-testid="stop-training-btn"
            >
              <Square className="mr-2 h-4 w-4" />
              Cancel
            </Button>
          </div>

          {/* Training Progress */}
          {trainingStatus && trainingStatus.progress && trainingActive && (
            <div className="p-4 bg-slate-700 rounded-lg" data-testid="training-progress">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium text-slate-300">Progress</span>
                <span className="text-sm text-slate-400">{trainingStatus.progress.progress}%</span>
              </div>
              <div className="w-full bg-slate-600 rounded-full h-2 mb-2">
                <div 
                  className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${trainingStatus.progress.progress}%` }}
                />
              </div>
              <p className="text-xs text-slate-400">{trainingStatus.progress.message}</p>
            </div>
          )}

          <div className="space-y-2">
            <Button
              onClick={saveModel}
              className="w-full bg-indigo-600 hover:bg-indigo-700"
              data-testid="save-model-btn"
            >
              Save Current Model
            </Button>

            <input
              ref={fileInputRef}
              type="file"
              accept=".pt"
              onChange={importModel}
              style={{ display: 'none' }}
              data-testid="file-input"
            />
            
            <Button
              onClick={() => fileInputRef.current?.click()}
              className="w-full bg-teal-600 hover:bg-teal-700"
              data-testid="import-model-btn"
            >
              <Upload className="mr-2 h-4 w-4" />
              Import Model
            </Button>
          </div>

          {trainingStatus && trainingStatus.recent_metrics && trainingStatus.recent_metrics.length > 0 && (
            <div className="p-4 bg-slate-700 rounded-lg" data-testid="training-metrics">
              <div className="text-sm font-medium mb-2 text-slate-300">Recent Training Metrics</div>
              <div className="text-xs space-y-1 text-slate-400">
                {trainingStatus.recent_metrics.slice(0, 3).map((metric, idx) => (
                  <div key={idx} className="flex justify-between">
                    <span>Epoch {metric.epoch}</span>
                    <span>Loss: {metric.loss?.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {models.length > 0 && (
            <div className="mt-4" data-testid="model-list">
              <div className="text-sm font-medium mb-2 text-slate-300">
                Saved Models {activeModel && <span className="text-xs text-slate-500">(Active: {activeModel})</span>}
              </div>
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {models.map((model) => (
                  <div key={model} className="flex gap-1 items-center">
                    <Button
                      onClick={() => activateModel(model)}
                      variant="outline"
                      className={`flex-1 text-left justify-start text-xs truncate ${
                        model === activeModel 
                          ? 'border-emerald-500 text-emerald-400 hover:bg-emerald-900' 
                          : 'border-slate-600 text-slate-300 hover:bg-slate-600'
                      }`}
                      data-testid={`activate-model-${model}`}
                      title={`${model} ${model === activeModel ? '(ACTIVE)' : ''}`}
                    >
                      {model === activeModel && <span className="mr-1">✓</span>}
                      {model}
                    </Button>
                    <Button
                      onClick={() => exportModel(model)}
                      variant="outline"
                      size="sm"
                      className="px-2 border-slate-600 text-slate-300 hover:bg-slate-600"
                      data-testid={`export-model-${model}`}
                      title="Export/Download"
                    >
                      <Download className="h-3 w-3" />
                    </Button>
                    <Button
                      onClick={() => deleteModel(model)}
                      variant="outline"
                      size="sm"
                      className="px-2 border-red-600 text-red-400 hover:bg-red-900"
                      data-testid={`delete-model-${model}`}
                      title="Delete"
                      disabled={model === activeModel}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Model Evaluation Section */}
      <Card className="shadow-xl bg-slate-800 border-slate-700" data-testid="evaluation-panel">
        <CardHeader className="bg-gradient-to-r from-amber-900 to-orange-900">
          <CardTitle className="flex items-center justify-between text-white">
            <div className="flex items-center gap-2">
              <span>Model Evaluation</span>
            </div>
            {evaluationActive && (
              <Badge variant="default" className="animate-pulse bg-amber-500" data-testid="evaluation-active-badge">
                Evaluating
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 pt-6">
          <div className="text-sm text-slate-300">
            <p className="mb-2">Run evaluation matches between models to compare performance.</p>
            <p className="text-xs text-slate-400">Evaluation uses 20 games with 400 MCTS simulations per move.</p>
          </div>

          {/* Evaluation Progress */}
          {evaluationStatus && evaluationStatus.progress && evaluationActive && (
            <div className="p-4 bg-slate-700 rounded-lg" data-testid="evaluation-progress">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium text-slate-300">Progress</span>
                <span className="text-sm text-slate-400">{evaluationStatus.progress.progress}%</span>
              </div>
              <div className="w-full bg-slate-600 rounded-full h-2 mb-2">
                <div 
                  className="bg-amber-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${evaluationStatus.progress.progress}%` }}
                />
              </div>
              <div className="flex justify-between items-center">
                <p className="text-xs text-slate-400">{evaluationStatus.progress.message}</p>
                <Button
                  onClick={stopEvaluation}
                  variant="destructive"
                  size="sm"
                  className="text-xs"
                  data-testid="stop-evaluation-btn"
                >
                  <Square className="mr-1 h-3 w-3" />
                  Cancel
                </Button>
              </div>
            </div>
          )}

          {evaluations.length > 0 && (
            <div data-testid="evaluation-history">
              <div className="text-sm font-medium mb-2 text-slate-300">Recent Evaluations</div>
              <div className="space-y-3 max-h-80 overflow-y-auto">
                {evaluations.map((evalData, idx) => (
                  <div key={idx} className="p-3 bg-slate-700 rounded-lg">
                    <div className="flex justify-between items-start mb-2">
                      <div className="text-xs text-slate-400">
                        {new Date(evalData.timestamp).toLocaleString()}
                      </div>
                      {evalData.promoted && (
                        <Badge variant="default" className="text-xs bg-green-600">
                          Promoted
                        </Badge>
                      )}
                    </div>
                    <div className="text-sm text-slate-300 mb-2">
                      <div className="font-medium">{evalData.challenger_name}</div>
                      <div className="text-xs">vs {evalData.champion_name}</div>
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div className="bg-slate-800 p-2 rounded">
                        <div className="text-slate-400">Win Rate</div>
                        <div className={`font-bold ${evalData.challenger_win_rate >= 0.55 ? 'text-green-400' : 'text-slate-300'}`}>
                          {(evalData.challenger_win_rate * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-slate-800 p-2 rounded">
                        <div className="text-slate-400">Games</div>
                        <div className="font-bold text-slate-300">{evalData.games_played}</div>
                      </div>
                      <div className="bg-slate-800 p-2 rounded">
                        <div className="text-slate-400">W-L-D</div>
                        <div className="font-bold text-slate-300">
                          {evalData.model1_wins}-{evalData.model2_wins}-{evalData.draws}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {evaluations.length === 0 && (
            <div className="text-center text-slate-500 text-sm py-4">
              No evaluations yet. Training will automatically evaluate new models.
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default TrainingPanel;