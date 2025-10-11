import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Alert, AlertDescription } from './ui/alert';
import { 
  Download, 
  Upload, 
  RefreshCw, 
  Database, 
  CheckCircle, 
  AlertCircle,
  Cpu,
  HardDrive,
  Calendar,
  Activity
} from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Separator } from './ui/separator';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ModelManagement = () => {
  const [models, setModels] = useState([]);
  const [currentModel, setCurrentModel] = useState(null);
  const [exports, setExports] = useState([]);
  const [deviceInfo, setDeviceInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [exportLoading, setExportLoading] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });
  const [selectedModel, setSelectedModel] = useState('');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        loadModels(),
        loadCurrentModel(),
        loadExports(),
        loadDeviceInfo()
      ]);
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadModels = async () => {
    try {
      const response = await axios.get(`${API}/model/list`);
      if (response.data.success) {
        setModels(response.data.models);
        if (response.data.models.length > 0 && !selectedModel) {
          setSelectedModel(response.data.models[0].name);
        }
      }
    } catch (error) {
      console.error('Error loading models:', error);
      showMessage('error', 'Failed to load models');
    }
  };

  const loadCurrentModel = async () => {
    try {
      const response = await axios.get(`${API}/model/current`);
      if (response.data.success && response.data.loaded) {
        setCurrentModel(response.data);
      } else {
        setCurrentModel(null);
      }
    } catch (error) {
      console.error('Error loading current model:', error);
    }
  };

  const loadExports = async () => {
    try {
      const response = await axios.get(`${API}/model/exports`);
      if (response.data.success) {
        setExports(response.data.exports);
      }
    } catch (error) {
      console.error('Error loading exports:', error);
    }
  };

  const loadDeviceInfo = async () => {
    try {
      const response = await axios.get(`${API}/device/info`);
      if (response.data.success) {
        setDeviceInfo(response.data);
      }
    } catch (error) {
      console.error('Error loading device info:', error);
    }
  };

  const showMessage = (type, text) => {
    setMessage({ type, text });
    setTimeout(() => setMessage({ type: '', text: '' }), 5000);
  };

  const handleExport = async (format) => {
    if (!selectedModel) {
      showMessage('error', 'Please select a model to export');
      return;
    }

    setExportLoading(true);
    try {
      const response = await axios.post(`${API}/model/export/${format}/${selectedModel}`, {
        metadata: {
          exported_by: 'Model Management UI',
          export_reason: 'User requested export'
        }
      });

      if (response.data.success) {
        showMessage('success', `Model exported successfully as ${format.toUpperCase()}`);
        await loadExports();
      }
    } catch (error) {
      console.error('Error exporting model:', error);
      showMessage('error', `Failed to export model: ${error.response?.data?.detail || error.message}`);
    } finally {
      setExportLoading(false);
    }
  };

  const handleLoadModel = async (modelName) => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/model/load/${modelName}`);
      if (response.data.success) {
        showMessage('success', `Model ${modelName} loaded successfully`);
        await loadCurrentModel();
      }
    } catch (error) {
      console.error('Error loading model:', error);
      showMessage('error', `Failed to load model: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = (filename) => {
    window.open(`${API}/model/download/${filename}`, '_blank');
    showMessage('success', 'Download started');
  };

  const formatDate = (dateString) => {
    if (!dateString || dateString === 'unknown') return 'Unknown';
    try {
      return new Date(dateString).toLocaleString();
    } catch {
      return dateString;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header with Device Info */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-5 w-5" />
                Model Export & Deployment
              </CardTitle>
              <CardDescription>
                Export, download, and manage trained AlphaZero models
              </CardDescription>
            </div>
            <Button 
              onClick={loadData} 
              variant="outline" 
              size="sm"
              disabled={loading}
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {deviceInfo && (
            <div className="flex items-center gap-6 text-sm">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-blue-500" />
                <span className="font-medium">Device:</span>
                <Badge variant={deviceInfo.is_gpu ? "default" : "secondary"}>
                  {deviceInfo.device_name}
                </Badge>
              </div>
              {deviceInfo.is_gpu && (
                <>
                  <div className="flex items-center gap-2">
                    <HardDrive className="h-4 w-4 text-green-500" />
                    <span className="font-medium">Memory:</span>
                    <span className="text-muted-foreground">
                      {deviceInfo.gpu_memory_allocated}
                    </span>
                  </div>
                </>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Messages */}
      {message.text && (
        <Alert variant={message.type === 'error' ? 'destructive' : 'default'}>
          {message.type === 'error' ? (
            <AlertCircle className="h-4 w-4" />
          ) : (
            <CheckCircle className="h-4 w-4" />
          )}
          <AlertDescription>{message.text}</AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="models" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="models">Available Models</TabsTrigger>
          <TabsTrigger value="export">Export Models</TabsTrigger>
          <TabsTrigger value="downloads">Downloads</TabsTrigger>
        </TabsList>

        {/* Available Models Tab */}
        <TabsContent value="models" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Trained Models</CardTitle>
              <CardDescription>
                View and load trained model versions
              </CardDescription>
            </CardHeader>
            <CardContent>
              {/* Current Model Display */}
              {currentModel && (
                <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-800">
                  <div className="flex items-center gap-2 mb-3">
                    <CheckCircle className="h-5 w-5 text-blue-600" />
                    <h3 className="font-semibold text-blue-900 dark:text-blue-100">
                      Currently Active Model
                    </h3>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">Name:</span>
                      <Badge variant="default">{currentModel.model_name}</Badge>
                    </div>
                    {currentModel.metadata?.version && (
                      <div className="flex items-center gap-2">
                        <span className="font-medium">Version:</span>
                        <span>{currentModel.metadata.version}</span>
                      </div>
                    )}
                    {currentModel.metadata?.training_date && (
                      <div className="flex items-center gap-2">
                        <Calendar className="h-4 w-4" />
                        <span className="font-medium">Trained:</span>
                        <span className="text-muted-foreground">
                          {formatDate(currentModel.metadata.training_date)}
                        </span>
                      </div>
                    )}
                    {currentModel.metadata?.win_rate && (
                      <div className="flex items-center gap-2">
                        <Activity className="h-4 w-4" />
                        <span className="font-medium">Win Rate:</span>
                        <span className="text-muted-foreground">
                          {(currentModel.metadata.win_rate * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              <Separator className="my-4" />

              {/* Models List */}
              <div className="space-y-3">
                {models.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    No models found. Train a model first.
                  </div>
                ) : (
                  models.map((model) => (
                    <div
                      key={model.name}
                      className={`p-4 border rounded-lg transition-all ${
                        model.is_current
                          ? 'border-blue-500 bg-blue-50 dark:bg-blue-950'
                          : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="space-y-2 flex-1">
                          <div className="flex items-center gap-2">
                            <h4 className="font-semibold">{model.name}</h4>
                            {model.is_current && (
                              <Badge variant="default">Active</Badge>
                            )}
                            {model.version !== 'unknown' && (
                              <Badge variant="outline">v{model.version}</Badge>
                            )}
                          </div>
                          <div className="text-sm text-muted-foreground space-y-1">
                            <div>Size: {model.file_size_mb} MB</div>
                            {model.training_date !== 'unknown' && (
                              <div>Trained: {formatDate(model.training_date)}</div>
                            )}
                          </div>
                        </div>
                        {!model.is_current && (
                          <Button
                            onClick={() => handleLoadModel(model.name)}
                            variant="outline"
                            size="sm"
                            disabled={loading}
                          >
                            <Upload className="h-4 w-4 mr-2" />
                            Load
                          </Button>
                        )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Export Tab */}
        <TabsContent value="export" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Export Model</CardTitle>
              <CardDescription>
                Export models in PyTorch (.pt) or ONNX (.onnx) format
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-3">
                <label className="text-sm font-medium">Select Model</label>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger>
                    <SelectValue placeholder="Choose a model" />
                  </SelectTrigger>
                  <SelectContent>
                    {models.map((model) => (
                      <SelectItem key={model.name} value={model.name}>
                        {model.name} {model.version !== 'unknown' && `(v${model.version})`}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <Separator />

              <div className="space-y-3">
                <h4 className="text-sm font-medium">Export Format</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card className="p-4">
                    <div className="space-y-3">
                      <h5 className="font-semibold">PyTorch (.pt)</h5>
                      <p className="text-sm text-muted-foreground">
                        Native PyTorch format with full model weights and metadata.
                        Best for Python/PyTorch environments.
                      </p>
                      <Button
                        onClick={() => handleExport('pytorch')}
                        disabled={exportLoading || !selectedModel}
                        className="w-full"
                        data-testid="export-pytorch-btn"
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Export as PyTorch
                      </Button>
                    </div>
                  </Card>

                  <Card className="p-4">
                    <div className="space-y-3">
                      <h5 className="font-semibold">ONNX (.onnx)</h5>
                      <p className="text-sm text-muted-foreground">
                        Open standard format for cross-platform deployment.
                        Compatible with multiple frameworks and languages.
                      </p>
                      <Button
                        onClick={() => handleExport('onnx')}
                        disabled={exportLoading || !selectedModel}
                        className="w-full"
                        variant="outline"
                        data-testid="export-onnx-btn"
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Export as ONNX
                      </Button>
                    </div>
                  </Card>
                </div>
              </div>

              {exportLoading && (
                <Alert>
                  <RefreshCw className="h-4 w-4 animate-spin" />
                  <AlertDescription>
                    Exporting model... This may take a moment.
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Downloads Tab */}
        <TabsContent value="downloads" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Exported Models</CardTitle>
              <CardDescription>
                Download previously exported models
              </CardDescription>
            </CardHeader>
            <CardContent>
              {exports.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  No exports available. Export a model first.
                </div>
              ) : (
                <div className="space-y-3">
                  {exports.map((exp, idx) => (
                    <div
                      key={idx}
                      className="p-4 border rounded-lg hover:border-gray-300 dark:hover:border-gray-600 transition-all"
                    >
                      <div className="flex items-center justify-between">
                        <div className="space-y-2 flex-1">
                          <div className="flex items-center gap-2">
                            <h4 className="font-semibold">{exp.filename}</h4>
                            <Badge variant={exp.format === 'pytorch' ? 'default' : 'secondary'}>
                              {exp.format.toUpperCase()}
                            </Badge>
                          </div>
                          <div className="text-sm text-muted-foreground space-y-1">
                            <div>Size: {exp.file_size_mb} MB</div>
                            <div>Exported: {formatDate(exp.export_date)}</div>
                            {exp.metadata?.device_used && (
                              <div>Device: {exp.metadata.device_used}</div>
                            )}
                          </div>
                        </div>
                        <Button
                          onClick={() => handleDownload(exp.filename)}
                          variant="outline"
                          size="sm"
                          data-testid={`download-${exp.filename}-btn`}
                        >
                          <Download className="h-4 w-4 mr-2" />
                          Download
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ModelManagement;
