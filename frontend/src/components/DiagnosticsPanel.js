import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Alert, AlertDescription } from './ui/alert';
import { FileText, Download, RefreshCw, CheckCircle, AlertTriangle, XCircle } from 'lucide-react';
import axios from 'axios';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const DiagnosticsPanel = () => {
  const [reports, setReports] = useState([]);
  const [selectedReport, setSelectedReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [validationStatus, setValidationStatus] = useState(null);

  useEffect(() => {
    loadDiagnostics();
    fetchValidationStatus();
  }, []);

  const loadDiagnostics = async () => {
    try {
      setLoading(true);
      if (window.electronAPI) {
        const data = await window.electronAPI.getDiagnostics();
        setReports(data);
        if (data.length > 0) {
          setSelectedReport(data[0]);
        }
      }
    } catch (error) {
      console.error('Failed to load diagnostics:', error);
      toast.error('Failed to load diagnostics');
    } finally {
      setLoading(false);
    }
  };

  const fetchValidationStatus = async () => {
    try {
      const response = await axios.get(`${API}/casv1/status`);
      setValidationStatus(response.data);
    } catch (error) {
      console.error('Failed to fetch validation status:', error);
    }
  };

  const runValidation = async () => {
    try {
      toast.info('Running CASV-1 validation...');
      const response = await axios.post(`${API}/casv1/validate`);
      toast.success('Validation complete!');
      await fetchValidationStatus();
      await loadDiagnostics();
    } catch (error) {
      console.error('Validation failed:', error);
      toast.error('Validation failed: ' + error.message);
    }
  };

  const downloadReport = (report) => {
    const blob = new Blob([report.content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = report.filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success('Report downloaded');
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'passed':
      case 'success':
        return <CheckCircle className="text-green-500" size={20} />;
      case 'warning':
        return <AlertTriangle className="text-yellow-500" size={20} />;
      case 'failed':
      case 'error':
        return <XCircle className="text-red-500" size={20} />;
      default:
        return <FileText className="text-blue-500" size={20} />;
    }
  };

  if (loading) {
    return (
      <div className="p-8 text-center text-slate-300">
        <RefreshCw className="animate-spin mx-auto mb-2" size={24} />
        Loading diagnostics...
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="diagnostics-panel">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">CASV-1 Diagnostics</h2>
          <p className="text-slate-400">System validation and compliance reports</p>
        </div>
        <div className="flex gap-2">
          <Button onClick={loadDiagnostics} variant="outline" className="border-slate-700">
            <RefreshCw size={16} className="mr-2" />
            Refresh
          </Button>
          <Button onClick={runValidation} className="bg-blue-600 hover:bg-blue-700">
            Run Validation
          </Button>
        </div>
      </div>

      {/* Validation Status Card */}
      {validationStatus && (
        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-white">
              {getStatusIcon(validationStatus.overall_status)}
              Overall System Status: {validationStatus.overall_status?.toUpperCase()}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-slate-900 p-3 rounded-lg">
                <p className="text-xs text-slate-400 mb-1">Last Check</p>
                <p className="text-white font-semibold">
                  {validationStatus.last_check ? new Date(validationStatus.last_check).toLocaleDateString() : 'N/A'}
                </p>
              </div>
              <div className="bg-slate-900 p-3 rounded-lg">
                <p className="text-xs text-slate-400 mb-1">Tests Passed</p>
                <p className="text-green-400 font-semibold">{validationStatus.tests_passed || 0}</p>
              </div>
              <div className="bg-slate-900 p-3 rounded-lg">
                <p className="text-xs text-slate-400 mb-1">Warnings</p>
                <p className="text-yellow-400 font-semibold">{validationStatus.warnings || 0}</p>
              </div>
              <div className="bg-slate-900 p-3 rounded-lg">
                <p className="text-xs text-slate-400 mb-1">Failures</p>
                <p className="text-red-400 font-semibold">{validationStatus.failures || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {reports.length === 0 ? (
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="p-12 text-center">
            <FileText className="mx-auto mb-4 text-slate-600" size={48} />
            <h3 className="text-xl text-slate-300 mb-2">No Diagnostic Reports</h3>
            <p className="text-slate-400 mb-4">Run a validation to generate your first report</p>
            <Button onClick={runValidation} className="bg-blue-600 hover:bg-blue-700">
              Run CASV-1 Validation
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Reports List */}
          <Card className="lg:col-span-1 bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Reports ({reports.length})</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 max-h-96 overflow-y-auto">
              {reports.map((report, index) => (
                <div
                  key={index}
                  onClick={() => setSelectedReport(report)}
                  className={`p-3 rounded-lg cursor-pointer transition-colors ${
                    selectedReport?.filename === report.filename
                      ? 'bg-blue-600/20 border-blue-500/50'
                      : 'bg-slate-900 hover:bg-slate-700'
                  } border`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <p className="text-white text-sm font-medium truncate">{report.filename}</p>
                      <p className="text-xs text-slate-400 mt-1">
                        {(report.content.length / 1024).toFixed(1)} KB
                      </p>
                    </div>
                    <FileText size={16} className="text-slate-400 ml-2" />
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Report Viewer */}
          <Card className="lg:col-span-2 bg-slate-800/50 border-slate-700">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-white">
                  {selectedReport?.filename || 'Select a report'}
                </CardTitle>
                {selectedReport && (
                  <Button
                    onClick={() => downloadReport(selectedReport)}
                    variant="outline"
                    size="sm"
                    className="border-slate-700"
                  >
                    <Download size={14} className="mr-2" />
                    Download
                  </Button>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {selectedReport ? (
                <div className="bg-slate-900 p-4 rounded-lg max-h-96 overflow-y-auto">
                  <pre className="text-sm text-slate-300 whitespace-pre-wrap font-mono">
                    {selectedReport.content}
                  </pre>
                </div>
              ) : (
                <div className="text-center py-12 text-slate-400">
                  <FileText size={48} className="mx-auto mb-4 text-slate-600" />
                  <p>Select a report to view its contents</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Information Alert */}
      <Alert>
        <AlertDescription className="text-sm text-slate-300">
          <strong>CASV-1 (Comprehensive AI System Validation v1.0)</strong> validates the integrity of all AlphaZero 
          subsystems including training pipeline, ethical governance, cognitive systems, and LLM integrations.
        </AlertDescription>
      </Alert>
    </div>
  );
};

export default DiagnosticsPanel;
