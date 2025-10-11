import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Download, RefreshCw, CheckCircle, XCircle } from 'lucide-react';
import { toast } from 'sonner';

const UpdateNotification = () => {
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const [updateInfo, setUpdateInfo] = useState(null);
  const [downloading, setDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [updateReady, setUpdateReady] = useState(false);

  useEffect(() => {
    if (!window.electronAPI) return;

    // Listen for update events
    window.electronAPI.onUpdateAvailable((info) => {
      setUpdateAvailable(true);
      setUpdateInfo(info);
      toast.info(`Update available: v${info.version}`);
    });

    window.electronAPI.onDownloadProgress((progress) => {
      setDownloadProgress(progress.percent);
    });

    window.electronAPI.onUpdateDownloaded((info) => {
      setDownloading(false);
      setUpdateReady(true);
      toast.success(`Update v${info.version} ready to install!`);
    });

    window.electronAPI.onUpdateError((error) => {
      setDownloading(false);
      toast.error(`Update error: ${error}`);
    });
  }, []);

  const handleDownload = async () => {
    setDownloading(true);
    try {
      await window.electronAPI.downloadUpdate();
    } catch (error) {
      console.error('Download error:', error);
      toast.error('Failed to download update');
      setDownloading(false);
    }
  };

  const handleInstall = () => {
    window.electronAPI.installUpdate();
  };

  const handleCheckUpdates = async () => {
    try {
      const result = await window.electronAPI.checkForUpdates();
      if (result.success) {
        toast.info('Checking for updates...');
      } else {
        toast.info('No updates available');
      }
    } catch (error) {
      toast.error('Failed to check for updates');
    }
  };

  if (updateReady) {
    return (
      <div className="fixed bottom-4 right-4 z-50 max-w-md">
        <Card className="bg-green-900 border-green-700 shadow-lg">
          <CardContent className="p-4">
            <div className="flex items-start gap-3">
              <CheckCircle className="text-green-400 flex-shrink-0" size={24} />
              <div className="flex-1">
                <h3 className="text-white font-semibold mb-1">
                  Update Ready to Install
                </h3>
                <p className="text-green-200 text-sm mb-3">
                  Version {updateInfo?.version} has been downloaded and is ready to install.
                </p>
                <div className="flex gap-2">
                  <Button
                    onClick={handleInstall}
                    size="sm"
                    className="bg-green-600 hover:bg-green-700 text-white"
                  >
                    Restart & Install
                  </Button>
                  <Button
                    onClick={() => setUpdateReady(false)}
                    size="sm"
                    variant="outline"
                    className="border-green-600 text-green-300 hover:bg-green-800"
                  >
                    Later
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (downloading) {
    return (
      <div className="fixed bottom-4 right-4 z-50 max-w-md">
        <Card className="bg-blue-900 border-blue-700 shadow-lg">
          <CardContent className="p-4">
            <div className="flex items-start gap-3">
              <Download className="text-blue-400 flex-shrink-0 animate-bounce" size={24} />
              <div className="flex-1">
                <h3 className="text-white font-semibold mb-1">
                  Downloading Update
                </h3>
                <p className="text-blue-200 text-sm mb-2">
                  Version {updateInfo?.version}
                </p>
                <div className="w-full bg-blue-800 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${downloadProgress}%` }}
                  />
                </div>
                <p className="text-xs text-blue-300 mt-1">
                  {Math.round(downloadProgress)}%
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (updateAvailable) {
    return (
      <div className="fixed bottom-4 right-4 z-50 max-w-md">
        <Card className="bg-slate-800 border-slate-600 shadow-lg">
          <CardContent className="p-4">
            <div className="flex items-start gap-3">
              <RefreshCw className="text-blue-400 flex-shrink-0" size={24} />
              <div className="flex-1">
                <h3 className="text-white font-semibold mb-1">
                  Update Available
                </h3>
                <p className="text-slate-300 text-sm mb-3">
                  Version {updateInfo?.version} is now available. Would you like to download it?
                </p>
                <div className="flex gap-2">
                  <Button
                    onClick={handleDownload}
                    size="sm"
                    className="bg-blue-600 hover:bg-blue-700 text-white"
                  >
                    Download
                  </Button>
                  <Button
                    onClick={() => setUpdateAvailable(false)}
                    size="sm"
                    variant="outline"
                    className="border-slate-600 text-slate-300 hover:bg-slate-700"
                  >
                    Skip
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Manual check button (always visible in desktop mode)
  if (window.electronAPI) {
    return (
      <Button
        onClick={handleCheckUpdates}
        variant="ghost"
        size="sm"
        className="fixed bottom-4 right-4 text-slate-400 hover:text-white"
        title="Check for updates"
      >
        <RefreshCw size={16} />
      </Button>
    );
  }

  return null;
};

export default UpdateNotification;
