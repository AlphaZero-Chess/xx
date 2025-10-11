import React from 'react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Download, Trash2, Circle, CheckCircle } from 'lucide-react';
import { toast } from 'sonner';

const ModelCardList = ({ models, activeModel, onActivate, onExport, onDelete }) => {
  if (!models || models.length === 0) {
    return (
      <div className="text-center py-8 text-slate-400">
        <p>No models available yet.</p>
        <p className="text-xs mt-2">Train a model to get started!</p>
      </div>
    );
  }

  const getEloBadgeColor = (elo) => {
    if (elo >= 1600) return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
    if (elo >= 1500) return 'bg-green-500/20 text-green-400 border-green-500/30';
    if (elo >= 1400) return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
    return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
  };

  const formatDate = (timestamp) => {
    if (!timestamp || timestamp === 'unknown') return 'Unknown';
    try {
      return new Date(timestamp).toLocaleString();
    } catch {
      return timestamp;
    }
  };

  return (
    <div className="space-y-2 max-h-60 overflow-y-auto" data-testid="model-card-list">
      {models.map((model) => {
        const isActive = model.name === activeModel || model.active;
        
        return (
          <div 
            key={model.name} 
            className={`p-3 rounded-lg border transition-all ${
              isActive 
                ? 'bg-emerald-900/20 border-emerald-500/50' 
                : 'bg-slate-700/50 border-slate-600 hover:border-slate-500'
            }`}
            data-testid={`model-card-${model.name}`}
          >
            {/* Model Header */}
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                {isActive ? (
                  <CheckCircle className="h-4 w-4 text-emerald-400" data-testid="active-badge" />
                ) : (
                  <Circle className="h-4 w-4 text-slate-500" />
                )}
                <span className={`text-sm font-medium truncate ${isActive ? 'text-emerald-300' : 'text-slate-300'}`}>
                  {model.name}
                </span>
              </div>
              
              <div className="flex items-center gap-1">
                {isActive && (
                  <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30 text-xs">
                    🟢 Active
                  </Badge>
                )}
                <Badge className={`${getEloBadgeColor(model.elo)} text-xs`} data-testid={`elo-badge-${model.name}`}>
                  ELO {model.elo}
                </Badge>
              </div>
            </div>

            {/* Model Info */}
            <div className="text-xs text-slate-400 mb-2 space-y-1">
              {model.timestamp && model.timestamp !== 'unknown' && (
                <div>Trained: {formatDate(model.timestamp)}</div>
              )}
              {model.file_size_mb && (
                <div>Size: {model.file_size_mb} MB</div>
              )}
            </div>

            {/* Action Buttons */}
            <div className="flex gap-1">
              {!isActive && (
                <Button
                  onClick={() => onActivate(model.name)}
                  variant="outline"
                  size="sm"
                  className="flex-1 text-xs border-slate-600 text-slate-300 hover:bg-slate-600"
                  data-testid={`activate-model-${model.name}`}
                >
                  Activate
                </Button>
              )}
              
              <Button
                onClick={() => onExport(model.name)}
                variant="outline"
                size="sm"
                className={`${isActive ? 'flex-1' : ''} px-2 border-slate-600 text-slate-300 hover:bg-slate-600`}
                data-testid={`export-model-${model.name}`}
                title="Export/Download"
              >
                <Download className="h-3 w-3" />
              </Button>
              
              <Button
                onClick={() => onDelete(model.name)}
                variant="outline"
                size="sm"
                className="px-2 border-red-600 text-red-400 hover:bg-red-900"
                data-testid={`delete-model-${model.name}`}
                title="Delete"
                disabled={isActive}
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default ModelCardList;
