"""
Enhanced FastAPI Server with Full AlphaZero Pipeline
Includes training, evaluation, model management, and export endpoints
"""
from fastapi import FastAPI, APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys

# Add backend to path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Import AlphaZero components
from neural_network import AlphaZeroNetwork, ModelManager
from self_play import SelfPlayManager
from trainer import AlphaZeroTrainer
from evaluator import ModelEvaluator
from model_export import ModelExporter, ModelLoader

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=2)

# Global state
training_status = {
    "active": False,
    "progress": 0,
    "message": "",
    "session_id": None,
    "start_time": None
}

evaluation_status = {
    "active": False,
    "progress": 0,
    "message": "",
    "results": None
}

# Model manager instances
model_manager = ModelManager()
model_exporter = ModelExporter()
model_loader = ModelLoader()

# ====================
# Data Models
# ====================

class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class TrainingConfig(BaseModel):
    num_games: int = Field(default=1, ge=1, le=100)
    num_epochs: int = Field(default=1, ge=1, le=50)
    batch_size: int = Field(default=64, ge=8, le=256)
    num_simulations: int = Field(default=10, ge=5, le=1000)
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1)

class EvaluationConfig(BaseModel):
    challenger_name: str
    champion_name: str
    num_games: int = Field(default=3, ge=1, le=50)
    num_simulations: int = Field(default=10, ge=5, le=1000)
    win_threshold: float = Field(default=0.55, ge=0.5, le=1.0)

class ExportRequest(BaseModel):
    metadata: Optional[Dict[str, Any]] = None

# ====================
# Basic Endpoints
# ====================

@api_router.get("/")
async def root():
    return {"message": "AlphaZero Chess API", "version": "1.0", "status": "active"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# ====================
# Training Endpoints
# ====================

def run_training_pipeline(config: TrainingConfig, session_id: str):
    """Run complete training pipeline in background"""
    try:
        global training_status
        training_status["message"] = "Initializing neural network..."
        training_status["progress"] = 5
        
        # Initialize network
        network = AlphaZeroNetwork()
        
        # Stage 1: Self-Play
        training_status["message"] = f"Generating self-play games ({config.num_games} games)..."
        training_status["progress"] = 10
        logger.info(f"Starting self-play: {config.num_games} games, {config.num_simulations} simulations")
        
        self_play_manager = SelfPlayManager(network, num_simulations=config.num_simulations)
        training_data, game_results = self_play_manager.generate_games(
            num_games=config.num_games,
            store_fen=True
        )
        
        logger.info(f"Self-play complete: {len(training_data)} positions generated")
        training_status["message"] = f"Self-play complete: {len(training_data)} positions"
        training_status["progress"] = 40
        
        # Store self-play data in MongoDB
        asyncio.run(db.self_play_positions.insert_many([
            {
                **pos,
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc),
                "source": "pipeline_test"
            }
            for pos in training_data[:100]  # Store sample
        ]))
        
        # Stage 2: Training
        training_status["message"] = f"Training network ({config.num_epochs} epochs)..."
        training_status["progress"] = 50
        logger.info(f"Starting training: {config.num_epochs} epochs, batch size {config.batch_size}")
        
        trainer = AlphaZeroTrainer(network, learning_rate=config.learning_rate)
        training_history = trainer.train(
            training_data,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size
        )
        
        logger.info(f"Training complete: {len(training_history)} epochs")
        training_status["message"] = "Training complete, saving model..."
        training_status["progress"] = 75
        
        # Store training metrics
        asyncio.run(db.training_metrics.insert_many([
            {
                **metrics,
                "session_id": session_id,
                "source": "pipeline_test"
            }
            for metrics in training_history
        ]))
        
        # Save model with version
        metadata = {
            "training_date": datetime.now(timezone.utc).isoformat(),
            "num_games": config.num_games,
            "num_epochs": config.num_epochs,
            "num_positions": len(training_data),
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_simulations": config.num_simulations,
            "training_session_id": session_id,
            "final_loss": training_history[-1]["loss"] if training_history else 0.0,
            "source": "pipeline_test"
        }
        
        model_path = model_manager.save_versioned_model(network, metadata=metadata)
        new_model_name = Path(model_path).stem
        logger.info(f"Model saved: {new_model_name}")
        
        training_status["message"] = f"Model saved: {new_model_name}"
        training_status["progress"] = 90
        
        # Stage 3: Evaluation (if there's an existing model)
        existing_models = model_manager.list_models()
        if len(existing_models) > 1:
            training_status["message"] = "Running evaluation..."
            logger.info("Starting model evaluation")
            
            # Get current active model
            active_model_doc = asyncio.run(db.active_model.find_one({}))
            champion_name = active_model_doc["model_name"] if active_model_doc else existing_models[0]
            
            # Load models for evaluation
            challenger_model, _ = model_manager.load_model(new_model_name)
            champion_model, _ = model_manager.load_model(champion_name)
            
            # Run evaluation (3 games, 10 simulations as per requirements)
            evaluator = ModelEvaluator(
                num_evaluation_games=3,
                num_simulations=10,
                win_threshold=0.55
            )
            
            eval_results, should_promote = evaluator.evaluate_models(
                challenger_model,
                champion_model,
                new_model_name,
                champion_name
            )
            
            # Store evaluation results
            asyncio.run(db.model_evaluations.insert_one({
                **eval_results,
                "timestamp": datetime.now(timezone.utc),
                "automatic": True,
                "session_id": session_id,
                "source": "pipeline_test"
            }))
            
            # Promote if threshold met
            if should_promote:
                asyncio.run(db.active_model.replace_one(
                    {},
                    {
                        "model_name": new_model_name,
                        "promoted_at": datetime.now(timezone.utc),
                        "win_rate": eval_results["challenger_win_rate"],
                        "previous_champion": champion_name,
                        "manual_activation": False
                    },
                    upsert=True
                ))
                logger.info(f"Model promoted: {new_model_name}")
                training_status["message"] = f"Model promoted! Win rate: {eval_results['challenger_win_rate']:.1%}"
            else:
                logger.info(f"Model not promoted. Win rate: {eval_results['challenger_win_rate']:.1%}")
                training_status["message"] = f"Model not promoted. Win rate: {eval_results['challenger_win_rate']:.1%}"
        else:
            # First model - make it active
            asyncio.run(db.active_model.replace_one(
                {},
                {
                    "model_name": new_model_name,
                    "promoted_at": datetime.now(timezone.utc),
                    "win_rate": 1.0,
                    "previous_champion": None,
                    "manual_activation": False
                },
                upsert=True
            ))
        
        training_status["progress"] = 100
        training_status["message"] = "Training pipeline complete!"
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training pipeline error: {str(e)}")
        training_status["message"] = f"Error: {str(e)}"
        training_status["progress"] = 0
        raise
    finally:
        training_status["active"] = False

@api_router.post("/training/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start training pipeline in background"""
    global training_status
    
    if training_status["active"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    session_id = str(uuid.uuid4())
    training_status = {
        "active": True,
        "progress": 0,
        "message": "Starting training...",
        "session_id": session_id,
        "start_time": datetime.now(timezone.utc)
    }
    
    # Run in background
    background_tasks.add_task(run_training_pipeline, config, session_id)
    
    return {
        "success": True,
        "message": "Training started",
        "session_id": session_id,
        "config": config.dict()
    }

@api_router.get("/training/status")
async def get_training_status():
    """Get current training status"""
    return {
        "active": training_status["active"],
        "progress": training_status["progress"],
        "message": training_status["message"],
        "session_id": training_status.get("session_id"),
        "start_time": training_status.get("start_time")
    }

@api_router.get("/training/history")
async def get_training_history(limit: int = 10):
    """Get training history from database"""
    sessions = await db.training_metrics.aggregate([
        {"$group": {
            "_id": "$session_id",
            "epochs": {"$sum": 1},
            "avg_loss": {"$avg": "$loss"},
            "timestamp": {"$first": "$timestamp"}
        }},
        {"$sort": {"timestamp": -1}},
        {"$limit": limit}
    ]).to_list(limit)
    
    return {"sessions": sessions}

# ====================
# Evaluation Endpoints
# ====================

def run_evaluation_pipeline(config: EvaluationConfig):
    """Run evaluation in background"""
    try:
        global evaluation_status
        evaluation_status["message"] = "Loading models..."
        evaluation_status["progress"] = 10
        
        # Load models
        challenger_model, challenger_meta = model_manager.load_model(config.challenger_name)
        champion_model, champion_meta = model_manager.load_model(config.champion_name)
        
        if not challenger_model or not champion_model:
            raise Exception("Failed to load one or both models")
        
        evaluation_status["message"] = f"Running evaluation ({config.num_games} games)..."
        evaluation_status["progress"] = 30
        
        # Run evaluation
        evaluator = ModelEvaluator(
            num_evaluation_games=config.num_games,
            num_simulations=config.num_simulations,
            win_threshold=config.win_threshold
        )
        
        results, should_promote = evaluator.evaluate_models(
            challenger_model,
            champion_model,
            config.challenger_name,
            config.champion_name
        )
        
        evaluation_status["progress"] = 80
        evaluation_status["message"] = "Storing results..."
        
        # Store results
        asyncio.run(db.model_evaluations.insert_one({
            **results,
            "timestamp": datetime.now(timezone.utc),
            "automatic": False,
            "source": "pipeline_test"
        }))
        
        evaluation_status["progress"] = 100
        evaluation_status["message"] = "Evaluation complete!"
        evaluation_status["results"] = results
        
        logger.info(f"Evaluation complete: {results['challenger_win_rate']:.1%} win rate")
        
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        evaluation_status["message"] = f"Error: {str(e)}"
        evaluation_status["progress"] = 0
        raise
    finally:
        evaluation_status["active"] = False

@api_router.post("/evaluation/run")
async def run_evaluation(config: EvaluationConfig, background_tasks: BackgroundTasks):
    """Run evaluation between two models"""
    global evaluation_status
    
    if evaluation_status["active"]:
        raise HTTPException(status_code=400, detail="Evaluation already in progress")
    
    evaluation_status = {
        "active": True,
        "progress": 0,
        "message": "Starting evaluation...",
        "results": None
    }
    
    background_tasks.add_task(run_evaluation_pipeline, config)
    
    return {
        "success": True,
        "message": "Evaluation started",
        "config": config.dict()
    }

@api_router.get("/evaluation/status")
async def get_evaluation_status():
    """Get current evaluation status"""
    return evaluation_status

@api_router.get("/evaluation/history")
async def get_evaluation_history(limit: int = 10):
    """Get evaluation history"""
    evaluations = await db.model_evaluations.find().sort("timestamp", -1).limit(limit).to_list(limit)
    return {"evaluations": evaluations}

# ====================
# Model Management Endpoints
# ====================

@api_router.get("/model/list")
async def list_models():
    """List all available models"""
    models = model_manager.list_models()
    
    # Get active model
    active_model_doc = await db.active_model.find_one({})
    active_model_name = active_model_doc["model_name"] if active_model_doc else None
    
    model_list = []
    for model_name in models:
        model_info = model_manager.get_model_info(model_name)
        if model_info:
            metadata = model_info.get("metadata", {})
            model_path = model_manager.get_model_path(model_name)
            file_size = os.path.getsize(model_path) / (1024 * 1024) if model_path.exists() else 0
            
            model_list.append({
                "name": model_name,
                "version": metadata.get("version", "unknown"),
                "training_date": metadata.get("training_date", "unknown"),
                "file_size_mb": round(file_size, 2),
                "metadata": metadata,
                "is_active": model_name == active_model_name
            })
    
    return {"success": True, "models": model_list, "count": len(model_list)}

@api_router.get("/model/current")
async def get_current_model():
    """Get current active model"""
    active_model_doc = await db.active_model.find_one({})
    return {
        "success": True,
        "active_model": active_model_doc["model_name"] if active_model_doc else None,
        "details": active_model_doc
    }

@api_router.post("/model/activate/{model_name}")
async def activate_model(model_name: str):
    """Activate a specific model"""
    # Verify model exists
    model_info = model_manager.get_model_info(model_name)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    # Update active model
    await db.active_model.replace_one(
        {},
        {
            "model_name": model_name,
            "promoted_at": datetime.now(timezone.utc),
            "manual_activation": True
        },
        upsert=True
    )
    
    return {"success": True, "message": f"Model {model_name} activated"}

@api_router.get("/model/info/{model_name}")
async def get_model_info(model_name: str):
    """Get detailed model information"""
    model_info = model_manager.get_model_info(model_name)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    return {"success": True, "model_info": model_info}

# ====================
# Model Export Endpoints
# ====================

@api_router.post("/model/export/{format}/{model_name}")
async def export_model(format: str, model_name: str, request: ExportRequest = ExportRequest()):
    """Export model in specified format"""
    if format not in ["pytorch", "onnx"]:
        raise HTTPException(status_code=400, detail="Format must be 'pytorch' or 'onnx'")
    
    try:
        if format == "pytorch":
            result = await asyncio.get_event_loop().run_in_executor(
                executor,
                model_exporter.export_pytorch,
                model_name,
                request.metadata
            )
        else:
            result = await asyncio.get_event_loop().run_in_executor(
                executor,
                model_exporter.export_onnx,
                model_name,
                request.metadata
            )
        
        return {
            "success": True,
            "message": f"Model exported successfully to {format} format",
            **result
        }
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/model/exports")
async def list_exports():
    """List all exported models"""
    exports = model_exporter.list_exports()
    return {"success": True, "exports": exports, "count": len(exports)}

@api_router.get("/model/download/{filename}")
async def download_export(filename: str):
    """Download an exported model file"""
    file_path = model_exporter.get_export_path(filename)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream"
    )

# ====================
# Statistics Endpoint
# ====================

@api_router.get("/stats")
async def get_stats():
    """Get overall statistics"""
    total_games = await db.self_play_positions.count_documents({})
    total_epochs = await db.training_metrics.count_documents({})
    total_evaluations = await db.model_evaluations.count_documents({})
    
    active_model_doc = await db.active_model.find_one({})
    
    return {
        "total_self_play_positions": total_games,
        "total_training_epochs": total_epochs,
        "total_evaluations": total_evaluations,
        "total_models": len(model_manager.list_models()),
        "active_model": active_model_doc["model_name"] if active_model_doc else None,
        "training_active": training_status["active"],
        "evaluation_active": evaluation_status["active"]
    }

# ====================
# Device Info
# ====================

@api_router.get("/device/info")
async def get_device_info():
    """Get device information"""
    from device_manager import device_manager
    
    return {
        "success": True,
        "device_type": str(device_manager.device),
        "device_name": device_manager.device_name,
        "is_gpu": device_manager.device.type == "cuda"
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
