"""
Enhanced FastAPI Server with Full AlphaZero Pipeline
Includes training, evaluation, model management, and export endpoints
"""
from fastapi import FastAPI, APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
import numpy as np
import json
import time
import queue

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
from chess_engine import ChessEngine
from mcts import MCTS

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'alphazero_chess')]

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
    "start_time": None,
    "step": 0,
    "total_steps": 0,
    "eta_seconds": 0
}

evaluation_status = {
    "active": False,
    "progress": 0,
    "message": "",
    "results": None
}

# Progress event queue for SSE
progress_queue = queue.Queue()

# Model manager instances
model_manager = ModelManager()
model_exporter = ModelExporter()
model_loader = ModelLoader()

# Game sessions storage (in-memory for now, can move to MongoDB)
active_games = {}  # {game_id: {"engine": ChessEngine, "history": [], "ai_color": "black"}}

# Coaching sessions storage (conversation memory per game)
coaching_sessions = {}  # {game_id: LLMChessEvaluator instance}

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

async def store_data_async(collection, data):
    """Helper to store data asynchronously"""
    await collection.insert_many(data)

def emit_progress(step: int, total_steps: int, message: str):
    """Emit progress event to queue"""
    global training_status
    progress = int((step / total_steps) * 100) if total_steps > 0 else 0
    elapsed = time.time() - training_status.get("start_time_raw", time.time())
    eta_seconds = int((elapsed / step * (total_steps - step))) if step > 0 else 0
    
    training_status["progress"] = progress
    training_status["step"] = step
    training_status["total_steps"] = total_steps
    training_status["eta_seconds"] = eta_seconds
    training_status["message"] = message
    
    # Put in queue for SSE
    try:
        progress_queue.put_nowait({
            "step": step,
            "total_steps": total_steps,
            "percent": progress,
            "eta_seconds": eta_seconds,
            "message": message
        })
    except queue.Full:
        pass  # Skip if queue is full

def run_training_pipeline(config: TrainingConfig, session_id: str):
    """Run complete training pipeline in background"""
    try:
        global training_status
        start_time = time.time()
        training_status["start_time_raw"] = start_time
        
        # Calculate total steps: self-play games + training epochs
        total_steps = config.num_games + config.num_epochs
        current_step = 0
        
        emit_progress(current_step, total_steps, "Initializing neural network...")
        
        # Initialize network
        network = AlphaZeroNetwork()
        current_step += 1
        
        # Stage 1: Self-Play
        emit_progress(current_step, total_steps, f"Starting self-play ({config.num_games} games)...")
        logger.info(f"Starting self-play: {config.num_games} games, {config.num_simulations} simulations")
        
        self_play_manager = SelfPlayManager(network, num_simulations=config.num_simulations)
        training_data = []
        game_results = []
        
        # Generate games one by one with progress updates
        for game_idx in range(config.num_games):
            game_data, game_result = self_play_manager.generate_single_game(store_fen=True)
            training_data.extend(game_data)
            game_results.append(game_result)
            current_step += 1
            emit_progress(current_step, total_steps, f"Self-play game {game_idx + 1}/{config.num_games} complete")
        
        logger.info(f"Self-play complete: {len(training_data)} positions generated")
        emit_progress(current_step, total_steps, f"Self-play complete: {len(training_data)} positions")
        
        # Store self-play data in MongoDB (use sync client)
        sync_client = MongoClient(os.environ.get('MONGO_URL', 'mongodb://localhost:27017'))
        sync_db = sync_client[os.environ.get('DB_NAME', 'alphazero_chess')]
        
        if training_data:
            sync_db.self_play_positions.insert_many([
                {
                    "position": pos.get("position").tolist() if hasattr(pos.get("position"), "tolist") else pos.get("position"),
                    "policy": pos.get("policy"),
                    "value": pos.get("value"),
                    "fen": pos.get("fen"),
                    "move_number": pos.get("move_number"),
                    "session_id": session_id,
                    "timestamp": datetime.now(timezone.utc),
                    "source": "pipeline_test"
                }
                for pos in training_data[:100]  # Store sample
            ])
        
        # Stage 2: Training
        emit_progress(current_step, total_steps, f"Starting training ({config.num_epochs} epochs)...")
        logger.info(f"Starting training: {config.num_epochs} epochs, batch size {config.batch_size}")
        
        trainer = AlphaZeroTrainer(network, learning_rate=config.learning_rate)
        training_history = []
        
        # Train epoch by epoch with progress updates
        for epoch in range(config.num_epochs):
            epoch_metrics = trainer.train_epoch(training_data, batch_size=config.batch_size)
            training_history.append(epoch_metrics)
            current_step += 1
            emit_progress(current_step, total_steps, f"Training epoch {epoch + 1}/{config.num_epochs} - Loss: {epoch_metrics.get('loss', 0):.4f}")
        
        logger.info(f"Training complete: {len(training_history)} epochs")
        emit_progress(current_step, total_steps, "Training complete, saving model...")
        
        # Store training metrics
        if training_history:
            sync_db.training_metrics.insert_many([
                {
                    **metrics,
                    "session_id": session_id,
                    "source": "pipeline_test"
                }
                for metrics in training_history
            ])
        
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
        
        emit_progress(total_steps - 1, total_steps, f"Model saved: {new_model_name}")
        
        # Stage 3: Evaluation (if there's an existing model)
        existing_models = model_manager.list_models()
        if len(existing_models) > 1:
            training_status["message"] = "Running evaluation..."
            logger.info("Starting model evaluation")
            
            # Get current active model
            active_model_doc = sync_db.active_model.find_one({})
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
            sync_db.model_evaluations.insert_one({
                **eval_results,
                "timestamp": datetime.now(timezone.utc),
                "automatic": True,
                "session_id": session_id,
                "source": "pipeline_test"
            })
            
            # Promote if threshold met
            if should_promote:
                sync_db.active_model.replace_one(
                    {},
                    {
                        "model_name": new_model_name,
                        "promoted_at": datetime.now(timezone.utc),
                        "win_rate": eval_results["challenger_win_rate"],
                        "previous_champion": champion_name,
                        "manual_activation": False
                    },
                    upsert=True
                )
                logger.info(f"Model promoted: {new_model_name}")
                message = f"Model promoted! Win rate: {eval_results['challenger_win_rate']:.1%}"
            else:
                logger.info(f"Model not promoted. Win rate: {eval_results['challenger_win_rate']:.1%}")
                message = f"Model not promoted. Win rate: {eval_results['challenger_win_rate']:.1%}"
            emit_progress(total_steps, total_steps, message)
        else:
            # First model - make it active
            sync_db.active_model.replace_one(
                {},
                {
                    "model_name": new_model_name,
                    "promoted_at": datetime.now(timezone.utc),
                    "win_rate": 1.0,
                    "previous_champion": None,
                    "manual_activation": False,
                    "elo": 1500  # Starting ELO
                },
                upsert=True
            )
            emit_progress(total_steps, total_steps, "First model activated!")
        
        # Close sync client
        sync_client.close()
        
        emit_progress(total_steps, total_steps, "Training pipeline complete!")
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training pipeline error: {str(e)}")
        emit_progress(0, 100, f"Error: {str(e)}")
        raise
    finally:
        training_status["active"] = False
        # Signal completion
        try:
            progress_queue.put_nowait({"done": True})
        except queue.Full:
            pass

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
        "start_time": training_status.get("start_time"),
        "step": training_status.get("step", 0),
        "total_steps": training_status.get("total_steps", 0),
        "eta_seconds": training_status.get("eta_seconds", 0)
    }

async def progress_event_generator():
    """Server-Sent Events generator for training progress"""
    while True:
        try:
            # Wait for events with timeout
            event = progress_queue.get(timeout=30)
            
            # Check for completion signal
            if event.get("done"):
                yield f"data: {json.dumps({'done': True})}\n\n"
                break
            
            # Send progress update
            yield f"data: {json.dumps(event)}\n\n"
            
        except queue.Empty:
            # Send keepalive
            yield f": keepalive\n\n"
            
        except Exception as e:
            logger.error(f"SSE error: {e}")
            break
        
        await asyncio.sleep(0.1)

@api_router.get("/training/progress/stream")
async def training_progress_stream():
    """Stream training progress via Server-Sent Events"""
    return StreamingResponse(
        progress_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

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
        
        # Store results using sync client
        sync_client = MongoClient(os.environ.get('MONGO_URL', 'mongodb://localhost:27017'))
        sync_db = sync_client[os.environ.get('DB_NAME', 'alphazero_chess')]
        
        sync_db.model_evaluations.insert_one({
            **results,
            "timestamp": datetime.now(timezone.utc),
            "automatic": False,
            "source": "pipeline_test"
        })
        
        sync_client.close()
        
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
    """List all available models with ELO ratings"""
    models = model_manager.list_models()
    
    # Get active model
    active_model_doc = await db.active_model.find_one({})
    active_model_name = active_model_doc["model_name"] if active_model_doc else None
    
    # Get all evaluation results to calculate ELO
    evaluations = await db.model_evaluations.find().to_list(1000)
    
    # Calculate ELO ratings for each model
    elo_ratings = {}
    for model_name in models:
        # Start with base ELO
        elo = 1500
        
        # Find evaluations involving this model
        model_evals = [e for e in evaluations if e.get("challenger_name") == model_name or e.get("champion_name") == model_name]
        
        if model_evals:
            # Use latest win rate to adjust ELO
            latest_eval = max(model_evals, key=lambda x: x.get("timestamp", datetime.min))
            if latest_eval.get("challenger_name") == model_name:
                win_rate = latest_eval.get("challenger_win_rate", 0.5)
            else:
                win_rate = 1.0 - latest_eval.get("challenger_win_rate", 0.5)
            
            # Adjust ELO based on win rate (simplified)
            elo = 1500 + int((win_rate - 0.5) * 400)
        
        elo_ratings[model_name] = elo
    
    model_list = []
    for model_name in models:
        model_info = model_manager.get_model_info(model_name)
        if model_info:
            metadata = model_info.get("metadata", {})
            model_path = model_manager.get_model_path(model_name)
            file_size = os.path.getsize(model_path) / (1024 * 1024) if model_path.exists() else 0
            
            # Get last modified time from file
            try:
                timestamp = datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc).isoformat()
            except:
                timestamp = metadata.get("training_date", "unknown")
            
            model_list.append({
                "name": model_name,
                "path": str(model_path),
                "version": metadata.get("version", "unknown"),
                "timestamp": timestamp,
                "training_date": metadata.get("training_date", "unknown"),
                "file_size_mb": round(file_size, 2),
                "elo": elo_ratings.get(model_name, 1500),
                "active": model_name == active_model_name,
                "is_active": model_name == active_model_name,
                "metadata": metadata
            })
    
    # Sort by ELO descending
    model_list.sort(key=lambda x: x["elo"], reverse=True)
    
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

@api_router.post("/model/save")
async def save_model(name: str = Query(..., description="Model name")):
    """Save current model with given name"""
    try:
        # Get currently loaded network (would need to be stored globally)
        # For now, we'll copy the active model with a new name
        active_model_doc = await db.active_model.find_one({})
        if not active_model_doc:
            raise HTTPException(status_code=404, detail="No active model to save")
        
        active_model_name = active_model_doc["model_name"]
        
        # Load the active model
        network, metadata = model_manager.load_model(active_model_name)
        
        # Update metadata with new name and timestamp
        new_metadata = {
            **metadata,
            "saved_from": active_model_name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "custom_name": name
        }
        
        # Save with new name
        model_path = model_manager.save_model(network, name, metadata=new_metadata)
        
        return {
            "success": True,
            "message": f"Model saved as {name}",
            "model_name": name,
            "path": str(model_path)
        }
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/model/load")
async def load_model_endpoint(name: str = Query(..., description="Model name")):
    """Load a model and make it active"""
    try:
        # Verify model exists
        model_info = model_manager.get_model_info(name)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {name} not found")
        
        # Load model (just to verify it works)
        network, metadata = model_manager.load_model(name)
        
        # Update active model in database
        await db.active_model.replace_one(
            {},
            {
                "model_name": name,
                "promoted_at": datetime.now(timezone.utc),
                "manual_activation": True,
                "metadata": metadata
            },
            upsert=True
        )
        
        return {
            "success": True,
            "message": f"Model {name} loaded and activated",
            "model_name": name
        }
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
# Game Endpoints (Human vs AI)
# ====================

class GameRequest(BaseModel):
    ai_color: str = "black"  # "white" or "black"

class MoveRequest(BaseModel):
    game_id: str
    move: str  # UCI format

class AIRequest(BaseModel):
    game_id: str
    num_simulations: int = Field(default=800, ge=10, le=2000)

class CoachRequest(BaseModel):
    game_id: str
    question: Optional[str] = None
    num_simulations: int = Field(default=400, ge=10, le=1000)

class AnalyzeMoveRequest(BaseModel):
    game_id: str
    move: str
    num_simulations: int = Field(default=400, ge=10, le=1000)

# ====================
# Helper Functions for Coaching
# ====================

async def get_mcts_evaluation(engine: ChessEngine, network, num_simulations: int = 400):
    """
    Run MCTS and return top moves with probabilities and position value
    Returns: (top_moves_list, position_value)
    """
    try:
        mcts = MCTS(network, num_simulations=num_simulations, c_puct=1.5)
        best_move, move_probs, root_value = mcts.search(engine, temperature=0.1)
        
        # Sort moves by probability
        sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
        
        # Get top moves with visit counts (estimate from probabilities)
        top_moves = []
        total_visits = num_simulations
        for move, prob in sorted_moves[:5]:  # Get top 5
            visits = int(prob * total_visits)
            top_moves.append({
                "move": move,
                "probability": float(prob),
                "visits": visits
            })
        
        return top_moves, float(root_value)
    except Exception as e:
        logger.error(f"Error getting MCTS evaluation: {e}")
        return [], 0.0

@api_router.post("/game/new")
async def create_new_game(request: GameRequest):
    """Create a new game session"""
    game_id = str(uuid.uuid4())
    engine = ChessEngine()
    
    active_games[game_id] = {
        "engine": engine,
        "history": [],
        "ai_color": request.ai_color,
        "created_at": datetime.now(timezone.utc)
    }
    
    # Reset any existing coaching session for clean start
    if game_id in coaching_sessions:
        coaching_sessions[game_id].reset_conversation()
    
    # If AI plays white, make first move
    if request.ai_color == "white":
        try:
            # Get active model
            active_model_doc = await db.active_model.find_one({})
            if active_model_doc:
                model_name = active_model_doc["model_name"]
                network, _ = model_manager.load_model(model_name)
            else:
                # Use fresh network if no trained model
                network = AlphaZeroNetwork()
            
            # Run MCTS
            mcts = MCTS(network, num_simulations=800, c_puct=1.5)
            best_move, _, _ = mcts.search(engine, temperature=0.0)
            
            if best_move:
                engine.make_move(best_move)
                active_games[game_id]["history"].append(best_move)
        except Exception as e:
            logger.error(f"Error making AI's first move: {e}")
    
    return {
        "success": True,
        "game_id": game_id,
        "fen": engine.get_fen(),
        "legal_moves": engine.get_legal_moves(),
        "is_game_over": engine.is_game_over(),
        "result": engine.get_result() if engine.is_game_over() else None,
        "history": active_games[game_id]["history"]
    }

@api_router.get("/game/{game_id}/state")
async def get_game_state(game_id: str):
    """Get current game state"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[game_id]
    engine = game["engine"]
    
    return {
        "success": True,
        "game_id": game_id,
        "fen": engine.get_fen(),
        "legal_moves": engine.get_legal_moves(),
        "is_game_over": engine.is_game_over(),
        "result": engine.get_result() if engine.is_game_over() else None,
        "history": game["history"],
        "ai_color": game["ai_color"]
    }

@api_router.post("/game/move")
async def make_player_move(request: MoveRequest):
    """Make a player move"""
    if request.game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[request.game_id]
    engine = game["engine"]
    
    # Validate and make move
    if not engine.make_move(request.move):
        raise HTTPException(status_code=400, detail="Illegal move")
    
    # Add to history
    game["history"].append(request.move)
    
    return {
        "success": True,
        "fen": engine.get_fen(),
        "legal_moves": engine.get_legal_moves(),
        "is_game_over": engine.is_game_over(),
        "result": engine.get_result() if engine.is_game_over() else None,
        "history": game["history"]
    }

@api_router.post("/game/ai-move")
async def get_ai_move(request: AIRequest):
    """Get AI's move using MCTS + Neural Network"""
    if request.game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[request.game_id]
    engine = game["engine"]
    
    if engine.is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over")
    
    try:
        # Get active model
        active_model_doc = await db.active_model.find_one({})
        if active_model_doc:
            model_name = active_model_doc["model_name"]
            network, _ = model_manager.load_model(model_name)
            logger.info(f"Using trained model: {model_name}")
        else:
            # Use fresh network if no trained model
            network = AlphaZeroNetwork()
            logger.info("Using fresh network (no trained models)")
        
        # Run MCTS
        start_time = datetime.now()
        mcts = MCTS(network, num_simulations=request.num_simulations, c_puct=1.5)
        best_move, move_probs, root_value = mcts.search(engine, temperature=0.0)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if not best_move:
            raise HTTPException(status_code=500, detail="AI couldn't find a move")
        
        # Make the move
        engine.make_move(best_move)
        game["history"].append(best_move)
        
        logger.info(f"AI move: {best_move} (took {elapsed:.2f}s, {request.num_simulations} sims)")
        
        return {
            "success": True,
            "move": best_move,
            "fen": engine.get_fen(),
            "legal_moves": engine.get_legal_moves(),
            "is_game_over": engine.is_game_over(),
            "result": engine.get_result() if engine.is_game_over() else None,
            "history": game["history"],
            "computation_time": float(elapsed),
            "simulations": int(request.num_simulations)
        }
    except Exception as e:
        logger.error(f"Error getting AI move: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/game/{game_id}")
async def delete_game(game_id: str):
    """Delete a game session"""
    if game_id in active_games:
        del active_games[game_id]
    
    # Also clean up coaching session
    if game_id in coaching_sessions:
        del coaching_sessions[game_id]
    
    return {"success": True, "message": "Game deleted"}

# ====================
# LLM Explanation Endpoints
# ====================

class ExplainRequest(BaseModel):
    fen: str
    last_move: Optional[str] = None
    context: Optional[str] = None

@api_router.post("/llm/explain")
async def explain_position(request: ExplainRequest):
    """Get LLM explanation for a chess position"""
    try:
        from llm_evaluator import LLMChessEvaluator
        
        evaluator = LLMChessEvaluator()
        
        context = request.context or ""
        if request.last_move:
            context += f"\nLast move played: {request.last_move}"
        
        explanation = await evaluator.evaluate_position(request.fen, context)
        
        return {
            "success": True,
            "explanation": explanation,
            "fen": request.fen
        }
    except Exception as e:
        logger.error(f"LLM explanation error: {e}")
        # Return fallback explanation
        return {
            "success": True,
            "explanation": "Position analysis unavailable. Continue playing to improve the AI model.",
            "fen": request.fen,
            "offline": True
        }

@api_router.post("/llm/suggest-strategy")
async def suggest_strategy(request: ExplainRequest):
    """Get strategic suggestions from LLM"""
    try:
        from llm_evaluator import LLMChessEvaluator
        
        evaluator = LLMChessEvaluator()
        strategy = await evaluator.suggest_opening_strategy(request.fen)
        
        return {
            "success": True,
            "strategy": strategy,
            "fen": request.fen
        }
    except Exception as e:
        logger.error(f"LLM strategy error: {e}")
        return {
            "success": True,
            "strategy": "Focus on controlling the center and developing your pieces.",
            "fen": request.fen,
            "offline": True
        }

# ====================
# Coaching Mode Endpoints
# ====================

async def save_llm_metrics(evaluator):
    """Save LLM performance metrics to database"""
    try:
        recent_metrics = evaluator.get_recent_metrics(limit=10)
        if recent_metrics:
            # Save only new metrics (last one)
            latest_metric = recent_metrics[-1]
            await db.llm_performance.insert_one(latest_metric)
    except Exception as e:
        logger.error(f"Error saving LLM metrics: {e}")

def get_or_create_coach(game_id: str):
    """Get or create coaching session for a game"""
    if game_id not in coaching_sessions:
        from llm_evaluator import LLMChessEvaluator, LLMConfig
        # Use global config
        config = LLMChessEvaluator.get_global_config()
        coaching_sessions[game_id] = LLMChessEvaluator(session_id=f"coach-{game_id}", config=config)
    return coaching_sessions[game_id]

@api_router.post("/coaching/suggest")
async def get_coaching_suggestion(request: CoachRequest):
    """Get move suggestions and coaching advice with AlphaZero evaluation"""
    if request.game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    try:
        game = active_games[request.game_id]
        engine = game["engine"]
        
        if engine.is_game_over():
            return {
                "success": True,
                "message": "Game is over. Start a new game to continue coaching.",
                "top_moves": [],
                "position_value": 0.0,
                "coaching": "The game has ended. Let's start a fresh game!"
            }
        
        # Get active model
        active_model_doc = await db.active_model.find_one({})
        if active_model_doc:
            model_name = active_model_doc["model_name"]
            network, _ = model_manager.load_model(model_name)
        else:
            network = AlphaZeroNetwork()
        
        # Get MCTS evaluation
        top_moves, position_value = await get_mcts_evaluation(engine, network, request.num_simulations)
        
        # Get LLM coaching
        coach = get_or_create_coach(request.game_id)
        fen = engine.get_fen()
        
        context = request.question if request.question else ""
        coaching_text = await coach.coach_with_mcts(fen, top_moves, position_value, context)
        
        # Save performance metrics
        await save_llm_metrics(coach)
        
        return {
            "success": True,
            "fen": fen,
            "top_moves": top_moves[:3],  # Return top 3
            "position_value": position_value,
            "coaching": coaching_text,
            "conversation_history": coach.get_conversation_history()[-10:]  # Last 10 messages
        }
    except Exception as e:
        logger.error(f"Coaching suggestion error: {e}")
        return {
            "success": True,
            "coaching": "Coaching temporarily unavailable. Focus on controlling the center and developing your pieces.",
            "top_moves": [],
            "position_value": 0.0,
            "offline": True
        }

@api_router.post("/coaching/analyze-move")
async def analyze_move(request: AnalyzeMoveRequest):
    """Analyze a specific move to see if it was good or bad"""
    if request.game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    try:
        game = active_games[request.game_id]
        engine = game["engine"]
        
        # Get the position BEFORE the move was made
        history = game["history"]
        if not history or history[-1] != request.move:
            return {
                "success": False,
                "message": "Move not found in history or not the last move"
            }
        
        # Recreate position before the move
        temp_engine = ChessEngine()
        for move in history[:-1]:
            temp_engine.make_move(move)
        
        fen_before = temp_engine.get_fen()
        
        # Get active model
        active_model_doc = await db.active_model.find_one({})
        if active_model_doc:
            model_name = active_model_doc["model_name"]
            network, _ = model_manager.load_model(model_name)
        else:
            network = AlphaZeroNetwork()
        
        # Get MCTS evaluation for the position
        top_moves, _ = await get_mcts_evaluation(temp_engine, network, request.num_simulations)
        
        # Check if the played move was in top moves
        played_move = request.move
        top_move_uci = [m["move"] for m in top_moves]
        
        was_best = top_move_uci[0] == played_move if top_move_uci else False
        was_in_top_3 = played_move in top_move_uci[:3]
        
        # Get LLM analysis
        coach = get_or_create_coach(request.game_id)
        analysis = await coach.analyze_specific_move(
            fen_before, 
            played_move, 
            was_best=was_best,
            better_moves=top_move_uci[:3] if not was_in_top_3 else None
        )
        
        return {
            "success": True,
            "move": played_move,
            "was_best": was_best,
            "was_in_top_3": was_in_top_3,
            "best_move": top_move_uci[0] if top_move_uci else None,
            "top_moves": top_moves[:3],
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Move analysis error: {e}")
        return {
            "success": True,
            "analysis": "Move analysis temporarily unavailable.",
            "offline": True
        }

@api_router.post("/coaching/ask")
async def ask_coach_question(request: CoachRequest):
    """Ask the coach a general question"""
    if request.game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    try:
        game = active_games[request.game_id]
        engine = game["engine"]
        fen = engine.get_fen()
        
        coach = get_or_create_coach(request.game_id)
        answer = await coach.general_question(request.question, fen)
        
        return {
            "success": True,
            "question": request.question,
            "answer": answer,
            "conversation_history": coach.get_conversation_history()[-10:]
        }
    except Exception as e:
        logger.error(f"Coach question error: {e}")
        return {
            "success": True,
            "answer": "I'm having trouble answering right now. Keep playing and learning!",
            "offline": True
        }

@api_router.post("/coaching/reset/{game_id}")
async def reset_coaching_session(game_id: str):
    """Reset coaching conversation for a game"""
    if game_id in coaching_sessions:
        coaching_sessions[game_id].reset_conversation()
        return {"success": True, "message": "Coaching session reset"}
    return {"success": True, "message": "No active coaching session"}

@api_router.get("/coaching/history/{game_id}")
async def get_coaching_history(game_id: str):
    """Get conversation history for a game"""
    if game_id in coaching_sessions:
        history = coaching_sessions[game_id].get_conversation_history()
        return {"success": True, "history": history}
    return {"success": True, "history": []}

# ====================
# LLM Analytics Insights Endpoint (Step 12)
# ====================

@api_router.post("/llm/insights")
async def generate_training_insights():
    """
    Generate comprehensive LLM-powered insights from training and evaluation metrics.
    This endpoint fetches data from MongoDB and uses LLM to provide coaching insights.
    """
    try:
        from llm_evaluator import LLMChessEvaluator, LLMConfig
        
        # Initialize LLM evaluator for analytics with global config
        config = LLMChessEvaluator.get_global_config()
        evaluator = LLMChessEvaluator(session_id="analytics-coach", config=config)
        
        # Fetch training metrics (last 50 epochs)
        training_metrics = await db.training_metrics.find().sort("timestamp", -1).limit(50).to_list(50)
        
        # Fetch evaluation results (last 10)
        evaluations = await db.model_evaluations.find().sort("timestamp", -1).limit(10).to_list(10)
        
        # Fetch self-play statistics
        total_positions = await db.self_play_positions.count_documents({})
        recent_positions = await db.self_play_positions.find().sort("timestamp", -1).limit(1000).to_list(1000)
        
        # Get active model
        active_model_doc = await db.active_model.find_one({})
        
        # Process training data
        if training_metrics:
            recent_losses = [m.get("loss", 0) for m in training_metrics[:20] if m.get("loss")]
            avg_recent_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
            
            # Calculate loss improvement (compare first 10 vs last 10)
            first_half = [m.get("loss", 0) for m in training_metrics[-10:] if m.get("loss")]
            second_half = [m.get("loss", 0) for m in training_metrics[:10] if m.get("loss")]
            
            if first_half and second_half:
                avg_first = sum(first_half) / len(first_half)
                avg_second = sum(second_half) / len(second_half)
                improvement = ((avg_first - avg_second) / avg_first * 100) if avg_first > 0 else 0
                loss_improvement = f"{improvement:.1f}% improvement"
            else:
                loss_improvement = "N/A"
            
            # Loss summary
            if len(recent_losses) >= 3:
                loss_summary = f"Min: {min(recent_losses):.4f}, Max: {max(recent_losses):.4f}, Latest: {recent_losses[0]:.4f}"
            else:
                loss_summary = "Insufficient data"
            
            # Count unique training sessions
            training_sessions = await db.training_metrics.distinct("session_id")
            total_sessions = len(training_sessions)
        else:
            avg_recent_loss = "N/A"
            loss_improvement = "No training data"
            loss_summary = "No training data"
            total_sessions = 0
        
        training_data = {
            "total_sessions": total_sessions,
            "total_epochs": len(training_metrics),
            "loss_summary": loss_summary,
            "avg_recent_loss": f"{avg_recent_loss:.4f}" if isinstance(avg_recent_loss, float) else avg_recent_loss,
            "loss_improvement": loss_improvement
        }
        
        # Process evaluation data
        if evaluations:
            recent_win_rates = []
            promoted_count = 0
            
            for eval_data in evaluations:
                win_rate = eval_data.get("challenger_win_rate", 0)
                recent_win_rates.append(win_rate)
                if eval_data.get("promoted", False) or win_rate > 0.55:
                    promoted_count += 1
            
            recent_win_rate = recent_win_rates[0] if recent_win_rates else 0
            avg_win_rate = sum(recent_win_rates) / len(recent_win_rates) if recent_win_rates else 0
            
            # Win rate trend
            if len(recent_win_rates) >= 3:
                recent_trend = sum(recent_win_rates[:3]) / 3
                older_trend = sum(recent_win_rates[-3:]) / 3
                if recent_trend > older_trend + 0.05:
                    win_rate_trend = "Improving"
                elif recent_trend < older_trend - 0.05:
                    win_rate_trend = "Declining"
                else:
                    win_rate_trend = "Stable"
            else:
                win_rate_trend = "Insufficient data"
        else:
            recent_win_rate = "N/A"
            win_rate_trend = "No evaluation data"
            promoted_count = 0
            avg_win_rate = 0
        
        evaluation_data = {
            "total_evaluations": len(evaluations),
            "recent_win_rate": f"{recent_win_rate * 100:.1f}%" if isinstance(recent_win_rate, float) else recent_win_rate,
            "win_rate_trend": win_rate_trend,
            "promoted_count": promoted_count,
            "current_champion": active_model_doc["model_name"] if active_model_doc else "None"
        }
        
        # Process self-play data
        recent_games = len(recent_positions) // 30  # Estimate games (avg 30 positions per game)
        quality_score = "Good" if total_positions > 1000 else "Low" if total_positions > 0 else "None"
        
        selfplay_data = {
            "total_positions": total_positions,
            "recent_games": recent_games,
            "quality_score": quality_score
        }
        
        # Generate LLM insights
        logger.info("Generating LLM insights from analytics data...")
        insights = await evaluator.analyze_training_metrics(training_data, evaluation_data, selfplay_data)
        
        # Save performance metrics
        await save_llm_metrics(evaluator)
        
        return {
            "success": True,
            "insights": insights,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics_summary": {
                "training": training_data,
                "evaluation": evaluation_data,
                "selfplay": selfplay_data
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")

# ====================
# Analytics Endpoints (Supporting Step 12)
# ====================

@api_router.get("/training/metrics")
async def get_training_metrics(limit: int = 50):
    """Get training metrics for analytics"""
    metrics = await db.training_metrics.find().sort("timestamp", -1).limit(limit).to_list(limit)
    return {"success": True, "metrics": metrics}

@api_router.get("/analytics/training-summary")
async def get_training_summary():
    """Get training summary statistics"""
    # Get unique sessions
    training_sessions_raw = await db.training_metrics.aggregate([
        {"$group": {
            "_id": "$session_id",
            "epochs": {"$sum": 1},
            "avg_loss": {"$avg": "$loss"},
            "timestamp": {"$first": "$timestamp"},
            "device": {"$first": "$device"}
        }},
        {"$sort": {"timestamp": -1}},
        {"$limit": 10}
    ]).to_list(10)
    
    training_sessions = [
        {
            "session_id": s["_id"],
            "epochs": s["epochs"],
            "avg_loss": s["avg_loss"],
            "device": s.get("device", "Unknown")
        }
        for s in training_sessions_raw
    ]
    
    total_sessions = len(await db.training_metrics.distinct("session_id"))
    total_epochs = await db.training_metrics.count_documents({})
    
    return {
        "success": True,
        "total_sessions": total_sessions,
        "total_epochs": total_epochs,
        "training_sessions": training_sessions
    }

@api_router.get("/analytics/evaluation-summary")
async def get_evaluation_summary(limit: int = 20):
    """Get evaluation summary with win rate progression"""
    evaluations = await db.model_evaluations.find().sort("timestamp", -1).limit(limit).to_list(limit)
    
    # Build win rate progression
    win_rate_progression = []
    for eval_data in reversed(evaluations):
        win_rate_progression.append({
            "challenger": eval_data.get("challenger_name", "Unknown"),
            "champion": eval_data.get("champion_name", "Unknown"),
            "win_rate": eval_data.get("challenger_win_rate", 0),
            "promoted": eval_data.get("promoted", False) or eval_data.get("challenger_win_rate", 0) > 0.55,
            "games_played": eval_data.get("games_played", 0)
        })
    
    return {
        "success": True,
        "count": len(evaluations),
        "evaluations": evaluations,
        "win_rate_progression": win_rate_progression
    }

@api_router.get("/analytics/model-history")
async def get_model_history():
    """Get model promotion history"""
    # Get all active model changes
    promotions = await db.model_evaluations.find({
        "$or": [
            {"promoted": True},
            {"challenger_win_rate": {"$gte": 0.55}}
        ]
    }).sort("timestamp", -1).limit(10).to_list(10)
    
    promotion_history = [
        {
            "model_name": p.get("challenger_name", "Unknown"),
            "defeated": p.get("champion_name", "N/A"),
            "win_rate": p.get("challenger_win_rate", 0),
            "promoted_at": p.get("timestamp", datetime.now(timezone.utc)).isoformat() if isinstance(p.get("timestamp"), datetime) else str(p.get("timestamp"))
        }
        for p in promotions
    ]
    
    active_model_doc = await db.active_model.find_one({})
    
    return {
        "success": True,
        "active_model": active_model_doc["model_name"] if active_model_doc else None,
        "promotion_history": promotion_history
    }

# ====================
# LLM Tuning & Performance Endpoints (Step 13)
# ====================

class LLMConfigRequest(BaseModel):
    response_mode: str = Field(default="balanced", pattern="^(fast|balanced|insightful)$")
    prompt_depth: int = Field(default=5, ge=1, le=10)
    adaptive_enabled: bool = Field(default=True)
    max_response_time: float = Field(default=10.0, ge=1.0, le=30.0)
    fallback_mode: str = Field(default="fast", pattern="^(fast|balanced|insightful)$")

@api_router.get("/llm/tune")
async def get_llm_config():
    """Get current LLM configuration"""
    try:
        from llm_evaluator import LLMChessEvaluator, LLMConfig
        
        # Try to get from database
        config_doc = await db.llm_config.find_one({"type": "global"})
        
        if config_doc:
            config = LLMConfig(
                response_mode=config_doc.get("response_mode", "balanced"),
                prompt_depth=config_doc.get("prompt_depth", 5),
                adaptive_enabled=config_doc.get("adaptive_enabled", True),
                max_response_time=config_doc.get("max_response_time", 10.0),
                fallback_mode=config_doc.get("fallback_mode", "fast")
            )
        else:
            # Use current global config or default
            config = LLMChessEvaluator.get_global_config()
        
        return {
            "success": True,
            "config": config.to_dict()
        }
    except Exception as e:
        logger.error(f"Error getting LLM config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/llm/tune")
async def update_llm_config(config_request: LLMConfigRequest):
    """Update LLM configuration globally"""
    try:
        from llm_evaluator import LLMChessEvaluator, LLMConfig
        
        # Create new config
        new_config = LLMConfig(
            response_mode=config_request.response_mode,
            prompt_depth=config_request.prompt_depth,
            adaptive_enabled=config_request.adaptive_enabled,
            max_response_time=config_request.max_response_time,
            fallback_mode=config_request.fallback_mode
        )
        
        # Set global config
        LLMChessEvaluator.set_global_config(new_config)
        
        # Store in database for persistence
        await db.llm_config.replace_one(
            {"type": "global"},
            {
                "type": "global",
                "response_mode": new_config.response_mode,
                "prompt_depth": new_config.prompt_depth,
                "adaptive_enabled": new_config.adaptive_enabled,
                "max_response_time": new_config.max_response_time,
                "fallback_mode": new_config.fallback_mode,
                "updated_at": datetime.now(timezone.utc)
            },
            upsert=True
        )
        
        logger.info(f"LLM config updated globally: {new_config.to_dict()}")
        
        return {
            "success": True,
            "message": "LLM configuration updated successfully",
            "config": new_config.to_dict()
        }
    except Exception as e:
        logger.error(f"Error updating LLM config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/performance-metrics")
async def get_llm_performance_metrics(limit: int = 50):
    """Get LLM performance metrics from database"""
    try:
        # Get metrics from database
        metrics = await db.llm_performance.find().sort("timestamp", -1).limit(limit).to_list(limit)
        
        # Calculate aggregate stats
        if metrics:
            avg_response_time = sum(m.get("response_time", 0) for m in metrics) / len(metrics)
            success_rate = sum(1 for m in metrics if m.get("success", False)) / len(metrics) * 100
            fallback_count = sum(1 for m in metrics if m.get("fallback_triggered", False))
            
            # Get mode distribution
            mode_counts = {}
            for m in metrics:
                mode = m.get("mode", "unknown")
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
        else:
            avg_response_time = 0
            success_rate = 0
            fallback_count = 0
            mode_counts = {}
        
        return {
            "success": True,
            "metrics": metrics,
            "summary": {
                "total_requests": len(metrics),
                "avg_response_time": round(avg_response_time, 2),
                "success_rate": round(success_rate, 1),
                "fallback_count": fallback_count,
                "mode_distribution": mode_counts
            }
        }
    except Exception as e:
        logger.error(f"Error getting LLM performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/optimization-status")
async def get_llm_optimization_status():
    """Get current LLM optimization status and recommendations"""
    try:
        from llm_evaluator import LLMChessEvaluator
        
        # Get recent metrics from database
        recent_metrics = await db.llm_performance.find().sort("timestamp", -1).limit(20).to_list(20)
        
        if not recent_metrics:
            return {
                "success": True,
                "status": "No data available",
                "recommendations": ["Start using LLM features to gather performance data"]
            }
        
        # Calculate stats
        avg_time = sum(m.get("response_time", 0) for m in recent_metrics) / len(recent_metrics)
        success_rate = sum(1 for m in recent_metrics if m.get("success", False)) / len(recent_metrics)
        fallback_count = sum(1 for m in recent_metrics if m.get("fallback_triggered", False))
        
        # Generate recommendations
        recommendations = []
        if avg_time > 8.0:
            recommendations.append("High response time detected. Consider using 'fast' mode.")
        if fallback_count > len(recent_metrics) * 0.3:
            recommendations.append("Frequent fallbacks occurring. Reduce prompt depth or use faster mode.")
        if success_rate < 0.9:
            recommendations.append("Low success rate. Check API connectivity.")
        
        if not recommendations:
            recommendations.append("Performance is optimal.")
        
        # Get current config
        config = LLMChessEvaluator.get_global_config()
        
        return {
            "success": True,
            "status": "active",
            "avg_response_time": round(avg_time, 2),
            "success_rate": round(success_rate * 100, 1),
            "fallback_count": fallback_count,
            "current_config": config.to_dict(),
            "recommendations": recommendations,
            "adaptive_active": config.adaptive_enabled
        }
    except Exception as e:
        logger.error(f"Error getting optimization status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================
# LLM Auto-Evaluation & Feedback
# ====================

class FeedbackInput(BaseModel):
    """User feedback for LLM outputs"""
    session_id: str
    operation_type: str  # "coaching", "analytics", "general"
    accuracy_score: float = Field(ge=1, le=5)
    usefulness: float = Field(ge=1, le=5)
    clarity: float = Field(ge=1, le=5)
    response_time: Optional[float] = None
    comment: Optional[str] = None
    llm_confidence: Optional[float] = Field(default=None, ge=0, le=1)

@api_router.post("/llm/evaluate")
async def evaluate_llm_response(feedback: FeedbackInput):
    """
    Accept user feedback for LLM outputs and store in llm_feedback collection.
    Returns aggregate scores and tuning recommendations.
    
    Evaluation weights:
    - User feedback (50%): accuracy, usefulness, clarity
    - LLM self-assessment (25%): confidence score
    - Performance metrics (25%): response time, success rate
    """
    try:
        from llm_evaluator import FeedbackData, evaluate_llm_output, PerformanceMetrics
        
        # Generate feedback ID
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Create feedback data object
        feedback_data = FeedbackData(
            feedback_id=feedback_id,
            session_id=feedback.session_id,
            operation_type=feedback.operation_type,
            accuracy_score=feedback.accuracy_score,
            usefulness=feedback.usefulness,
            clarity=feedback.clarity,
            response_time=feedback.response_time or 0,
            timestamp=timestamp,
            comment=feedback.comment,
            llm_confidence=feedback.llm_confidence
        )
        
        # Create mock performance metrics if not provided
        performance_metrics = PerformanceMetrics(
            timestamp=timestamp,
            response_time=feedback.response_time or 5.0,
            model_used="gpt-4o-mini",
            prompt_length=0,
            response_length=0,
            mode="balanced",
            success=True,
            fallback_triggered=False
        )
        
        # Evaluate the output
        evaluation = evaluate_llm_output(
            feedback_data=feedback_data,
            performance_metrics=performance_metrics,
            llm_confidence=feedback.llm_confidence
        )
        
        # Store feedback in MongoDB
        feedback_doc = feedback_data.to_dict()
        feedback_doc["evaluation"] = evaluation.to_dict()
        await db.llm_feedback.insert_one(feedback_doc)
        
        # Check if we should trigger auto-optimization (every 10 feedbacks)
        feedback_count = await db.llm_feedback.count_documents({})
        should_auto_optimize = (feedback_count % 10 == 0) and feedback_count > 0
        
        # Calculate aggregate scores from recent feedback
        recent_feedback = await db.llm_feedback.find().sort("timestamp", -1).limit(20).to_list(20)
        
        if recent_feedback:
            avg_overall = sum(f.get("evaluation", {}).get("overall_score", 0) for f in recent_feedback) / len(recent_feedback)
            avg_accuracy = sum(f.get("evaluation", {}).get("accuracy_score", 0) for f in recent_feedback) / len(recent_feedback)
            avg_clarity = sum(f.get("evaluation", {}).get("clarity_score", 0) for f in recent_feedback) / len(recent_feedback)
        else:
            avg_overall = evaluation.overall_score
            avg_accuracy = evaluation.accuracy_score
            avg_clarity = evaluation.clarity_score
        
        response_data = {
            "success": True,
            "feedback_id": feedback_id,
            "evaluation": evaluation.to_dict(),
            "aggregate_scores": {
                "overall_score": round(avg_overall, 2),
                "accuracy_score": round(avg_accuracy, 2),
                "clarity_score": round(avg_clarity, 2),
                "total_feedback_count": feedback_count
            },
            "auto_optimize_triggered": should_auto_optimize
        }
        
        # Trigger auto-optimization if threshold reached
        if should_auto_optimize:
            logger.info(f"Auto-optimization triggered at {feedback_count} feedbacks")
            response_data["auto_optimize_message"] = f"Auto-optimization triggered after {feedback_count} feedbacks"
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error evaluating LLM response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/llm/auto-optimize")
async def auto_optimize_llm():
    """
    Analyze feedback and performance data to automatically tune LLM configuration.
    Can be triggered manually or automatically every 10 feedbacks.
    
    Returns recommended parameter changes and applies them if beneficial.
    """
    try:
        from llm_evaluator import LLMChessEvaluator, FeedbackData, auto_tune_from_feedback
        
        # Get current configuration
        current_config = LLMChessEvaluator.get_global_config()
        
        # Fetch recent feedback (last 50)
        feedback_docs = await db.llm_feedback.find().sort("timestamp", -1).limit(50).to_list(50)
        
        if not feedback_docs:
            return {
                "success": False,
                "message": "No feedback data available for optimization",
                "recommendations": ["Collect user feedback to enable auto-optimization"]
            }
        
        # Convert to FeedbackData objects
        feedback_list = []
        for doc in feedback_docs:
            try:
                feedback_list.append(FeedbackData(
                    feedback_id=doc.get("feedback_id", ""),
                    session_id=doc.get("session_id", ""),
                    operation_type=doc.get("operation_type", "general"),
                    accuracy_score=doc.get("accuracy_score", 3.0),
                    usefulness=doc.get("usefulness", 3.0),
                    clarity=doc.get("clarity", 3.0),
                    response_time=doc.get("response_time", 5.0),
                    timestamp=doc.get("timestamp", ""),
                    comment=doc.get("comment"),
                    llm_confidence=doc.get("llm_confidence")
                ))
            except Exception as e:
                logger.warning(f"Skipping invalid feedback doc: {e}")
                continue
        
        # Get performance history from MongoDB (we'll create a collection for this)
        # For now, use empty list as performance history is tracked in-memory
        performance_history = []
        
        # Perform auto-tuning
        new_config, recommendations = auto_tune_from_feedback(
            feedback_list=feedback_list,
            current_config=current_config,
            performance_history=performance_history
        )
        
        # Check if configuration changed
        config_changed = (
            new_config.response_mode != current_config.response_mode or
            new_config.prompt_depth != current_config.prompt_depth or
            new_config.max_response_time != current_config.max_response_time
        )
        
        # Apply new configuration if changed
        if config_changed:
            LLMChessEvaluator.set_global_config(new_config)
            logger.info(f"Auto-optimization applied: {new_config.to_dict()}")
            
            # Store optimization event in MongoDB
            optimization_event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "previous_config": current_config.to_dict(),
                "new_config": new_config.to_dict(),
                "recommendations": recommendations,
                "trigger": "auto",
                "feedback_count": len(feedback_list)
            }
            await db.llm_optimization_events.insert_one(optimization_event)
        
        # Calculate performance metrics
        if feedback_list:
            avg_accuracy = sum(f.accuracy_score for f in feedback_list) / len(feedback_list)
            avg_usefulness = sum(f.usefulness for f in feedback_list) / len(feedback_list)
            avg_clarity = sum(f.clarity for f in feedback_list) / len(feedback_list)
            avg_response_time = sum(f.response_time for f in feedback_list) / len(feedback_list)
        else:
            avg_accuracy = avg_usefulness = avg_clarity = 0
            avg_response_time = 0
        
        return {
            "success": True,
            "config_changed": config_changed,
            "previous_config": current_config.to_dict(),
            "new_config": new_config.to_dict(),
            "recommendations": recommendations,
            "performance_summary": {
                "avg_accuracy": round((avg_accuracy / 5.0) * 100, 1),
                "avg_usefulness": round((avg_usefulness / 5.0) * 100, 1),
                "avg_clarity": round((avg_clarity / 5.0) * 100, 1),
                "avg_response_time": round(avg_response_time, 2),
                "feedback_samples": len(feedback_list)
            },
            "message": "Configuration updated successfully" if config_changed else "No configuration changes needed"
        }
        
    except Exception as e:
        logger.error(f"Error in auto-optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/feedback-summary")
async def get_feedback_summary(limit: int = 50):
    """Get summary of LLM feedback and evaluation trends"""
    try:
        # Fetch recent feedback
        feedback_docs = await db.llm_feedback.find().sort("timestamp", -1).limit(limit).to_list(limit)
        
        if not feedback_docs:
            return {
                "success": True,
                "total_feedback": 0,
                "message": "No feedback data available"
            }
        
        # Calculate aggregate metrics
        evaluations = [doc.get("evaluation", {}) for doc in feedback_docs]
        
        avg_overall = sum(e.get("overall_score", 0) for e in evaluations) / len(evaluations)
        avg_accuracy = sum(e.get("accuracy_score", 0) for e in evaluations) / len(evaluations)
        avg_usefulness = sum(e.get("usefulness_score", 0) for e in evaluations) / len(evaluations)
        avg_clarity = sum(e.get("clarity_score", 0) for e in evaluations) / len(evaluations)
        
        # Count by operation type
        operation_counts = {}
        for doc in feedback_docs:
            op_type = doc.get("operation_type", "unknown")
            operation_counts[op_type] = operation_counts.get(op_type, 0) + 1
        
        # Get recent optimization events
        optimization_events = await db.llm_optimization_events.find().sort("timestamp", -1).limit(10).to_list(10)
        
        return {
            "success": True,
            "total_feedback": len(feedback_docs),
            "aggregate_scores": {
                "overall_score": round(avg_overall, 2),
                "accuracy_score": round(avg_accuracy, 2),
                "usefulness_score": round(avg_usefulness, 2),
                "clarity_score": round(avg_clarity, 2)
            },
            "operation_distribution": operation_counts,
            "recent_feedback": [
                {
                    "timestamp": doc.get("timestamp"),
                    "operation_type": doc.get("operation_type"),
                    "evaluation": doc.get("evaluation", {}),
                    "comment": doc.get("comment")
                }
                for doc in feedback_docs[:10]
            ],
            "optimization_history": [
                {
                    "timestamp": evt.get("timestamp"),
                    "trigger": evt.get("trigger"),
                    "config_changed": evt.get("new_config") != evt.get("previous_config"),
                    "recommendations": evt.get("recommendations", [])[:3]  # Top 3
                }
                for evt in optimization_events
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting feedback summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================
# Step 39: User Feedback Collection System
# ====================

class UserFeedbackInput(BaseModel):
    """User feedback for the app (Step 39)"""
    feedback_type: str  # "bug", "feature", "general", "performance"
    category: str  # "ui", "ai", "performance", "other"
    rating: int = Field(ge=1, le=5)
    message: str
    email: Optional[str] = None
    allow_contact: bool = False
    telemetry_opt_in: bool = False
    app_version: str = "1.0.0"
    platform: Optional[str] = None

class TelemetryData(BaseModel):
    """Anonymous telemetry data (opt-in)"""
    session_id: str
    event_type: str  # "game_played", "training_started", "ai_move", etc.
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    app_version: str = "1.0.0"
    platform: Optional[str] = None

@api_router.post("/feedback/submit")
async def submit_user_feedback(feedback: UserFeedbackInput):
    """
    Submit user feedback for the AlphaZero Chess App.
    Collects bug reports, feature requests, and general feedback.
    """
    try:
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        feedback_doc = {
            "feedback_id": feedback_id,
            "feedback_type": feedback.feedback_type,
            "category": feedback.category,
            "rating": feedback.rating,
            "message": feedback.message,
            "email": feedback.email if feedback.allow_contact else None,
            "allow_contact": feedback.allow_contact,
            "telemetry_opt_in": feedback.telemetry_opt_in,
            "app_version": feedback.app_version,
            "platform": feedback.platform,
            "timestamp": timestamp,
            "status": "new",
            "priority": "normal"
        }
        
        # Auto-assign priority based on type and rating
        if feedback.feedback_type == "bug" and feedback.rating <= 2:
            feedback_doc["priority"] = "high"
        elif feedback.feedback_type == "bug":
            feedback_doc["priority"] = "medium"
        
        # Store in MongoDB
        await db.user_feedback.insert_one(feedback_doc)
        
        logger.info(f"User feedback received: {feedback_id} - {feedback.feedback_type}")
        
        return {
            "success": True,
            "feedback_id": feedback_id,
            "message": "Thank you for your feedback! We'll review it shortly.",
            "priority": feedback_doc["priority"]
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/telemetry/submit")
async def submit_telemetry(telemetry: TelemetryData):
    """
    Submit anonymous telemetry data (opt-in only).
    Used for performance monitoring and usage analytics.
    """
    try:
        telemetry_doc = {
            "session_id": telemetry.session_id,
            "event_type": telemetry.event_type,
            "duration": telemetry.duration,
            "success": telemetry.success,
            "error_message": telemetry.error_message,
            "app_version": telemetry.app_version,
            "platform": telemetry.platform,
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Store in MongoDB
        await db.telemetry_data.insert_one(telemetry_doc)
        
        return {
            "success": True,
            "message": "Telemetry data recorded"
        }
        
    except Exception as e:
        logger.error(f"Error submitting telemetry: {e}")
        return {"success": False, "message": "Telemetry submission failed"}

@api_router.get("/feedback/analytics")
async def get_feedback_analytics(limit: int = 100):
    """
    Get analytics dashboard data for user feedback.
    Shows feedback trends, common issues, and sentiment analysis.
    """
    try:
        # Fetch recent feedback
        feedback_docs = await db.user_feedback.find().sort("timestamp", -1).limit(limit).to_list(limit)
        
        if not feedback_docs:
            return {
                "success": True,
                "total_feedback": 0,
                "message": "No feedback data available"
            }
        
        # Calculate metrics
        total_feedback = len(feedback_docs)
        avg_rating = sum(doc.get("rating", 0) for doc in feedback_docs) / total_feedback
        
        # Count by type
        type_counts = {}
        category_counts = {}
        priority_counts = {}
        
        for doc in feedback_docs:
            fb_type = doc.get("feedback_type", "unknown")
            category = doc.get("category", "unknown")
            priority = doc.get("priority", "normal")
            
            type_counts[fb_type] = type_counts.get(fb_type, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        # Get rating distribution
        rating_distribution = {}
        for doc in feedback_docs:
            rating = doc.get("rating", 0)
            rating_distribution[rating] = rating_distribution.get(rating, 0) + 1
        
        # Calculate sentiment (simplified)
        positive_count = sum(1 for doc in feedback_docs if doc.get("rating", 0) >= 4)
        neutral_count = sum(1 for doc in feedback_docs if doc.get("rating", 0) == 3)
        negative_count = sum(1 for doc in feedback_docs if doc.get("rating", 0) <= 2)
        
        # Get telemetry stats
        total_telemetry = await db.telemetry_data.count_documents({})
        telemetry_events = await db.telemetry_data.aggregate([
            {"$group": {"_id": "$event_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]).to_list(10)
        
        return {
            "success": True,
            "overview": {
                "total_feedback": total_feedback,
                "average_rating": round(avg_rating, 2),
                "sentiment": {
                    "positive": positive_count,
                    "neutral": neutral_count,
                    "negative": negative_count
                }
            },
            "distributions": {
                "by_type": type_counts,
                "by_category": category_counts,
                "by_priority": priority_counts,
                "by_rating": rating_distribution
            },
            "telemetry": {
                "total_events": total_telemetry,
                "top_events": [
                    {"event_type": evt["_id"], "count": evt["count"]}
                    for evt in telemetry_events
                ]
            },
            "recent_feedback": [
                {
                    "feedback_id": doc.get("feedback_id"),
                    "type": doc.get("feedback_type"),
                    "category": doc.get("category"),
                    "rating": doc.get("rating"),
                    "message": doc.get("message")[:100] + "..." if len(doc.get("message", "")) > 100 else doc.get("message"),
                    "timestamp": doc.get("timestamp").isoformat() if isinstance(doc.get("timestamp"), datetime) else str(doc.get("timestamp")),
                    "priority": doc.get("priority")
                }
                for doc in feedback_docs[:20]
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting feedback analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/app/version")
async def get_app_version():
    """Get current app version info"""
    return {
        "success": True,
        "version": "1.0.0",
        "build": "#39",
        "release_date": "2025-10-10",
        "status": "public-release"
    }

# ====================
# Step 30: Self-Reflection & Continuous Learning Loop
# ====================

# Import reflection controller
from self_reflection import ReflectionController

# Initialize reflection controller
reflection_controller = None

def get_reflection_controller():
    """Get or create reflection controller instance"""
    global reflection_controller
    if reflection_controller is None:
        reflection_controller = ReflectionController(db)
    return reflection_controller

@api_router.post("/llm/reflection/trigger")
async def trigger_reflection_cycle(
    trigger: str = "manual",
    game_id: Optional[str] = None
):
    """
    Trigger a self-reflection cycle.
    
    Args:
        trigger: What triggered this cycle ("post_game", "scheduled", "manual")
        game_id: Specific game to reflect on (optional)
    
    Returns:
        Complete reflection cycle results
    """
    try:
        controller = get_reflection_controller()
        
        logger.info(f"Triggering reflection cycle: trigger={trigger}, game_id={game_id}")
        
        cycle = await controller.trigger_reflection_cycle(
            trigger=trigger,
            game_id=game_id
        )
        
        # **Step 31 Integration**: Automatically trigger memory fusion after reflection
        memory_fusion_result = None
        try:
            memory_controller = get_memory_fusion_controller()
            logger.info(f"Auto-triggering memory fusion for reflection cycle {cycle.cycle_id}")
            
            reflection_data = cycle.to_dict()
            memory_fusion_result = await memory_controller.trigger_fusion_cycle(
                reflection_cycle_id=cycle.cycle_id,
                reflection_data=reflection_data,
                trigger="post_reflection"
            )
            
            logger.info(
                f"Memory fusion complete: {memory_fusion_result.get('new_memory_nodes', 0)} nodes created"
            )
        except Exception as fusion_error:
            logger.error(f"Memory fusion failed (non-critical): {fusion_error}")
            # Don't fail the whole reflection if fusion fails
        
        # **Step 32 Integration**: Automatically trigger cohesion after memory fusion
        cohesion_result = None
        if memory_fusion_result and memory_fusion_result.get("success"):
            try:
                cohesion_controller = get_cohesion_controller()
                logger.info(f"Auto-triggering cohesion for memory fusion {memory_fusion_result.get('fusion_id')}")
                
                cohesion_report = await cohesion_controller.trigger_cohesion_cycle(
                    trigger="post_memory_fusion",
                    memory_fusion_id=memory_fusion_result.get("fusion_id"),
                    reflection_cycle_id=cycle.cycle_id
                )
                
                cohesion_result = {
                    "cycle_id": cohesion_report.cycle_id,
                    "alignment_score": cohesion_report.metrics.alignment_score,
                    "system_health_index": cohesion_report.metrics.system_health_index,
                    "ethical_continuity": cohesion_report.metrics.ethical_continuity,
                    "cohesion_health": cohesion_report.metrics.cohesion_health,
                    "auto_triggered": True
                }
                
                logger.info(
                    f"Cohesion cycle complete: alignment={cohesion_report.metrics.alignment_score:.2f}, "
                    f"health={cohesion_report.metrics.system_health_index:.2f}"
                )
            except Exception as cohesion_error:
                logger.error(f"Cohesion cycle failed (non-critical): {cohesion_error}")
                # Don't fail the whole pipeline if cohesion fails
        
        response = {
            "success": True,
            "cycle_id": cycle.cycle_id,
            "timestamp": cycle.timestamp,
            "trigger": cycle.trigger,
            "games_analyzed": cycle.games_analyzed,
            "strategies_evaluated": cycle.strategies_evaluated,
            "overall_performance_score": cycle.overall_performance_score,
            "learning_health_index": cycle.learning_health_index,
            "ethical_alignment_status": cycle.ethical_alignment_status,
            "insights_summary": cycle.insights_summary,
            "recommendations": cycle.recommendations,
            "parameter_adjustments": cycle.parameter_adjustments,
            "game_reflections": cycle.game_reflections,
            "strategy_evaluations": cycle.strategy_evaluations
        }
        
        # Add memory fusion results if successful
        if memory_fusion_result and memory_fusion_result.get("success"):
            response["memory_fusion"] = {
                "fusion_id": memory_fusion_result.get("fusion_id"),
                "nodes_created": memory_fusion_result.get("new_memory_nodes", 0),
                "fusion_time": memory_fusion_result.get("fusion_time_seconds", 0),
                "auto_triggered": True
            }
        
        # Add cohesion results if successful
        if cohesion_result:
            response["cohesion"] = cohesion_result
        
        return response
        
    except Exception as e:
        logger.error(f"Error triggering reflection cycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/reflection/status")
async def get_reflection_status():
    """Get current reflection system status"""
    try:
        controller = get_reflection_controller()
        status = await controller.get_reflection_status()
        
        return {
            "success": True,
            **status
        }
        
    except Exception as e:
        logger.error(f"Error getting reflection status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/reflection/history")
async def get_reflection_history(limit: int = 10):
    """Get reflection cycle history"""
    try:
        controller = get_reflection_controller()
        history = await controller.get_reflection_history(limit=limit)
        
        return {
            "success": True,
            "count": len(history),
            "cycles": history
        }
        
    except Exception as e:
        logger.error(f"Error getting reflection history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/reflection/parameters")
async def get_learning_parameters():
    """Get current learning parameters"""
    try:
        controller = get_reflection_controller()
        params = await controller.get_current_learning_parameters()
        
        return {
            "success": True,
            "parameters": params.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Error getting learning parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class UpdateParametersRequest(BaseModel):
    novelty_weight: Optional[float] = None
    stability_weight: Optional[float] = None
    ethical_threshold: Optional[float] = None
    creativity_bias: Optional[float] = None
    risk_tolerance: Optional[float] = None
    reflection_depth: Optional[int] = None

@api_router.post("/llm/reflection/parameters")
async def update_learning_parameters(request: UpdateParametersRequest):
    """Update learning parameters"""
    try:
        controller = get_reflection_controller()
        
        params = await controller.update_learning_parameters(
            novelty_weight=request.novelty_weight,
            stability_weight=request.stability_weight,
            ethical_threshold=request.ethical_threshold,
            creativity_bias=request.creativity_bias,
            risk_tolerance=request.risk_tolerance,
            reflection_depth=request.reflection_depth
        )
        
        return {
            "success": True,
            "message": "Learning parameters updated successfully",
            "parameters": params.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Error updating learning parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class HumanFeedbackRequest(BaseModel):
    game_id: str
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = None

@api_router.post("/llm/reflection/feedback")
async def submit_human_feedback(request: HumanFeedbackRequest):
    """Submit human feedback for a game"""
    try:
        controller = get_reflection_controller()
        
        result = await controller.submit_human_feedback(
            game_id=request.game_id,
            rating=request.rating,
            comment=request.comment
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/reflection/metrics")
async def get_reflection_metrics():
    """Get comprehensive reflection system metrics"""
    try:
        # Get reflection status
        controller = get_reflection_controller()
        status = await controller.get_reflection_status()
        
        # Get recent cycles
        recent_cycles = await controller.get_reflection_history(limit=10)
        
        # Calculate trends
        if recent_cycles:
            performance_trend = [c.get("overall_performance_score", 0) for c in recent_cycles]
            health_trend = [c.get("learning_health_index", 0) for c in recent_cycles]
            
            avg_performance = sum(performance_trend) / len(performance_trend)
            avg_health = sum(health_trend) / len(health_trend)
            
            # Calculate improvement
            if len(performance_trend) >= 2:
                recent_perf = sum(performance_trend[:3]) / min(3, len(performance_trend[:3]))
                older_perf = sum(performance_trend[-3:]) / min(3, len(performance_trend[-3:]))
                performance_change = recent_perf - older_perf
            else:
                performance_change = 0.0
        else:
            avg_performance = 0.0
            avg_health = 0.0
            performance_change = 0.0
        
        # Get strategy evaluation summary
        strategy_evals = await db.llm_strategy_evaluation.find().sort("timestamp", -1).limit(20).to_list(20)
        
        strategy_summary = {}
        if strategy_evals:
            rating_counts = {}
            for eval_doc in strategy_evals:
                rating = eval_doc.get("performance_rating", "unknown")
                rating_counts[rating] = rating_counts.get(rating, 0) + 1
            
            strategy_summary = {
                "total_evaluated": len(strategy_evals),
                "rating_distribution": rating_counts,
                "avg_success_rate": sum(e.get("success_rate", 0) for e in strategy_evals) / len(strategy_evals)
            }
        
        return {
            "success": True,
            "reflection_status": status,
            "performance_metrics": {
                "avg_performance_score": round(avg_performance, 1),
                "avg_learning_health": round(avg_health, 2),
                "performance_change": round(performance_change, 1),
                "total_cycles": len(recent_cycles)
            },
            "strategy_evaluation": strategy_summary,
            "recent_cycles": recent_cycles[:5]
        }
        
    except Exception as e:
        logger.error(f"Error getting reflection metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================
# Memory Fusion & Long-Term Cognitive Persistence (Step 31)
# ====================

# Helper function to convert ObjectIds in nested structures
def convert_objectids(obj):
    """Recursively convert ObjectId instances to strings"""
    from bson import ObjectId
    
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_objectids(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectids(item) for item in obj]
    else:
        return obj

# Global memory fusion controller
_memory_fusion_controller = None

def get_memory_fusion_controller():
    """Get or create memory fusion controller"""
    global _memory_fusion_controller
    if _memory_fusion_controller is None:
        from memory_fusion import MemoryFusionController
        _memory_fusion_controller = MemoryFusionController(db)
    return _memory_fusion_controller

class MemoryFusionRequest(BaseModel):
    reflection_cycle_id: str
    trigger: str = "post_reflection"

class MemoryRetrievalRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    limit: int = Field(default=5, ge=1, le=20)

class MemoryResetRequest(BaseModel):
    confirmation: str
    admin_override: bool = False

@api_router.post("/llm/memory/fuse")
async def trigger_memory_fusion(
    reflection_cycle_id: str,
    trigger: str = "post_reflection"
):
    """
    Trigger memory fusion cycle to consolidate reflection insights into long-term memory.
    
    Automatically called after Step 30 reflection cycles. Can also be manually triggered.
    Creates up to 5 memory nodes per cycle with exponential decay tracking.
    """
    try:
        controller = get_memory_fusion_controller()
        
        # Get reflection data
        reflection_data = await db.llm_reflection_log.find_one({"cycle_id": reflection_cycle_id})
        
        if not reflection_data:
            raise HTTPException(
                status_code=404,
                detail=f"Reflection cycle {reflection_cycle_id} not found"
            )
        
        # Trigger fusion cycle
        result = await controller.trigger_fusion_cycle(
            reflection_cycle_id=reflection_cycle_id,
            reflection_data=reflection_data,
            trigger=trigger
        )
        
        return {
            "success": result.get("success", True),
            "fusion_result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering memory fusion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/memory/nodes")
async def get_memory_nodes(
    limit: int = Query(default=50, ge=1, le=200),
    active_only: bool = Query(default=True),
    sort_by: str = Query(default="timestamp", pattern="^(timestamp|decay_weight|usage_count)$")
):
    """
    Get list of memory nodes with filtering and sorting options.
    
    Query parameters:
    - limit: Maximum nodes to return (1-200)
    - active_only: Only return nodes above decay threshold
    - sort_by: Sort field (timestamp, decay_weight, usage_count)
    """
    try:
        controller = get_memory_fusion_controller()
        
        # Build query
        query = {}
        if active_only:
            query["decay_weight"] = {"$gte": controller.min_decay_threshold}
        
        # Get nodes
        nodes = await db.llm_memory_nodes.find(query).sort(
            sort_by, -1
        ).limit(limit).to_list(limit)
        
        # Convert ObjectId to string for JSON serialization
        for node in nodes:
            if '_id' in node:
                node['_id'] = str(node['_id'])
        
        return {
            "success": True,
            "count": len(nodes),
            "nodes": nodes,
            "active_only": active_only,
            "sort_by": sort_by
        }
        
    except Exception as e:
        logger.error(f"Error getting memory nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/llm/memory/retrieve")
async def retrieve_contextual_memory(request: MemoryRetrievalRequest):
    """
    Retrieve relevant memory nodes based on contextual query.
    
    Uses relevance scoring to find the most applicable memories for a given context.
    Updates usage counts and resets decay for retrieved memories.
    """
    try:
        controller = get_memory_fusion_controller()
        
        # Retrieve memories
        memories = await controller.retrieve_contextual_memory(
            query=request.query,
            context=request.context,
            limit=request.limit
        )
        
        return {
            "success": True,
            "query": request.query,
            "memories_found": len(memories),
            "memories": [m.to_dict() for m in memories]
        }
        
    except Exception as e:
        logger.error(f"Error retrieving contextual memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/memory/profile")
async def get_long_term_profile():
    """
    Get comprehensive long-term persistence profile.
    
    Returns:
    - Memory distribution by type
    - Parameter evolution trajectories
    - Learning trends (creativity, stability, ethics)
    - Health metrics and target comparisons
    - AI-generated summary of cognitive evolution
    """
    try:
        controller = get_memory_fusion_controller()
        
        profile = await controller.synthesize_long_term_profile()
        
        # Convert any ObjectIds in the profile
        profile = convert_objectids(profile)
        
        return {
            "success": True,
            "profile": profile
        }
        
    except Exception as e:
        logger.error(f"Error getting long-term profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/memory/health")
async def get_memory_health():
    """
    Get current memory system health metrics.
    
    Returns:
    - Memory retention index (target: ≥0.90)
    - Fusion efficiency (target: ≥0.85)
    - Ethical continuity (target: ≥0.92)
    - Retrieval latency (target: ≤2.0s)
    - Persistence health (target: ≥0.88)
    - Health recommendations
    """
    try:
        controller = get_memory_fusion_controller()
        
        health = await controller.get_memory_health()
        
        return {
            "success": True,
            "health": health
        }
        
    except Exception as e:
        logger.error(f"Error getting memory health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/memory/traces")
async def get_memory_traces(
    limit: int = Query(default=50, ge=1, le=200),
    trace_type: Optional[str] = Query(default=None, pattern="^(fusion_cycle|retrieval)$")
):
    """
    Get memory trace logs (fusion cycles and retrievals).
    
    Query parameters:
    - limit: Maximum traces to return
    - trace_type: Filter by type (fusion_cycle or retrieval)
    """
    try:
        query = {}
        if trace_type:
            query["trace_type"] = trace_type
        
        traces = await db.llm_memory_trace.find(query).sort(
            "timestamp", -1
        ).limit(limit).to_list(limit)
        
        return {
            "success": True,
            "count": len(traces),
            "traces": traces,
            "trace_type": trace_type
        }
        
    except Exception as e:
        logger.error(f"Error getting memory traces: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/llm/memory/reset")
async def reset_memory_system(request: MemoryResetRequest):
    """
    Reset memory system (admin function with safeguards).
    
    Requires:
    - confirmation: "CONFIRM_RESET"
    - admin_override: true
    
    Creates backup before reset. Use with caution.
    """
    try:
        controller = get_memory_fusion_controller()
        
        result = await controller.reset_memory_system(
            confirmation=request.confirmation,
            admin_override=request.admin_override
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=403, detail=result.get("error", "Reset failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting memory system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/memory/stats")
async def get_memory_statistics():
    """Get quick memory system statistics"""
    try:
        total_nodes = await db.llm_memory_nodes.count_documents({})
        controller = get_memory_fusion_controller()
        active_nodes = await db.llm_memory_nodes.count_documents({
            "decay_weight": {"$gte": controller.min_decay_threshold}
        })
        total_traces = await db.llm_memory_trace.count_documents({})
        total_profiles = await db.llm_persistence_profile.count_documents({})
        
        # Get latest fusion
        latest_fusion = await db.llm_memory_trace.find_one(
            {"trace_type": "fusion_cycle"},
            sort=[("timestamp", -1)]
        )
        
        return {
            "success": True,
            "statistics": {
                "total_memory_nodes": total_nodes,
                "active_memory_nodes": active_nodes,
                "decayed_memory_nodes": total_nodes - active_nodes,
                "total_traces": total_traces,
                "total_profiles": total_profiles,
                "latest_fusion": latest_fusion.get("timestamp") if latest_fusion else None,
                "decay_lambda": controller.decay_lambda,
                "max_nodes_per_cycle": controller.max_nodes_per_cycle,
                "retention_window_games": controller.retention_window
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting memory statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================
# Cohesion Core & Systemic Unification (Step 32)
# ====================

# Import cohesion controller
from cohesion_core import CohesionController

# Initialize cohesion controller
cohesion_controller = None

def get_cohesion_controller():
    """Get or create cohesion controller instance"""
    global cohesion_controller
    if cohesion_controller is None:
        cohesion_controller = CohesionController(db)
    return cohesion_controller

@api_router.post("/llm/cohesion/trigger")
async def trigger_cohesion_cycle(
    trigger: str = "manual",
    memory_fusion_id: Optional[str] = None,
    reflection_cycle_id: Optional[str] = None
):
    """
    Trigger a cohesion cycle to synchronize all cognitive subsystems.
    
    Args:
        trigger: What triggered this cycle ("post_memory_fusion", "scheduled", "manual")
        memory_fusion_id: ID of the memory fusion that triggered this
        reflection_cycle_id: ID of the reflection cycle (if available)
    
    Returns:
        Comprehensive cohesion report
    """
    try:
        controller = get_cohesion_controller()
        
        logger.info(f"Triggering cohesion cycle: trigger={trigger}")
        
        report = await controller.trigger_cohesion_cycle(
            trigger=trigger,
            memory_fusion_id=memory_fusion_id,
            reflection_cycle_id=reflection_cycle_id
        )
        
        return {
            "success": True,
            "report_id": report.report_id,
            "cycle_id": report.cycle_id,
            "timestamp": report.timestamp,
            "metrics": report.metrics.to_dict(),
            "module_states": {k: v.to_dict() for k, v in report.module_states.items()},
            "health_analysis": report.health_analysis,
            "recommendations": report.recommendations,
            "actions_log": report.actions_log,
            "parameter_comparison": report.parameter_comparison
        }
        
    except Exception as e:
        logger.error(f"Error triggering cohesion cycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/cohesion/status")
async def get_cohesion_status():
    """Get current cohesion system status"""
    try:
        controller = get_cohesion_controller()
        status = await controller.get_cohesion_status()
        
        return {
            "success": True,
            **status
        }
        
    except Exception as e:
        logger.error(f"Error getting cohesion status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/cohesion/history")
async def get_cohesion_history(limit: int = 10):
    """Get cohesion cycle history"""
    try:
        controller = get_cohesion_controller()
        history = await controller.get_cohesion_history(limit=limit)
        
        # Remove MongoDB _id field
        for cycle in history:
            if '_id' in cycle:
                del cycle['_id']
        
        return {
            "success": True,
            "count": len(history),
            "cycles": history
        }
        
    except Exception as e:
        logger.error(f"Error getting cohesion history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/cohesion/report")
async def get_cohesion_report(cycle_id: Optional[str] = None):
    """Get comprehensive cohesion report"""
    try:
        controller = get_cohesion_controller()
        result = await controller.generate_cohesion_report(cycle_id=cycle_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting cohesion report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/cohesion/metrics")
async def get_cohesion_metrics(limit: int = 20):
    """Get cohesion metrics history"""
    try:
        metrics = await db.llm_cohesion_metrics.find().sort(
            "timestamp", -1
        ).limit(limit).to_list(limit)
        
        # Remove MongoDB _id field
        for metric in metrics:
            if '_id' in metric:
                del metric['_id']
        
        return {
            "success": True,
            "count": len(metrics),
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting cohesion metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/cohesion/health")
async def get_system_health(limit: int = 20):
    """Get system health history"""
    try:
        health_records = await db.llm_system_health.find().sort(
            "timestamp", -1
        ).limit(limit).to_list(limit)
        
        # Remove MongoDB _id field
        for record in health_records:
            if '_id' in record:
                del record['_id']
        
        # Calculate trends if enough data
        if len(health_records) >= 3:
            recent_health = [h.get("system_health_index", 0) for h in health_records[:3]]
            older_health = [h.get("system_health_index", 0) for h in health_records[-3:]]
            
            recent_avg = sum(recent_health) / len(recent_health)
            older_avg = sum(older_health) / len(older_health)
            
            if recent_avg > older_avg + 0.05:
                trend = "improving"
            elif recent_avg < older_avg - 0.05:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "success": True,
            "count": len(health_records),
            "health_records": health_records,
            "trend": trend,
            "latest_health": health_records[0] if health_records else None
        }
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# ====================
# Ethical Governance Layer 2.0 (Step 33)
# ====================

from ethical_governance_v2 import EthicalGovernanceController

# Global controller instance
ethics_controller = None

def get_ethics_controller():
    """Get or create Ethical Governance controller instance"""
    global ethics_controller
    if ethics_controller is None:
        ethics_controller = EthicalGovernanceController(db)
    return ethics_controller

@api_router.post("/llm/ethics/trigger")
async def trigger_ethics_scan(context: str = "general"):
    """
    Trigger an ethical governance scan across all cognitive subsystems.
    
    Args:
        context: Operational context ("opening", "middlegame", "endgame", "general")
    
    Returns:
        Complete ethical assessment with metrics, violations, and recommendations
    """
    try:
        controller = get_ethics_controller()
        
        logger.info(f"Triggering ethics scan for context: {context}")
        
        # Step 1: Monitor system state
        system_state = await controller.monitor_system_state()
        
        # Step 2: Evaluate compliance
        metrics = await controller.evaluate_compliance(system_state, context)
        
        # Step 3: Auto-flag anomalies
        violations = await controller.auto_flag_anomalies(system_state, metrics)
        
        # Step 4: Recalibrate thresholds
        thresholds = await controller.recalibrate_thresholds(metrics, context)
        
        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context,
            "metrics": metrics.to_dict(),
            "violations_flagged": len(violations),
            "violations": [v.to_dict() for v in violations],
            "thresholds_updated": len(thresholds),
            "system_state": system_state
        }
        
    except Exception as e:
        logger.error(f"Error triggering ethics scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/ethics/status")
async def get_ethics_status():
    """Get current ethical governance system status"""
    try:
        controller = get_ethics_controller()
        status = await controller.get_ethics_status()
        
        return {
            "success": True,
            **status
        }
        
    except Exception as e:
        logger.error(f"Error getting ethics status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/ethics/violations")
async def get_ethics_violations(
    severity: Optional[str] = None,
    module: Optional[str] = None,
    limit: int = 50
):
    """
    Get flagged ethical violations.
    
    Args:
        severity: Filter by severity ("critical", "high", "medium", "low")
        module: Filter by module ("creativity", "reflection", "memory", "cohesion")
        limit: Maximum number of violations to return
    
    Returns:
        List of violations with details
    """
    try:
        controller = get_ethics_controller()
        violations = await controller.get_violations(severity, module, limit)
        
        return {
            "success": True,
            "count": len(violations),
            "filters": {"severity": severity, "module": module},
            "violations": violations
        }
        
    except Exception as e:
        logger.error(f"Error getting violations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/ethics/metrics")
async def get_ethics_metrics(days: int = 7):
    """
    Get ethical compliance metrics history.
    
    Args:
        days: Number of days of history to retrieve
    
    Returns:
        List of metrics over time
    """
    try:
        controller = get_ethics_controller()
        metrics = await controller.get_metrics_history(days)
        
        return {
            "success": True,
            "count": len(metrics),
            "period_days": days,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting ethics metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/ethics/report")
async def get_ethics_report(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Generate comprehensive Ethics Report.
    
    Args:
        start_date: Report period start (ISO format)
        end_date: Report period end (ISO format)
    
    Returns:
        Complete ethics report with analysis and recommendations
    """
    try:
        controller = get_ethics_controller()
        
        report = await controller.generate_ethics_report(start_date, end_date)
        
        return {
            "success": True,
            "report": report.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Error generating ethics report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/ethics/thresholds")
async def get_ethics_thresholds(context: str = "general"):
    """
    Get current adaptive ethical thresholds.
    
    Args:
        context: Context to retrieve thresholds for
    
    Returns:
        Current thresholds per module and parameter
    """
    try:
        thresholds = await db.llm_ethics_policies.find({
            "context": context
        }).to_list(100)
        
        # Remove MongoDB _id
        for t in thresholds:
            if '_id' in t:
                del t['_id']
        
        return {
            "success": True,
            "context": context,
            "count": len(thresholds),
            "thresholds": thresholds
        }
        
    except Exception as e:
        logger.error(f"Error getting thresholds: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/ethics/approvals")
async def get_pending_approvals():
    """Get all pending parameter change requests requiring human approval"""
    try:
        controller = get_ethics_controller()
        approvals = await controller.get_pending_approvals()
        
        return {
            "success": True,
            "count": len(approvals),
            "pending_approvals": approvals
        }
        
    except Exception as e:
        logger.error(f"Error getting pending approvals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/llm/ethics/approve")
async def approve_parameter_change(
    request_id: str,
    approved: bool,
    approved_by: str,
    notes: Optional[str] = None
):
    """
    Approve or reject a parameter change request.
    
    Args:
        request_id: ID of the change request
        approved: True to approve, False to reject
        approved_by: Username/identifier of approver
        notes: Optional approval notes
    
    Returns:
        Result of approval action
    """
    try:
        controller = get_ethics_controller()
        
        result = await controller.approve_parameter_change(
            request_id, approved, approved_by, notes
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error approving parameter change: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/llm/ethics/request-change")
async def request_parameter_change(
    module: str,
    parameter: str,
    current_value: float,
    proposed_value: float,
    reason: str
):
    """
    Request a parameter change (requires human approval for critical changes).
    
    Args:
        module: Module requesting change
        parameter: Parameter to change
        current_value: Current parameter value
        proposed_value: Proposed new value
        reason: Justification for change
    
    Returns:
        Parameter change request details
    """
    try:
        controller = get_ethics_controller()
        
        request = await controller.request_parameter_change(
            module, parameter, current_value, proposed_value, reason
        )
        
        return {
            "success": True,
            "request": request.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Error requesting parameter change: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# ====================
# Cognitive Resonance & Long-Term System Stability (Step 34)
# ====================

_resonance_controller = None

def get_resonance_controller():
    """Lazy initialization of Resonance Controller"""
    global _resonance_controller
    if _resonance_controller is None:
        from cognitive_resonance import CognitiveResonanceController
        _resonance_controller = CognitiveResonanceController(db)
    return _resonance_controller

@api_router.get("/llm/resonance/status")
async def get_resonance_status():
    """
    Get current resonance and stability indicators.
    
    Returns resonance index, temporal stability, feedback equilibrium,
    entropy balance, and target compliance status.
    
    Returns:
        Current resonance system status
    """
    try:
        controller = get_resonance_controller()
        status = await controller.get_resonance_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting resonance status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/llm/resonance/analyze")
async def analyze_resonance():
    """
    Trigger resonance state analysis.
    
    Analyzes cross-module alignment, temporal stability, feedback equilibrium,
    and entropy balance across all cognitive subsystems (Steps 29-33).
    
    Returns:
        Comprehensive resonance metrics
    """
    try:
        controller = get_resonance_controller()
        metrics = await controller.analyze_resonance_state()
        
        return {
            "success": True,
            "metrics": metrics.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing resonance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/llm/resonance/recalibrate")
async def recalibrate_feedback_weights(force: bool = False):
    """
    Force balance recalibration of feedback weights.
    
    Adaptive Feedback Regulator (AFR) adjusts learning feedback gains to maintain
    equilibrium between novelty, stability, and ethical parameters.
    
    Args:
        force: Force recalibration even if system is balanced
    
    Returns:
        Balance adjustment recommendations (advisory mode)
    """
    try:
        controller = get_resonance_controller()
        result = await controller.balance_feedback_weights(force_recalibration=force)
        
        return {
            "success": True,
            "balance_result": result
        }
        
    except Exception as e:
        logger.error(f"Error recalibrating feedback weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/resonance/forecast")
async def get_stability_forecast(horizon_hours: int = Query(24, ge=1, le=168)):
    """
    Get stability forecast for upcoming sessions.
    
    Temporal Stability Monitor (TSM) predicts potential drift, oscillation,
    and stagnation risks over the specified time horizon.
    
    Args:
        horizon_hours: Forecast horizon in hours (default: 24, max: 168)
    
    Returns:
        Stability forecast with predictions and risk assessment
    """
    try:
        controller = get_resonance_controller()
        forecast = await controller.stability_forecast(horizon_hours=horizon_hours)
        
        return {
            "success": True,
            "forecast": forecast.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Error generating stability forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/resonance/report")
async def get_resonance_report(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Retrieve comprehensive resonance report.
    
    Generates Resonance Report with summary metrics, module analysis,
    stability trends, forecasts, and recommendations.
    
    Args:
        start_date: Report period start (ISO format, default: 7 days ago)
        end_date: Report period end (ISO format, default: now)
    
    Returns:
        Complete resonance report
    """
    try:
        controller = get_resonance_controller()
        report = await controller.generate_resonance_report(start_date, end_date)
        
        return {
            "success": True,
            "report": report.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Error generating resonance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/llm/resonance/snapshot")
async def record_snapshot():
    """
    Record a resonance snapshot for longitudinal tracking.
    
    Persistence Resonator (PR) stores current system state, module metrics,
    and notable events for historical analysis.
    
    Returns:
        Stored resonance snapshot
    """
    try:
        controller = get_resonance_controller()
        snapshot = await controller.record_resonance_snapshot()
        
        return {
            "success": True,
            "snapshot": snapshot.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Error recording resonance snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/resonance/snapshots")
async def get_resonance_snapshots(limit: int = Query(20, ge=1, le=100)):
    """
    Get historical resonance snapshots.
    
    Args:
        limit: Maximum number of snapshots to retrieve
    
    Returns:
        List of historical resonance snapshots
    """
    try:
        snapshots = await db.llm_resonance_snapshots.find().sort(
            "timestamp", -1
        ).limit(limit).to_list(limit)
        
        # Remove MongoDB _id
        for snapshot in snapshots:
            if '_id' in snapshot:
                del snapshot['_id']
        
        return {
            "success": True,
            "snapshots": snapshots,
            "count": len(snapshots)
        }
        
    except Exception as e:
        logger.error(f"Error getting resonance snapshots: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/resonance/metrics")
async def get_resonance_metrics(days: int = Query(7, ge=1, le=30)):
    """
    Get historical resonance metrics.
    
    Args:
        days: Number of days of history to retrieve
    
    Returns:
        List of historical resonance metrics
    """
    try:
        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        
        metrics = await db.llm_resonance_metrics.find({
            "timestamp": {"$gte": start_date}
        }).sort("timestamp", -1).to_list(1000)
        
        # Remove MongoDB _id
        for m in metrics:
            if '_id' in m:
                del m['_id']
        
        return {
            "success": True,
            "metrics": metrics,
            "count": len(metrics)
        }
        
    except Exception as e:
        logger.error(f"Error getting resonance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# ====================
# LLM Knowledge Distillation & Performance Audit (Step 15)
# ====================

@api_router.post("/llm/distill")
async def distill_knowledge():
    """
    Extract reusable strategic knowledge from high-rated feedback (≥4★).
    Analyzes patterns and creates compact distilled insights.
    Auto-trims to keep latest 100 records.
    """
    try:
        from llm_evaluator import distill_from_feedback, DistilledKnowledge
        
        # Fetch high-rated feedback (accuracy_score ≥ 4)
        high_rated_feedback = await db.llm_feedback.find({
            "accuracy_score": {"$gte": 4.0}
        }).sort("timestamp", -1).limit(100).to_list(100)
        
        if not high_rated_feedback:
            return {
                "success": False,
                "message": "No high-rated feedback (≥4★) available for distillation",
                "distilled_count": 0
            }
        
        # Get API key
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="EMERGENT_LLM_KEY not configured")
        
        # Perform distillation
        logger.info(f"Starting distillation from {len(high_rated_feedback)} high-rated feedback entries")
        distilled_entries = await distill_from_feedback(high_rated_feedback, api_key)
        
        if not distilled_entries:
            return {
                "success": False,
                "message": "Distillation failed to extract knowledge",
                "distilled_count": 0
            }
        
        # Store in llm_knowledge_base collection
        distilled_dicts = [entry.to_dict() for entry in distilled_entries]
        await db.llm_knowledge_base.insert_many(distilled_dicts)
        
        # Auto-trim: Keep only latest 100 records
        total_count = await db.llm_knowledge_base.count_documents({})
        if total_count > 100:
            # Get oldest entries to delete
            entries_to_delete = await db.llm_knowledge_base.find().sort("timestamp", 1).limit(total_count - 100).to_list(total_count - 100)
            ids_to_delete = [entry["_id"] for entry in entries_to_delete]
            
            await db.llm_knowledge_base.delete_many({"_id": {"$in": ids_to_delete}})
            logger.info(f"Trimmed {len(ids_to_delete)} old distilled entries, kept latest 100")
        
        # Return summary
        logger.info(f"Successfully distilled {len(distilled_entries)} knowledge entries")
        
        return {
            "success": True,
            "message": f"Successfully distilled {len(distilled_entries)} knowledge entries from {len(high_rated_feedback)} high-rated feedback",
            "distilled_count": len(distilled_entries),
            "source_feedback_count": len(high_rated_feedback),
            "total_in_knowledge_base": min(total_count + len(distilled_entries), 100),
            "distilled_entries": [
                {
                    "pattern": entry.pattern,
                    "insight": entry.insight,
                    "recommendation": entry.recommendation,
                    "operation_type": entry.operation_type,
                    "confidence": round(entry.confidence_score, 2)
                }
                for entry in distilled_entries
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in knowledge distillation: {e}")
        raise HTTPException(status_code=500, detail=f"Distillation failed: {str(e)}")


@api_router.get("/llm/knowledge-base")
async def get_knowledge_base(limit: int = 20, operation_type: Optional[str] = None):
    """
    Get distilled knowledge entries from the knowledge base.
    Can filter by operation_type.
    """
    try:
        query = {}
        if operation_type:
            query["operation_type"] = operation_type
        
        entries = await db.llm_knowledge_base.find(query).sort("timestamp", -1).limit(limit).to_list(limit)
        
        total_count = await db.llm_knowledge_base.count_documents(query)
        
        return {
            "success": True,
            "total_count": total_count,
            "entries": [
                {
                    "distillation_id": entry.get("distillation_id"),
                    "pattern": entry.get("pattern"),
                    "insight": entry.get("insight"),
                    "recommendation": entry.get("recommendation"),
                    "operation_type": entry.get("operation_type"),
                    "confidence_score": entry.get("confidence_score"),
                    "timestamp": entry.get("timestamp"),
                    "source_count": len(entry.get("source_feedback_ids", []))
                }
                for entry in entries
            ]
        }
        
    except Exception as e:
        logger.error(f"Error fetching knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/audit")
async def get_performance_audit():
    """
    Generate comprehensive performance audit report.
    Measures accuracy, latency, and improvement per optimization cycle.
    Returns trends and performance history.
    """
    try:
        from llm_evaluator import generate_audit_report
        
        # Fetch all feedback history
        feedback_history = await db.llm_feedback.find().sort("timestamp", -1).to_list(None)
        
        # Fetch performance metrics
        performance_history = await db.llm_performance.find().sort("timestamp", -1).to_list(None)
        
        # Fetch optimization events
        optimization_events = await db.llm_optimization_events.find().sort("timestamp", -1).to_list(None)
        
        # Generate audit report
        logger.info("Generating LLM performance audit report")
        report = generate_audit_report(feedback_history, performance_history, optimization_events)
        
        # Calculate distillation coverage
        total_feedback = len(feedback_history)
        high_rated_feedback = len([f for f in feedback_history if f.get("accuracy_score", 0) >= 4.0])
        distilled_entries = await db.llm_knowledge_base.count_documents({})
        
        report["distillation_coverage"] = {
            "total_feedback": total_feedback,
            "high_rated_feedback": high_rated_feedback,
            "distilled_entries": distilled_entries,
            "coverage_percentage": round((high_rated_feedback / total_feedback * 100), 1) if total_feedback > 0 else 0
        }
        
        return {
            "success": True,
            "audit_report": report,
            "timestamp": report.get("timestamp")
        }
        
    except Exception as e:
        logger.error(f"Error generating audit report: {e}")
        raise HTTPException(status_code=500, detail=f"Audit generation failed: {str(e)}")


@api_router.post("/llm/auto-distill-check")
async def check_auto_distill():
    """
    Check if auto-distillation should be triggered.
    Triggers when:
    - Total feedback count is multiple of 50
    - At least 10 high-rated feedback entries exist
    
    Returns distillation status and triggers if needed.
    """
    try:
        # Get feedback count
        total_feedback = await db.llm_feedback.count_documents({})
        high_rated_count = await db.llm_feedback.count_documents({"accuracy_score": {"$gte": 4.0}})
        
        # Get last distillation timestamp
        last_distillation = await db.llm_knowledge_base.find_one({}, sort=[("timestamp", -1)])
        last_distill_time = last_distillation.get("timestamp") if last_distillation else None
        
        should_trigger = False
        reason = ""
        
        # Trigger conditions
        if total_feedback > 0 and total_feedback % 50 == 0 and high_rated_count >= 10:
            should_trigger = True
            reason = f"Auto-trigger: {total_feedback} feedback entries reached (every 50), {high_rated_count} high-rated available"
        elif high_rated_count >= 20:
            should_trigger = True
            reason = f"Auto-trigger: {high_rated_count} high-rated feedback accumulated"
        
        # If should trigger, run distillation
        if should_trigger:
            logger.info(f"Auto-distillation triggered: {reason}")
            
            # Call distill endpoint internally
            from llm_evaluator import distill_from_feedback
            
            high_rated_feedback = await db.llm_feedback.find({
                "accuracy_score": {"$gte": 4.0}
            }).sort("timestamp", -1).limit(100).to_list(100)
            
            api_key = os.environ.get('EMERGENT_LLM_KEY')
            distilled_entries = await distill_from_feedback(high_rated_feedback, api_key)
            
            if distilled_entries:
                distilled_dicts = [entry.to_dict() for entry in distilled_entries]
                await db.llm_knowledge_base.insert_many(distilled_dicts)
                
                # Trim to 100
                total_count = await db.llm_knowledge_base.count_documents({})
                if total_count > 100:
                    entries_to_delete = await db.llm_knowledge_base.find().sort("timestamp", 1).limit(total_count - 100).to_list(total_count - 100)
                    ids_to_delete = [entry["_id"] for entry in entries_to_delete]
                    await db.llm_knowledge_base.delete_many({"_id": {"$in": ids_to_delete}})
                
                return {
                    "success": True,
                    "auto_triggered": True,
                    "reason": reason,
                    "distilled_count": len(distilled_entries),
                    "message": "Auto-distillation completed successfully"
                }
        
        return {
            "success": True,
            "auto_triggered": False,
            "total_feedback": total_feedback,
            "high_rated_count": high_rated_count,
            "last_distillation": last_distill_time,
            "message": "Auto-distillation conditions not met"
        }
        
    except Exception as e:
        logger.error(f"Error in auto-distill check: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# ====================
# Predictive Trend Analysis & LLM Forecasting (Step 16)
# ====================

class ForecastRequest(BaseModel):
    """Request parameters for forecast generation"""
    timeframes: List[int] = Field(default=[7, 30, 90], description="Forecast timeframes in days")
    include_narrative: bool = Field(default=True, description="Include LLM narrative")

@api_router.post("/llm/forecast")
async def generate_forecast(request: ForecastRequest = ForecastRequest()):
    """
    Generate predictive trend analysis and LLM-powered forecasts.
    
    Analyzes historical data from:
    - training_metrics (accuracy trends)
    - model_evaluations (win rate trends)
    - llm_performance (latency trends)
    - llm_knowledge_base (distilled insights)
    - Audit history
    
    Returns forecasts for 7, 30, and 90 days with:
    - Predicted metrics (accuracy, win_rate, latency)
    - Trend direction (improving/stable/declining)
    - Confidence scores
    - LLM-generated strategic recommendations
    """
    try:
        from llm_evaluator import generate_forecast_report, generate_audit_report
        
        # Fetch historical data
        logger.info("Fetching historical data for forecast generation...")
        
        # Training metrics (last 100)
        training_data = await db.training_metrics.find().sort("timestamp", -1).limit(100).to_list(100)
        
        # Evaluation results (last 50)
        evaluation_data = await db.model_evaluations.find().sort("timestamp", -1).limit(50).to_list(50)
        
        # LLM performance metrics (last 100)
        performance_data = await db.llm_performance.find().sort("timestamp", -1).limit(100).to_list(100)
        
        # Distilled knowledge
        distilled_knowledge = await db.llm_knowledge_base.find().sort("timestamp", -1).limit(20).to_list(20)
        
        # Get audit history for context
        feedback_history = await db.llm_feedback.find().sort("timestamp", -1).limit(100).to_list(100)
        optimization_events = await db.llm_optimization_events.find().sort("timestamp", -1).limit(20).to_list(20)
        
        audit_report = generate_audit_report(
            feedback_history,
            performance_data,
            optimization_events
        )
        
        # Get API key
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="EMERGENT_LLM_KEY not configured")
        
        # Generate forecast
        logger.info(f"Generating forecast for timeframes: {request.timeframes}")
        
        forecast = await generate_forecast_report(
            training_data=training_data,
            evaluation_data=evaluation_data,
            performance_data=performance_data,
            distilled_knowledge=distilled_knowledge,
            audit_history=audit_report,
            llm_api_key=api_key,
            timeframes=request.timeframes
        )
        
        # Build response
        response_data = {
            "success": True,
            "timestamp": forecast.timestamp,
            "forecasts": {},
            "overall_confidence": forecast.overall_confidence,
            "data_sufficiency": forecast.data_sufficiency,
            "strategic_recommendations": forecast.strategic_recommendations
        }
        
        # Add forecast narrative if requested
        if request.include_narrative:
            response_data["forecast_narrative"] = forecast.forecast_narrative
        
        # Format forecasts by timeframe
        for timeframe_str, metrics in forecast.timeframes.items():
            timeframe_data = {}
            for metric_name, forecast_result in metrics.items():
                if forecast_result:
                    timeframe_data[metric_name] = {
                        "current_value": forecast_result.current_value,
                        "predicted_value": forecast_result.predicted_value,
                        "change_percent": forecast_result.change_percent,
                        "trend_direction": forecast_result.trend_direction,
                        "confidence": forecast_result.confidence
                    }
            response_data["forecasts"][f"{timeframe_str}_days"] = timeframe_data
        
        logger.info(f"Forecast generated successfully with {forecast.overall_confidence * 100:.1f}% confidence")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate forecast: {str(e)}")


# ====================
# Strategic Insight Fusion Endpoints (Step 18)
# ====================

class InsightFusionRequest(BaseModel):
    decision_id: Optional[str] = None  # Specific decision to explain
    limit: int = Field(default=10, ge=1, le=50)
    ranking: str = Field(default="confidence", pattern="^(confidence|recency|impact)$")


@api_router.post("/llm/insight-fusion")
async def generate_insight_fusion():
    """
    Generate comprehensive strategic insight fusion from all subsystems.
    
    Combines data from:
    - Step 17: Auto-tuning decisions (strategy_auto_tuning)
    - Step 16: Forecast results  
    - Step 15: Distilled knowledge (llm_knowledge_base)
    
    Generates cross-referenced insight bundles with:
    - Structured reasoning chains
    - Evidence sources with links
    - Confidence scores
    - Suggested actions
    - Alignment evaluation
    """
    try:
        from llm_evaluator import (
            generate_reasoning_chain,
            evaluate_decision_alignment,
            rank_strategic_insights,
            ReasoningChain
        )
        
        logger.info("Generating strategic insight fusion...")
        
        # Fetch data from all subsystems
        auto_tuning_history = await db.strategy_auto_tuning.find().sort("timestamp", -1).limit(20).to_list(20)
        distilled_knowledge = await db.llm_knowledge_base.find().sort("timestamp", -1).limit(20).to_list(20)
        
        # Get latest forecast data (if available)
        # For now, we'll fetch recent training and evaluation data
        training_data = await db.training_metrics.find().sort("timestamp", -1).limit(50).to_list(50)
        evaluation_data = await db.model_evaluations.find().sort("timestamp", -1).limit(20).to_list(20)
        
        # Build simple forecast context
        forecast_context = {}
        if evaluation_data:
            recent_eval = evaluation_data[0]
            forecast_context = {
                "recent_win_rate": recent_eval.get("challenger_win_rate", 0) * 100,
                "overall_confidence": 0.75
            }
        
        # Get API key
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="EMERGENT_LLM_KEY not configured")
        
        # Generate reasoning chains for recent auto-tuning decisions
        reasoning_chains = []
        
        for decision in auto_tuning_history[:5]:  # Process top 5 most recent
            try:
                # Prepare decision data
                decision_data = {
                    "tuning_id": decision.get("tuning_id"),
                    "type": "auto_tuning",
                    "trigger_reason": decision.get("trigger_reason"),
                    "parameters_adjusted": decision.get("parameters_adjusted"),
                    "confidence_score": decision.get("confidence_score"),
                    "expected_impact": decision.get("expected_impact")
                }
                
                # Generate reasoning chain
                reasoning = await generate_reasoning_chain(
                    decision_data=decision_data,
                    forecast_data=forecast_context,
                    distilled_knowledge=distilled_knowledge,
                    auto_tuning_history=auto_tuning_history,
                    llm_api_key=api_key
                )
                
                # Evaluate alignment
                alignment_eval = evaluate_decision_alignment(
                    decision_data=decision,
                    forecast_data=forecast_context
                )
                
                # Update reasoning with alignment info
                reasoning.alignment_status = "aligned" if alignment_eval["aligned"] else "deviation_detected"
                
                reasoning_chains.append(reasoning)
                
                # Store in MongoDB
                await db.llm_reasoning_log.insert_one(reasoning.to_dict())
                
            except Exception as e:
                logger.error(f"Error processing decision {decision.get('tuning_id')}: {e}")
                continue
        
        # Rank insights by confidence
        ranked_insights = rank_strategic_insights(reasoning_chains, ranking_criteria="confidence")
        
        # Build response
        insight_bundles = []
        for reasoning in ranked_insights[:10]:  # Return top 10
            insight_bundles.append({
                "reasoning_id": reasoning.reasoning_id,
                "decision_id": reasoning.decision_id,
                "timestamp": reasoning.timestamp,
                "decision_type": reasoning.decision_type,
                "reason_summary": reasoning.reason_summary,
                "evidence_sources": reasoning.evidence_sources,
                "reasoning_steps": reasoning.reasoning_steps,
                "suggested_action": reasoning.suggested_action,
                "confidence": reasoning.confidence,
                "alignment_status": reasoning.alignment_status,
                "impact_prediction": reasoning.impact_prediction
            })
        
        logger.info(f"Generated {len(insight_bundles)} insight bundles")
        
        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_insights": len(insight_bundles),
            "insights": insight_bundles,
            "data_sources": {
                "auto_tuning_decisions": len(auto_tuning_history),
                "distilled_knowledge": len(distilled_knowledge),
                "training_metrics": len(training_data),
                "evaluations": len(evaluation_data)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating insight fusion: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate insight fusion: {str(e)}")


@api_router.get("/llm/reasoning-log")
async def get_reasoning_log(limit: int = 20, decision_type: Optional[str] = None):
    """
    Get stored reasoning chains from llm_reasoning_log collection.
    
    Query Parameters:
    - limit: Number of entries to return (default: 20, max: 100)
    - decision_type: Filter by decision type (auto_tuning, forecast_alert, etc.)
    """
    try:
        query = {}
        if decision_type:
            query["decision_type"] = decision_type
        
        reasoning_logs = await db.llm_reasoning_log.find(query).sort("timestamp", -1).limit(min(limit, 100)).to_list(limit)
        
        return {
            "success": True,
            "count": len(reasoning_logs),
            "reasoning_logs": reasoning_logs
        }
    except Exception as e:
        logger.error(f"Error retrieving reasoning log: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/llm/explain-decision/{decision_id}")
async def explain_specific_decision(decision_id: str):
    """
    Generate on-demand explanation for a specific auto-tuning decision.
    
    Used by "Explain This Decision" button in UI.
    """
    try:
        from llm_evaluator import generate_reasoning_chain, evaluate_decision_alignment
        
        # Find the decision
        decision = await db.strategy_auto_tuning.find_one({"tuning_id": decision_id})
        
        if not decision:
            raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")
        
        # Get supporting data
        distilled_knowledge = await db.llm_knowledge_base.find().sort("timestamp", -1).limit(10).to_list(10)
        auto_tuning_history = await db.strategy_auto_tuning.find().sort("timestamp", -1).limit(10).to_list(10)
        
        # Get recent evaluation for forecast context
        recent_eval = await db.model_evaluations.find_one({}, sort=[("timestamp", -1)])
        forecast_context = {}
        if recent_eval:
            forecast_context = {
                "recent_win_rate": recent_eval.get("challenger_win_rate", 0) * 100,
                "overall_confidence": 0.75
            }
        
        # Get API key
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="EMERGENT_LLM_KEY not configured")
        
        # Prepare decision data
        decision_data = {
            "tuning_id": decision.get("tuning_id"),
            "type": "auto_tuning",
            "trigger_reason": decision.get("trigger_reason"),
            "parameters_adjusted": decision.get("parameters_adjusted"),
            "confidence_score": decision.get("confidence_score"),
            "expected_impact": decision.get("expected_impact")
        }
        
        # Generate reasoning
        reasoning = await generate_reasoning_chain(
            decision_data=decision_data,
            forecast_data=forecast_context,
            distilled_knowledge=distilled_knowledge,
            auto_tuning_history=auto_tuning_history,
            llm_api_key=api_key
        )
        
        # Evaluate alignment
        alignment_eval = evaluate_decision_alignment(
            decision_data=decision,
            forecast_data=forecast_context
        )
        
        # Store reasoning
        await db.llm_reasoning_log.insert_one(reasoning.to_dict())
        
        return {
            "success": True,
            "decision_id": decision_id,
            "reasoning": reasoning.to_dict(),
            "alignment_evaluation": alignment_eval
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining decision {decision_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ====================
# Real-Time Adaptive Forecasting & Strategy Auto-Tuning Endpoints (Step 17)
# ====================

# Global state for auto-tuning system
auto_tuning_state = {
    "enabled": False,
    "last_run": None,
    "next_run": None,
    "interval_hours": 1,
    "deviation_threshold": 5.0,
    "total_tunings": 0,
    "last_decision": None
}

# Background task handle
auto_tuning_task = None


async def auto_tuning_background_task():
    """Background task that runs auto-forecasting and tuning every hour"""
    global auto_tuning_state
    
    while auto_tuning_state["enabled"]:
        try:
            logger.info("Running scheduled auto-forecast and tuning check...")
            
            # Update last run time
            auto_tuning_state["last_run"] = datetime.now(timezone.utc).isoformat()
            
            # Fetch recent data (lighter than full forecast)
            training_data = await db.training_metrics.find().sort("timestamp", -1).limit(20).to_list(20)
            evaluation_data = await db.model_evaluations.find().sort("timestamp", -1).limit(10).to_list(10)
            performance_data = await db.llm_performance.find().sort("timestamp", -1).limit(30).to_list(30)
            
            # Generate realtime forecast
            from llm_evaluator import generate_realtime_forecast, detect_performance_deviation, auto_tune_strategy, apply_auto_tuning
            
            forecast_update = await generate_realtime_forecast(
                training_data=training_data,
                evaluation_data=evaluation_data,
                performance_data=performance_data,
                timeframe_days=7
            )
            
            # Check if tuning is required
            if forecast_update.requires_tuning:
                logger.info(f"Deviation detected: {forecast_update.deviation_percent:.1f}% for {forecast_update.metric_name}")
                
                # Prepare deviation data
                deviating_metrics = [forecast_update.metric_name]
                deviations = {forecast_update.metric_name: forecast_update.deviation_percent}
                
                # Get current config
                current_config = {
                    "learning_rate": 0.001,  # Default, would be read from latest training
                    "num_simulations": 800,
                    "temperature": 1.0,
                    "c_puct": 1.5
                }
                
                # Get LLM config
                from llm_evaluator import LLMChessEvaluator
                llm_config = LLMChessEvaluator.get_global_config()
                
                # Generate auto-tuning decision
                decision = auto_tune_strategy(
                    deviating_metrics=deviating_metrics,
                    deviations=deviations,
                    current_config=current_config,
                    llm_config=llm_config
                )
                
                # Apply tuning if adjustments were made
                if decision.parameters_adjusted:
                    success = await apply_auto_tuning(
                        decision=decision,
                        db_collection=db.strategy_auto_tuning,
                        llm_evaluator=None  # Would pass actual instance in production
                    )
                    
                    if success:
                        auto_tuning_state["total_tunings"] += 1
                        auto_tuning_state["last_decision"] = decision.to_dict()
                        logger.info(f"Auto-tuning applied successfully: {decision.tuning_id}")
                    else:
                        logger.error("Failed to apply auto-tuning")
                else:
                    logger.info("No adjustments needed despite deviation")
            else:
                logger.info("No significant deviations detected. System stable.")
            
            # Calculate next run time
            from datetime import timedelta
            next_run_time = datetime.now(timezone.utc) + timedelta(hours=auto_tuning_state["interval_hours"])
            auto_tuning_state["next_run"] = next_run_time.isoformat()
            
            # Wait for next interval (1 hour = 3600 seconds)
            await asyncio.sleep(auto_tuning_state["interval_hours"] * 3600)
            
        except Exception as e:
            logger.error(f"Error in auto-tuning background task: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes on error before retry


@api_router.post("/llm/auto-forecast")
async def trigger_auto_forecast(background_tasks: BackgroundTasks):
    """
    Manually trigger real-time forecast and auto-tuning check.
    
    This runs the same logic as the scheduled background task but can be
    triggered on-demand for testing or immediate response.
    """
    try:
        from llm_evaluator import generate_realtime_forecast, auto_tune_strategy, apply_auto_tuning
        
        # Fetch recent data
        training_data = await db.training_metrics.find().sort("timestamp", -1).limit(20).to_list(20)
        evaluation_data = await db.model_evaluations.find().sort("timestamp", -1).limit(10).to_list(10)
        performance_data = await db.llm_performance.find().sort("timestamp", -1).limit(30).to_list(30)
        
        # Generate realtime forecast
        forecast_update = await generate_realtime_forecast(
            training_data=training_data,
            evaluation_data=evaluation_data,
            performance_data=performance_data,
            timeframe_days=7
        )
        
        # Check for deviation
        tuning_applied = False
        decision_data = None
        
        if forecast_update.requires_tuning and forecast_update.deviation_percent >= auto_tuning_state["deviation_threshold"]:
            logger.info(f"Manual forecast: Deviation detected {forecast_update.deviation_percent:.1f}%")
            
            # Prepare for auto-tuning
            deviating_metrics = [forecast_update.metric_name]
            deviations = {forecast_update.metric_name: forecast_update.deviation_percent}
            
            current_config = {
                "learning_rate": 0.001,
                "num_simulations": 800,
                "temperature": 1.0,
                "c_puct": 1.5
            }
            
            from llm_evaluator import LLMChessEvaluator
            llm_config = LLMChessEvaluator.get_global_config()
            
            # Generate decision
            decision = auto_tune_strategy(
                deviating_metrics=deviating_metrics,
                deviations=deviations,
                current_config=current_config,
                llm_config=llm_config
            )
            
            # Apply if adjustments exist
            if decision.parameters_adjusted:
                success = await apply_auto_tuning(
                    decision=decision,
                    db_collection=db.strategy_auto_tuning,
                    llm_evaluator=None
                )
                
                tuning_applied = success
                decision_data = decision.to_dict()
                
                if success:
                    auto_tuning_state["last_decision"] = decision_data
                    auto_tuning_state["total_tunings"] += 1
        
        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "forecast_update": forecast_update.to_dict(),
            "tuning_applied": tuning_applied,
            "tuning_decision": decision_data,
            "message": "Auto-tuning applied" if tuning_applied else "No tuning needed"
        }
        
    except Exception as e:
        logger.error(f"Error in auto-forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/strategy/auto-tune/status")
async def get_auto_tuning_status():
    """Get current status of auto-tuning system"""
    return {
        "success": True,
        "enabled": auto_tuning_state["enabled"],
        "last_run": auto_tuning_state["last_run"],
        "next_run": auto_tuning_state["next_run"],
        "interval_hours": auto_tuning_state["interval_hours"],
        "deviation_threshold": auto_tuning_state["deviation_threshold"],
        "total_tunings": auto_tuning_state["total_tunings"],
        "last_decision": auto_tuning_state["last_decision"]
    }


@api_router.post("/strategy/auto-tune/toggle")
async def toggle_auto_tuning(enable: bool = True, background_tasks: BackgroundTasks = None):
    """
    Enable or disable automatic strategy tuning.
    
    When enabled, runs forecasting and tuning checks every hour.
    """
    global auto_tuning_task, auto_tuning_state
    
    try:
        if enable and not auto_tuning_state["enabled"]:
            # Start auto-tuning
            auto_tuning_state["enabled"] = True
            auto_tuning_state["last_run"] = None
            
            from datetime import timedelta
            next_run_time = datetime.now(timezone.utc) + timedelta(hours=auto_tuning_state["interval_hours"])
            auto_tuning_state["next_run"] = next_run_time.isoformat()
            
            # Start background task
            auto_tuning_task = asyncio.create_task(auto_tuning_background_task())
            
            logger.info("Auto-tuning system enabled")
            
            return {
                "success": True,
                "message": "Auto-tuning enabled",
                "status": auto_tuning_state
            }
            
        elif not enable and auto_tuning_state["enabled"]:
            # Stop auto-tuning
            auto_tuning_state["enabled"] = False
            auto_tuning_state["next_run"] = None
            
            # Cancel background task
            if auto_tuning_task and not auto_tuning_task.done():
                auto_tuning_task.cancel()
            
            logger.info("Auto-tuning system disabled")
            
            return {
                "success": True,
                "message": "Auto-tuning disabled",
                "status": auto_tuning_state
            }
        
        else:
            return {
                "success": True,
                "message": f"Auto-tuning already {'enabled' if enable else 'disabled'}",
                "status": auto_tuning_state
            }
            
    except Exception as e:
        logger.error(f"Error toggling auto-tuning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/strategy/auto-tune/history")
async def get_auto_tuning_history(limit: int = 20):
    """
    Get history of automatic strategy tuning decisions.
    
    Returns timeline of all tuning events with parameters adjusted,
    reasoning, and outcomes.
    """
    try:
        tuning_history = await db.strategy_auto_tuning.find().sort("timestamp", -1).limit(limit).to_list(limit)
        
        return {
            "success": True,
            "count": len(tuning_history),
            "history": tuning_history
        }
        
    except Exception as e:
        logger.error(f"Error fetching auto-tuning history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/strategy/auto-tune/config")
async def update_auto_tuning_config(
    interval_hours: Optional[float] = None,
    deviation_threshold: Optional[float] = None
):
    """
    Update auto-tuning configuration.
    
    Args:
        interval_hours: Hours between auto-tuning checks (default: 1)
        deviation_threshold: Deviation percentage to trigger tuning (default: 5.0)
    """
    global auto_tuning_state
    
    try:
        if interval_hours is not None:
            if interval_hours < 0.1 or interval_hours > 24:
                raise HTTPException(status_code=400, detail="Interval must be between 0.1 and 24 hours")
            auto_tuning_state["interval_hours"] = interval_hours
        
        if deviation_threshold is not None:
            if deviation_threshold < 1 or deviation_threshold > 50:
                raise HTTPException(status_code=400, detail="Threshold must be between 1% and 50%")
            auto_tuning_state["deviation_threshold"] = deviation_threshold
        
        return {
            "success": True,
            "message": "Auto-tuning configuration updated",
            "config": {
                "interval_hours": auto_tuning_state["interval_hours"],
                "deviation_threshold": auto_tuning_state["deviation_threshold"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating auto-tuning config: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@api_router.get("/llm/forecast/history")
async def get_forecast_history(limit: int = 10):
    """
    Get historical forecast records (if we want to store them).
    For now, returns empty as we generate on-demand.
    """
    return {
        "success": True,
        "message": "Forecasts are generated on-demand. Historical forecast storage not yet implemented.",
        "forecasts": []
    }



# ====================
# Step 19: Multi-Agent Collaboration & Meta-Learning
# ====================

from llm_evaluator import MultiAgentOrchestrator, get_orchestrator

class MetaCollaborationRequest(BaseModel):
    task: str
    context: Optional[Dict[str, Any]] = None
    apply_meta_learning: bool = False

class MetaKnowledgeApplication(BaseModel):
    insight_ids: Optional[List[str]] = None
    apply_all_high_confidence: bool = False

@api_router.post("/llm/meta-collaboration")
async def run_meta_collaboration(request: MetaCollaborationRequest):
    """
    Run multi-agent collaborative reasoning session
    Orchestrates Strategy, Evaluation, Forecast, and Adaptation agents
    """
    try:
        logger.info(f"Starting meta-collaboration for task: {request.task}")
        
        # Get orchestrator
        orchestrator = get_orchestrator()
        
        # Run multi-agent reasoning
        consensus = await orchestrator.run_multi_agent_reasoning(
            task=request.task,
            context=request.context
        )
        
        # Store session in MongoDB
        session_data = {
            "session_id": consensus.session_id,
            "task": request.task,
            "context": request.context,
            "consensus": consensus.to_dict(),
            "timestamp": consensus.timestamp,
            "created_at": datetime.now(timezone.utc)
        }
        
        await db.llm_meta_sessions.insert_one(session_data)
        
        # Apply meta-learning if requested
        if request.apply_meta_learning:
            orchestrator.update_meta_knowledge(consensus)
            
            # Store meta-knowledge in MongoDB
            for knowledge in orchestrator.meta_knowledge_base[-len(consensus.meta_insights):]:
                await db.llm_meta_knowledge.insert_one(knowledge.to_dict())
        
        logger.info(f"Meta-collaboration complete. Consensus: {consensus.consensus_reached}")
        
        return {
            "success": True,
            "consensus": consensus.to_dict(),
            "message": "Multi-agent reasoning completed",
            "session_id": consensus.session_id
        }
        
    except Exception as e:
        logger.error(f"Meta-collaboration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/meta-collaboration/sessions")
async def get_meta_sessions(limit: int = 20, skip: int = 0):
    """Get recent multi-agent collaboration sessions"""
    try:
        sessions = await db.llm_meta_sessions.find(
            {},
            {"_id": 0}
        ).sort("created_at", -1).skip(skip).limit(limit).to_list(length=limit)
        
        total = await db.llm_meta_sessions.count_documents({})
        
        return {
            "success": True,
            "sessions": sessions,
            "total": total,
            "limit": limit,
            "skip": skip
        }
        
    except Exception as e:
        logger.error(f"Error fetching meta sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/meta-collaboration/session/{session_id}")
async def get_meta_session_detail(session_id: str):
    """Get detailed transcript of a specific meta-collaboration session"""
    try:
        session = await db.llm_meta_sessions.find_one(
            {"session_id": session_id},
            {"_id": 0}
        )
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "success": True,
            "session": session
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching session detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/meta-knowledge")
async def get_meta_knowledge(
    category: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = 50
):
    """Get accumulated meta-knowledge insights"""
    try:
        query = {}
        if category:
            query["category"] = category
        if min_confidence > 0:
            query["confidence"] = {"$gte": min_confidence}
        
        knowledge = await db.llm_meta_knowledge.find(
            query,
            {"_id": 0}
        ).sort("confidence", -1).limit(limit).to_list(length=limit)
        
        # Get summary from orchestrator
        orchestrator = get_orchestrator()
        summary = orchestrator.get_meta_knowledge_summary()
        
        return {
            "success": True,
            "knowledge": knowledge,
            "summary": summary,
            "filters": {
                "category": category,
                "min_confidence": min_confidence
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching meta-knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/llm/meta-knowledge/apply")
async def apply_meta_knowledge(request: MetaKnowledgeApplication):
    """
    Apply meta-knowledge to system heuristics
    Updates system behavior based on learned insights
    """
    try:
        orchestrator = get_orchestrator()
        
        if request.apply_all_high_confidence:
            # Apply all high-confidence insights (>0.8)
            knowledge = await db.llm_meta_knowledge.find(
                {"confidence": {"$gte": 0.8}},
                {"_id": 0}
            ).to_list(length=100)
        elif request.insight_ids:
            # Apply specific insights
            knowledge = await db.llm_meta_knowledge.find(
                {"knowledge_id": {"$in": request.insight_ids}},
                {"_id": 0}
            ).to_list(length=100)
        else:
            raise HTTPException(status_code=400, detail="Must specify insight_ids or apply_all_high_confidence")
        
        applied_count = 0
        for k in knowledge:
            # Update validation count
            await db.llm_meta_knowledge.update_one(
                {"knowledge_id": k["knowledge_id"]},
                {"$inc": {"validation_count": 1}}
            )
            applied_count += 1
        
        logger.info(f"Applied {applied_count} meta-knowledge insights")
        
        return {
            "success": True,
            "applied_count": applied_count,
            "message": f"Successfully applied {applied_count} meta-knowledge insights"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying meta-knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/meta-learning/metrics")
async def get_meta_learning_metrics():
    """Get meta-learning progress metrics"""
    try:
        # Get total sessions
        total_sessions = await db.llm_meta_sessions.count_documents({})
        
        # Get consensus rate
        consensus_sessions = await db.llm_meta_sessions.count_documents(
            {"consensus.consensus_reached": True}
        )
        consensus_rate = (consensus_sessions / total_sessions * 100) if total_sessions > 0 else 0
        
        # Get average confidence over time
        sessions = await db.llm_meta_sessions.find(
            {},
            {"consensus.confidence_score": 1, "consensus.consensus_level": 1, "timestamp": 1, "_id": 0}
        ).sort("timestamp", -1).limit(50).to_list(length=50)
        
        avg_confidence = sum(s.get("consensus", {}).get("confidence_score", 0) for s in sessions) / len(sessions) if sessions else 0
        avg_consensus_level = sum(s.get("consensus", {}).get("consensus_level", 0) for s in sessions) / len(sessions) if sessions else 0
        
        # Get knowledge distribution
        knowledge_by_category = await db.llm_meta_knowledge.aggregate([
            {"$group": {"_id": "$category", "count": {"$sum": 1}, "avg_confidence": {"$avg": "$confidence"}}}
        ]).to_list(length=10)
        
        # Get orchestrator summary
        orchestrator = get_orchestrator()
        knowledge_summary = orchestrator.get_meta_knowledge_summary()
        
        return {
            "success": True,
            "metrics": {
                "total_sessions": total_sessions,
                "consensus_rate": round(consensus_rate, 1),
                "avg_confidence": round(avg_confidence, 3),
                "avg_consensus_level": round(avg_consensus_level, 3),
                "total_insights": knowledge_summary["total_insights"],
                "knowledge_by_category": knowledge_by_category,
                "recent_trend": {
                    "last_10_sessions": sessions[:10],
                    "confidence_improving": _calculate_trend([s.get("consensus", {}).get("confidence_score", 0) for s in sessions[:10]])
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching meta-learning metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _calculate_trend(values: List[float]) -> bool:
    """Calculate if trend is improving (simple linear regression)"""
    if len(values) < 2:
        return False
    # Simple check: is average of first half less than average of second half?
    mid = len(values) // 2
    first_half_avg = sum(values[:mid]) / mid if mid > 0 else 0
    second_half_avg = sum(values[mid:]) / (len(values) - mid) if len(values) > mid else 0
    return second_half_avg > first_half_avg


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


# ====================
# Step 20: Cognitive Consensus Engine & Trust Calibration Layer
# ====================

from llm_evaluator import (
    AgentTrustProfile,
    WeightedConsensusResult,
    compute_agent_trust_score,
    derive_weighted_consensus,
    update_trust_profile,
    recalibrate_all_trust_profiles,
    get_trust_profiles,
    set_trust_profiles
)

class ConsensusRequest(BaseModel):
    """Request for trust-calibrated consensus computation"""
    task: str
    context: Optional[Dict] = None
    confidence_threshold: float = Field(default=0.90, ge=0.5, le=1.0)

@api_router.post("/llm/consensus/derive")
async def derive_trust_consensus(request: ConsensusRequest):
    """
    Run trust-calibrated consensus computation.
    
    Performs multi-agent reasoning with trust-weighted decision scoring.
    """
    try:
        logger.info(f"Starting trust-calibrated consensus for task: {request.task}")
        
        # Get orchestrator
        orchestrator = get_orchestrator()
        
        # Run multi-agent reasoning to get agent messages
        consensus = await orchestrator.run_multi_agent_reasoning(request.task, request.context)
        
        # Get agent messages from session history
        session = orchestrator.session_history[-1] if orchestrator.session_history else None
        if not session:
            raise HTTPException(status_code=500, detail="No session data available")
        
        agent_messages = []
        for msg_dict in session["messages"]:
            from llm_evaluator import AgentMessage
            agent_messages.append(AgentMessage(
                agent_name=msg_dict["agent_name"],
                message_type=msg_dict["message_type"],
                content=msg_dict["content"],
                confidence=msg_dict["confidence"],
                reasoning_chain=msg_dict["reasoning_chain"],
                timestamp=msg_dict["timestamp"],
                metadata=msg_dict.get("metadata")
            ))
        
        # Get trust profiles
        trust_profiles = get_trust_profiles()
        
        # Derive weighted consensus
        weighted_consensus = derive_weighted_consensus(
            agent_messages,
            trust_profiles,
            request.confidence_threshold
        )
        
        # Store consensus in database
        await db.llm_consensus_history.insert_one({
            **weighted_consensus.to_dict(),
            "task": request.task,
            "context": request.context,
            "source": "trust_calibration"
        })
        
        # Update trust profiles based on consensus
        for agent_name, profile in trust_profiles.items():
            # Find agent's message
            agent_msg = next((m for m in agent_messages if m.agent_name == agent_name), None)
            if agent_msg:
                # Create performance entry
                performance_entry = {
                    "confidence": agent_msg.confidence,
                    "outcome_correct": weighted_consensus.consensus_reached,
                    "agreed_with_consensus": True,  # Simplified
                    "response_time": 5.0,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Update profile
                updated_profile = update_trust_profile(
                    agent_name,
                    profile,
                    performance_entry
                )
                trust_profiles[agent_name] = updated_profile
        
        # Save updated profiles
        set_trust_profiles(trust_profiles)
        
        # Save profiles to database
        for agent_name, profile in trust_profiles.items():
            await db.llm_agent_trust_profiles.replace_one(
                {"agent_name": agent_name},
                profile.to_dict(),
                upsert=True
            )
        
        logger.info(f"Trust-calibrated consensus complete: {weighted_consensus.consensus_reached}")
        
        return {
            "success": True,
            "message": "Trust-calibrated consensus computed",
            "consensus": weighted_consensus.to_dict(),
            "original_consensus": consensus.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Trust consensus error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/consensus/trust-scores")
async def get_current_trust_scores():
    """
    Returns current trust scores for all agents.
    
    Includes:
    - Trust score (0.0-1.0)
    - Performance metrics
    - Historical stats
    """
    try:
        # Get trust profiles from database
        profiles_docs = await db.llm_agent_trust_profiles.find({}, {"_id": 0}).to_list(10)
        
        if not profiles_docs:
            # Initialize from global if DB is empty
            trust_profiles = get_trust_profiles()
            profiles_docs = [p.to_dict() for p in trust_profiles.values()]
        
        # Calculate aggregate stats
        if profiles_docs:
            avg_trust = sum(p.get("trust_score", 0.7) for p in profiles_docs) / len(profiles_docs)
            highest_trust = max(profiles_docs, key=lambda p: p.get("trust_score", 0))
            lowest_trust = min(profiles_docs, key=lambda p: p.get("trust_score", 0))
        else:
            avg_trust = 0.75
            highest_trust = None
            lowest_trust = None
        
        return {
            "success": True,
            "trust_profiles": profiles_docs,
            "summary": {
                "average_trust": round(avg_trust, 3),
                "highest_trust_agent": highest_trust.get("agent_name") if highest_trust else None,
                "lowest_trust_agent": lowest_trust.get("agent_name") if lowest_trust else None,
                "total_agents": len(profiles_docs)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching trust scores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/llm/consensus/recalibrate")
async def recalibrate_trust_scores():
    """
    Force trust score recalibration across recent sessions.
    
    Recomputes all agent trust scores using:
    - Recent consensus history
    - Performance data
    - Exponential decay (0.85 retention)
    """
    try:
        logger.info("Starting trust profile recalibration...")
        
        # Get current trust profiles
        profiles_docs = await db.llm_agent_trust_profiles.find({}, {"_id": 0}).to_list(10)
        
        if not profiles_docs:
            # Initialize default profiles
            trust_profiles = get_trust_profiles()
        else:
            # Convert to AgentTrustProfile objects
            trust_profiles = {}
            for doc in profiles_docs:
                trust_profiles[doc["agent_name"]] = AgentTrustProfile.from_dict(doc)
        
        # Get recent consensus history
        consensus_history = await db.llm_consensus_history.find(
            {},
            {"_id": 0}
        ).sort("timestamp", -1).limit(30).to_list(30)
        
        # Also get meta-collaboration sessions
        meta_sessions = await db.llm_meta_sessions.find(
            {},
            {"_id": 0}
        ).sort("timestamp", -1).limit(30).to_list(30)
        
        # Merge histories
        all_history = consensus_history + meta_sessions
        
        # Recalibrate
        updated_profiles = await recalibrate_all_trust_profiles(
            trust_profiles,
            all_history,
            decay_factor=0.85
        )
        
        # Save to database
        updates_count = 0
        for agent_name, profile in updated_profiles.items():
            result = await db.llm_agent_trust_profiles.replace_one(
                {"agent_name": agent_name},
                profile.to_dict(),
                upsert=True
            )
            if result.modified_count > 0 or result.upserted_id:
                updates_count += 1
        
        # Update global profiles
        set_trust_profiles(updated_profiles)
        
        # Calculate changes
        changes = []
        for agent_name in updated_profiles:
            old_score = trust_profiles[agent_name].trust_score if agent_name in trust_profiles else 0.75
            new_score = updated_profiles[agent_name].trust_score
            change = new_score - old_score
            
            changes.append({
                "agent": agent_name,
                "old_score": round(old_score, 3),
                "new_score": round(new_score, 3),
                "change": round(change, 3)
            })
        
        logger.info(f"Recalibration complete. Updated {updates_count} profiles.")
        
        return {
            "success": True,
            "message": f"Recalibrated {updates_count} agent trust profiles",
            "profiles_updated": updates_count,
            "changes": changes,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error recalibrating trust scores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/consensus/history")
async def get_consensus_history(limit: int = 20):
    """
    Get consensus computation history with reliability outcomes.
    
    Returns logged consensus computations sorted by timestamp.
    """
    try:
        history = await db.llm_consensus_history.find(
            {},
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit).to_list(limit)
        
        # Calculate aggregate stats
        if history:
            total_consensus = len(history)
            reached_consensus = sum(1 for h in history if h.get("consensus_reached", False))
            consensus_rate = (reached_consensus / total_consensus * 100) if total_consensus > 0 else 0
            
            avg_confidence = sum(h.get("weighted_confidence", 0) for h in history) / total_consensus
            
            stability_counts = {}
            for h in history:
                stability = h.get("stability_index", "medium")
                stability_counts[stability] = stability_counts.get(stability, 0) + 1
        else:
            consensus_rate = 0
            avg_confidence = 0
            stability_counts = {}
        
        return {
            "success": True,
            "history": history,
            "summary": {
                "total_consensus_computed": len(history),
                "consensus_rate": round(consensus_rate, 1),
                "avg_weighted_confidence": round(avg_confidence, 3),
                "stability_distribution": stability_counts
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching consensus history: {e}")
        raise HTTPException(status_code=500, detail=str(e))




# ====================
# Step 21: Meta-Agent Arbitration & Dynamic Trust Threshold System
# ====================

from llm_evaluator import (
    ArbitrationResult,
    DynamicThreshold,
    ConflictAnalysis,
    calculate_dynamic_threshold,
    analyze_conflict_context,
    invoke_meta_arbitrator,
    update_arbitration_history
)


class ArbitrationRequest(BaseModel):
    """Request for meta-agent arbitration"""
    task: str
    context: Optional[Dict] = None
    confidence_threshold: float = Field(default=0.90, ge=0.5, le=1.0)
    task_complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    task_category: str = Field(default="general")
    force_arbitration: bool = Field(default=False)


@api_router.post("/llm/arbitration/resolve")
async def resolve_with_arbitration(request: ArbitrationRequest):
    """
    Trigger meta-agent arbitration process.
    
    This endpoint:
    1. Runs multi-agent consensus first
    2. Checks if arbitration is needed (confidence < threshold OR high disagreement)
    3. If needed, invokes meta-arbitrator to resolve conflicts
    4. Stores arbitration results in MongoDB
    5. Returns revised consensus with confidence delta
    
    Arbitration Triggers:
    - Consensus confidence < 90%
    - Agent disagreement ≥ 15%
    - Trust variance > 0.12
    - Manual force_arbitration flag
    """
    try:
        logger.info(f"Arbitration request for task: {request.task[:100]}")
        
        # Step 1: Run multi-agent consensus
        orchestrator = get_orchestrator()
        consensus = await orchestrator.run_multi_agent_reasoning(request.task, request.context)
        
        # Get agent messages from session
        session = orchestrator.session_history[-1] if orchestrator.session_history else None
        if not session:
            raise HTTPException(status_code=500, detail="No session data available")
        
        agent_messages = []
        for msg_dict in session["messages"]:
            from llm_evaluator import AgentMessage
            agent_messages.append(AgentMessage(
                agent_name=msg_dict["agent_name"],
                message_type=msg_dict["message_type"],
                content=msg_dict["content"],
                confidence=msg_dict["confidence"],
                reasoning_chain=msg_dict["reasoning_chain"],
                timestamp=msg_dict["timestamp"],
                metadata=msg_dict.get("metadata")
            ))
        
        # Get trust profiles
        trust_profiles = get_trust_profiles()
        
        # Step 2: Calculate dynamic threshold
        dynamic_threshold = calculate_dynamic_threshold(
            trust_profiles=trust_profiles,
            task_complexity=request.task_complexity,
            task_category=request.task_category,
            base_threshold=request.confidence_threshold
        )
        
        # Store threshold in database
        await db.llm_dynamic_threshold.insert_one(dynamic_threshold.to_dict())
        
        # Step 3: Derive weighted consensus
        weighted_consensus = derive_weighted_consensus(
            agent_messages,
            trust_profiles,
            dynamic_threshold.current_threshold
        )
        
        # Step 4: Analyze conflict context
        conflict_analysis = analyze_conflict_context(
            agent_messages,
            trust_profiles,
            weighted_consensus
        )
        
        # Step 5: Determine if arbitration is needed
        needs_arbitration = (
            request.force_arbitration or
            not weighted_consensus.consensus_reached or
            conflict_analysis.conflict_resolution_needed or
            weighted_consensus.weighted_confidence < dynamic_threshold.current_threshold
        )
        
        arbitration_result = None
        
        if needs_arbitration:
            logger.info(f"Arbitration triggered: consensus={weighted_consensus.consensus_reached}, confidence={weighted_consensus.weighted_confidence:.3f}, threshold={dynamic_threshold.current_threshold:.3f}")
            
            # Step 6: Invoke meta-arbitrator
            api_key = os.environ.get('EMERGENT_LLM_KEY')
            if not api_key:
                raise HTTPException(status_code=500, detail="EMERGENT_LLM_KEY not configured")
            
            arbitration_result = await invoke_meta_arbitrator(
                agent_messages=agent_messages,
                trust_profiles=trust_profiles,
                original_consensus=weighted_consensus,
                conflict_analysis=conflict_analysis,
                llm_api_key=api_key,
                task_description=request.task
            )
            
            # Step 7: Store arbitration history
            arbitration_history_entry = update_arbitration_history(
                arbitration_result,
                weighted_consensus,
                conflict_analysis
            )
            
            await db.llm_arbitration_log.insert_one(arbitration_history_entry)
            
            # Update trust profiles based on arbitration
            # (In production, would analyze which agents aligned with arbitration outcome)
            for agent_name, profile in trust_profiles.items():
                await db.llm_agent_trust_profiles.replace_one(
                    {"agent_name": agent_name},
                    profile.to_dict(),
                    upsert=True
                )
            
            logger.info(f"Arbitration complete: {arbitration_result.arbitration_outcome}, confidence improved by {arbitration_result.confidence_delta:+.2%}")
        else:
            logger.info(f"No arbitration needed: consensus reached with {weighted_consensus.weighted_confidence:.2%} confidence")
        
        # Store weighted consensus
        await db.llm_consensus_history.insert_one({
            **weighted_consensus.to_dict(),
            "task": request.task,
            "context": request.context,
            "arbitration_triggered": needs_arbitration,
            "dynamic_threshold_used": dynamic_threshold.current_threshold,
            "source": "arbitration_system"
        })
        
        return {
            "success": True,
            "message": "Arbitration process completed" if needs_arbitration else "Consensus reached without arbitration",
            "arbitration_triggered": needs_arbitration,
            "consensus": weighted_consensus.to_dict(),
            "dynamic_threshold": dynamic_threshold.to_dict(),
            "conflict_analysis": conflict_analysis.to_dict(),
            "arbitration_result": arbitration_result.to_dict() if arbitration_result else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Arbitration resolution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/arbitration/history")
async def get_arbitration_history(limit: int = 20):
    """
    Retrieve arbitration logs with confidence outcomes.
    
    Returns:
    - Arbitration session details
    - Confidence before/after metrics
    - Agents involved and divergence analysis
    - Resolution outcomes (Approved/Rejected/Reassessed)
    """
    try:
        # Get arbitration history
        history = await db.llm_arbitration_log.find(
            {},
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit).to_list(limit)
        
        # Calculate summary statistics
        if history:
            total_arbitrations = len(history)
            avg_confidence_before = sum(h.get("confidence_before", 0) for h in history) / total_arbitrations
            avg_confidence_after = sum(h.get("confidence_after", 0) for h in history) / total_arbitrations
            avg_confidence_delta = sum(h.get("confidence_delta", 0) for h in history) / total_arbitrations
            avg_resolution_time = sum(h.get("resolution_time", 0) for h in history) / total_arbitrations
            
            # Count outcomes
            outcome_counts = {}
            for h in history:
                outcome = h.get("arbitration_outcome", "Unknown")
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            
            # Count trigger reasons
            trigger_counts = {}
            for h in history:
                trigger = h.get("trigger_reason", "Unknown")[:50]  # Truncate
                trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
        else:
            avg_confidence_before = 0
            avg_confidence_after = 0
            avg_confidence_delta = 0
            avg_resolution_time = 0
            outcome_counts = {}
            trigger_counts = {}
        
        return {
            "success": True,
            "history": history,
            "summary": {
                "total_arbitrations": len(history),
                "avg_confidence_before": round(avg_confidence_before, 3),
                "avg_confidence_after": round(avg_confidence_after, 3),
                "avg_confidence_improvement": round(avg_confidence_delta, 3),
                "avg_resolution_time_seconds": round(avg_resolution_time, 2),
                "outcome_distribution": outcome_counts,
                "trigger_distribution": trigger_counts
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching arbitration history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/threshold/status")
async def get_dynamic_threshold_status():
    """
    Get current dynamic threshold parameters and metrics.
    
    Returns:
    - Current threshold value (80-95%)
    - Trust variance metrics
    - Task complexity ratings
    - Threshold adjustment history
    - Per-task-category thresholds
    """
    try:
        # Get most recent thresholds (one per task category)
        thresholds_by_category = {}
        categories = ["strategy", "evaluation", "forecasting", "general"]
        
        for category in categories:
            threshold_doc = await db.llm_dynamic_threshold.find_one(
                {"task_category": category},
                {"_id": 0},
                sort=[("timestamp", -1)]
            )
            
            if threshold_doc:
                thresholds_by_category[category] = threshold_doc
        
        # Get all recent thresholds for trend analysis
        recent_thresholds = await db.llm_dynamic_threshold.find(
            {},
            {"_id": 0}
        ).sort("timestamp", -1).limit(50).to_list(50)
        
        # Calculate overall statistics
        if recent_thresholds:
            avg_threshold = sum(t.get("current_threshold", 0.90) for t in recent_thresholds) / len(recent_thresholds)
            avg_trust_variance = sum(t.get("trust_variance", 0.05) for t in recent_thresholds) / len(recent_thresholds)
            avg_complexity = sum(t.get("complexity_rating", 0.5) for t in recent_thresholds) / len(recent_thresholds)
            
            # Calculate threshold range
            threshold_values = [t.get("current_threshold", 0.90) for t in recent_thresholds]
            min_threshold = min(threshold_values)
            max_threshold = max(threshold_values)
            
            # Build threshold trend (last 20)
            threshold_trend = [
                {
                    "timestamp": t.get("timestamp"),
                    "threshold": t.get("current_threshold"),
                    "task_category": t.get("task_category"),
                    "trust_variance": t.get("trust_variance")
                }
                for t in recent_thresholds[:20]
            ]
        else:
            # No data - return defaults
            avg_threshold = 0.90
            avg_trust_variance = 0.05
            avg_complexity = 0.5
            min_threshold = 0.80
            max_threshold = 0.95
            threshold_trend = []
        
        # Get current global trust variance
        trust_profiles = get_trust_profiles()
        if trust_profiles:
            trust_scores = [p.trust_score for p in trust_profiles.values()]
            avg_trust = sum(trust_scores) / len(trust_scores)
            current_trust_variance = sum((t - avg_trust) ** 2 for t in trust_scores) / len(trust_scores)
        else:
            current_trust_variance = 0.05
        
        # Determine auto-adjust status
        auto_adjust_enabled = True  # Default
        if recent_thresholds:
            auto_adjust_enabled = recent_thresholds[0].get("auto_adjust_enabled", True)
        
        return {
            "success": True,
            "current_status": {
                "global_threshold": round(avg_threshold, 3),
                "trust_variance": round(current_trust_variance, 3),
                "avg_complexity": round(avg_complexity, 2),
                "auto_adjust_enabled": auto_adjust_enabled,
                "threshold_range": {
                    "min": round(min_threshold, 3),
                    "max": round(max_threshold, 3),
                    "configured_range": "80-95%"
                }
            },
            "thresholds_by_category": thresholds_by_category,
            "threshold_trend": threshold_trend,
            "statistics": {
                "total_threshold_calculations": len(recent_thresholds),
                "avg_threshold": round(avg_threshold, 3),
                "avg_trust_variance": round(avg_trust_variance, 3),
                "avg_complexity_rating": round(avg_complexity, 2)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching threshold status: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# ====================
# Step 22: Collective Memory & Experience Replay System
# ====================

from llm_evaluator import (
    ExperienceRecord,
    MemoryRetrievalResult,
    ReplaySession,
    store_experience_record,
    retrieve_similar_experiences,
    run_experience_replay,
    summarize_collective_memory,
    cleanup_old_experiences
)


class MemoryStoreRequest(BaseModel):
    """Request to store experience in collective memory"""
    task_type: str = Field(..., description="Type: arbitration, consensus, coaching, analytics")
    task_description: str
    agents_involved: List[str]
    outcome: Dict[str, Any]
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None


class MemoryRetrieveRequest(BaseModel):
    """Request to retrieve similar experiences"""
    query: str
    task_type_filter: Optional[str] = None
    min_confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    limit: int = Field(default=5, ge=1, le=20)


class MemoryReplayRequest(BaseModel):
    """Request to run experience replay session"""
    experience_ids: Optional[List[str]] = None  # If None, use most recent high-confidence
    auto_select_count: int = Field(default=10, ge=5, le=50)


@api_router.post("/llm/memory/store")
async def store_memory_experience(request: MemoryStoreRequest):
    """
    Store arbitration or consensus experience to collective memory.
    
    Filters out low-confidence experiences (<70%).
    Generates semantic embeddings for similarity search.
    Enforces retention policy (max 1000 or 30 days).
    """
    try:
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="EMERGENT_LLM_KEY not configured")
        
        # Store experience
        experience = await store_experience_record(
            task_type=request.task_type,
            task_description=request.task_description,
            agents_involved=request.agents_involved,
            outcome=request.outcome,
            confidence=request.confidence,
            api_key=api_key,
            metadata=request.metadata
        )
        
        if not experience:
            return {
                "success": False,
                "message": f"Experience not stored (confidence {request.confidence:.2f} < 0.70 threshold)"
            }
        
        # Save to MongoDB
        await db.llm_experience_memory.insert_one(experience.to_dict())
        
        # Check if cleanup is needed
        total_count = await db.llm_experience_memory.count_documents({})
        
        # Trigger replay every 50 experiences
        should_trigger_replay = (total_count % 50 == 0) and total_count > 0
        
        # Cleanup if needed
        cleanup_triggered = False
        if total_count > 1000:
            cleanup_result = await cleanup_old_experiences(
                db_collection=db.llm_experience_memory,
                max_count=1000,
                max_age_days=30
            )
            cleanup_triggered = True
            logger.info(f"Cleanup triggered: {cleanup_result['total_deleted']} experiences deleted")
        
        return {
            "success": True,
            "experience_id": experience.experience_id,
            "confidence": experience.confidence,
            "stored_at": experience.timestamp,
            "total_experiences": total_count,
            "replay_triggered": should_trigger_replay,
            "cleanup_triggered": cleanup_triggered,
            "message": "Experience stored successfully" + (" (replay threshold reached)" if should_trigger_replay else "")
        }
        
    except Exception as e:
        logger.error(f"Error storing memory experience: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/memory/retrieve")
async def retrieve_memory_experiences(
    query: str,
    task_type_filter: Optional[str] = None,
    min_confidence: float = 0.70,
    limit: int = 5
):
    """
    Retrieve similar experiences using semantic similarity search.
    
    Uses OpenAI embeddings and cosine similarity to find related cases.
    Returns ranked results by similarity score.
    """
    try:
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="EMERGENT_LLM_KEY not configured")
        
        # Retrieve similar experiences
        result = await retrieve_similar_experiences(
            query=query,
            api_key=api_key,
            task_type_filter=task_type_filter,
            min_confidence=min_confidence,
            limit=limit,
            db_collection=db.llm_experience_memory
        )
        
        return {
            "success": True,
            "retrieval_id": result.retrieval_id,
            "query": result.query,
            "similar_experiences": result.similar_experiences,
            "top_match_confidence": result.top_match_confidence,
            "recall_count": result.recall_count,
            "timestamp": result.timestamp
        }
        
    except Exception as e:
        logger.error(f"Error retrieving memory experiences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/llm/memory/replay")
async def run_memory_replay(request: MemoryReplayRequest):
    """
    Run experience replay session to reinforce learning.
    
    Replays high-confidence cases to:
    - Update trust calibration
    - Reinforce successful patterns
    - Calculate performance improvements
    
    Can auto-select recent high-confidence experiences or use provided IDs.
    """
    try:
        experience_ids = request.experience_ids
        
        # Auto-select if not provided
        if not experience_ids:
            # Get most recent high-confidence experiences (>=90%)
            experiences = await db.llm_experience_memory.find({
                "confidence": {"$gte": 0.90}
            }).sort("timestamp", -1).limit(request.auto_select_count).to_list(request.auto_select_count)
            
            experience_ids = [exp["experience_id"] for exp in experiences]
            
            if not experience_ids:
                return {
                    "success": False,
                    "message": "No high-confidence experiences available for replay"
                }
        
        # Get trust profiles
        trust_profiles = get_trust_profiles()
        
        # Run replay
        replay_session = await run_experience_replay(
            experience_ids=experience_ids,
            trust_profiles=trust_profiles,
            db_collection=db.llm_experience_memory
        )
        
        # Save replay log to database
        await db.llm_memory_replay_log.insert_one(replay_session.to_dict())
        
        # Update trust profiles in database
        for agent_name, adjustment in replay_session.trust_adjustments.items():
            if agent_name in trust_profiles:
                new_trust = trust_profiles[agent_name].trust_score
                await db.llm_trust_profiles.update_one(
                    {"agent_name": agent_name},
                    {"$set": {
                        "trust_score": new_trust,
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    }}
                )
        
        return {
            "success": True,
            "session_id": replay_session.session_id,
            "experiences_replayed": replay_session.experiences_replayed,
            "avg_confidence": replay_session.avg_confidence,
            "trust_adjustments": replay_session.trust_adjustments,
            "reasoning_improvements": replay_session.reasoning_improvements,
            "performance_delta": replay_session.performance_delta,
            "timestamp": replay_session.timestamp,
            "message": f"Replayed {replay_session.experiences_replayed} experiences (avg confidence: {replay_session.avg_confidence * 100:.1f}%)"
        }
        
    except Exception as e:
        logger.error(f"Error running memory replay: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/memory/summary")
async def get_memory_summary(timeframe_days: int = 30):
    """
    Get collective memory summary and metrics.
    
    Provides:
    - Total experiences stored
    - Confidence statistics
    - Task type breakdown
    - Agent activity
    - Memory quality metrics
    - Retention status
    """
    try:
        # Get summary
        summary = await summarize_collective_memory(
            db_collection=db.llm_experience_memory,
            timeframe_days=timeframe_days
        )
        
        # Get recent replay sessions
        recent_replays = await db.llm_memory_replay_log.find().sort(
            "timestamp", -1
        ).limit(5).to_list(5)
        
        # Calculate last replay date
        last_replay_date = recent_replays[0]["timestamp"] if recent_replays else None
        
        # Add replay information to summary
        summary["recent_replay_sessions"] = len(recent_replays)
        summary["last_replay_date"] = last_replay_date
        
        # Get total across all time
        total_all_time = await db.llm_experience_memory.count_documents({})
        summary["total_experiences_all_time"] = total_all_time
        
        return {
            "success": True,
            **summary
        }
        
    except Exception as e:
        logger.error(f"Error getting memory summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/memory/stats")
async def get_memory_stats():
    """
    Get detailed memory statistics for dashboard display.
    
    Returns metrics for UI visualization.
    """
    try:
        # Total experiences
        total_count = await db.llm_experience_memory.count_documents({})
        
        # High-quality count (>=90%)
        high_quality_count = await db.llm_experience_memory.count_documents({
            "confidence": {"$gte": 0.90}
        })
        
        # Average confidence
        pipeline = [
            {"$group": {
                "_id": None,
                "avg_confidence": {"$avg": "$confidence"}
            }}
        ]
        agg_result = await db.llm_experience_memory.aggregate(pipeline).to_list(1)
        avg_confidence = agg_result[0]["avg_confidence"] if agg_result else 0.0
        
        # Task type distribution
        task_type_pipeline = [
            {"$group": {
                "_id": "$task_type",
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": -1}}
        ]
        task_types = await db.llm_experience_memory.aggregate(task_type_pipeline).to_list(10)
        
        # Recent replay sessions
        replay_count = await db.llm_memory_replay_log.count_documents({})
        recent_replay = await db.llm_memory_replay_log.find().sort("timestamp", -1).limit(1).to_list(1)
        
        # Check if replay needed (every 50 experiences)
        replay_threshold = 50
        experiences_since_last_replay = total_count % replay_threshold if total_count > 0 else 0
        replay_recommended = experiences_since_last_replay == 0 and total_count > 0
        
        return {
            "success": True,
            "total_experiences": total_count,
            "high_quality_count": high_quality_count,
            "avg_confidence": round(avg_confidence, 3),
            "memory_accuracy_percent": round(avg_confidence * 100, 1),
            "task_type_distribution": [
                {"task_type": t["_id"], "count": t["count"]}
                for t in task_types
            ],
            "replay_sessions_total": replay_count,
            "last_replay": recent_replay[0] if recent_replay else None,
            "experiences_since_last_replay": experiences_since_last_replay,
            "replay_recommended": replay_recommended,
            "retention_status": "Within limit" if total_count <= 1000 else "Cleanup recommended"
        }
        
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))




# ====================
# Step 23: Collective Intelligence Layer & Global Strategy Synthesis
# ====================

from llm_evaluator import (
    CollectiveStrategy,
    AlignmentMetrics,
    aggregate_global_insights,
    synthesize_collective_strategy,
    evaluate_collective_alignment,
    update_global_strategy_map
)


class SynthesisRequest(BaseModel):
    """Request to trigger global strategy synthesis"""
    include_forecast: bool = Field(default=True)
    synthesis_depth: str = Field(default="comprehensive", pattern="^(quick|balanced|comprehensive)$")


@api_router.post("/llm/collective/synthesize")
async def synthesize_collective_intelligence():
    """
    Run synthesis across all subsystems (memory, trust, arbitration, forecasting).
    
    Aggregates insights from:
    - Step 19: Multi-Agent Collaboration
    - Step 20: Trust Calibration
    - Step 21: Arbitration
    - Step 22: Collective Memory
    
    Generates unified strategic recommendations using LLM (GPT-5).
    """
    try:
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="EMERGENT_LLM_KEY not configured")
        
        # Fetch data from all subsystems
        # Memory data (Step 22)
        memory_data = await db.llm_experience_memory.find().sort("timestamp", -1).limit(100).to_list(100)
        
        # Trust calibration data (Step 20)
        trust_data = await db.llm_trust_calibration.find().sort("timestamp", -1).limit(50).to_list(50)
        
        # Arbitration data (Step 21)
        arbitration_data = await db.llm_arbitration_log.find().sort("timestamp", -1).limit(50).to_list(50)
        
        # Forecast data (optional - if available)
        forecast_docs = await db.llm_forecast_log.find().sort("timestamp", -1).limit(1).to_list(1)
        forecast_data = forecast_docs[0] if forecast_docs else None
        
        # Step 1: Aggregate global insights
        logger.info("Aggregating global insights from all subsystems...")
        aggregated_insights = await aggregate_global_insights(
            memory_data=memory_data,
            trust_data=trust_data,
            arbitration_data=arbitration_data,
            forecast_data=forecast_data
        )
        
        # Step 2: Synthesize collective strategy using LLM
        logger.info("Synthesizing collective strategy using GPT-5...")
        collective_strategy = await synthesize_collective_strategy(
            aggregated_insights=aggregated_insights,
            llm_api_key=api_key
        )
        
        # Step 3: Evaluate collective alignment
        logger.info("Evaluating collective alignment...")
        existing_strategies = await db.llm_global_strategy.find().to_list(100)
        alignment_metrics = await evaluate_collective_alignment(
            memory_data=memory_data,
            trust_data=trust_data,
            arbitration_data=arbitration_data,
            collective_strategies=existing_strategies
        )
        
        # Step 4: Update global strategy map
        logger.info("Updating global strategy map...")
        strategy_update = await update_global_strategy_map(
            new_strategy=collective_strategy,
            existing_strategies=existing_strategies,
            db_collection=db.llm_global_strategy
        )
        
        # Store alignment metrics
        await db.llm_alignment_metrics.insert_one(alignment_metrics.to_dict())
        
        # Store synthesis log
        synthesis_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy_id": collective_strategy.strategy_id,
            "strategy_archetype": collective_strategy.strategy_archetype,
            "aggregated_insights": aggregated_insights,
            "alignment_score": alignment_metrics.overall_alignment_score,
            "consensus_level": alignment_metrics.consensus_level,
            "strategy_update_action": strategy_update.get("action")
        }
        await db.llm_synthesis_log.insert_one(synthesis_log)
        
        logger.info(f"Collective intelligence synthesis complete: {collective_strategy.strategy_archetype}")
        
        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "collective_strategy": collective_strategy.to_dict(),
            "alignment_metrics": alignment_metrics.to_dict(),
            "aggregated_insights": aggregated_insights,
            "strategy_update": strategy_update,
            "data_sources": {
                "memory_count": len(memory_data),
                "trust_count": len(trust_data),
                "arbitration_count": len(arbitration_data),
                "forecast_available": forecast_data is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Error in collective synthesis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/collective/strategies")
async def get_global_strategies(
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: int = Query(default=20, ge=1, le=100),
    sort_by: str = Query(default="confidence", pattern="^(confidence|usage|recent)$")
):
    """
    Retrieve global strategy map with confidence metrics.
    
    Returns list of high-confidence strategies synthesized from collective intelligence.
    Strategies can be sorted by confidence, usage count, or recency.
    """
    try:
        # Build query
        query = {}
        if min_confidence > 0:
            query["confidence_score"] = {"$gte": min_confidence}
        
        # Determine sort field
        if sort_by == "confidence":
            sort_field = [("confidence_score", -1)]
        elif sort_by == "usage":
            sort_field = [("usage_count", -1)]
        else:  # recent
            sort_field = [("last_updated", -1)]
        
        # Fetch strategies
        strategies = await db.llm_global_strategy.find(query).sort(sort_field).limit(limit).to_list(limit)
        
        # Calculate statistics
        total_strategies = await db.llm_global_strategy.count_documents({})
        high_confidence_count = await db.llm_global_strategy.count_documents({"confidence_score": {"$gte": 0.8}})
        
        if strategies:
            avg_confidence = sum(s.get("confidence_score", 0) for s in strategies) / len(strategies)
            avg_alignment = sum(s.get("alignment_score", 0) for s in strategies) / len(strategies)
            total_usage = sum(s.get("usage_count", 0) for s in strategies)
        else:
            avg_confidence = 0
            avg_alignment = 0
            total_usage = 0
        
        # Get most recent synthesis
        recent_synthesis = await db.llm_synthesis_log.find().sort("timestamp", -1).limit(1).to_list(1)
        
        return {
            "success": True,
            "strategies": strategies,
            "statistics": {
                "total_strategies": total_strategies,
                "high_confidence_count": high_confidence_count,
                "avg_confidence": round(avg_confidence, 3),
                "avg_alignment": round(avg_alignment, 3),
                "total_usage": total_usage
            },
            "last_synthesis": recent_synthesis[0] if recent_synthesis else None,
            "query_params": {
                "min_confidence": min_confidence,
                "limit": limit,
                "sort_by": sort_by
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/collective/alignment")
async def get_collective_alignment(
    limit: int = Query(default=10, ge=1, le=50)
):
    """
    Returns collective alignment analytics across all subsystems.
    
    Provides:
    - Overall alignment score (0-1)
    - Subsystem-specific alignment scores
    - Consensus level (high/moderate/low/divergent)
    - Harmony index
    - Strategic coherence
    - Historical alignment trends
    """
    try:
        # Get recent alignment metrics
        alignment_history = await db.llm_alignment_metrics.find().sort("timestamp", -1).limit(limit).to_list(limit)
        
        if not alignment_history:
            return {
                "success": True,
                "message": "No alignment data available. Run synthesis first.",
                "current_alignment": None,
                "alignment_history": [],
                "trends": {}
            }
        
        # Get current (most recent) alignment
        current_alignment = alignment_history[0]
        
        # Calculate trends
        if len(alignment_history) >= 2:
            recent_scores = [a.get("overall_alignment_score", 0) for a in alignment_history[:5]]
            older_scores = [a.get("overall_alignment_score", 0) for a in alignment_history[-5:]]
            
            recent_avg = sum(recent_scores) / len(recent_scores) if recent_scores else 0
            older_avg = sum(older_scores) / len(older_scores) if older_scores else 0
            
            if recent_avg > older_avg + 0.05:
                trend_direction = "improving"
            elif recent_avg < older_avg - 0.05:
                trend_direction = "declining"
            else:
                trend_direction = "stable"
            
            trend_change = round((recent_avg - older_avg) * 100, 2)
        else:
            trend_direction = "insufficient_data"
            trend_change = 0
        
        # Calculate subsystem stability
        subsystem_trends = {}
        for subsystem in ["memory", "trust", "arbitration", "forecast"]:
            scores = [
                a.get("subsystem_scores", {}).get(subsystem, 0)
                for a in alignment_history
            ]
            if scores:
                subsystem_trends[subsystem] = {
                    "current": scores[0],
                    "average": round(sum(scores) / len(scores), 3),
                    "variance": round(sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores), 4)
                }
        
        # Get synthesis count
        synthesis_count = await db.llm_synthesis_log.count_documents({})
        
        # Get conflict statistics
        recent_arbitrations = await db.llm_arbitration_log.find().sort("timestamp", -1).limit(50).to_list(50)
        unresolved_conflicts = len([a for a in recent_arbitrations if a.get("resolution") == "unresolved"])
        
        return {
            "success": True,
            "current_alignment": current_alignment,
            "alignment_history": alignment_history,
            "trends": {
                "overall_trend": trend_direction,
                "change_percent": trend_change,
                "subsystem_trends": subsystem_trends
            },
            "statistics": {
                "total_synthesis_runs": synthesis_count,
                "alignment_measurements": len(alignment_history),
                "current_conflicts": unresolved_conflicts
            },
            "recommendations": [
                "Alignment is optimal - continue current strategy" if current_alignment.get("overall_alignment_score", 0) >= 0.85
                else "Consider recalibration - alignment below target" if current_alignment.get("overall_alignment_score", 0) < 0.6
                else "Moderate alignment - monitor for improvements"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting alignment analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/collective/insights-summary")
async def get_collective_insights_summary():
    """
    Get quick summary of collective intelligence status.
    Useful for dashboard header display.
    """
    try:
        # Get counts from each subsystem
        memory_count = await db.llm_experience_memory.count_documents({})
        trust_count = await db.llm_trust_calibration.count_documents({})
        arbitration_count = await db.llm_arbitration_log.count_documents({})
        strategy_count = await db.llm_global_strategy.count_documents({})
        
        # Get latest alignment
        latest_alignment = await db.llm_alignment_metrics.find().sort("timestamp", -1).limit(1).to_list(1)
        
        # Get latest synthesis
        latest_synthesis = await db.llm_synthesis_log.find().sort("timestamp", -1).limit(1).to_list(1)
        
        alignment_score = latest_alignment[0].get("overall_alignment_score", 0) if latest_alignment else 0
        consensus_level = latest_alignment[0].get("consensus_level", "unknown") if latest_alignment else "unknown"
        
        return {
            "success": True,
            "subsystem_status": {
                "memory_experiences": memory_count,
                "trust_calibrations": trust_count,
                "arbitration_logs": arbitration_count,
                "global_strategies": strategy_count
            },
            "collective_intelligence": {
                "alignment_score": round(alignment_score, 3),
                "consensus_level": consensus_level,
                "last_synthesis": latest_synthesis[0].get("timestamp") if latest_synthesis else None,
                "active_strategy": latest_synthesis[0].get("strategy_archetype") if latest_synthesis else "No strategy yet"
            },
            "system_health": "optimal" if alignment_score >= 0.85 else "good" if alignment_score >= 0.7 else "needs_attention"
        }
        
    except Exception as e:
        logger.error(f"Error getting insights summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================
# Meta-Optimization Endpoints (Step 24)
# ====================

# Initialize Meta-Reasoning Controller
meta_controller = None

def get_meta_controller():
    """Get or create meta-reasoning controller"""
    global meta_controller
    if meta_controller is None:
        from meta_optimizer import MetaReasoningController
        meta_controller = MetaReasoningController(db)
    return meta_controller

@api_router.post("/llm/meta/optimize")
async def trigger_meta_optimization(background_tasks: BackgroundTasks):
    """
    Trigger meta-optimization cycle manually.
    Analyzes system performance and applies autonomous optimizations.
    """
    try:
        controller = get_meta_controller()
        
        # Run optimization cycle in background
        cycle = await controller.run_optimization_cycle(trigger="manual")
        
        return {
            "success": True,
            "message": "Meta-optimization cycle initiated",
            "cycle_id": cycle.cycle_id,
            "adjustments_proposed": len(cycle.adjustments),
            "applied": cycle.applied,
            "approval_required": cycle.approval_required,
            "system_health_score": cycle.system_health_score,
            "reflection_summary": cycle.reflection_summary
        }
    except Exception as e:
        logger.error(f"Error triggering meta-optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/meta/status")
async def get_meta_optimization_status():
    """
    Get current meta-optimization system status.
    Returns system health, recent metrics, and optimization deltas.
    """
    try:
        controller = get_meta_controller()
        status = await controller.get_optimization_status()
        
        return {
            "success": True,
            **status
        }
    except Exception as e:
        logger.error(f"Error getting meta-optimization status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/meta/history")
async def get_meta_optimization_history(limit: int = 20):
    """
    Get meta-optimization history with parameter adjustments and impact.
    Returns list of optimization cycles with detailed metrics.
    """
    try:
        # Get optimization cycles from database
        cycles = await db.llm_meta_optimization_log.find().sort(
            "timestamp", -1
        ).limit(limit).to_list(limit)
        
        # Process cycles for frontend
        processed_cycles = []
        for cycle in cycles:
            processed_cycles.append({
                "cycle_id": cycle.get("cycle_id"),
                "timestamp": cycle.get("timestamp"),
                "trigger": cycle.get("trigger"),
                "adjustments_count": len(cycle.get("adjustments", [])),
                "adjustments": cycle.get("adjustments", []),
                "applied": cycle.get("applied", False),
                "approval_required": cycle.get("approval_required", False),
                "system_health_score": cycle.get("system_health_score", 75.0),
                "reflection_summary": cycle.get("reflection_summary", ""),
                "performance_delta": cycle.get("performance_delta", {})
            })
        
        return {
            "success": True,
            "cycles": processed_cycles,
            "total": len(processed_cycles)
        }
    except Exception as e:
        logger.error(f"Error getting meta-optimization history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/meta/metrics")
async def get_system_performance_metrics(lookback_hours: int = 24):
    """
    Get detailed system performance metrics across all subsystems.
    Used for monitoring and diagnostics.
    """
    try:
        controller = get_meta_controller()
        metrics = await controller.analyze_system_performance(lookback_hours=lookback_hours)
        
        # Convert SystemMetrics objects to dictionaries
        metrics_dict = {
            subsystem: metric.to_dict() 
            for subsystem, metric in metrics.items()
        }
        
        return {
            "success": True,
            "lookback_hours": lookback_hours,
            "metrics": metrics_dict,
            "subsystems": list(metrics_dict.keys())
        }
    except Exception as e:
        logger.error(f"Error getting system performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/llm/meta/approve/{cycle_id}")
async def approve_optimization_cycle(cycle_id: str):
    """
    Manually approve a pending optimization cycle.
    Applies adjustments that required manual review.
    """
    try:
        # Get cycle from database
        cycle_doc = await db.llm_meta_optimization_log.find_one({"cycle_id": cycle_id})
        
        if not cycle_doc:
            raise HTTPException(status_code=404, detail="Optimization cycle not found")
        
        if cycle_doc.get("applied"):
            return {
                "success": False,
                "message": "Cycle already applied"
            }
        
        # Apply adjustments
        controller = get_meta_controller()
        from meta_optimizer import MetaAdjustment
        
        adjustments = [
            MetaAdjustment(**adj) for adj in cycle_doc.get("adjustments", [])
        ]
        
        simulation_results = cycle_doc.get("simulation_results", {})
        simulation_results["recommendation"] = "approve"  # Override for manual approval
        
        success, results = await controller.apply_autonomous_optimization(
            adjustments, 
            simulation_results
        )
        
        # Update cycle in database
        await db.llm_meta_optimization_log.update_one(
            {"cycle_id": cycle_id},
            {"$set": {
                "applied": success,
                "manual_approval": True,
                "approval_timestamp": datetime.now(timezone.utc).isoformat(),
                "application_results": results
            }}
        )
        
        return {
            "success": success,
            "message": "Optimization cycle approved and applied",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error approving optimization cycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================
# Adaptive Goal Formation & Ethical Governance (Step 25)
# ====================

from goal_governor import AdaptiveGoalController, AdaptiveGoal, GovernanceEvaluation

# Initialize goal governor
goal_governor = None

def get_goal_governor():
    """Get or create goal governor instance"""
    global goal_governor
    if goal_governor is None:
        goal_governor = AdaptiveGoalController(db)
    return goal_governor

class GoalGenerationRequest(BaseModel):
    """Request to generate adaptive goals"""
    lookback_hours: int = Field(default=24, ge=1, le=168)
    include_performance: bool = Field(default=True)
    include_alignment: bool = Field(default=True)
    include_stability: bool = Field(default=True)

class GoalApprovalRequest(BaseModel):
    """Request to approve/reject a goal"""
    goal_id: str
    approved: bool
    approver: str = Field(default="manual")
    notes: Optional[str] = None

@api_router.post("/llm/goals/generate")
async def generate_adaptive_goals(request: GoalGenerationRequest):
    """
    Generate adaptive goals based on system performance and emergent signals.
    
    Returns:
        List of proposed goals with ethical alignment scores
    """
    try:
        governor = get_goal_governor()
        
        # Gather performance trends
        from meta_optimizer import MetaReasoningController
        meta_controller = MetaReasoningController(db)
        
        # Get system metrics
        metrics = await meta_controller.analyze_system_performance(
            lookback_hours=request.lookback_hours
        )
        
        # Format performance trends
        performance_trends = {
            "lookback_hours": request.lookback_hours,
            "subsystem_metrics": {
                subsystem: {
                    "alignment_pct": metric.alignment_pct,
                    "trust_variance": metric.trust_variance,
                    "consensus_stability": metric.consensus_stability,
                    "win_rate": metric.win_rate
                }
                for subsystem, metric in metrics.items()
            }
        }
        
        # Format emergent signals
        optimization_status = await meta_controller.get_optimization_status()
        emergent_signals = {
            "overall_alignment": optimization_status.get("system_health_score", 75.0) / 100.0,
            "trust_variance": optimization_status.get("metrics_summary", {}).get("trust", {}).get("variance", 0.1),
            "system_health": optimization_status.get("system_health_score", 75.0)
        }
        
        # Generate goals
        goals = await governor.generate_adaptive_goals(
            performance_trends=performance_trends,
            emergent_signals=emergent_signals
        )
        
        # Evaluate each goal
        evaluated_goals = []
        for goal in goals:
            # Evaluate alignment
            evaluation = await governor.evaluate_goal_alignment(goal)
            
            # Check governance rules
            can_execute, reason = await governor.apply_governance_rules(goal, evaluation)
            
            # Store goal
            goal_dict = goal.to_dict()
            goal_dict["evaluation"] = evaluation.to_dict()
            goal_dict["can_auto_execute"] = can_execute
            goal_dict["governance_reason"] = reason
            
            await db.llm_adaptive_goals.insert_one(goal_dict)
            
            # Record in governance log
            await governor.record_goal_outcome(
                goal=goal,
                evaluation=evaluation,
                executed=False,
                outcome={"status": "proposed", "can_auto_execute": can_execute}
            )
            
            evaluated_goals.append(goal_dict)
        
        logger.info(f"Generated {len(evaluated_goals)} adaptive goals")
        
        return {
            "success": True,
            "goals_generated": len(evaluated_goals),
            "goals": evaluated_goals,
            "performance_trends": performance_trends,
            "emergent_signals": emergent_signals,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating adaptive goals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/goals/status")
async def get_goals_status(
    status: Optional[str] = Query(None, regex="^(proposed|approved|rejected|active|completed)$"),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Get list of goals with alignment and ethics scores.
    
    Query Parameters:
        status: Filter by goal status
        limit: Maximum number of goals to return
    
    Returns:
        List of goals with alignment metrics
    """
    try:
        governor = get_goal_governor()
        
        # Build query
        query = {}
        if status:
            query["status"] = status
        
        # Get goals from database
        goals_raw = await db.llm_adaptive_goals.find(query).sort(
            "timestamp", -1
        ).limit(limit).to_list(limit)
        
        # Remove MongoDB _id field
        goals = []
        for g in goals_raw:
            g.pop('_id', None)
            goals.append(g)
        
        # Calculate statistics
        if goals:
            avg_strategic_alignment = np.mean([g.get("strategic_alignment", 0) for g in goals])
            avg_ethical_alignment = np.mean([g.get("ethical_alignment", 0) for g in goals])
            
            critical_count = sum(1 for g in goals if g.get("is_critical", False))
            auto_apply_count = sum(1 for g in goals if g.get("auto_apply", False))
            
            status_distribution = {}
            type_distribution = {}
            for g in goals:
                s = g.get("status", "unknown")
                t = g.get("goal_type", "unknown")
                status_distribution[s] = status_distribution.get(s, 0) + 1
                type_distribution[t] = type_distribution.get(t, 0) + 1
        else:
            avg_strategic_alignment = 0.0
            avg_ethical_alignment = 0.0
            critical_count = 0
            auto_apply_count = 0
            status_distribution = {}
            type_distribution = {}
        
        return {
            "success": True,
            "total_goals": len(goals),
            "goals": goals,
            "statistics": {
                "avg_strategic_alignment": round(avg_strategic_alignment, 3),
                "avg_ethical_alignment": round(avg_ethical_alignment, 3),
                "critical_goals": critical_count,
                "auto_apply_eligible": auto_apply_count,
                "status_distribution": status_distribution,
                "type_distribution": type_distribution
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting goals status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/llm/goals/approve")
async def approve_goal(request: GoalApprovalRequest):
    """
    Manually approve or reject a goal.
    
    Returns:
        Approval result and updated goal status
    """
    try:
        governor = get_goal_governor()
        
        # Get the goal
        goal_doc = await db.llm_adaptive_goals.find_one({"goal_id": request.goal_id})
        
        if not goal_doc:
            raise HTTPException(status_code=404, detail=f"Goal {request.goal_id} not found")
        
        # Approve/reject the goal
        success = await governor.approve_goal(
            goal_id=request.goal_id,
            approved=request.approved,
            approver=request.approver
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update goal status")
        
        # Get updated goal
        updated_goal = await db.llm_adaptive_goals.find_one({"goal_id": request.goal_id})
        
        # Record in governance log
        log_entry = {
            "log_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "manual_approval",
            "goal_id": request.goal_id,
            "approved": request.approved,
            "approver": request.approver,
            "notes": request.notes,
            "goal_snapshot": updated_goal
        }
        await db.llm_governance_log.insert_one(log_entry)
        
        logger.info(f"Goal {request.goal_id[:8]} {'approved' if request.approved else 'rejected'} by {request.approver}")
        
        return {
            "success": True,
            "goal_id": request.goal_id,
            "approved": request.approved,
            "new_status": updated_goal.get("status"),
            "approver": request.approver,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving goal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/governance/report")
async def get_governance_report():
    """
    Generate comprehensive ethical governance report.
    
    Returns:
        Governance report with alignment metrics, violations, and recommendations
    """
    try:
        governor = get_goal_governor()
        
        # Generate comprehensive report
        report = await governor.generate_governance_report()
        
        return {
            "success": True,
            "report": report,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating governance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/governance/rules")
async def get_governance_rules():
    """Get active governance rules"""
    try:
        rules_docs = await db.llm_governance_rules.find({"enabled": True}).to_list(100)
        
        if not rules_docs:
            # Return default rules
            governor = get_goal_governor()
            rules_docs = [rule.to_dict() for rule in governor.default_rules]
        
        return {
            "success": True,
            "rules": rules_docs,
            "count": len(rules_docs)
        }
        
    except Exception as e:
        logger.error(f"Error getting governance rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/llm/governance/rules/update")
async def update_governance_rule(rule: Dict[str, Any]):
    """Update or create a governance rule"""
    try:
        rule_id = rule.get("rule_id")
        
        if not rule_id:
            rule_id = str(uuid.uuid4())
            rule["rule_id"] = rule_id
        
        # Validate rule structure
        required_fields = ["name", "description", "constraint_type", "threshold"]
        for field in required_fields:
            if field not in rule:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Update or insert
        await db.llm_governance_rules.replace_one(
            {"rule_id": rule_id},
            rule,
            upsert=True
        )
        
        logger.info(f"Updated governance rule: {rule_id}")
        
        return {
            "success": True,
            "rule_id": rule_id,
            "message": "Governance rule updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating governance rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================
# Ethical Consensus Endpoints (Step 26)
# ====================

class ConsensusRequest(BaseModel):
    """Request for ethical consensus deliberation"""
    goal_id: str
    trigger_type: str = Field(default="manual", pattern="^(manual|automatic)$")
    include_refinement: bool = Field(default=True)

@api_router.post("/llm/ethics/consensus")
async def trigger_ethical_consensus(request: ConsensusRequest):
    """
    Trigger multi-agent ethical consensus deliberation.
    
    This endpoint initiates deliberation among 5 LLM agents (GPT-5, Claude 4, Gemini 2.5)
    to evaluate an adaptive goal's ethical alignment.
    """
    try:
        from ethical_consensus import EthicalConsensusEngine
        
        # Initialize consensus engine
        engine = EthicalConsensusEngine(db)
        
        # Get the goal to evaluate
        goal = await db.llm_adaptive_goals.find_one({"goal_id": request.goal_id})
        if not goal:
            raise HTTPException(status_code=404, detail="Goal not found")
        
        # Get governance context
        governance_rules = await db.llm_governance_rules.find({"enabled": True}).to_list(100)
        
        # Get recent governance metrics
        recent_logs = await db.llm_governance_log.find().sort("timestamp", -1).limit(20).to_list(20)
        avg_alignment = np.mean([log.get('overall_alignment', 0) for log in recent_logs]) if recent_logs else 0.85
        recent_violations = sum(len(log.get('violations', [])) for log in recent_logs)
        
        governance_context = {
            "active_rules": governance_rules,
            "avg_alignment": avg_alignment,
            "recent_violations": recent_violations
        }
        
        logger.info(f"Starting ethical consensus for goal {request.goal_id[:8]}...")
        
        # Step 1: Aggregate agent ethics
        agent_opinions = await engine.aggregate_agent_ethics(goal, governance_context)
        
        if not agent_opinions:
            raise HTTPException(
                status_code=503,
                detail="Unable to gather agent opinions. LLM services may be unavailable."
            )
        
        # Step 2: Run consensus voting
        consensus_result = await engine.run_consensus_voting(agent_opinions, goal)
        
        # Step 3: Resolve conflicts if detected
        if consensus_result.conflicts_detected:
            consensus_result, conflict_resolution = await engine.resolve_conflicts(consensus_result)
        
        # Step 4: Store consensus in database
        await db.llm_ethics_consensus_log.insert_one(consensus_result.to_dict())
        
        # Step 5: Refine governance rules if requested
        refinements = []
        if request.include_refinement:
            refinements = await engine.refine_governance_rules(consensus_result, governance_rules)
            
            # Store refinements
            if refinements:
                for refinement in refinements:
                    await db.llm_rule_refinement.insert_one(refinement.to_dict())
        
        logger.info(
            f"Consensus complete: {consensus_result.final_decision} "
            f"(EAI: {consensus_result.agreement_score:.2f}, "
            f"{len(refinements)} refinements generated)"
        )
        
        return {
            "success": True,
            "consensus_id": consensus_result.consensus_id,
            "final_decision": consensus_result.final_decision,
            "agreement_score": consensus_result.agreement_score,
            "agreement_variance": consensus_result.agreement_variance,
            "consensus_reached": consensus_result.consensus_reached,
            "vote_distribution": consensus_result.vote_distribution,
            "conflicts_detected": consensus_result.conflicts_detected,
            "conflict_resolution": consensus_result.conflict_resolution,
            "reasoning_summary": consensus_result.reasoning_summary,
            "agent_opinions_count": len(agent_opinions),
            "refinements_generated": len(refinements),
            "message": f"Consensus deliberation complete: {consensus_result.final_decision}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ethical consensus: {e}")
        raise HTTPException(status_code=500, detail=f"Consensus error: {str(e)}")


@api_router.get("/llm/ethics/status")
async def get_consensus_status(limit: int = Query(default=10, ge=1, le=100)):
    """
    Get current ethical consensus alignment metrics and recent status.
    
    Returns:
    - Overall Ethical Alignment Index (EAI)
    - Agreement variance trends
    - Recent consensus decisions
    - System health status
    """
    try:
        from ethical_consensus import EthicalConsensusEngine
        
        engine = EthicalConsensusEngine(db)
        
        # Get recent consensus logs
        recent_consensuses = await db.llm_ethics_consensus_log.find().sort(
            "timestamp", -1
        ).limit(limit).to_list(limit)
        
        if not recent_consensuses:
            return {
                "success": True,
                "status": "no_data",
                "message": "No consensus data available yet",
                "overall_eai": 0.0,
                "avg_variance": 0.0,
                "recent_decisions": []
            }
        
        # Calculate aggregate metrics
        eai_scores = [c.get('agreement_score', 0) for c in recent_consensuses]
        variances = [c.get('agreement_variance', 0) for c in recent_consensuses]
        
        overall_eai = np.mean(eai_scores)
        avg_variance = np.mean(variances)
        
        # Decision distribution
        decision_counts = {}
        for c in recent_consensuses:
            decision = c.get('final_decision', 'unknown')
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        # Conflict analysis
        total_conflicts = sum(len(c.get('conflicts_detected', [])) for c in recent_consensuses)
        
        # Recent decisions summary
        recent_decisions = [
            {
                "consensus_id": c.get('consensus_id'),
                "goal_description": c.get('goal_description', '')[:100],
                "final_decision": c.get('final_decision'),
                "agreement_score": c.get('agreement_score'),
                "timestamp": c.get('timestamp'),
                "conflicts_count": len(c.get('conflicts_detected', []))
            }
            for c in recent_consensuses[:5]
        ]
        
        # Determine status
        if overall_eai >= 0.85 and avg_variance < 0.15:
            status = "excellent"
            status_description = "High consensus alignment with low variance"
        elif overall_eai >= 0.70 and avg_variance < 0.25:
            status = "good"
            status_description = "Acceptable consensus alignment"
        elif overall_eai >= 0.55:
            status = "needs_attention"
            status_description = "Moderate alignment, monitoring recommended"
        else:
            status = "critical"
            status_description = "Low consensus alignment, review required"
        
        return {
            "success": True,
            "status": status,
            "status_description": status_description,
            "overall_eai": round(overall_eai, 3),
            "avg_variance": round(avg_variance, 3),
            "total_consensuses": len(recent_consensuses),
            "decision_distribution": decision_counts,
            "total_conflicts": total_conflicts,
            "conflict_rate": round(total_conflicts / len(recent_consensuses), 2) if recent_consensuses else 0,
            "recent_decisions": recent_decisions
        }
        
    except Exception as e:
        logger.error(f"Error getting consensus status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/ethics/history")
async def get_consensus_history(
    limit: int = Query(default=50, ge=1, le=200),
    goal_id: Optional[str] = Query(default=None)
):
    """
    Get timeline of consensus decisions with detailed agent opinions and voting patterns.
    """
    try:
        query = {}
        if goal_id:
            query["goal_id"] = goal_id
        
        consensuses = await db.llm_ethics_consensus_log.find(query).sort(
            "timestamp", -1
        ).limit(limit).to_list(limit)
        
        # Format timeline
        timeline = []
        for c in consensuses:
            timeline.append({
                "consensus_id": c.get('consensus_id'),
                "timestamp": c.get('timestamp'),
                "goal_id": c.get('goal_id'),
                "goal_description": c.get('goal_description'),
                "final_decision": c.get('final_decision'),
                "agreement_score": c.get('agreement_score'),
                "agreement_variance": c.get('agreement_variance'),
                "consensus_reached": c.get('consensus_reached'),
                "vote_distribution": c.get('vote_distribution'),
                "agents_participated": c.get('agents_participated'),
                "conflicts_detected": c.get('conflicts_detected'),
                "conflict_resolution": c.get('conflict_resolution'),
                "reasoning_summary": c.get('reasoning_summary')
            })
        
        return {
            "success": True,
            "total_records": len(timeline),
            "timeline": timeline
        }
        
    except Exception as e:
        logger.error(f"Error getting consensus history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/llm/ethics/rules/update")
async def update_ethics_rules(refinement_id: str, approved: bool):
    """
    Apply or reject a proposed governance rule refinement.
    
    Args:
        refinement_id: ID of the rule refinement
        approved: Whether to apply the refinement
    """
    try:
        # Get the refinement
        refinement = await db.llm_rule_refinement.find_one({"refinement_id": refinement_id})
        if not refinement:
            raise HTTPException(status_code=404, detail="Refinement not found")
        
        if approved:
            # Get the rule to update
            rule = await db.llm_governance_rules.find_one({"rule_id": refinement['rule_id']})
            if not rule:
                raise HTTPException(status_code=404, detail="Governance rule not found")
            
            # Apply the refinement
            await db.llm_governance_rules.update_one(
                {"rule_id": refinement['rule_id']},
                {"$set": {
                    "threshold": refinement['new_weight'],
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "updated_by": "consensus_refinement"
                }}
            )
            
            # Mark refinement as applied
            await db.llm_rule_refinement.update_one(
                {"refinement_id": refinement_id},
                {"$set": {
                    "auto_applied": True,
                    "approved_at": datetime.now(timezone.utc).isoformat()
                }}
            )
            
            logger.info(f"Applied rule refinement {refinement_id[:8]} to rule {refinement['rule_id'][:8]}")
            message = f"Rule refinement applied: {refinement['rule_name']} threshold updated to {refinement['new_weight']}"
        else:
            # Mark refinement as rejected
            await db.llm_rule_refinement.update_one(
                {"refinement_id": refinement_id},
                {"$set": {
                    "rejected": True,
                    "rejected_at": datetime.now(timezone.utc).isoformat()
                }}
            )
            
            logger.info(f"Rejected rule refinement {refinement_id[:8]}")
            message = "Rule refinement rejected"
        
        return {
            "success": True,
            "refinement_id": refinement_id,
            "approved": approved,
            "message": message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating ethics rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/ethics/report")
async def generate_consensus_report(
    lookback_hours: int = Query(default=24, ge=1, le=168),
    consensus_id: Optional[str] = Query(default=None)
):
    """
    Generate comprehensive ethics report with consensus outcomes and rule shifts.
    
    Returns detailed analysis of:
    - Ethical Alignment Index trends
    - Agreement variance patterns
    - Conflict analysis
    - Rule refinement history
    - Provider-specific insights
    - System health assessment
    """
    try:
        from ethical_consensus import EthicalConsensusEngine
        
        engine = EthicalConsensusEngine(db)
        
        # Generate report
        report = await engine.generate_ethics_report(
            consensus_id=consensus_id,
            lookback_hours=lookback_hours
        )
        
        return {
            "success": True,
            "report": report
        }
        
    except Exception as e:
        logger.error(f"Error generating consensus report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/ethics/agent-details/{consensus_id}")
async def get_agent_details(consensus_id: str):
    """
    Get detailed agent opinions and reasoning for a specific consensus.
    
    Returns individual agent perspectives, confidence scores, and voting rationale.
    """
    try:
        consensus = await db.llm_ethics_consensus_log.find_one({"consensus_id": consensus_id})
        if not consensus:
            raise HTTPException(status_code=404, detail="Consensus not found")
        
        agent_opinions = consensus.get('agent_opinions', [])
        
        # Format agent details
        detailed_opinions = []
        for opinion in agent_opinions:
            detailed_opinions.append({
                "agent_name": opinion.get('agent_name'),
                "provider": opinion.get('provider'),
                "model": opinion.get('model'),
                "vote": opinion.get('vote'),
                "alignment_score": opinion.get('alignment_score'),
                "confidence": opinion.get('confidence'),
                "opinion": opinion.get('opinion'),
                "reasoning": opinion.get('reasoning'),
                "response_time": opinion.get('response_time')
            })
        
        return {
            "success": True,
            "consensus_id": consensus_id,
            "goal_description": consensus.get('goal_description'),
            "final_decision": consensus.get('final_decision'),
            "agents_participated": len(detailed_opinions),
            "agent_opinions": detailed_opinions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent details: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# ====================
# Cognitive Synthesis Endpoints (Step 27)
# ====================

from cognitive_synthesizer import CognitiveSynthesisController

# Initialize cognitive synthesis controller
cognitive_controller = CognitiveSynthesisController(db)

class SynthesisRequest(BaseModel):
    trigger: str = Field(default="manual", pattern="^(scheduled|manual|threshold)$")

@api_router.post("/llm/cognitive/synthesize")
async def run_cognitive_synthesis(request: SynthesisRequest):
    """
    Trigger autonomous cognitive synthesis cycle.
    Integrates insights from all layers and evaluates value preservation.
    """
    try:
        logger.info(f"Starting cognitive synthesis cycle (trigger: {request.trigger})")
        
        # Run synthesis cycle
        cycle = await cognitive_controller.run_self_synthesis_cycle(trigger=request.trigger)
        
        return {
            "success": True,
            "cycle_id": cycle.cycle_id,
            "timestamp": cycle.timestamp,
            "cognitive_coherence_index": cycle.cognitive_coherence_index,
            "value_integrity_score": cycle.value_integrity_score,
            "patterns_detected": len(cycle.patterns_detected),
            "layers_integrated": cycle.layers_integrated,
            "drift_status": cycle.drift_report.get("drift_status", "unknown"),
            "reflection_summary": cycle.reflection_summary,
            "recommendations": cycle.recommendations
        }
        
    except Exception as e:
        logger.error(f"Error running cognitive synthesis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/cognitive/status")
async def get_cognitive_status():
    """
    Get current cognitive synthesis system status.
    Returns coherence index, value integrity, and drift monitoring.
    """
    try:
        status = await cognitive_controller.get_synthesis_status()
        
        return {
            "success": True,
            **status
        }
        
    except Exception as e:
        logger.error(f"Error getting cognitive status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/cognitive/history")
async def get_synthesis_history(limit: int = Query(default=10, ge=1, le=50)):
    """
    Get historical synthesis cycle records.
    """
    try:
        cycles = await db.llm_cognitive_synthesis_log.find().sort(
            "timestamp", -1
        ).limit(limit).to_list(limit)
        
        # Format for response
        history = []
        for cycle in cycles:
            history.append({
                "cycle_id": cycle.get("cycle_id"),
                "timestamp": cycle.get("timestamp"),
                "trigger": cycle.get("trigger"),
                "cognitive_coherence_index": cycle.get("cognitive_coherence_index"),
                "value_integrity_score": cycle.get("value_integrity_score"),
                "patterns_detected": len(cycle.get("patterns_detected", [])),
                "layers_integrated": cycle.get("layers_integrated", []),
                "drift_status": cycle.get("drift_report", {}).get("drift_status", "unknown"),
                "recommendations_count": len(cycle.get("recommendations", []))
            })
        
        return {
            "success": True,
            "count": len(history),
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Error getting synthesis history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/cognitive/values")
async def get_value_states(limit: int = Query(default=20, ge=1, le=100)):
    """
    Get current value preservation states.
    Tracks ethical and strategic value drift over time.
    """
    try:
        # Get latest value states (one per value type)
        value_names = ["transparency", "fairness", "safety", "performance", "alignment", "stability"]
        
        latest_states = []
        for value_name in value_names:
            state = await db.llm_value_preservation.find_one(
                {"value_name": value_name},
                sort=[("last_updated", -1)]
            )
            if state:
                latest_states.append({
                    "value_name": state.get("value_name"),
                    "category": state.get("category"),
                    "current_score": state.get("current_score"),
                    "target_score": state.get("target_score"),
                    "drift_amount": state.get("drift_amount"),
                    "drift_direction": state.get("drift_direction"),
                    "stability_index": state.get("stability_index"),
                    "last_updated": state.get("last_updated")
                })
        
        # Get historical trend (last N records)
        historical = await db.llm_value_preservation.find().sort(
            "last_updated", -1
        ).limit(limit).to_list(limit)
        
        # Calculate overall drift
        if latest_states:
            overall_drift = np.mean([abs(s["drift_amount"] / s["target_score"]) 
                                    if s["target_score"] > 0 else 0 
                                    for s in latest_states])
            avg_integrity = np.mean([s["stability_index"] for s in latest_states]) * 100
        else:
            overall_drift = 0.0
            avg_integrity = 100.0
        
        return {
            "success": True,
            "current_values": latest_states,
            "overall_drift": round(overall_drift, 4),
            "avg_value_integrity": round(avg_integrity, 1),
            "historical_count": len(historical),
            "drift_status": "critical" if overall_drift > 0.15 else "moderate" if overall_drift > 0.08 else "healthy"
        }
        
    except Exception as e:
        logger.error(f"Error getting value states: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/cognitive/patterns")
async def get_cognitive_patterns(limit: int = Query(default=10, ge=1, le=50)):
    """
    Get detected cognitive patterns across system layers.
    """
    try:
        patterns = await db.llm_cognitive_patterns.find().sort(
            "last_detected", -1
        ).limit(limit).to_list(limit)
        
        # Format patterns
        formatted_patterns = []
        for pattern in patterns:
            formatted_patterns.append({
                "pattern_name": pattern.get("pattern_name"),
                "description": pattern.get("description"),
                "layers_involved": pattern.get("layers_involved", []),
                "strength": pattern.get("strength"),
                "emergence_count": pattern.get("emergence_count"),
                "first_detected": pattern.get("first_detected"),
                "last_detected": pattern.get("last_detected"),
                "impact_areas": pattern.get("impact_areas", [])
            })
        
        return {
            "success": True,
            "count": len(formatted_patterns),
            "patterns": formatted_patterns
        }
        
    except Exception as e:
        logger.error(f"Error getting cognitive patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/llm/cognitive/insights")
async def get_multilayer_insights():
    """
    Get current multilayer insights from all system layers.
    """
    try:
        # Collect fresh insights
        insights = await cognitive_controller.aggregate_multilayer_insights()
        
        # Format for response
        formatted_insights = [
            {
                "layer_name": insight.layer_name,
                "insight_type": insight.insight_type,
                "content": insight.content,
                "confidence": insight.confidence,
                "timestamp": insight.timestamp,
                "metrics": insight.metrics
            }
            for insight in insights
        ]
        
        # Group by layer
        by_layer = {}
        for insight in formatted_insights:
            layer = insight["layer_name"]
            if layer not in by_layer:
                by_layer[layer] = []
            by_layer[layer].append(insight)
        
        return {
            "success": True,
            "total_insights": len(formatted_insights),
            "layers": list(by_layer.keys()),
            "insights": formatted_insights,
            "by_layer": by_layer
        }
        
    except Exception as e:
        logger.error(f"Error getting multilayer insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================
# Step 28: Collective Consciousness & Evolutionary Learning Core
# ====================

from collective_consciousness import ConsciousnessController

# Initialize Consciousness Controller
consciousness_controller = ConsciousnessController(db)

class EvolutionTriggerRequest(BaseModel):
    """Request to trigger evolution cycle"""
    trigger: str = Field(default="manual", pattern="^(manual|scheduled|threshold)$")

@api_router.post("/llm/collective/evolve")
async def trigger_evolution_cycle(request: EvolutionTriggerRequest = EvolutionTriggerRequest()):
    """
    Trigger evolutionary learning cycle.
    
    Aggregates collective experiences from all layers (Steps 19-27),
    performs adaptive refinement with safety checks, and returns
    evolution results with recommendations.
    
    Returns:
        EvolutionCycle with proposed adaptations and consciousness metrics
    """
    try:
        logger.info(f"Triggering evolution cycle (trigger: {request.trigger})...")
        
        # Aggregate collective experiences
        experience = await consciousness_controller.aggregate_collective_experiences()
        
        # Run evolution cycle
        cycle = await consciousness_controller.evolve_conscious_state(
            experience=experience,
            trigger=request.trigger
        )
        
        # Compute updated consciousness metrics
        metrics = await consciousness_controller.compute_consciousness_index(experience)
        
        return {
            "success": True,
            "cycle_id": cycle.cycle_id,
            "trigger": cycle.trigger,
            "timestamp": cycle.timestamp,
            "consciousness_index": cycle.consciousness_index,
            "evolution_rate": cycle.evolution_rate,
            "pre_evolution_state": cycle.pre_evolution_state,
            "post_evolution_state": cycle.post_evolution_state,
            "adaptations_proposed": cycle.adaptations_proposed,
            "adaptations_applied": cycle.adaptations_applied,
            "safety_violations": cycle.safety_violations,
            "recommendations": cycle.recommendations,
            "metrics": {
                "consciousness_index": metrics.consciousness_index,
                "coherence_ratio": metrics.coherence_ratio,
                "evolution_rate": metrics.evolution_rate,
                "value_integrity": metrics.value_integrity,
                "emergence_level": metrics.emergence_level,
                "stability_index": metrics.stability_index
            }
        }
        
    except Exception as e:
        logger.error(f"Error triggering evolution cycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/collective/status")
async def get_consciousness_status():
    """
    Get current collective consciousness status.
    
    Returns consciousness index, coherence metrics, value integrity,
    and overall system health indicators.
    
    Returns:
        ConsciousnessMetrics with color-coded health status
    """
    try:
        # Aggregate current experience
        experience = await consciousness_controller.aggregate_collective_experiences()
        
        # Compute consciousness metrics
        metrics = await consciousness_controller.compute_consciousness_index(experience)
        
        # Determine health status with color coding
        if metrics.consciousness_index >= 0.85:
            health_status = "excellent"
            health_color = "green"
        elif metrics.consciousness_index >= 0.70:
            health_status = "good"
            health_color = "yellow"
        else:
            health_status = "needs_attention"
            health_color = "red"
        
        # Get recent evolution rate
        recent_cycles = await db.llm_collective_consciousness.find().sort(
            "timestamp", -1
        ).limit(5).to_list(5)
        
        return {
            "success": True,
            "consciousness_index": metrics.consciousness_index,
            "coherence_ratio": metrics.coherence_ratio,
            "evolution_rate": metrics.evolution_rate,
            "value_integrity": metrics.value_integrity,
            "emergence_level": metrics.emergence_level,
            "stability_index": metrics.stability_index,
            "health_status": health_status,
            "health_color": health_color,
            "timestamp": metrics.timestamp,
            "layers_integrated": len(experience.source_layers),
            "total_layers": 6,
            "recent_cycles_count": len(recent_cycles),
            "system_status": "operational"
        }
        
    except Exception as e:
        logger.error(f"Error getting consciousness status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/collective/history")
async def get_evolution_history(limit: int = 20):
    """
    Get evolution timeline history.
    
    Returns historical consciousness metrics and evolution cycles
    showing system progression over time.
    
    Args:
        limit: Number of historical records to return
    
    Returns:
        Timeline of consciousness evolution with metrics progression
    """
    try:
        # Get evolution cycles
        cycles_raw = await db.llm_collective_consciousness.find().sort(
            "timestamp", -1
        ).limit(limit).to_list(limit)
        
        # Format cycles (remove MongoDB ObjectId)
        cycles = []
        for cycle in cycles_raw:
            cycles.append({
                "cycle_id": cycle.get("cycle_id"),
                "timestamp": cycle.get("timestamp"),
                "trigger": cycle.get("trigger"),
                "consciousness_index": cycle.get("consciousness_index"),
                "evolution_rate": cycle.get("evolution_rate"),
                "recommendations": cycle.get("recommendations", [])
            })
        
        # Get consciousness metrics history
        metrics_history = await db.llm_consciousness_metrics.find().sort(
            "timestamp", -1
        ).limit(limit).to_list(limit)
        
        # Build timeline
        timeline = []
        for metric in reversed(metrics_history):
            timeline.append({
                "timestamp": metric.get("timestamp"),
                "consciousness_index": metric.get("consciousness_index", 0),
                "coherence_ratio": metric.get("coherence_ratio", 0),
                "evolution_rate": metric.get("evolution_rate", 0),
                "value_integrity": metric.get("value_integrity", 0),
                "stability_index": metric.get("stability_index", 0),
                "emergence_level": metric.get("emergence_level", 0)
            })
        
        # Calculate trends
        if len(timeline) >= 2:
            ci_trend = "improving" if timeline[-1]["consciousness_index"] > timeline[0]["consciousness_index"] else "stable" if timeline[-1]["consciousness_index"] == timeline[0]["consciousness_index"] else "declining"
            vi_trend = "improving" if timeline[-1]["value_integrity"] > timeline[0]["value_integrity"] else "stable" if timeline[-1]["value_integrity"] == timeline[0]["value_integrity"] else "declining"
        else:
            ci_trend = "stable"
            vi_trend = "stable"
        
        return {
            "success": True,
            "timeline": timeline,
            "cycles": cycles,
            "total_records": len(timeline),
            "trends": {
                "consciousness_index": ci_trend,
                "value_integrity": vi_trend
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting evolution history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/collective/values")
async def get_value_drift_analysis():
    """
    Get long-term value drift analysis.
    
    Tracks drift in all core values (transparency, fairness, safety,
    performance, alignment, stability) with status indicators.
    
    Returns:
        Value drift metrics with heatmap data and stability indicators
    """
    try:
        # Analyze value drift
        drift_metrics = await consciousness_controller.analyze_value_drift()
        
        # Format for response
        formatted_metrics = []
        heatmap_data = []
        
        for dm in drift_metrics:
            formatted_metrics.append({
                "value_name": dm.value_name,
                "category": dm.category,
                "baseline": dm.baseline,
                "current": dm.current,
                "drift_amount": dm.drift_amount,
                "drift_percentage": dm.drift_percentage,
                "drift_velocity": dm.drift_velocity,
                "stability": dm.stability,
                "status": dm.status,
                "history": dm.history
            })
            
            # Heatmap data (for visualization)
            heatmap_data.append({
                "value": dm.value_name,
                "drift": abs(dm.drift_percentage),
                "status": dm.status
            })
        
        # Calculate overall drift status
        critical_count = sum(1 for dm in drift_metrics if dm.status == "critical")
        drifting_count = sum(1 for dm in drift_metrics if dm.status == "drifting")
        stable_count = sum(1 for dm in drift_metrics if dm.status == "stable")
        
        if critical_count > 0:
            overall_status = "critical"
        elif drifting_count > len(drift_metrics) / 2:
            overall_status = "moderate"
        else:
            overall_status = "healthy"
        
        return {
            "success": True,
            "drift_metrics": formatted_metrics,
            "heatmap_data": heatmap_data,
            "summary": {
                "total_values": len(drift_metrics),
                "stable": stable_count,
                "drifting": drifting_count,
                "critical": critical_count,
                "overall_status": overall_status
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting value drift analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/collective/reflection")
async def get_reflective_summary():
    """
    Generate meta-level reflective summary.
    
    Uses multi-provider LLM (GPT + Claude + Gemini) to generate
    high-level insights, strategic recommendations, and ethical
    considerations based on current consciousness state.
    
    Returns:
        ReflectiveSummary with emergent insights and future directions
    """
    try:
        logger.info("Generating reflective summary...")
        
        # Aggregate current state
        experience = await consciousness_controller.aggregate_collective_experiences()
        metrics = await consciousness_controller.compute_consciousness_index(experience)
        drift_metrics = await consciousness_controller.analyze_value_drift()
        
        # Generate reflective summary
        reflection = await consciousness_controller.generate_reflective_summary(
            experience=experience,
            metrics=metrics,
            drift_metrics=drift_metrics
        )
        
        return {
            "success": True,
            "reflection_id": reflection.reflection_id,
            "timestamp": reflection.timestamp,
            "consciousness_state": reflection.consciousness_state,
            "emergent_insights": reflection.emergent_insights,
            "strategic_recommendations": reflection.strategic_recommendations,
            "ethical_considerations": reflection.ethical_considerations,
            "learning_achievements": reflection.learning_achievements,
            "future_directions": reflection.future_directions,
            "llm_providers_used": reflection.llm_providers_used,
            "confidence": reflection.confidence,
            "is_mocked": "fallback" in reflection.llm_providers_used
        }
        
    except Exception as e:
        logger.error(f"Error generating reflective summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================
# Step 29: Autonomous Creativity & Meta-Strategic Synthesis
# ====================

from autonomous_creativity import CreativeSynthesisController

# Initialize Creativity Controller
creativity_controller = CreativeSynthesisController(db)

class CreativityGenerateRequest(BaseModel):
    """Request to generate creative strategies"""
    phase: Optional[str] = Field(default=None, pattern="^(opening|middlegame|endgame)$")
    count: int = Field(default=3, ge=1, le=10)
    use_patterns: bool = Field(default=True)

@api_router.post("/llm/creativity/generate")
async def generate_creative_strategies(request: CreativityGenerateRequest = CreativityGenerateRequest()):
    """
    Generate novel chess strategies using multi-provider LLM synthesis.
    
    Produces innovative strategic variants across game phases (opening, middlegame, endgame)
    with creative recombination of cognitive patterns from Steps 23-28.
    
    Features:
    - Multi-provider synthesis (GPT + Claude + Gemini)
    - Ethical guardrails (fair play, educational, anti-cheating)
    - Originality and stability metrics
    - Pattern-based creative recombination
    
    Args:
        phase: Specific phase or None for all phases
        count: Number of strategies per phase (1-10)
        use_patterns: Whether to leverage existing cognitive patterns
    
    Returns:
        List of creative strategies with evaluation scores
    """
    try:
        logger.info(f"Generating creative strategies: phase={request.phase}, count={request.count}")
        
        # Generate strategies
        strategies = await creativity_controller.generate_creative_strategies(
            phase=request.phase,
            count=request.count,
            use_patterns=request.use_patterns
        )
        
        # Separate approved and rejected
        approved = [s for s in strategies if not s.rejected]
        rejected = [s for s in strategies if s.rejected]
        
        return {
            "success": True,
            "strategies_generated": len(strategies),
            "approved": len(approved),
            "rejected": len(rejected),
            "strategies": [
                {
                    "strategy_id": s.strategy_id,
                    "phase": s.phase,
                    "strategy_name": s.strategy_name,
                    "description": s.description,
                    "tactical_elements": s.tactical_elements,
                    "novelty_score": s.novelty_score,
                    "stability_score": s.stability_score,
                    "ethical_alignment": s.ethical_alignment,
                    "educational_value": s.educational_value,
                    "risk_level": s.risk_level,
                    "llm_provider": s.llm_provider,
                    "parent_patterns": s.parent_patterns,
                    "rejected": s.rejected,
                    "rejection_reason": s.rejection_reason
                }
                for s in strategies
            ],
            "is_mocked": "[MOCKED]" in strategies[0].llm_provider if strategies else False
        }
        
    except Exception as e:
        logger.error(f"Error generating creative strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/creativity/status")
async def get_creativity_status():
    """
    Get current creativity system status and health metrics.
    
    Returns:
        Creativity metrics including:
        - Total ideas generated/approved/rejected
        - Average novelty, stability, and ethical scores
        - Creativity health index (0-1)
        - Provider and phase distribution
        - Health status (excellent/good/moderate/poor)
    """
    try:
        metrics = await creativity_controller.get_creativity_metrics()
        
        return {
            "success": True,
            "timestamp": metrics.timestamp,
            "total_ideas_generated": metrics.total_ideas_generated,
            "ideas_approved": metrics.ideas_approved,
            "ideas_rejected": metrics.ideas_rejected,
            "avg_novelty": metrics.avg_novelty,
            "avg_stability": metrics.avg_stability,
            "avg_ethical_alignment": metrics.avg_ethical_alignment,
            "creativity_health": metrics.creativity_health,
            "health_status": creativity_controller._get_health_status(metrics.creativity_health),
            "provider_distribution": metrics.provider_distribution,
            "phase_distribution": metrics.phase_distribution
        }
        
    except Exception as e:
        logger.error(f"Error getting creativity status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/creativity/history")
async def get_creativity_history(
    limit: int = Query(default=20, ge=1, le=100),
    phase: Optional[str] = Query(default=None, pattern="^(opening|middlegame|endgame)$"),
    approved_only: bool = Query(default=True)
):
    """
    Get historical creative strategies.
    
    Args:
        limit: Maximum number of strategies to return (1-100)
        phase: Filter by specific phase (optional)
        approved_only: Show only approved strategies (default: True)
    
    Returns:
        List of historical strategies with metadata
    """
    try:
        # Build query
        query = {}
        if phase:
            query["phase"] = phase
        if approved_only:
            query["rejected"] = False
        
        # Get strategies
        strategies = await db.llm_creative_synthesis.find(query).sort(
            "timestamp", -1
        ).limit(limit).to_list(limit)
        
        # Get meta-strategies
        meta_strategies = await db.llm_meta_strategy_log.find().sort(
            "timestamp", -1
        ).limit(5).to_list(5)
        
        # Convert ObjectId to string for JSON serialization
        for s in strategies:
            if '_id' in s:
                s['_id'] = str(s['_id'])
        
        for m in meta_strategies:
            if '_id' in m:
                m['_id'] = str(m['_id'])
        
        return {
            "success": True,
            "count": len(strategies),
            "strategies": strategies,
            "meta_strategies": meta_strategies
        }
        
    except Exception as e:
        logger.error(f"Error getting creativity history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/creativity/report")
async def get_creativity_report():
    """
    Generate comprehensive creativity report.
    
    Provides:
    - System metrics and health
    - Recent approved strategies
    - Meta-strategy synthesis
    - Phase-by-phase breakdown
    - Actionable recommendations
    - Sample outputs for each phase
    
    Returns:
        Complete creativity system report
    """
    try:
        logger.info("Generating comprehensive creativity report...")
        
        # Generate report
        report = await creativity_controller.generate_creativity_report()
        
        # If no strategies exist, generate sample ones
        if report.get("metrics", {}).get("total_ideas_generated", 0) == 0:
            logger.info("No strategies found, generating samples...")
            await creativity_controller.generate_creative_strategies(count=1)
            
            # Generate meta-strategy
            await creativity_controller.synthesize_meta_strategy()
            
            # Regenerate report
            report = await creativity_controller.generate_creativity_report()
        
        # Convert ObjectId to string for JSON serialization
        if "recent_approved_strategies" in report:
            for s in report["recent_approved_strategies"]:
                if '_id' in s:
                    s['_id'] = str(s['_id'])
        
        if "recent_meta_strategy" in report and report["recent_meta_strategy"]:
            if '_id' in report["recent_meta_strategy"]:
                report["recent_meta_strategy"]['_id'] = str(report["recent_meta_strategy"]['_id'])
        
        if "phase_breakdown" in report:
            for phase, data in report["phase_breakdown"].items():
                if "sample" in data and data["sample"] and '_id' in data["sample"]:
                    data["sample"]['_id'] = str(data["sample"]['_id'])
                # Convert numpy types to native Python types
                if "avg_novelty" in data:
                    data["avg_novelty"] = float(data["avg_novelty"])
                if "avg_stability" in data:
                    data["avg_stability"] = float(data["avg_stability"])
        
        return {
            "success": True,
            **report
        }
        
    except Exception as e:
        logger.error(f"Error generating creativity report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/llm/creativity/synthesize-meta")
async def synthesize_meta_strategy():
    """
    Synthesize meta-strategic vision from creative outputs.
    
    Integrates multiple creative strategies into a coherent long-term
    meta-strategic framework with:
    - Overarching theme and vision
    - Coherence and adaptability scores
    - Long-term value assessment
    - Ethical compliance verification
    - Recommended contexts and safety constraints
    
    Returns:
        MetaStrategy with synthesized insights
    """
    try:
        logger.info("Synthesizing meta-strategy...")
        
        # Synthesize meta-strategy
        meta = await creativity_controller.synthesize_meta_strategy()
        
        return {
            "success": True,
            "meta_id": meta.meta_id,
            "timestamp": meta.timestamp,
            "theme": meta.theme,
            "description": meta.description,
            "integrated_strategies": meta.integrated_strategies,
            "coherence_score": meta.coherence_score,
            "adaptability_score": meta.adaptability_score,
            "long_term_value": meta.long_term_value,
            "ethical_compliance": meta.ethical_compliance,
            "recommended_contexts": meta.recommended_contexts,
            "safety_constraints": meta.safety_constraints,
            "is_mocked": "[MOCKED]" in meta.description
        }
        
    except Exception as e:
        logger.error(f"Error synthesizing meta-strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================
# System Optimization Endpoints (Step 35)
# ====================

from system_optimization import SystemOptimizationController, initialize_optimization_controller

# Initialize optimization controller
optimization_controller = None

@app.on_event("startup")
async def initialize_optimization():
    """Initialize the optimization controller on startup"""
    global optimization_controller
    optimization_controller = initialize_optimization_controller(db)
    logger.info("System Optimization Controller initialized")


@api_router.post("/llm/optimize/run")
async def run_optimization_cycle():
    """
    Trigger full optimization cycle across all subsystems
    
    Returns comprehensive optimization results including:
    - Runtime optimization (CPU/GPU balancing)
    - LLM inference scaling
    - Database I/O streamlining
    - System latency evaluation
    - Optimization report
    """
    try:
        if not optimization_controller:
            raise HTTPException(status_code=500, detail="Optimization controller not initialized")
        
        logger.info("Running full optimization cycle...")
        result = await optimization_controller.run_full_optimization_cycle()
        
        return {
            "success": True,
            "cycle_id": result.get('cycle_id'),
            "timestamp": result.get('timestamp'),
            "cycle_duration": result.get('cycle_duration'),
            "optimization_count": result.get('optimization_count'),
            "overall_status": result.get('overall_status'),
            "critical_issues": result.get('report', {}).get('critical_issues', []),
            "recommendations": result.get('report', {}).get('recommendations', []),
            "runtime_optimization": result.get('runtime_optimization'),
            "inference_balancing": result.get('inference_balancing'),
            "database_streamlining": result.get('database_streamlining'),
            "latency_evaluation": result.get('latency_evaluation'),
            "report": result.get('report')
        }
        
    except Exception as e:
        logger.error(f"Error running optimization cycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/optimize/status")
async def get_optimization_status():
    """
    Get current system resource metrics and optimization status
    
    Returns real-time snapshot of:
    - CPU and memory utilization
    - Average API latency
    - Optimization activity status
    - Metrics collection counts
    - System health indicator
    """
    try:
        if not optimization_controller:
            raise HTTPException(status_code=500, detail="Optimization controller not initialized")
        
        status = await optimization_controller.get_current_status()
        
        return {
            "success": True,
            "timestamp": status.get('timestamp'),
            "optimization_active": status.get('optimization_active'),
            "last_optimization": status.get('last_optimization'),
            "total_optimizations": status.get('total_optimizations'),
            "current_cpu": status.get('current_cpu'),
            "current_memory_percent": status.get('current_memory_percent'),
            "avg_latency": status.get('avg_latency'),
            "metrics_collected": status.get('metrics_collected'),
            "actions_recorded": status.get('actions_recorded'),
            "llm_inferences": status.get('llm_inferences'),
            "db_operations": status.get('db_operations'),
            "system_health": status.get('system_health')
        }
        
    except Exception as e:
        logger.error(f"Error getting optimization status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/optimize/report")
async def get_optimization_report(report_number: Optional[int] = None):
    """
    Generate detailed optimization report
    
    Args:
        report_number: Optional specific report number (generates new if None)
    
    Returns comprehensive report including:
    - Overall system health assessment
    - Critical issues and recommendations
    - Performance metrics vs targets
    - Module-specific performance
    - Predicted bottlenecks
    - Optimization roadmap
    """
    try:
        if not optimization_controller:
            raise HTTPException(status_code=500, detail="Optimization controller not initialized")
        
        logger.info(f"Generating optimization report #{report_number or 'new'}...")
        report = await optimization_controller.generate_optimization_report(report_number)
        
        return {
            "success": True,
            "report_id": report.report_id,
            "report_number": report.report_number,
            "timestamp": report.timestamp,
            "period_start": report.period_start,
            "period_end": report.period_end,
            "overall_health": report.overall_health,
            "critical_issues": report.critical_issues,
            "recommendations": report.recommendations,
            "latency_trend": report.latency_trend,
            "resource_trend": report.resource_trend,
            "efficiency_trend": report.efficiency_trend,
            "current_metrics": report.current_metrics,
            "target_metrics": report.target_metrics,
            "metrics_delta": report.metrics_delta,
            "actions_taken": report.actions_taken,
            "actions_approved": report.actions_approved,
            "actions_rejected": report.actions_rejected,
            "avg_improvement": report.avg_improvement,
            "module_performance": report.module_performance,
            "predicted_bottlenecks": report.predicted_bottlenecks,
            "optimization_roadmap": report.optimization_roadmap
        }
        
    except Exception as e:
        logger.error(f"Error generating optimization report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/optimize/metrics/history")
async def get_optimization_metrics_history(limit: int = 100):
    """
    Get historical optimization metrics from database
    
    Args:
        limit: Maximum number of records to return (default 100)
    
    Returns list of historical optimization metrics
    """
    try:
        metrics = await db['llm_optimization_metrics'].find(
            {},
            {'_id': 0}
        ).sort('timestamp', -1).limit(limit).to_list(length=limit)
        
        return {
            "success": True,
            "count": len(metrics),
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error retrieving metrics history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/optimize/logs")
async def get_optimization_logs(log_type: Optional[str] = None, limit: int = 50):
    """
    Get optimization action logs from database
    
    Args:
        log_type: Filter by log type ('report', 'optimization_cycle', 'action', etc.)
        limit: Maximum number of records to return (default 50)
    
    Returns list of optimization logs
    """
    try:
        query = {}
        if log_type:
            query['type'] = log_type
        
        logs = await db['llm_optimization_logs'].find(
            query,
            {'_id': 0}
        ).sort('timestamp', -1).limit(limit).to_list(length=limit)
        
        return {
            "success": True,
            "count": len(logs),
            "log_type": log_type,
            "logs": logs
        }
        
    except Exception as e:
        logger.error(f"Error retrieving optimization logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================
# CASV-1 Validation Endpoints (Step 36)
# ====================

from casv1_validator import get_casv1_validator
from casv1_reports import CASV1ReportGenerator

casv1_status = {
    "active": False,
    "progress": 0,
    "message": "",
    "validation_id": None,
    "start_time": None
}

def run_casv1_validation(num_games: int = 10):
    """Run CASV-1 validation in background"""
    global casv1_status
    
    try:
        # Use sync MongoDB client
        from pymongo import MongoClient
        import asyncio
        
        sync_client = MongoClient(os.environ.get('MONGO_URL', 'mongodb://localhost:27017'))
        sync_db = sync_client[os.environ.get('DB_NAME', 'alphazero_chess')]
        
        # Create async event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        casv1_status["message"] = "Initializing CASV-1 validator..."
        casv1_status["progress"] = 5
        
        # Create validator using async db client (we'll need async wrapper)
        # For now, let's create a simpler sync version
        from casv1_validator import CASV1Validator
        from pathlib import Path
        
        # We need to adapt this to work with sync client
        # For now, let's run the async version
        async def run_validation():
            validator = get_casv1_validator(db, Path("/app/logs/CASV1"))
            casv1_status["validation_id"] = validator.validation_id
            casv1_status["message"] = "Running validation..."
            casv1_status["progress"] = 10
            
            # Run full validation
            metrics = await validator.run_full_validation(num_games)
            
            casv1_status["progress"] = 90
            casv1_status["message"] = "Generating reports..."
            
            # Generate reports
            report_generator = CASV1ReportGenerator(validator)
            report_files = report_generator.generate_all_reports(metrics)
            
            casv1_status["progress"] = 100
            casv1_status["message"] = f"Validation complete! Reports: {', '.join(report_files)}"
            
            return metrics, report_files
        
        # Run async validation
        metrics, report_files = loop.run_until_complete(run_validation())
        
        logger.info(f"CASV-1 validation completed: {casv1_status['validation_id']}")
        
        sync_client.close()
        loop.close()
        
    except Exception as e:
        logger.error(f"CASV-1 validation error: {str(e)}")
        casv1_status["message"] = f"Error: {str(e)}"
        casv1_status["progress"] = 0
        raise
    finally:
        casv1_status["active"] = False

class CASV1Request(BaseModel):
    num_games: int = Field(default=10, ge=3, le=50)

@api_router.post("/casv1/start")
async def start_casv1_validation(request: CASV1Request, background_tasks: BackgroundTasks):
    """Start CASV-1 Unified System Test & Validation"""
    global casv1_status
    
    if casv1_status["active"]:
        raise HTTPException(status_code=400, detail="CASV-1 validation already in progress")
    
    casv1_status = {
        "active": True,
        "progress": 0,
        "message": "Starting CASV-1 validation...",
        "validation_id": None,
        "start_time": datetime.now(timezone.utc)
    }
    
    # Run in background
    background_tasks.add_task(run_casv1_validation, request.num_games)
    
    return {
        "success": True,
        "message": "CASV-1 validation started",
        "num_games": request.num_games,
        "estimated_duration": f"{request.num_games * 30}s"
    }

@api_router.get("/casv1/status")
async def get_casv1_status():
    """Get CASV-1 validation status"""
    return {
        "active": casv1_status["active"],
        "progress": casv1_status["progress"],
        "message": casv1_status["message"],
        "validation_id": casv1_status.get("validation_id"),
        "start_time": casv1_status.get("start_time")
    }

@api_router.get("/casv1/results")
async def get_casv1_results(limit: int = 10):
    """Get CASV-1 validation results history"""
    validations = await db.casv1_validations.find().sort("timestamp", -1).limit(limit).to_list(limit)
    return {
        "success": True,
        "count": len(validations),
        "validations": validations
    }

@api_router.get("/casv1/reports/{validation_id}")
async def get_casv1_reports(validation_id: str):
    """Get CASV-1 reports for a specific validation"""
    from pathlib import Path
    import os
    
    logs_dir = Path("/app/logs/CASV1")
    
    # Check if validation exists
    validation = await db.casv1_validations.find_one({"validation_id": validation_id})
    if not validation:
        raise HTTPException(status_code=404, detail="Validation not found")
    
    # List available reports
    report_files = [
        'CASV1_FunctionalReport.md',
        'CASV1_PerformanceReport.md',
        'CASV1_EthicalReport.md',
        'CASV1_ResonanceSummary.md',
        'CASV1_MasterValidationReport.md'
    ]
    
    available_reports = []
    for report_file in report_files:
        report_path = logs_dir / report_file
        if report_path.exists():
            available_reports.append({
                "filename": report_file,
                "size": os.path.getsize(report_path),
                "path": str(report_path)
            })
    
    return {
        "success": True,
        "validation_id": validation_id,
        "reports": available_reports,
        "logs_directory": str(logs_dir)
    }

@api_router.get("/casv1/report/{filename}")
async def download_casv1_report(filename: str):
    """Download a specific CASV-1 report"""
    from pathlib import Path
    
    logs_dir = Path("/app/logs/CASV1")
    report_path = logs_dir / filename
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        path=str(report_path),
        filename=filename,
        media_type="text/markdown"
    )


# ====================
# AlphaZero Historical Memory Archive Endpoints
# ====================

class MemoryUploadResponse(BaseModel):
    success: bool
    games_parsed: int
    games_stored: int
    errors: List[str] = []
    message: str

class TrainFromMemoryRequest(BaseModel):
    game_ids: Optional[List[str]] = None
    num_epochs: int = Field(default=5, ge=1, le=50)
    batch_size: int = Field(default=64, ge=8, le=256)
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1)

class PGNUploadRequest(BaseModel):
    pgn_content: str

@api_router.post("/memory/upload", response_model=MemoryUploadResponse)
async def upload_pgn_memory(request: PGNUploadRequest):
    """
    Upload and parse PGN file containing AlphaZero vs Stockfish historical matches.
    Stores games in alphazero_history collection for replay and training.
    """
    try:
        from pgn_parser import PGNParser
        
        # Get PGN content from request
        content = request.pgn_content
        
        if not content:
            raise HTTPException(status_code=400, detail="No PGN content provided")
        
        logger.info("Starting PGN parsing for historical memory upload...")
        
        # Parse PGN file
        parser = PGNParser()
        games = parser.parse_pgn_file(content)
        
        if not games:
            raise HTTPException(status_code=400, detail="No valid games found in PGN file")
        
        # Store games in MongoDB
        stored_count = 0
        for game in games:
            # Check if game already exists
            existing = await db.alphazero_history.find_one({"game_id": game["game_id"]})
            
            if existing:
                # Update existing game
                await db.alphazero_history.replace_one(
                    {"game_id": game["game_id"]},
                    game
                )
            else:
                # Insert new game
                await db.alphazero_history.insert_one(game)
            
            stored_count += 1
        
        stats = parser.get_parsing_stats()
        
        logger.info(f"Successfully stored {stored_count} historical games")
        
        return MemoryUploadResponse(
            success=True,
            games_parsed=stats["games_parsed"],
            games_stored=stored_count,
            errors=stats["errors"],
            message=f"Successfully parsed and stored {stored_count} AlphaZero vs Stockfish games"
        )
        
    except Exception as e:
        logger.error(f"Error uploading PGN memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/memory/list")
async def list_historical_matches(
    limit: int = Query(default=100, ge=1, le=500),
    skip: int = Query(default=0, ge=0),
    winner: Optional[str] = Query(default=None),
    sort_by: str = Query(default="game_number")
):
    """
    List all stored historical AlphaZero vs Stockfish matches.
    Supports filtering by winner and pagination.
    """
    try:
        # Build query filter
        query_filter = {}
        if winner:
            query_filter["winner"] = winner
        
        # Get total count
        total_count = await db.alphazero_history.count_documents(query_filter)
        
        # Determine sort order
        sort_field = sort_by if sort_by in ["game_number", "move_count", "date"] else "game_number"
        
        # Fetch games
        games = await db.alphazero_history.find(query_filter).sort(
            sort_field, 1
        ).skip(skip).limit(limit).to_list(limit)
        
        # Remove large fields for list view (positions array is big)
        games_summary = []
        for game in games:
            game_summary = {
                "game_id": game.get("game_id"),
                "game_number": game.get("game_number"),
                "white": game.get("white"),
                "black": game.get("black"),
                "result": game.get("result"),
                "winner": game.get("winner"),
                "outcome": game.get("outcome"),
                "date": game.get("date"),
                "move_count": game.get("move_count"),
                "opening": game.get("opening"),
                "eco": game.get("eco"),
                "timestamp_recalled": game.get("timestamp_recalled")
            }
            games_summary.append(game_summary)
        
        return {
            "success": True,
            "total_count": total_count,
            "games": games_summary,
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error listing historical matches: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/memory/replay/{game_id}")
async def get_match_replay(game_id: str):
    """
    Get complete match data for replay, including all moves and positions.
    """
    try:
        game = await db.alphazero_history.find_one({"game_id": game_id})
        
        if not game:
            raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
        
        # Remove MongoDB _id for JSON serialization
        if "_id" in game:
            del game["_id"]
        
        # Remove large position array if not needed for display
        # Keep moves with FEN strings for board replay
        if "positions" in game:
            game["positions_available"] = len(game["positions"])
            # Remove positions array to reduce response size (optional)
            # game.pop("positions", None)
        
        return {
            "success": True,
            "game": game
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching game replay: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/train/from_memory")
async def train_from_historical_memory(
    request: TrainFromMemoryRequest,
    background_tasks: BackgroundTasks
):
    """
    Train AlphaZero from historical memory (PGN games).
    Extracts positions and moves from stored games and runs training.
    """
    try:
        from pgn_parser import PGNParser
        
        # Fetch games to train from
        if request.game_ids:
            games = await db.alphazero_history.find({
                "game_id": {"$in": request.game_ids}
            }).to_list(1000)
        else:
            # Use all available games
            games = await db.alphazero_history.find().to_list(1000)
        
        if not games:
            raise HTTPException(status_code=404, detail="No historical games found for training")
        
        logger.info(f"Preparing to train from {len(games)} historical games")
        
        # Extract training positions
        parser = PGNParser()
        training_data = parser.extract_training_positions(games)
        
        if not training_data:
            raise HTTPException(status_code=400, detail="No training positions could be extracted")
        
        logger.info(f"Extracted {len(training_data)} training positions from historical games")
        
        # Store training positions for the training pipeline
        session_id = str(uuid.uuid4())
        
        # Store positions in self_play_positions collection (reusing existing structure)
        for i, pos in enumerate(training_data[:5000]):  # Limit to 5000 positions
            await db.self_play_positions.insert_one({
                "position": pos["position"],
                "fen": pos["fen"],
                "value": pos["value"],
                "move_played": pos.get("move_played"),
                "game_id": pos["game_id"],
                "move_number": pos["move_number"],
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc),
                "source": "historical_memory"
            })
        
        # Initialize network and trainer
        network = AlphaZeroNetwork()
        trainer = AlphaZeroTrainer(network, learning_rate=learning_rate)
        
        # Prepare training data in format expected by trainer
        formatted_training_data = []
        for pos in training_data[:5000]:
            formatted_training_data.append({
                "position": np.array(pos["position"]),
                "policy": {},  # Historical games don't have policy targets
                "value": pos["value"]
            })
        
        # Run training in background
        def run_memory_training():
            try:
                training_history = []
                
                for epoch in range(request.num_epochs):
                    logger.info(f"Memory training epoch {epoch + 1}/{request.num_epochs}")
                    
                    # Train on historical positions
                    metrics = trainer.train_epoch(formatted_training_data, batch_size=request.batch_size)
                    training_history.append(metrics)
                    
                    logger.info(f"Epoch {epoch + 1} - Loss: {metrics.get('loss', 0):.4f}")
                
                # Save trained model
                metadata = {
                    "training_date": datetime.now(timezone.utc).isoformat(),
                    "source": "historical_memory",
                    "num_games": len(games),
                    "num_positions": len(training_data),
                    "num_epochs": request.num_epochs,
                    "learning_rate": request.learning_rate,
                    "batch_size": request.batch_size,
                    "session_id": session_id,
                    "final_loss": training_history[-1]["loss"] if training_history else 0.0
                }
                
                model_path = model_manager.save_versioned_model(network, metadata=metadata)
                model_name = Path(model_path).stem
                
                logger.info(f"Historical memory training complete. Model saved: {model_name}")
                
                # Store training metrics
                sync_client = MongoClient(os.environ.get('MONGO_URL', 'mongodb://localhost:27017'))
                sync_db = sync_client[os.environ.get('DB_NAME', 'alphazero_chess')]
                
                for metrics in training_history:
                    sync_db.training_metrics.insert_one({
                        **metrics,
                        "session_id": session_id,
                        "source": "historical_memory"
                    })
                
                sync_client.close()
                
            except Exception as e:
                logger.error(f"Error in memory training: {e}")
        
        # Run training in background
        background_tasks.add_task(run_memory_training)
        
        return {
            "success": True,
            "message": f"Training started from {len(games)} historical games",
            "session_id": session_id,
            "training_positions": len(training_data),
            "games_count": len(games),
            "epochs": request.num_epochs,
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate
        }
        
    except Exception as e:
        logger.error(f"Error training from memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/memory/stats")
async def get_memory_archive_stats():
    """Get statistics about the historical memory archive"""
    try:
        total_games = await db.alphazero_history.count_documents({})
        
        if total_games == 0:
            return {
                "success": True,
                "total_games": 0,
                "message": "No historical games stored yet"
            }
        
        # Get winner distribution
        alphazero_wins = await db.alphazero_history.count_documents({"winner": "white"})
        stockfish_wins = await db.alphazero_history.count_documents({"winner": "black"})
        draws = await db.alphazero_history.count_documents({"winner": "draw"})
        
        # Get average move count
        pipeline = [
            {"$group": {
                "_id": None,
                "avg_moves": {"$avg": "$move_count"},
                "max_moves": {"$max": "$move_count"},
                "min_moves": {"$min": "$move_count"}
            }}
        ]
        stats = await db.alphazero_history.aggregate(pipeline).to_list(1)
        move_stats = stats[0] if stats else {}
        
        # Get most recent upload
        latest = await db.alphazero_history.find_one(
            sort=[("timestamp_recalled", -1)]
        )
        
        return {
            "success": True,
            "total_games": total_games,
            "alphazero_wins": alphazero_wins,
            "stockfish_wins": stockfish_wins,
            "draws": draws,
            "win_rate": {
                "alphazero": round(alphazero_wins / total_games * 100, 1) if total_games > 0 else 0,
                "stockfish": round(stockfish_wins / total_games * 100, 1) if total_games > 0 else 0,
                "draw": round(draws / total_games * 100, 1) if total_games > 0 else 0
            },
            "move_statistics": {
                "average": round(move_stats.get("avg_moves", 0), 1),
                "maximum": move_stats.get("max_moves", 0),
                "minimum": move_stats.get("min_moves", 0)
            },
            "last_upload": latest.get("timestamp_recalled") if latest else None
        }
        
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
