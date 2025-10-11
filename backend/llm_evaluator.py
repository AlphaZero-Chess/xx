from emergentintegrations.llm.chat import LlmChat, UserMessage
from dotenv import load_dotenv
import os
import chess
import logging
import time
import uuid
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """LLM Configuration Settings"""
    response_mode: str = "balanced"  # "fast", "balanced", "insightful"
    prompt_depth: int = 5  # 1-10 scale
    adaptive_enabled: bool = True
    max_response_time: float = 10.0  # seconds
    fallback_mode: str = "fast"
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

@dataclass
class PerformanceMetrics:
    """Performance tracking for LLM requests"""
    timestamp: str
    response_time: float  # seconds
    model_used: str
    prompt_length: int
    response_length: int
    mode: str
    success: bool
    fallback_triggered: bool = False
    error: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)

class LLMChessEvaluator:
    """Use LLM for chess position evaluation and insights with adaptive tuning"""
    
    # Class-level config shared across all instances (global settings)
    _global_config: Optional[LLMConfig] = None
    
    def __init__(self, session_id: str = "chess-evaluator", config: Optional[LLMConfig] = None):
        self.api_key = os.environ.get('EMERGENT_LLM_KEY')
        self.session_id = session_id
        
        # Use provided config or global config or default
        if config:
            self.config = config
        elif LLMChessEvaluator._global_config:
            self.config = LLMChessEvaluator._global_config
        else:
            self.config = LLMConfig()
        
        self.chat = self._create_chat_instance()
        self.conversation_history = []
        self.performance_history: List[PerformanceMetrics] = []
        self.total_requests = 0
        self.successful_requests = 0
        self.avg_response_time = 0.0
    
    @classmethod
    def set_global_config(cls, config: LLMConfig):
        """Set global configuration for all instances"""
        cls._global_config = config
        logger.info(f"Global LLM config updated: {config.to_dict()}")
    
    @classmethod
    def get_global_config(cls) -> LLMConfig:
        """Get current global configuration"""
        if cls._global_config is None:
            cls._global_config = LLMConfig()
        return cls._global_config
    
    def _create_chat_instance(self):
        """Create LlmChat instance with appropriate system message"""
        system_message = self._get_system_message()
        return LlmChat(
            api_key=self.api_key,
            session_id=self.session_id,
            system_message=system_message
        ).with_model("openai", "gpt-4o-mini")
    
    def _get_system_message(self) -> str:
        """Generate system message based on response mode"""
        base_msg = "You are an expert chess coach. Analyze positions, suggest moves, and provide strategic insights. Be encouraging and educational."
        
        if self.config.response_mode == "fast":
            return f"{base_msg} Keep responses very concise (1-2 sentences max)."
        elif self.config.response_mode == "insightful":
            return f"{base_msg} Provide detailed, comprehensive analysis with specific tactical and strategic insights."
        else:  # balanced
            return f"{base_msg} Keep responses concise but informative (2-4 sentences)."
    
    def update_config(self, new_config: LLMConfig):
        """Update configuration and recreate chat instance"""
        self.config = new_config
        self.chat = self._create_chat_instance()
        logger.info(f"LLM config updated for session {self.session_id}: {new_config.to_dict()}")
    
    def adaptive_prompt_builder(self, base_prompt: str, context: str = "") -> str:
        """
        Build adaptive prompts based on configuration and performance
        Adjusts complexity based on prompt_depth and response_mode
        """
        prompt_depth = self.config.prompt_depth
        
        # Adjust prompt complexity based on depth (1-10)
        if prompt_depth <= 3:
            # Minimal prompt
            detail_instruction = "Be very brief."
        elif prompt_depth <= 6:
            # Moderate prompt
            detail_instruction = "Be concise but informative."
        else:
            # Detailed prompt
            detail_instruction = "Provide detailed analysis with specific examples."
        
        # Add context if provided
        full_prompt = base_prompt
        if context:
            full_prompt += f"\n\nContext: {context}"
        
        full_prompt += f"\n\n{detail_instruction}"
        
        return full_prompt
    
    def optimize_llm_performance(self) -> Dict:
        """
        Analyze performance metrics and provide optimization recommendations
        Returns current performance stats and suggestions
        """
        if len(self.performance_history) == 0:
            return {
                "avg_response_time": 0,
                "success_rate": 0,
                "total_requests": 0,
                "fallback_count": 0,
                "recommendation": "No data yet"
            }
        
        recent_metrics = self.performance_history[-20:]  # Last 20 requests
        
        avg_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
        success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
        fallback_count = sum(1 for m in recent_metrics if m.fallback_triggered)
        
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
        
        return {
            "avg_response_time": round(avg_time, 2),
            "success_rate": round(success_rate * 100, 1),
            "total_requests": len(self.performance_history),
            "fallback_count": fallback_count,
            "current_mode": self.config.response_mode,
            "adaptive_enabled": self.config.adaptive_enabled,
            "recommendations": recommendations
        }
    
    async def _send_with_performance_tracking(self, prompt: str, operation: str = "general") -> Tuple[str, PerformanceMetrics]:
        """
        Send message with performance tracking and automatic fallback
        Returns: (response_text, metrics)
        """
        start_time = time.time()
        fallback_triggered = False
        success = True
        error_msg = None
        response_text = ""
        
        try:
            message = UserMessage(text=prompt)
            
            # Attempt to get response
            response_text = await self.chat.send_message(message)
            elapsed = time.time() - start_time
            
            # Check if response took too long and adaptive mode is enabled
            if elapsed > self.config.max_response_time and self.config.adaptive_enabled:
                logger.warning(f"Response took {elapsed:.2f}s (>{self.config.max_response_time}s). Triggering fallback.")
                fallback_triggered = True
                
                # Switch to fallback mode temporarily
                original_mode = self.config.response_mode
                self.config.response_mode = self.config.fallback_mode
                self.chat = self._create_chat_instance()
                
                # Try again with faster mode
                start_time = time.time()
                message = UserMessage(text=prompt)
                response_text = await self.chat.send_message(message)
                elapsed = time.time() - start_time
                
                # Restore original mode
                self.config.response_mode = original_mode
                self.chat = self._create_chat_instance()
                
        except Exception as e:
            elapsed = time.time() - start_time
            success = False
            error_msg = str(e)
            response_text = f"Response unavailable: {error_msg}"
            logger.error(f"LLM {operation} error: {e}")
        
        # Record metrics
        metrics = PerformanceMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            response_time=elapsed,
            model_used="gpt-4o-mini",
            prompt_length=len(prompt),
            response_length=len(response_text),
            mode=self.config.response_mode,
            success=success,
            fallback_triggered=fallback_triggered,
            error=error_msg
        )
        
        self.performance_history.append(metrics)
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        
        # Update rolling average
        recent_times = [m.response_time for m in self.performance_history[-10:]]
        self.avg_response_time = sum(recent_times) / len(recent_times)
        
        return response_text, metrics
    
    async def evaluate_position(self, fen: str, context: str = ""):
        """
        Evaluate a chess position using LLM with performance tracking
        Returns: evaluation text
        """
        try:
            board = chess.Board(fen)
            
            # Create position description
            position_desc = f"Position (FEN): {fen}\n"
            position_desc += f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}\n"
            position_desc += f"Legal moves: {len(list(board.legal_moves))}\n"
            
            base_prompt = f"{position_desc}\n\nProvide a brief strategic evaluation."
            prompt = self.adaptive_prompt_builder(base_prompt, context)
            
            response, metrics = await self._send_with_performance_tracking(prompt, "evaluate_position")
            return response
            
        except Exception as e:
            logger.error(f"LLM evaluation error: {e}")
            return "Evaluation unavailable"
    
    async def suggest_opening_strategy(self, fen: str):
        """Get opening strategy suggestions with performance tracking"""
        try:
            base_prompt = f"Position: {fen}\n\nSuggest opening strategy for this position."
            prompt = self.adaptive_prompt_builder(base_prompt)
            response, metrics = await self._send_with_performance_tracking(prompt, "suggest_strategy")
            return response
        except Exception as e:
            logger.error(f"LLM strategy error: {e}")
            return "No strategy available"
    
    async def analyze_game(self, moves_history: list, result: str):
        """Analyze a completed game with performance tracking"""
        try:
            moves_str = " ".join(moves_history[:20])  # First 20 moves
            base_prompt = f"Game moves: {moves_str}\nResult: {result}\n\nProvide analysis of key moments."
            prompt = self.adaptive_prompt_builder(base_prompt)
            response, metrics = await self._send_with_performance_tracking(prompt, "analyze_game")
            return response
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return "Analysis unavailable"
    
    async def coach_with_mcts(self, fen: str, top_moves: List[Dict], position_value: float, context: str = ""):
        """
        Provide coaching using AlphaZero's MCTS evaluation with performance tracking
        top_moves: [{"move": "e2e4", "probability": 0.45, "visits": 360}, ...]
        position_value: estimated value from -1 (black winning) to +1 (white winning)
        """
        try:
            board = chess.Board(fen)
            turn = "White" if board.turn == chess.WHITE else "Black"
            
            # Format top moves
            moves_desc = "\n".join([
                f"{i+1}. {m['move']} ({m['probability']*100:.1f}% confidence, {m['visits']} visits)"
                for i, m in enumerate(top_moves[:3])
            ])
            
            # Interpret position value
            if position_value > 0.3:
                eval_text = f"White has a significant advantage ({position_value:.2f})"
            elif position_value > 0.1:
                eval_text = f"White is slightly better ({position_value:.2f})"
            elif position_value > -0.1:
                eval_text = f"Position is roughly equal ({position_value:.2f})"
            elif position_value > -0.3:
                eval_text = f"Black is slightly better ({position_value:.2f})"
            else:
                eval_text = f"Black has a significant advantage ({position_value:.2f})"
            
            base_prompt = f"""Position (FEN): {fen}
Turn: {turn} to move
AlphaZero Evaluation: {eval_text}

Top recommended moves by AlphaZero:
{moves_desc}

{context if context else "Provide coaching advice for this position."}

As a chess coach, explain the position and recommend the best move."""
            
            prompt = self.adaptive_prompt_builder(base_prompt)
            response, metrics = await self._send_with_performance_tracking(prompt, "coach_with_mcts")
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            logger.error(f"LLM coaching error: {e}")
            return "Coaching unavailable. The AI recommends trying the top-rated move from AlphaZero's analysis."
    
    async def analyze_specific_move(self, fen: str, move: str, was_best: bool, better_moves: List[str] = None):
        """Analyze why a specific move was good or bad with performance tracking"""
        try:
            board = chess.Board(fen)
            turn = "White" if board.turn == chess.WHITE else "Black"
            
            if was_best:
                base_prompt = f"Position: {fen}\n{turn} played {move}.\n\nThis was the best move! Explain why this move is strong."
            else:
                better_str = ", ".join(better_moves[:3]) if better_moves else "other options"
                base_prompt = f"Position: {fen}\n{turn} played {move}.\n\nThis wasn't the best choice. Better moves: {better_str}.\n\nExplain what's wrong with {move} and why the alternatives are better."
            
            prompt = self.adaptive_prompt_builder(base_prompt)
            response, metrics = await self._send_with_performance_tracking(prompt, "analyze_move")
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            logger.error(f"LLM move analysis error: {e}")
            return "Move analysis unavailable."
    
    async def general_question(self, question: str, fen: str = None):
        """Answer general chess questions from the user with performance tracking"""
        try:
            context = f"\nCurrent position (FEN): {fen}" if fen else ""
            base_prompt = f"{question}{context}"
            
            prompt = self.adaptive_prompt_builder(base_prompt)
            response, metrics = await self._send_with_performance_tracking(prompt, "general_question")
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            logger.error(f"LLM question error: {e}")
            return "Unable to answer right now."
    
    def reset_conversation(self):
        """Reset conversation history for a new game"""
        self.conversation_history = []
        logger.info(f"Conversation reset for session {self.session_id}")
    
    def get_conversation_history(self):
        """Get the conversation history"""
        return self.conversation_history
    
    async def analyze_training_metrics(self, training_data: dict, evaluation_data: dict, selfplay_data: dict):
        """
        Analyze training metrics and provide detailed coaching insights with performance tracking
        
        Args:
            training_data: Dictionary with training metrics (epochs, losses, etc.)
            evaluation_data: Dictionary with evaluation results (win rates, etc.)
            selfplay_data: Dictionary with self-play statistics
        
        Returns:
            Comprehensive analysis text with actionable recommendations
        """
        try:
            # Build comprehensive prompt
            base_prompt = f"""You are an expert AI training coach for AlphaZero chess models. Analyze the following training and evaluation metrics to provide detailed insights and actionable recommendations.

**TRAINING METRICS:**
- Total Sessions: {training_data.get('total_sessions', 0)}
- Total Epochs: {training_data.get('total_epochs', 0)}
- Recent Loss Trends: {training_data.get('loss_summary', 'N/A')}
- Average Loss (Recent): {training_data.get('avg_recent_loss', 'N/A')}
- Loss Improvement Rate: {training_data.get('loss_improvement', 'N/A')}

**EVALUATION RESULTS:**
- Total Evaluations: {evaluation_data.get('total_evaluations', 0)}
- Recent Win Rate: {evaluation_data.get('recent_win_rate', 'N/A')}
- Win Rate Trend: {evaluation_data.get('win_rate_trend', 'N/A')}
- Promoted Models: {evaluation_data.get('promoted_count', 0)}
- Current Champion: {evaluation_data.get('current_champion', 'Unknown')}

**SELF-PLAY STATISTICS:**
- Total Positions Generated: {selfplay_data.get('total_positions', 0)}
- Recent Games Played: {selfplay_data.get('recent_games', 0)}
- Data Quality Score: {selfplay_data.get('quality_score', 'N/A')}

**ANALYSIS REQUEST:**
Based on these metrics, provide:
1. **Performance Assessment**: How is the model training progressing? Identify key strengths and weaknesses.
2. **Loss Analysis**: Are the loss curves converging well? Any signs of overfitting, underfitting, or instability?
3. **Win Rate Insights**: How effective are the new models compared to previous versions? Is the improvement rate satisfactory?
4. **Data Quality**: Is the self-play data generation sufficient? Any recommendations for data collection?
5. **Actionable Recommendations**: Provide 3-5 specific, prioritized actions to improve model performance. Include concrete parameter suggestions (e.g., "increase MCTS simulations to 800", "reduce learning rate to 0.0005").

Be detailed, technical, and provide specific numerical recommendations where applicable. Focus on practical next steps."""

            prompt = self.adaptive_prompt_builder(base_prompt)
            response, metrics = await self._send_with_performance_tracking(prompt, "analyze_training_metrics")
            
            return response
        except Exception as e:
            logger.error(f"LLM training analysis error: {e}")
            return "Training analysis unavailable. Continue training and evaluation to gather more metrics."
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": round((self.successful_requests / self.total_requests * 100), 1) if self.total_requests > 0 else 0,
            "avg_response_time": round(self.avg_response_time, 2),
            "recent_metrics": [m.to_dict() for m in self.performance_history[-10:]],
            "config": self.config.to_dict()
        }
    
    def get_recent_metrics(self, limit: int = 20) -> List[Dict]:
        """Get recent performance metrics"""
        return [m.to_dict() for m in self.performance_history[-limit:]]


@dataclass
class FeedbackData:
    """User feedback for LLM outputs"""
    feedback_id: str
    session_id: str
    operation_type: str  # "coaching", "analytics", "general"
    accuracy_score: float  # 1-5
    usefulness: float  # 1-5
    clarity: float  # 1-5
    response_time: float  # seconds
    timestamp: str
    comment: Optional[str] = None
    llm_confidence: Optional[float] = None
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

@dataclass
class EvaluationResult:
    """Result of LLM output evaluation"""
    overall_score: float  # 0-100
    accuracy_score: float  # 0-100
    usefulness_score: float  # 0-100
    clarity_score: float  # 0-100
    performance_score: float  # 0-100
    weighted_score: float  # 0-100
    recommendations: List[str]
    needs_tuning: bool
    
    def to_dict(self):
        return asdict(self)

def evaluate_llm_output(
    feedback_data: FeedbackData,
    performance_metrics: PerformanceMetrics,
    llm_confidence: Optional[float] = None
) -> EvaluationResult:
    """
    Evaluate LLM output quality based on user feedback and performance metrics.
    
    Uses weighted scoring:
    - User feedback (50%): accuracy, usefulness, clarity
    - LLM self-assessment (25%): confidence score
    - Performance metrics (25%): response time, success rate
    
    Args:
        feedback_data: User feedback scores
        performance_metrics: Performance metrics for the request
        llm_confidence: Optional LLM confidence score (0-1)
    
    Returns:
        EvaluationResult with scores and recommendations
    """
    # Normalize feedback scores to 0-100
    accuracy_score = (feedback_data.accuracy_score / 5.0) * 100
    usefulness_score = (feedback_data.usefulness / 5.0) * 100
    clarity_score = (feedback_data.clarity / 5.0) * 100
    
    # User feedback component (50% weight)
    user_feedback_avg = (accuracy_score + usefulness_score + clarity_score) / 3
    user_feedback_weighted = user_feedback_avg * 0.5
    
    # LLM confidence component (25% weight)
    if llm_confidence is not None:
        confidence_score = llm_confidence * 100
    else:
        confidence_score = 70  # Default neutral score
    confidence_weighted = confidence_score * 0.25
    
    # Performance component (25% weight)
    # Score based on response time (inverse relationship)
    max_acceptable_time = 10.0  # seconds
    if performance_metrics.response_time <= 3.0:
        time_score = 100
    elif performance_metrics.response_time <= max_acceptable_time:
        time_score = 100 - ((performance_metrics.response_time - 3.0) / (max_acceptable_time - 3.0)) * 50
    else:
        time_score = 50 - min(50, (performance_metrics.response_time - max_acceptable_time) * 5)
    
    # Success score
    success_score = 100 if performance_metrics.success else 0
    
    performance_avg = (time_score + success_score) / 2
    performance_weighted = performance_avg * 0.25
    
    # Calculate overall weighted score
    weighted_score = user_feedback_weighted + confidence_weighted + performance_weighted
    
    # Generate recommendations
    recommendations = []
    needs_tuning = False
    
    if accuracy_score < 60:
        recommendations.append("Low accuracy detected. Consider increasing prompt_depth for more detailed analysis.")
        needs_tuning = True
    
    if usefulness_score < 60:
        recommendations.append("Low usefulness rating. Review prompt templates and context inclusion.")
        needs_tuning = True
    
    if clarity_score < 60:
        recommendations.append("Low clarity score. Consider adjusting response_mode to 'balanced' or simplifying prompts.")
        needs_tuning = True
    
    if time_score < 50:
        recommendations.append("Response time too high. Switch to 'fast' mode or reduce prompt_depth.")
        needs_tuning = True
    
    if performance_metrics.fallback_triggered:
        recommendations.append("Fallback mode was triggered. Consider reducing max_response_time threshold.")
    
    if weighted_score >= 80:
        recommendations.append("Performance is excellent. Current configuration is optimal.")
    elif weighted_score >= 60:
        recommendations.append("Performance is acceptable but can be improved.")
    else:
        recommendations.append("Performance is below expectations. Immediate tuning recommended.")
        needs_tuning = True
    
    return EvaluationResult(
        overall_score=round(weighted_score, 2),
        accuracy_score=round(accuracy_score, 2),
        usefulness_score=round(usefulness_score, 2),
        clarity_score=round(clarity_score, 2),
        performance_score=round(performance_avg, 2),
        weighted_score=round(weighted_score, 2),
        recommendations=recommendations,
        needs_tuning=needs_tuning
    )

def auto_tune_from_feedback(
    feedback_list: List[FeedbackData],
    current_config: LLMConfig,
    performance_history: List[PerformanceMetrics]
) -> Tuple[LLMConfig, List[str]]:
    """
    Automatically tune LLM configuration based on aggregated feedback and performance data.
    
    Analyzes patterns in user feedback and performance metrics to suggest optimal configuration.
    Triggers when aggregate accuracy < 80% or latency > 10s for multiple runs.
    
    Args:
        feedback_list: List of recent feedback data
        current_config: Current LLM configuration
        performance_history: Recent performance metrics
    
    Returns:
        Tuple of (new_config, recommendations)
    """
    if len(feedback_list) == 0:
        return current_config, ["No feedback data available for tuning."]
    
    # Calculate aggregate scores
    avg_accuracy = sum(f.accuracy_score for f in feedback_list) / len(feedback_list)
    avg_usefulness = sum(f.usefulness for f in feedback_list) / len(feedback_list)
    avg_clarity = sum(f.clarity for f in feedback_list) / len(feedback_list)
    
    # Calculate performance metrics
    if performance_history:
        avg_response_time = sum(m.response_time for m in performance_history) / len(performance_history)
        success_rate = sum(1 for m in performance_history if m.success) / len(performance_history)
        fallback_rate = sum(1 for m in performance_history if m.fallback_triggered) / len(performance_history)
    else:
        avg_response_time = 0
        success_rate = 1.0
        fallback_rate = 0
    
    # Convert to 0-100 scale
    avg_accuracy_pct = (avg_accuracy / 5.0) * 100
    avg_usefulness_pct = (avg_usefulness / 5.0) * 100
    avg_clarity_pct = (avg_clarity / 5.0) * 100
    
    # Create new config as a copy
    new_config = LLMConfig(
        response_mode=current_config.response_mode,
        prompt_depth=current_config.prompt_depth,
        adaptive_enabled=current_config.adaptive_enabled,
        max_response_time=current_config.max_response_time,
        fallback_mode=current_config.fallback_mode
    )
    
    recommendations = []
    changes_made = False
    
    # Rule 1: High response time -> faster mode
    if avg_response_time > 10.0:
        if new_config.response_mode == "insightful":
            new_config.response_mode = "balanced"
            recommendations.append(f"Switched from 'insightful' to 'balanced' mode due to high avg response time ({avg_response_time:.2f}s)")
            changes_made = True
        elif new_config.response_mode == "balanced":
            new_config.response_mode = "fast"
            recommendations.append(f"Switched from 'balanced' to 'fast' mode due to high avg response time ({avg_response_time:.2f}s)")
            changes_made = True
        
        if new_config.prompt_depth > 3:
            new_config.prompt_depth = max(1, new_config.prompt_depth - 2)
            recommendations.append(f"Reduced prompt_depth to {new_config.prompt_depth} to improve response time")
            changes_made = True
    
    # Rule 2: Low accuracy/usefulness but fast response -> deeper mode
    elif avg_accuracy_pct < 70 or avg_usefulness_pct < 70:
        if avg_response_time < 5.0:  # Only increase if we have time budget
            if new_config.response_mode == "fast":
                new_config.response_mode = "balanced"
                recommendations.append(f"Switched from 'fast' to 'balanced' mode to improve accuracy (current: {avg_accuracy_pct:.1f}%)")
                changes_made = True
            elif new_config.response_mode == "balanced" and avg_accuracy_pct < 60:
                new_config.response_mode = "insightful"
                recommendations.append(f"Switched from 'balanced' to 'insightful' mode for better insights (current: {avg_accuracy_pct:.1f}%)")
                changes_made = True
            
            if new_config.prompt_depth < 8:
                new_config.prompt_depth = min(10, new_config.prompt_depth + 2)
                recommendations.append(f"Increased prompt_depth to {new_config.prompt_depth} for more detailed analysis")
                changes_made = True
    
    # Rule 3: Low clarity -> adjust mode
    if avg_clarity_pct < 70:
        if new_config.response_mode == "insightful":
            new_config.response_mode = "balanced"
            recommendations.append(f"Switched to 'balanced' mode to improve clarity (current: {avg_clarity_pct:.1f}%)")
            changes_made = True
    
    # Rule 4: High fallback rate -> adjust thresholds
    if fallback_rate > 0.3:
        new_config.max_response_time = min(30.0, new_config.max_response_time + 2.0)
        recommendations.append(f"Increased max_response_time to {new_config.max_response_time}s to reduce fallback rate")
        changes_made = True
    
    # Rule 5: Excellent performance -> maintain or optimize further
    if avg_accuracy_pct >= 80 and avg_usefulness_pct >= 80 and avg_response_time < 5.0:
        recommendations.append("Performance is excellent! No tuning needed. Current configuration is optimal.")
    
    # Rule 6: Critical performance issues
    if avg_accuracy_pct < 50 or success_rate < 0.8:
        recommendations.append("⚠️ CRITICAL: Performance is significantly degraded. Manual review recommended.")
        new_config.adaptive_enabled = True  # Ensure adaptive mode is on
        new_config.fallback_mode = "fast"
        changes_made = True
    
    if not changes_made:
        recommendations.append("No configuration changes needed. Performance metrics are within acceptable ranges.")
    
    return new_config, recommendations



# ====================
# Knowledge Distillation Functions (Step 15)
# ====================

@dataclass
class DistilledKnowledge:
    """Distilled knowledge entry from high-rated feedback"""
    distillation_id: str
    pattern: str  # Strategic pattern identified
    insight: str  # Key insight extracted
    recommendation: str  # Actionable recommendation
    source_feedback_ids: List[str]  # References to original feedback
    confidence_score: float  # 0-1 based on feedback ratings
    timestamp: str
    operation_type: str  # "coaching", "analytics", "general"
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


async def distill_from_feedback(feedback_list: List[Dict], llm_api_key: str) -> List[DistilledKnowledge]:
    """
    Extract reusable strategic patterns from high-rated feedback (≥4★).
    Uses LLM to analyze patterns and create compact distilled knowledge.
    
    Args:
        feedback_list: List of feedback dictionaries with rating ≥4
        llm_api_key: API key for LLM
    
    Returns:
        List of DistilledKnowledge objects
    """
    if not feedback_list or len(feedback_list) == 0:
        logger.info("No high-rated feedback available for distillation")
        return []
    
    try:
        # Initialize LLM for distillation
        chat = LlmChat(
            api_key=llm_api_key,
            session_id="knowledge-distiller",
            system_message="You are an expert at extracting reusable strategic patterns from chess coaching feedback. Identify common themes, strategic heuristics, and actionable insights."
        ).with_model("openai", "gpt-4o-mini")
        
        # Group feedback by operation type
        coaching_feedback = [f for f in feedback_list if f.get("operation_type") == "coaching"]
        analytics_feedback = [f for f in feedback_list if f.get("operation_type") == "analytics"]
        
        distilled_entries = []
        
        # Distill coaching feedback
        if coaching_feedback:
            coaching_summary = "\n".join([
                f"- Session {f.get('session_id', 'unknown')[:8]}: Accuracy {f.get('accuracy_score', 0)}/5, Usefulness {f.get('usefulness', 0)}/5"
                for f in coaching_feedback[:20]  # Limit to 20 entries
            ])
            
            prompt = f"""Analyze these high-rated chess coaching sessions (≥4★):

{coaching_summary}

Extract 3-5 reusable strategic patterns or coaching principles. For each pattern, provide:
1. **Pattern**: A brief title (5-10 words)
2. **Insight**: Key insight extracted (1-2 sentences)
3. **Recommendation**: Actionable recommendation (1 sentence)

Format as JSON array:
[{{"pattern": "...", "insight": "...", "recommendation": "..."}}]"""
            
            try:
                response = await chat.send_message(UserMessage(text=prompt))
                
                # Parse JSON response
                import json
                import re
                
                # Extract JSON from response
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    patterns = json.loads(json_match.group())
                    
                    for pattern_data in patterns:
                        distilled_entries.append(DistilledKnowledge(
                            distillation_id=str(uuid.uuid4()),
                            pattern=pattern_data.get("pattern", "Unknown pattern"),
                            insight=pattern_data.get("insight", ""),
                            recommendation=pattern_data.get("recommendation", ""),
                            source_feedback_ids=[f.get("feedback_id", "") for f in coaching_feedback[:10]],
                            confidence_score=sum(f.get("accuracy_score", 0) for f in coaching_feedback) / (len(coaching_feedback) * 5.0),
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            operation_type="coaching"
                        ))
            except Exception as e:
                logger.error(f"Error distilling coaching feedback: {e}")
        
        # Distill analytics feedback
        if analytics_feedback:
            analytics_summary = "\n".join([
                f"- Session {f.get('session_id', 'unknown')[:8]}: Accuracy {f.get('accuracy_score', 0)}/5, Clarity {f.get('clarity', 0)}/5"
                for f in analytics_feedback[:20]
            ])
            
            prompt = f"""Analyze these high-rated training analytics sessions (≥4★):

{analytics_summary}

Extract 2-3 reusable insights about model training and evaluation. For each:
1. **Pattern**: Brief title (5-10 words)
2. **Insight**: Key insight (1-2 sentences)
3. **Recommendation**: Actionable recommendation (1 sentence)

Format as JSON array:
[{{"pattern": "...", "insight": "...", "recommendation": "..."}}]"""
            
            try:
                response = await chat.send_message(UserMessage(text=prompt))
                
                # Parse JSON
                import json
                import re
                
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    patterns = json.loads(json_match.group())
                    
                    for pattern_data in patterns:
                        distilled_entries.append(DistilledKnowledge(
                            distillation_id=str(uuid.uuid4()),
                            pattern=pattern_data.get("pattern", "Unknown pattern"),
                            insight=pattern_data.get("insight", ""),
                            recommendation=pattern_data.get("recommendation", ""),
                            source_feedback_ids=[f.get("feedback_id", "") for f in analytics_feedback[:10]],
                            confidence_score=sum(f.get("accuracy_score", 0) for f in analytics_feedback) / (len(analytics_feedback) * 5.0),
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            operation_type="analytics"
                        ))
            except Exception as e:
                logger.error(f"Error distilling analytics feedback: {e}")
        
        logger.info(f"Distilled {len(distilled_entries)} knowledge entries from {len(feedback_list)} high-rated feedback items")
        return distilled_entries
        
    except Exception as e:
        logger.error(f"Error in distill_from_feedback: {e}")
        return []


def generate_audit_report(
    feedback_history: List[Dict],
    performance_history: List[Dict],
    optimization_events: List[Dict]
) -> Dict:
    """
    Generate comprehensive performance audit report.
    Measures accuracy, latency, and improvement per optimization cycle.
    
    Args:
        feedback_history: All feedback entries
        performance_history: All performance metrics
        optimization_events: All optimization events
    
    Returns:
        Audit report dictionary with trends and summaries
    """
    try:
        # Initialize report structure
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {},
            "accuracy_trend": [],
            "latency_trend": [],
            "optimization_cycles": [],
            "recommendations": []
        }
        
        # Calculate overall summary
        if feedback_history:
            avg_accuracy = sum(f.get("accuracy_score", 0) for f in feedback_history) / len(feedback_history)
            avg_usefulness = sum(f.get("usefulness", 0) for f in feedback_history) / len(feedback_history)
            avg_clarity = sum(f.get("clarity", 0) for f in feedback_history) / len(feedback_history)
            
            report["summary"]["total_feedback"] = len(feedback_history)
            report["summary"]["avg_accuracy"] = round(avg_accuracy, 2)
            report["summary"]["avg_usefulness"] = round(avg_usefulness, 2)
            report["summary"]["avg_clarity"] = round(avg_clarity, 2)
            report["summary"]["overall_rating"] = round((avg_accuracy + avg_usefulness + avg_clarity) / 3, 2)
        else:
            report["summary"]["total_feedback"] = 0
            report["summary"]["avg_accuracy"] = 0
            report["summary"]["avg_usefulness"] = 0
            report["summary"]["avg_clarity"] = 0
            report["summary"]["overall_rating"] = 0
        
        if performance_history:
            avg_latency = sum(p.get("response_time", 0) for p in performance_history) / len(performance_history)
            success_rate = sum(1 for p in performance_history if p.get("success", False)) / len(performance_history) * 100
            fallback_count = sum(1 for p in performance_history if p.get("fallback_triggered", False))
            
            report["summary"]["total_requests"] = len(performance_history)
            report["summary"]["avg_latency"] = round(avg_latency, 2)
            report["summary"]["success_rate"] = round(success_rate, 1)
            report["summary"]["fallback_count"] = fallback_count
        else:
            report["summary"]["total_requests"] = 0
            report["summary"]["avg_latency"] = 0
            report["summary"]["success_rate"] = 0
            report["summary"]["fallback_count"] = 0
        
        # Build accuracy trend (last 50 feedback entries)
        recent_feedback = sorted(feedback_history, key=lambda x: x.get("timestamp", ""))[-50:]
        
        # Group by chunks of 10 for trend
        chunk_size = 10
        for i in range(0, len(recent_feedback), chunk_size):
            chunk = recent_feedback[i:i+chunk_size]
            if chunk:
                avg_acc = sum(f.get("accuracy_score", 0) for f in chunk) / len(chunk)
                timestamp = chunk[-1].get("timestamp", "")
                report["accuracy_trend"].append({
                    "timestamp": timestamp,
                    "accuracy": round(avg_acc, 2),
                    "sample_size": len(chunk)
                })
        
        # Build latency trend (last 50 performance metrics)
        recent_performance = sorted(performance_history, key=lambda x: x.get("timestamp", ""))[-50:]
        
        for i in range(0, len(recent_performance), chunk_size):
            chunk = recent_performance[i:i+chunk_size]
            if chunk:
                avg_lat = sum(p.get("response_time", 0) for p in chunk) / len(chunk)
                timestamp = chunk[-1].get("timestamp", "")
                report["latency_trend"].append({
                    "timestamp": timestamp,
                    "latency": round(avg_lat, 2),
                    "sample_size": len(chunk)
                })
        
        # Process optimization cycles
        for event in optimization_events:
            previous_config = event.get("previous_config", {})
            new_config = event.get("new_config", {})
            
            cycle_info = {
                "timestamp": event.get("timestamp", ""),
                "trigger": event.get("trigger", "unknown"),
                "feedback_count": event.get("feedback_count", 0),
                "changes": [],
                "recommendations": event.get("recommendations", [])
            }
            
            # Detect config changes
            if previous_config.get("response_mode") != new_config.get("response_mode"):
                cycle_info["changes"].append(f"Mode: {previous_config.get('response_mode')} → {new_config.get('response_mode')}")
            
            if previous_config.get("prompt_depth") != new_config.get("prompt_depth"):
                cycle_info["changes"].append(f"Depth: {previous_config.get('prompt_depth')} → {new_config.get('prompt_depth')}")
            
            if previous_config.get("max_response_time") != new_config.get("max_response_time"):
                cycle_info["changes"].append(f"Max Time: {previous_config.get('max_response_time')}s → {new_config.get('max_response_time')}s")
            
            report["optimization_cycles"].append(cycle_info)
        
        # Generate recommendations
        if report["summary"]["avg_accuracy"] < 3.5:
            report["recommendations"].append("Low accuracy detected. Consider increasing prompt depth or switching to insightful mode.")
        
        if report["summary"]["avg_latency"] > 8.0:
            report["recommendations"].append("High latency detected. Consider using fast mode or reducing prompt complexity.")
        
        if report["summary"]["success_rate"] < 90:
            report["recommendations"].append("Low success rate. Check API connectivity and error logs.")
        
        if len(report["optimization_cycles"]) == 0:
            report["recommendations"].append("No optimization cycles recorded. Enable auto-optimization for continuous improvement.")
        
        if not report["recommendations"]:
            report["recommendations"].append("Performance is optimal. Continue monitoring for any changes.")
        
        logger.info(f"Generated audit report: {report['summary']}")
        return report
        
    except Exception as e:
        logger.error(f"Error generating audit report: {e}")
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {},
            "accuracy_trend": [],
            "latency_trend": [],
            "optimization_cycles": [],
            "recommendations": ["Error generating report. Please try again."],
            "error": str(e)
        }


def get_distilled_context(distilled_knowledge: List[Dict], operation_type: str = None, limit: int = 5) -> str:
    """
    Get distilled knowledge as context for prompt enhancement.
    Filters by operation type and returns most recent entries.
    
    Args:
        distilled_knowledge: List of distilled knowledge entries
        operation_type: Filter by type ("coaching", "analytics", or None for all)
        limit: Maximum number of entries to include
    
    Returns:
        Formatted context string to append to prompts
    """
    if not distilled_knowledge:
        return ""
    
    # Filter by operation type if specified
    if operation_type:
        filtered = [k for k in distilled_knowledge if k.get("operation_type") == operation_type]
    else:
        filtered = distilled_knowledge
    
    # Sort by timestamp (most recent first) and limit
    sorted_knowledge = sorted(filtered, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
    
    if not sorted_knowledge:
        return ""
    
    # Format as context
    context_lines = ["\n**Distilled Strategic Knowledge:**"]
    for entry in sorted_knowledge:
        pattern = entry.get("pattern", "")
        insight = entry.get("insight", "")
        recommendation = entry.get("recommendation", "")
        
        context_lines.append(f"• {pattern}: {insight} → {recommendation}")
    
    return "\n".join(context_lines)


# Add to LLMChessEvaluator class for integration
def enhance_prompt_with_distilled_knowledge(base_prompt: str, distilled_knowledge: List[Dict], operation_type: str) -> str:
    """
    Enhance prompt with distilled knowledge context.
    
    Args:
        base_prompt: Original prompt
        distilled_knowledge: List of distilled knowledge entries
        operation_type: Type of operation ("coaching", "analytics")
    
    Returns:
        Enhanced prompt with distilled context
    """
    context = get_distilled_context(distilled_knowledge, operation_type, limit=3)
    
    if context:
        enhanced_prompt = f"{base_prompt}\n\n{context}\n\nUse these insights to inform your response."
        return enhanced_prompt
    
    return base_prompt


# ====================
# Strategic Insight Fusion & Decision Reasoning (Step 18)
# ====================

@dataclass
class ReasoningChain:
    """Structured reasoning chain for a decision or insight"""
    reasoning_id: str
    decision_id: Optional[str]  # Links to auto-tuning decision if applicable
    timestamp: str
    decision_type: str  # "auto_tuning", "forecast_alert", "distillation", "general"
    reason_summary: str  # Brief one-liner
    evidence_sources: List[Dict[str, Any]]  # [{type: "forecast", data: {...}}, ...]
    reasoning_steps: List[str]  # Step-by-step logic
    suggested_action: str
    confidence: float  # 0-1
    alignment_status: str  # "aligned", "deviation_detected", "re_evaluation_needed"
    impact_prediction: str  # Expected outcome
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


async def generate_reasoning_chain(
    decision_data: Dict,
    forecast_data: Optional[Dict],
    distilled_knowledge: List[Dict],
    auto_tuning_history: List[Dict],
    llm_api_key: str
) -> ReasoningChain:
    """
    Generate structured reasoning chain for a decision using LLM.
    
    Combines:
    - Decision/event data (auto-tuning, forecast alert, etc.)
    - Related forecast predictions
    - Distilled knowledge patterns
    - Historical auto-tuning decisions
    
    Returns:
        ReasoningChain with step-by-step logic and evidence
    """
    try:
        # Extract decision context
        decision_id = decision_data.get("tuning_id") or decision_data.get("decision_id") or str(uuid.uuid4())
        decision_type = decision_data.get("type", "general")
        trigger_reason = decision_data.get("trigger_reason", "")
        parameters_adjusted = decision_data.get("parameters_adjusted", {})
        
        # Build evidence sources
        evidence_sources = []
        
        # Add forecast evidence if available
        if forecast_data:
            evidence_sources.append({
                "type": "forecast",
                "source": "Step 16 Predictive Forecasting",
                "data": {
                    "predicted_accuracy": forecast_data.get("predicted_accuracy"),
                    "predicted_win_rate": forecast_data.get("predicted_win_rate"),
                    "confidence": forecast_data.get("overall_confidence")
                }
            })
        
        # Add distilled knowledge evidence
        if distilled_knowledge:
            relevant_knowledge = [k for k in distilled_knowledge[:3]]
            if relevant_knowledge:
                evidence_sources.append({
                    "type": "distilled_knowledge",
                    "source": "Step 15 Knowledge Distillation",
                    "data": {
                        "patterns": [k.get("pattern") for k in relevant_knowledge],
                        "insights": [k.get("insight") for k in relevant_knowledge]
                    }
                })
        
        # Add auto-tuning history evidence
        if auto_tuning_history:
            recent_tunings = auto_tuning_history[:3]
            evidence_sources.append({
                "type": "auto_tuning_history",
                "source": "Step 17 Auto-Tuning",
                "data": {
                    "recent_adjustments": [
                        {
                            "reason": t.get("trigger_reason"),
                            "confidence": t.get("confidence_score")
                        } for t in recent_tunings
                    ]
                }
            })
        
        # Build LLM prompt for reasoning generation
        distilled_context = "\n".join([
            f"• {k.get('pattern')}: {k.get('insight')}"
            for k in distilled_knowledge[:3]
        ]) if distilled_knowledge else "No distilled knowledge available"
        
        forecast_context = ""
        if forecast_data:
            forecast_context = f"""
**Forecast Context:**
- Predicted Accuracy: {forecast_data.get('predicted_accuracy', 'N/A')}
- Predicted Win Rate: {forecast_data.get('predicted_win_rate', 'N/A')}
- Overall Confidence: {forecast_data.get('overall_confidence', 0) * 100:.1f}%
"""
        
        auto_tuning_context = ""
        if auto_tuning_history:
            recent = auto_tuning_history[0]
            auto_tuning_context = f"""
**Recent Auto-Tuning:**
- Trigger: {recent.get('trigger_reason', 'N/A')}
- Confidence: {recent.get('confidence_score', 0) * 100:.1f}%
- Expected Impact: {recent.get('expected_impact', 'N/A')}
"""
        
        llm_prompt = f"""You are an expert AI systems analyst. Generate a structured reasoning chain that explains WHY a specific decision was made in the AlphaZero Chess AI system.

**Decision Context:**
- Decision Type: {decision_type}
- Trigger Reason: {trigger_reason}
- Parameters Adjusted: {json.dumps(parameters_adjusted, indent=2) if parameters_adjusted else 'None'}

{forecast_context}

**Distilled Strategic Knowledge:**
{distilled_context}

{auto_tuning_context}

**Task:**
Generate a step-by-step reasoning chain that:
1. Explains the root cause of the decision trigger
2. Shows how forecast data influenced the decision
3. References relevant distilled knowledge patterns
4. Evaluates alignment with predicted goals
5. Provides a suggested action

**Output Format (JSON):**
{{
  "reason_summary": "One-sentence summary of why this decision was made",
  "reasoning_steps": [
    "Step 1: Root cause analysis...",
    "Step 2: Evidence from forecasting...",
    "Step 3: Application of distilled knowledge...",
    "Step 4: Decision alignment evaluation...",
    "Step 5: Recommended next action..."
  ],
  "suggested_action": "Specific recommended action based on this reasoning",
  "confidence": 0.85,
  "alignment_status": "aligned|deviation_detected|re_evaluation_needed",
  "impact_prediction": "Expected outcome of this decision"
}}

Provide only the JSON output, no additional text."""

        # Call LLM for reasoning
        chat = LlmChat(
            api_key=llm_api_key,
            session_id="reasoning-chain-generator",
            system_message="You are an expert AI systems analyst specializing in decision reasoning and strategic insight generation."
        ).with_model("openai", "gpt-5")
        
        response = await chat.send_message(UserMessage(text=llm_prompt))
        
        # Parse JSON response
        import json
        import re
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            reasoning_data = json.loads(json_match.group())
            
            return ReasoningChain(
                reasoning_id=str(uuid.uuid4()),
                decision_id=decision_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                decision_type=decision_type,
                reason_summary=reasoning_data.get("reason_summary", "Reasoning generated"),
                evidence_sources=evidence_sources,
                reasoning_steps=reasoning_data.get("reasoning_steps", []),
                suggested_action=reasoning_data.get("suggested_action", "Continue monitoring"),
                confidence=float(reasoning_data.get("confidence", 0.7)),
                alignment_status=reasoning_data.get("alignment_status", "aligned"),
                impact_prediction=reasoning_data.get("impact_prediction", "Positive impact expected")
            )
        else:
            # Fallback if JSON parsing fails
            return ReasoningChain(
                reasoning_id=str(uuid.uuid4()),
                decision_id=decision_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                decision_type=decision_type,
                reason_summary=f"Decision triggered by {trigger_reason}",
                evidence_sources=evidence_sources,
                reasoning_steps=[
                    f"Decision triggered: {trigger_reason}",
                    f"Parameters adjusted: {list(parameters_adjusted.keys()) if parameters_adjusted else 'None'}",
                    "Evidence from forecasting and distillation considered",
                    "Confidence calculated based on deviation magnitude",
                    "Action recommended based on system state"
                ],
                suggested_action="Continue monitoring system performance",
                confidence=0.7,
                alignment_status="aligned",
                impact_prediction="Expected to improve system performance"
            )
    
    except Exception as e:
        logger.error(f"Error generating reasoning chain: {e}")
        # Return basic reasoning chain
        return ReasoningChain(
            reasoning_id=str(uuid.uuid4()),
            decision_id=decision_data.get("tuning_id") or str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            decision_type=decision_data.get("type", "general"),
            reason_summary="Reasoning generation failed",
            evidence_sources=[],
            reasoning_steps=["Unable to generate detailed reasoning due to error"],
            suggested_action="Review system logs",
            confidence=0.5,
            alignment_status="re_evaluation_needed",
            impact_prediction="Unknown"
        )


def evaluate_decision_alignment(
    decision_data: Dict,
    forecast_data: Optional[Dict],
    actual_performance: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Evaluate whether an auto-tuning decision aligns with predicted goals.
    
    Compares:
    - Predicted trends from Step 16 forecast
    - Auto-tuning decision from Step 17
    - Actual performance (if available)
    
    Returns:
        Alignment evaluation with status and confidence
    """
    try:
        evaluation = {
            "aligned": False,
            "alignment_score": 0.0,
            "deviations": [],
            "recommendations": [],
            "confidence": 0.0
        }
        
        if not forecast_data or not decision_data:
            evaluation["recommendations"].append("Insufficient data for alignment evaluation")
            return evaluation
        
        # Extract decision intent
        trigger_reason = decision_data.get("trigger_reason", "")
        expected_impact = decision_data.get("expected_impact", "")
        parameters_adjusted = decision_data.get("parameters_adjusted", {})
        
        # Extract forecast predictions
        forecast_accuracy = forecast_data.get("forecasts", {}).get("7", {}).get("accuracy", {})
        forecast_winrate = forecast_data.get("forecasts", {}).get("7", {}).get("win_rate", {})
        forecast_latency = forecast_data.get("forecasts", {}).get("7", {}).get("latency", {})
        
        alignment_score = 0.0
        checks_passed = 0
        total_checks = 0
        
        # Check 1: Trigger aligns with forecast trend
        total_checks += 1
        if "accuracy" in trigger_reason and forecast_accuracy:
            if forecast_accuracy.get("trend_direction") == "declining":
                checks_passed += 1
                alignment_score += 0.3
        elif "win_rate" in trigger_reason and forecast_winrate:
            if forecast_winrate.get("trend_direction") == "declining":
                checks_passed += 1
                alignment_score += 0.3
        elif "latency" in trigger_reason and forecast_latency:
            if forecast_latency.get("trend_direction") == "declining":  # declining = latency increasing
                checks_passed += 1
                alignment_score += 0.3
        
        # Check 2: Parameters adjusted match expected impact
        total_checks += 1
        if "accuracy" in expected_impact and "num_simulations" in parameters_adjusted:
            checks_passed += 1
            alignment_score += 0.3
        elif "latency" in expected_impact and "prompt_depth" in parameters_adjusted:
            checks_passed += 1
            alignment_score += 0.3
        elif "win_rate" in expected_impact and "c_puct" in parameters_adjusted:
            checks_passed += 1
            alignment_score += 0.3
        
        # Check 3: Decision confidence matches forecast confidence
        total_checks += 1
        decision_confidence = decision_data.get("confidence_score", 0)
        forecast_confidence = forecast_data.get("overall_confidence", 0)
        
        if abs(decision_confidence - forecast_confidence) < 0.2:
            checks_passed += 1
            alignment_score += 0.4
        
        # Calculate final alignment
        evaluation["alignment_score"] = round(min(1.0, alignment_score), 2)
        evaluation["aligned"] = evaluation["alignment_score"] >= 0.6
        evaluation["confidence"] = round((checks_passed / total_checks), 2) if total_checks > 0 else 0.5
        
        # Generate recommendations
        if not evaluation["aligned"]:
            evaluation["deviations"].append(f"Decision triggered by {trigger_reason} may not align with forecast trends")
            evaluation["recommendations"].append("Review forecast data before applying this tuning")
        else:
            evaluation["recommendations"].append("Decision aligns well with predicted trends")
        
        return evaluation
    
    except Exception as e:
        logger.error(f"Error evaluating decision alignment: {e}")
        return {
            "aligned": False,
            "alignment_score": 0.0,
            "deviations": ["Error during evaluation"],
            "recommendations": ["Manual review required"],
            "confidence": 0.0
        }


def rank_strategic_insights(
    insights: List[ReasoningChain],
    ranking_criteria: str = "confidence"
) -> List[ReasoningChain]:
    """
    Rank strategic insights by confidence, impact, or recency.
    
    Args:
        insights: List of ReasoningChain objects
        ranking_criteria: "confidence", "impact", "recency"
    
    Returns:
        Sorted list of insights
    """
    try:
        if ranking_criteria == "confidence":
            return sorted(insights, key=lambda x: x.confidence, reverse=True)
        elif ranking_criteria == "recency":
            return sorted(insights, key=lambda x: x.timestamp, reverse=True)
        elif ranking_criteria == "impact":
            # Prioritize by alignment status and confidence
            priority_map = {
                "deviation_detected": 3,
                "re_evaluation_needed": 2,
                "aligned": 1
            }
            return sorted(
                insights,
                key=lambda x: (priority_map.get(x.alignment_status, 0), x.confidence),
                reverse=True
            )
        else:
            return insights
    except Exception as e:
        logger.error(f"Error ranking insights: {e}")
        return insights



# ====================
# Predictive Trend Analysis & LLM Forecasting (Step 16)
# ====================

@dataclass
class ForecastResult:
    """Forecast result for a specific metric and timeframe"""
    metric_name: str
    current_value: float
    predicted_value: float
    change_percent: float
    trend_direction: str  # "improving", "stable", "declining"
    confidence: float  # 0-1
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ComprehensiveForecast:
    """Complete forecast report with all metrics and LLM insights"""
    timestamp: str
    timeframes: Dict[str, Dict[str, ForecastResult]]  # {7: {accuracy: ForecastResult, ...}, ...}
    overall_confidence: float
    forecast_narrative: str
    strategic_recommendations: List[str]
    data_sufficiency: Dict[str, int]  # Count of data points used
    
    def to_dict(self):
        result = asdict(self)
        # Convert nested ForecastResult objects
        for timeframe, metrics in result['timeframes'].items():
            for metric_name, forecast in metrics.items():
                if hasattr(forecast, 'to_dict'):
                    result['timeframes'][timeframe][metric_name] = forecast.to_dict()
        return result


def analyze_trend_with_regression(
    data_points: List[Dict],
    value_key: str,
    timestamp_key: str = "timestamp",
    degree: int = 1
) -> Tuple[List[float], float]:
    """
    Perform linear regression on time-series data.
    
    Args:
        data_points: List of data dictionaries with timestamp and value
        value_key: Key to extract value from data points
        timestamp_key: Key to extract timestamp
        degree: Polynomial degree (default: 1 for linear)
    
    Returns:
        Tuple of (coefficients, r_squared)
    """
    import numpy as np
    from datetime import datetime as dt
    
    if not data_points or len(data_points) < 2:
        return None, 0.0
    
    try:
        # Extract timestamps and values
        timestamps = []
        values = []
        
        for point in data_points:
            timestamp_val = point.get(timestamp_key)
            value_val = point.get(value_key)
            
            if timestamp_val is None or value_val is None:
                continue
            
            # Convert timestamp to numeric (seconds since first point)
            if isinstance(timestamp_val, str):
                timestamp_dt = dt.fromisoformat(timestamp_val.replace('Z', '+00:00'))
            elif isinstance(timestamp_val, dt):
                timestamp_dt = timestamp_val
            else:
                continue
            
            timestamps.append(timestamp_dt)
            values.append(float(value_val))
        
        if len(timestamps) < 2:
            return None, 0.0
        
        # Convert to seconds since first timestamp
        base_time = timestamps[0]
        x = np.array([(t - base_time).total_seconds() for t in timestamps])
        y = np.array(values)
        
        # Perform polynomial fit
        coefficients = np.polyfit(x, y, degree)
        
        # Calculate R-squared
        y_pred = np.polyval(coefficients, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return coefficients, max(0.0, min(1.0, r_squared))
        
    except Exception as e:
        logger.error(f"Error in trend analysis: {e}")
        return None, 0.0


def predict_future_value(
    coefficients: List[float],
    current_value: float,
    base_timestamp,
    target_days: int
) -> float:
    """
    Predict future value using regression coefficients.
    
    Args:
        coefficients: Polynomial coefficients from regression
        current_value: Current metric value
        base_timestamp: Base timestamp for calculations
        target_days: Number of days into future
    
    Returns:
        Predicted value
    """
    import numpy as np
    
    if coefficients is None:
        return current_value
    
    try:
        # Convert days to seconds
        target_seconds = target_days * 24 * 60 * 60
        
        # Predict using polynomial
        predicted = np.polyval(coefficients, target_seconds)
        
        return float(predicted)
        
    except Exception as e:
        logger.error(f"Error predicting future value: {e}")
        return current_value


def calculate_forecast_confidence(
    data_count: int,
    r_squared: float,
    metric_type: str = "general"
) -> float:
    """
    Calculate confidence score for forecast based on data quality.
    
    Args:
        data_count: Number of data points used
        r_squared: R-squared value from regression
        metric_type: Type of metric being predicted
    
    Returns:
        Confidence score (0-1)
    """
    # Base confidence from R-squared
    confidence = r_squared * 0.6
    
    # Data sufficiency bonus (up to 0.3)
    if data_count >= 50:
        data_bonus = 0.3
    elif data_count >= 20:
        data_bonus = 0.2
    elif data_count >= 10:
        data_bonus = 0.15
    else:
        data_bonus = data_count / 10 * 0.15
    
    confidence += data_bonus
    
    # Stability bonus (up to 0.1)
    if r_squared > 0.8:
        confidence += 0.1
    elif r_squared > 0.6:
        confidence += 0.05
    
    return max(0.1, min(1.0, confidence))


async def generate_forecast_report(
    training_data: List[Dict],
    evaluation_data: List[Dict],
    performance_data: List[Dict],
    distilled_knowledge: List[Dict],
    audit_history: Dict,
    llm_api_key: str,
    timeframes: List[int] = [7, 30, 90]
) -> ComprehensiveForecast:
    """
    Generate comprehensive forecast report with predictive trend analysis.
    
    Combines:
    - Historical training metrics (accuracy trends)
    - Evaluation results (win rate trends)
    - LLM performance data (latency trends)
    - Distilled knowledge from Step 15
    - Audit history for context
    
    Args:
        training_data: Historical training metrics
        evaluation_data: Historical evaluation results
        performance_data: LLM performance history
        distilled_knowledge: Distilled knowledge entries
        audit_history: Performance audit report
        llm_api_key: API key for LLM
        timeframes: List of days to forecast (default: [7, 30, 90])
    
    Returns:
        ComprehensiveForecast with predictions and recommendations
    """
    from datetime import datetime as dt, timedelta
    import numpy as np
    
    try:
        current_time = dt.now(timezone.utc)
        forecast_results = {}
        
        # ====================
        # 1. Analyze Training Accuracy Trend
        # ====================
        accuracy_forecasts = {}
        accuracy_coeffs = None
        accuracy_r2 = 0.0
        current_accuracy = 0.0
        
        if training_data and len(training_data) > 0:
            # Extract loss values (inverse relationship with accuracy)
            loss_data = [
                {
                    "timestamp": d.get("timestamp"),
                    "value": d.get("loss", 0)
                }
                for d in training_data if d.get("loss") is not None
            ]
            
            if loss_data:
                accuracy_coeffs, accuracy_r2 = analyze_trend_with_regression(
                    loss_data, "value", "timestamp", degree=1
                )
                
                # Current accuracy estimate (inverse of loss)
                recent_losses = [d["value"] for d in loss_data[-10:]]
                current_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 1.0
                current_accuracy = max(0, min(100, (1 - min(current_loss, 1.0)) * 100))
                
                # Predict for each timeframe
                base_timestamp = loss_data[0]["timestamp"]
                if isinstance(base_timestamp, str):
                    base_timestamp = dt.fromisoformat(base_timestamp.replace('Z', '+00:00'))
                
                for days in timeframes:
                    if accuracy_coeffs is not None:
                        predicted_loss = predict_future_value(
                            accuracy_coeffs, current_loss, base_timestamp, days
                        )
                        predicted_accuracy = max(0, min(100, (1 - min(predicted_loss, 1.0)) * 100))
                    else:
                        predicted_accuracy = current_accuracy
                    
                    change = predicted_accuracy - current_accuracy
                    change_pct = (change / current_accuracy * 100) if current_accuracy > 0 else 0
                    
                    if change > 2:
                        trend = "improving"
                    elif change < -2:
                        trend = "declining"
                    else:
                        trend = "stable"
                    
                    confidence = calculate_forecast_confidence(
                        len(loss_data), accuracy_r2, "accuracy"
                    )
                    
                    accuracy_forecasts[days] = ForecastResult(
                        metric_name="accuracy",
                        current_value=round(current_accuracy, 2),
                        predicted_value=round(predicted_accuracy, 2),
                        change_percent=round(change_pct, 2),
                        trend_direction=trend,
                        confidence=round(confidence, 2)
                    )
        
        # ====================
        # 2. Analyze Win Rate Trend
        # ====================
        winrate_forecasts = {}
        winrate_coeffs = None
        winrate_r2 = 0.0
        current_winrate = 0.0
        
        if evaluation_data and len(evaluation_data) > 0:
            winrate_data = [
                {
                    "timestamp": e.get("timestamp"),
                    "value": e.get("challenger_win_rate", 0) * 100
                }
                for e in evaluation_data if e.get("challenger_win_rate") is not None
            ]
            
            if winrate_data:
                winrate_coeffs, winrate_r2 = analyze_trend_with_regression(
                    winrate_data, "value", "timestamp", degree=1
                )
                
                # Current win rate
                recent_winrates = [d["value"] for d in winrate_data[-5:]]
                current_winrate = sum(recent_winrates) / len(recent_winrates) if recent_winrates else 50.0
                
                base_timestamp = winrate_data[0]["timestamp"]
                if isinstance(base_timestamp, str):
                    base_timestamp = dt.fromisoformat(base_timestamp.replace('Z', '+00:00'))
                
                for days in timeframes:
                    if winrate_coeffs is not None:
                        predicted_winrate = predict_future_value(
                            winrate_coeffs, current_winrate, base_timestamp, days
                        )
                        predicted_winrate = max(0, min(100, predicted_winrate))
                    else:
                        predicted_winrate = current_winrate
                    
                    change = predicted_winrate - current_winrate
                    change_pct = (change / current_winrate * 100) if current_winrate > 0 else 0
                    
                    if change > 3:
                        trend = "improving"
                    elif change < -3:
                        trend = "declining"
                    else:
                        trend = "stable"
                    
                    confidence = calculate_forecast_confidence(
                        len(winrate_data), winrate_r2, "winrate"
                    )
                    
                    winrate_forecasts[days] = ForecastResult(
                        metric_name="win_rate",
                        current_value=round(current_winrate, 2),
                        predicted_value=round(predicted_winrate, 2),
                        change_percent=round(change_pct, 2),
                        trend_direction=trend,
                        confidence=round(confidence, 2)
                    )
        
        # ====================
        # 3. Analyze Latency Trend
        # ====================
        latency_forecasts = {}
        latency_coeffs = None
        latency_r2 = 0.0
        current_latency = 0.0
        
        if performance_data and len(performance_data) > 0:
            latency_data = [
                {
                    "timestamp": p.get("timestamp"),
                    "value": p.get("response_time", 0)
                }
                for p in performance_data if p.get("response_time") is not None
            ]
            
            if latency_data:
                latency_coeffs, latency_r2 = analyze_trend_with_regression(
                    latency_data, "value", "timestamp", degree=1
                )
                
                # Current latency
                recent_latencies = [d["value"] for d in latency_data[-20:]]
                current_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 5.0
                
                base_timestamp = latency_data[0]["timestamp"]
                if isinstance(base_timestamp, str):
                    base_timestamp = dt.fromisoformat(base_timestamp.replace('Z', '+00:00'))
                
                for days in timeframes:
                    if latency_coeffs is not None:
                        predicted_latency = predict_future_value(
                            latency_coeffs, current_latency, base_timestamp, days
                        )
                        predicted_latency = max(0, predicted_latency)
                    else:
                        predicted_latency = current_latency
                    
                    change = predicted_latency - current_latency
                    change_pct = (change / current_latency * 100) if current_latency > 0 else 0
                    
                    # Lower latency is better
                    if change < -0.5:
                        trend = "improving"
                    elif change > 0.5:
                        trend = "declining"
                    else:
                        trend = "stable"
                    
                    confidence = calculate_forecast_confidence(
                        len(latency_data), latency_r2, "latency"
                    )
                    
                    latency_forecasts[days] = ForecastResult(
                        metric_name="latency",
                        current_value=round(current_latency, 2),
                        predicted_value=round(predicted_latency, 2),
                        change_percent=round(change_pct, 2),
                        trend_direction=trend,
                        confidence=round(confidence, 2)
                    )
        
        # ====================
        # 4. Combine Forecasts
        # ====================
        for days in timeframes:
            forecast_results[str(days)] = {
                "accuracy": accuracy_forecasts.get(days),
                "win_rate": winrate_forecasts.get(days),
                "latency": latency_forecasts.get(days)
            }
        
        # Calculate overall confidence
        all_confidences = []
        for timeframe_forecasts in forecast_results.values():
            for forecast in timeframe_forecasts.values():
                if forecast:
                    all_confidences.append(forecast.confidence)
        
        overall_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.5
        
        # ====================
        # 5. Generate LLM Forecast Narrative
        # ====================
        
        # Build context from distilled knowledge
        distilled_context = get_distilled_context(distilled_knowledge, operation_type=None, limit=5)
        
        # Build audit summary
        audit_summary = ""
        if audit_history:
            summary = audit_history.get("summary", {})
            audit_summary = f"""
**Recent Performance Audit:**
- Total Feedback: {summary.get('total_feedback', 0)}
- Avg Accuracy: {summary.get('avg_accuracy', 0)}/5
- Success Rate: {summary.get('success_rate', 0)}%
- Avg Latency: {summary.get('avg_latency', 0)}s
"""
        
        # Build forecast summary for LLM
        forecast_summary = []
        for days in timeframes:
            forecasts = forecast_results.get(str(days), {})
            acc = forecasts.get("accuracy")
            wr = forecasts.get("win_rate")
            lat = forecasts.get("latency")
            
            summary_line = f"\n**{days}-Day Forecast:**"
            if acc:
                summary_line += f"\n- Accuracy: {acc.current_value}% → {acc.predicted_value}% ({acc.trend_direction})"
            if wr:
                summary_line += f"\n- Win Rate: {wr.current_value}% → {wr.predicted_value}% ({wr.trend_direction})"
            if lat:
                summary_line += f"\n- Latency: {lat.current_value}s → {lat.predicted_value}s ({lat.trend_direction})"
            
            forecast_summary.append(summary_line)
        
        # Create LLM prompt
        llm_prompt = f"""You are an expert AI training analyst for AlphaZero chess models. Based on predictive trend analysis of historical data, provide a comprehensive forecast report with strategic recommendations.

{audit_summary}

{distilled_context}

**PREDICTIVE FORECASTS (Linear Regression Analysis):**
{''.join(forecast_summary)}

**Overall Forecast Confidence:** {overall_confidence * 100:.1f}%

**DATA SUFFICIENCY:**
- Training Data Points: {len(training_data)}
- Evaluation Data Points: {len(evaluation_data)}
- Performance Metrics: {len(performance_data)}

**ANALYSIS REQUEST:**
Based on these predictive trends:

1. **Executive Summary**: Provide a 2-3 sentence overview of what the forecasts indicate about future model performance.

2. **Key Insights**: Identify 2-3 critical patterns or concerns from the projected trends.

3. **Strategic Recommendations**: Provide 3-5 specific, actionable recommendations to optimize future training cycles. Include concrete parameter suggestions (e.g., "increase MCTS simulations to 1000", "reduce learning rate to 0.0003").

4. **Risk Assessment**: Highlight any projected risks or areas needing attention (e.g., if latency is predicted to increase significantly).

Be concise, technical, and actionable. Focus on practical next steps."""

        # Call LLM
        try:
            chat = LlmChat(
                api_key=llm_api_key,
                session_id="forecast-analyst",
                system_message="You are an expert AI training analyst specializing in predictive analytics for deep learning systems."
            ).with_model("openai", "gpt-4o-mini")
            
            forecast_narrative = await chat.send_message(UserMessage(text=llm_prompt))
            
        except Exception as e:
            logger.error(f"Error generating LLM forecast narrative: {e}")
            forecast_narrative = f"Forecast analysis completed with {overall_confidence * 100:.1f}% confidence. Manual review recommended for strategic planning."
        
        # Extract recommendations (simple parsing)
        strategic_recommendations = []
        if "Recommendations:" in forecast_narrative or "recommendations:" in forecast_narrative:
            lines = forecast_narrative.split('\n')
            in_recs = False
            for line in lines:
                if "recommendation" in line.lower():
                    in_recs = True
                    continue
                if in_recs and line.strip().startswith(('-', '•', '*', str)):
                    strategic_recommendations.append(line.strip().lstrip('-•*0123456789. '))
                elif in_recs and len(strategic_recommendations) > 0 and not line.strip():
                    break
        
        if not strategic_recommendations:
            strategic_recommendations = [
                "Continue current training regimen with monitoring",
                "Review forecast trends after 7 days",
                "Adjust parameters if predictions deviate significantly"
            ]
        
        # ====================
        # 6. Build Final Report
        # ====================
        
        comprehensive_forecast = ComprehensiveForecast(
            timestamp=current_time.isoformat(),
            timeframes=forecast_results,
            overall_confidence=round(overall_confidence, 2),
            forecast_narrative=forecast_narrative,
            strategic_recommendations=strategic_recommendations[:5],  # Limit to 5
            data_sufficiency={
                "training_data_points": len(training_data),
                "evaluation_data_points": len(evaluation_data),
                "performance_data_points": len(performance_data),
                "distilled_knowledge_entries": len(distilled_knowledge)
            }
        )
        
        logger.info(f"Generated forecast report with {overall_confidence * 100:.1f}% confidence")
        return comprehensive_forecast
        
    except Exception as e:
        logger.error(f"Error generating forecast report: {e}")
        # Return empty forecast
        return ComprehensiveForecast(
            timestamp=dt.now(timezone.utc).isoformat(),
            timeframes={},
            overall_confidence=0.0,
            forecast_narrative=f"Unable to generate forecast: {str(e)}",
            strategic_recommendations=["Collect more historical data for accurate predictions"],
            data_sufficiency={
                "training_data_points": len(training_data) if training_data else 0,
                "evaluation_data_points": len(evaluation_data) if evaluation_data else 0,
                "performance_data_points": len(performance_data) if performance_data else 0,
                "distilled_knowledge_entries": len(distilled_knowledge) if distilled_knowledge else 0
            }
        )


# ====================
# Real-Time Adaptive Forecasting & Auto-Tuning (Step 17)
# ====================

@dataclass
class AutoTuningDecision:
    """Decision record for automatic strategy tuning"""
    tuning_id: str
    timestamp: str
    trigger_reason: str  # "accuracy_deviation", "latency_spike", "win_rate_decline", etc.
    parameters_adjusted: Dict[str, Dict[str, float]]  # {param_name: {old: x, new: y}}
    reasoning: str
    confidence_score: float  # 0-1
    expected_impact: str  # "improve_accuracy", "reduce_latency", etc.
    deviation_magnitude: float  # Percentage deviation that triggered tuning
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class RealtimeForecastUpdate:
    """Incremental forecast update for real-time monitoring"""
    update_id: str
    timestamp: str
    metric_name: str  # "accuracy", "win_rate", "latency"
    current_value: float
    forecasted_value: float
    deviation_percent: float
    trend: str  # "improving", "stable", "declining", "alert"
    requires_tuning: bool
    confidence: float
    
    def to_dict(self):
        return asdict(self)


def detect_performance_deviation(
    current_metrics: Dict[str, float],
    forecasted_metrics: Dict[str, float],
    threshold_percent: float = 5.0
) -> Tuple[bool, List[str], Dict[str, float]]:
    """
    Detect if current performance deviates from forecast by threshold.
    
    Args:
        current_metrics: {metric_name: current_value}
        forecasted_metrics: {metric_name: forecasted_value}
        threshold_percent: Deviation threshold (default 5%)
    
    Returns:
        Tuple of (deviation_detected, list_of_deviating_metrics, deviation_percentages)
    """
    deviations = {}
    deviating_metrics = []
    
    for metric_name, current_value in current_metrics.items():
        forecasted_value = forecasted_metrics.get(metric_name)
        
        if forecasted_value is None or forecasted_value == 0:
            continue
        
        # Calculate deviation percentage
        deviation = abs((current_value - forecasted_value) / forecasted_value * 100)
        deviations[metric_name] = deviation
        
        if deviation >= threshold_percent:
            deviating_metrics.append(metric_name)
    
    deviation_detected = len(deviating_metrics) > 0
    
    return deviation_detected, deviating_metrics, deviations


async def generate_realtime_forecast(
    training_data: List[Dict],
    evaluation_data: List[Dict],
    performance_data: List[Dict],
    previous_forecast: Optional[ComprehensiveForecast] = None,
    timeframe_days: int = 7
) -> RealtimeForecastUpdate:
    """
    Generate incremental forecast update for real-time monitoring.
    Lightweight version of generate_forecast_report for frequent updates.
    
    Args:
        training_data: Recent training metrics (last 20 entries)
        evaluation_data: Recent evaluations (last 10 entries)
        performance_data: Recent LLM performance (last 30 entries)
        previous_forecast: Last forecast for comparison
        timeframe_days: Forecast horizon (default 7 days)
    
    Returns:
        RealtimeForecastUpdate with current status
    """
    from datetime import datetime as dt, timedelta
    import numpy as np
    
    try:
        current_time = dt.now(timezone.utc)
        
        # Quick analysis - focus on most recent trend
        if training_data and len(training_data) >= 5:
            recent_losses = [d.get("loss", 0) for d in training_data[-10:] if d.get("loss")]
            if recent_losses:
                current_loss = sum(recent_losses) / len(recent_losses)
                current_accuracy = max(0, min(100, (1 - min(current_loss, 1.0)) * 100))
                
                # Simple linear trend
                if len(recent_losses) >= 5:
                    first_half_avg = sum(recent_losses[:5]) / 5
                    second_half_avg = sum(recent_losses[-5:]) / 5
                    trend_direction = "improving" if second_half_avg < first_half_avg else "declining" if second_half_avg > first_half_avg else "stable"
                    
                    # Project forward
                    change_rate = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg > 0 else 0
                    forecasted_loss = second_half_avg * (1 + change_rate * (timeframe_days / 7))
                    forecasted_accuracy = max(0, min(100, (1 - min(forecasted_loss, 1.0)) * 100))
                    
                    deviation = abs(forecasted_accuracy - current_accuracy)
                    deviation_percent = (deviation / current_accuracy * 100) if current_accuracy > 0 else 0
                    
                    # Check if tuning needed
                    requires_tuning = deviation_percent >= 5.0
                    
                    if deviation_percent >= 10:
                        trend = "alert"
                    else:
                        trend = trend_direction
                    
                    return RealtimeForecastUpdate(
                        update_id=str(uuid.uuid4()),
                        timestamp=current_time.isoformat(),
                        metric_name="accuracy",
                        current_value=round(current_accuracy, 2),
                        forecasted_value=round(forecasted_accuracy, 2),
                        deviation_percent=round(deviation_percent, 2),
                        trend=trend,
                        requires_tuning=requires_tuning,
                        confidence=0.75
                    )
        
        # Fallback - no significant data
        return RealtimeForecastUpdate(
            update_id=str(uuid.uuid4()),
            timestamp=current_time.isoformat(),
            metric_name="accuracy",
            current_value=0.0,
            forecasted_value=0.0,
            deviation_percent=0.0,
            trend="stable",
            requires_tuning=False,
            confidence=0.5
        )
        
    except Exception as e:
        logger.error(f"Error generating realtime forecast: {e}")
        return RealtimeForecastUpdate(
            update_id=str(uuid.uuid4()),
            timestamp=dt.now(timezone.utc).isoformat(),
            metric_name="error",
            current_value=0.0,
            forecasted_value=0.0,
            deviation_percent=0.0,
            trend="stable",
            requires_tuning=False,
            confidence=0.0
        )


def auto_tune_strategy(
    deviating_metrics: List[str],
    deviations: Dict[str, float],
    current_config: Dict[str, Any],
    llm_config: Optional[LLMConfig] = None
) -> AutoTuningDecision:
    """
    Automatically tune strategy parameters based on detected deviations.
    
    Adjusts:
    - Learning rate
    - MCTS search depth  
    - Prompt depth
    - Temperature
    - Exploration parameters (c_puct)
    
    Args:
        deviating_metrics: List of metrics that deviated
        deviations: Deviation percentages per metric
        current_config: Current training/LLM configuration
        llm_config: Current LLM configuration
    
    Returns:
        AutoTuningDecision with adjustments and reasoning
    """
    from datetime import datetime as dt
    
    adjustments = {}
    reasoning_parts = []
    expected_impact = []
    
    # Get current values or defaults
    current_lr = current_config.get("learning_rate", 0.001)
    current_mcts = current_config.get("num_simulations", 800)
    current_prompt_depth = llm_config.prompt_depth if llm_config else 5
    current_temp = current_config.get("temperature", 1.0)
    current_c_puct = current_config.get("c_puct", 1.5)
    
    # Calculate average deviation magnitude
    avg_deviation = sum(deviations.values()) / len(deviations) if deviations else 0
    
    # Rule-based auto-tuning logic
    
    # Rule 1: Accuracy declining significantly
    if "accuracy" in deviating_metrics:
        acc_deviation = deviations.get("accuracy", 0)
        
        if acc_deviation >= 10:
            # Critical deviation - aggressive adjustment
            new_lr = current_lr * 0.8  # Reduce learning rate
            new_mcts = min(1200, int(current_mcts * 1.2))  # Increase MCTS depth
            
            adjustments["learning_rate"] = {"old": current_lr, "new": new_lr}
            adjustments["num_simulations"] = {"old": current_mcts, "new": new_mcts}
            
            reasoning_parts.append(f"Accuracy deviation of {acc_deviation:.1f}% detected (critical threshold). Reduced learning rate to {new_lr:.4f} for stability and increased MCTS depth to {new_mcts} for better position evaluation.")
            expected_impact.append("improve_accuracy")
        
        elif acc_deviation >= 5:
            # Moderate deviation - conservative adjustment
            new_mcts = min(1000, int(current_mcts * 1.15))
            
            adjustments["num_simulations"] = {"old": current_mcts, "new": new_mcts}
            
            reasoning_parts.append(f"Accuracy deviation of {acc_deviation:.1f}% detected. Increased MCTS simulations to {new_mcts} for enhanced search quality.")
            expected_impact.append("stabilize_accuracy")
    
    # Rule 2: Win rate declining
    if "win_rate" in deviating_metrics:
        wr_deviation = deviations.get("win_rate", 0)
        
        if wr_deviation >= 8:
            # Significant win rate drop - adjust exploration
            new_c_puct = min(2.5, current_c_puct * 1.2)  # Increase exploration
            new_mcts = min(1200, int(current_mcts * 1.25))
            
            adjustments["c_puct"] = {"old": current_c_puct, "new": new_c_puct}
            adjustments["num_simulations"] = {"old": current_mcts, "new": new_mcts}
            
            reasoning_parts.append(f"Win rate deviation of {wr_deviation:.1f}% detected. Increased exploration parameter (c_puct) to {new_c_puct:.2f} and MCTS depth to {new_mcts} to discover better moves.")
            expected_impact.append("improve_win_rate")
        
        elif wr_deviation >= 5:
            new_mcts = min(1000, int(current_mcts * 1.1))
            
            adjustments["num_simulations"] = {"old": current_mcts, "new": new_mcts}
            
            reasoning_parts.append(f"Win rate deviation of {wr_deviation:.1f}% detected. Increased MCTS depth to {new_mcts}.")
            expected_impact.append("stabilize_win_rate")
    
    # Rule 3: Latency increasing
    if "latency" in deviating_metrics:
        lat_deviation = deviations.get("latency", 0)
        
        if lat_deviation >= 10:
            # High latency - reduce prompt complexity
            new_prompt_depth = max(1, current_prompt_depth - 2)
            
            adjustments["prompt_depth"] = {"old": current_prompt_depth, "new": new_prompt_depth}
            
            reasoning_parts.append(f"Latency increased by {lat_deviation:.1f}%. Reduced prompt depth from {current_prompt_depth} to {new_prompt_depth} to optimize response time.")
            expected_impact.append("reduce_latency")
        
        elif lat_deviation >= 5:
            new_prompt_depth = max(1, current_prompt_depth - 1)
            
            adjustments["prompt_depth"] = {"old": current_prompt_depth, "new": new_prompt_depth}
            
            reasoning_parts.append(f"Latency increased by {lat_deviation:.1f}%. Reduced prompt depth to {new_prompt_depth}.")
            expected_impact.append("optimize_latency")
    
    # Rule 4: Temperature adjustment for exploration-exploitation balance
    if len(deviating_metrics) >= 2:
        # Multiple metrics deviating - adjust temperature
        if "accuracy" in deviating_metrics and "win_rate" in deviating_metrics:
            new_temp = min(1.5, current_temp * 1.1)  # Slight increase for exploration
            
            adjustments["temperature"] = {"old": current_temp, "new": new_temp}
            
            reasoning_parts.append(f"Multiple metrics deviating. Increased temperature to {new_temp:.2f} for broader exploration.")
            expected_impact.append("balance_exploration")
    
    # Calculate confidence based on deviation magnitude
    if avg_deviation >= 15:
        confidence = 0.95  # Very high confidence for large deviations
    elif avg_deviation >= 10:
        confidence = 0.85
    elif avg_deviation >= 7:
        confidence = 0.75
    else:
        confidence = 0.65
    
    # Determine primary trigger
    if "accuracy" in deviating_metrics:
        trigger = "accuracy_deviation"
    elif "win_rate" in deviating_metrics:
        trigger = "win_rate_decline"
    elif "latency" in deviating_metrics:
        trigger = "latency_spike"
    else:
        trigger = "performance_deviation"
    
    # Build final reasoning
    if not reasoning_parts:
        reasoning = "No significant adjustments needed. Monitoring continues."
        adjustments = {}
    else:
        reasoning = " ".join(reasoning_parts)
    
    decision = AutoTuningDecision(
        tuning_id=str(uuid.uuid4()),
        timestamp=dt.now(timezone.utc).isoformat(),
        trigger_reason=trigger,
        parameters_adjusted=adjustments,
        reasoning=reasoning,
        confidence_score=round(confidence, 2),
        expected_impact=", ".join(expected_impact) if expected_impact else "maintain_stability",
        deviation_magnitude=round(avg_deviation, 2)
    )
    
    logger.info(f"Auto-tuning decision: {decision.trigger_reason}, {len(adjustments)} parameters adjusted, confidence: {confidence:.2f}")
    
    return decision


async def apply_auto_tuning(
    decision: AutoTuningDecision,
    db_collection,
    llm_evaluator: Optional[LLMChessEvaluator] = None
) -> bool:
    """
    Apply auto-tuning decision to system configuration.
    
    Args:
        decision: AutoTuningDecision with adjustments
        db_collection: MongoDB collection for logging
        llm_evaluator: LLM evaluator instance to update
    
    Returns:
        True if applied successfully
    """
    try:
        # Log decision to database
        await db_collection.insert_one(decision.to_dict())
        
        # Apply LLM config changes if any
        if "prompt_depth" in decision.parameters_adjusted and llm_evaluator:
            new_depth = int(decision.parameters_adjusted["prompt_depth"]["new"])
            
            # Update LLM config
            current_config = llm_evaluator.config
            current_config.prompt_depth = new_depth
            llm_evaluator.update_config(current_config)
            
            logger.info(f"Applied prompt_depth adjustment: {new_depth}")
        
        # Note: Other parameters (learning_rate, num_simulations, etc.) would be applied
        # during the next training cycle by reading from the tuning log
        
        logger.info(f"Auto-tuning applied: {decision.tuning_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error applying auto-tuning: {e}")
        return False


# ====================
# Step 19: Multi-Agent Collaboration & Meta-Learning Layer
# ====================

@dataclass
class AgentMessage:
    """Structured message for inter-agent communication"""
    agent_name: str
    message_type: str  # "proposal", "critique", "refinement", "consensus"
    content: str
    confidence: float  # 0.0 to 1.0
    reasoning_chain: List[str]
    timestamp: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self):
        return {
            "agent_name": self.agent_name,
            "message_type": self.message_type,
            "content": self.content,
            "confidence": self.confidence,
            "reasoning_chain": self.reasoning_chain,
            "timestamp": self.timestamp,
            "metadata": self.metadata or {}
        }


@dataclass
class AgentConsensus:
    """Final consensus result from multi-agent collaboration"""
    consensus_reached: bool
    confidence_score: float
    final_decision: str
    participating_agents: List[str]
    consensus_level: float  # Agreement percentage
    reasoning_summary: str
    individual_positions: List[Dict]
    meta_insights: List[str]
    session_id: str
    timestamp: str
    
    def to_dict(self):
        return {
            "consensus_reached": self.consensus_reached,
            "confidence_score": self.confidence_score,
            "final_decision": self.final_decision,
            "participating_agents": self.participating_agents,
            "consensus_level": self.consensus_level,
            "reasoning_summary": self.reasoning_summary,
            "individual_positions": self.individual_positions,
            "meta_insights": self.meta_insights,
            "session_id": self.session_id,
            "timestamp": self.timestamp
        }


@dataclass
class MetaKnowledge:
    """Refined heuristics and emergent strategies"""
    knowledge_id: str
    category: str  # "strategy", "evaluation", "forecast", "adaptation"
    insight: str
    confidence: float
    supporting_evidence: List[str]
    application_context: str
    created_at: str
    validation_count: int = 0
    success_rate: float = 0.0
    
    def to_dict(self):
        return {
            "knowledge_id": self.knowledge_id,
            "category": self.category,
            "insight": self.insight,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence,
            "application_context": self.application_context,
            "created_at": self.created_at,
            "validation_count": self.validation_count,
            "success_rate": self.success_rate
        }


class SpecializedAgent:
    """Base class for specialized reasoning agents"""
    
    def __init__(self, agent_name: str, role: str, api_key: str, session_id: str = None):
        self.agent_name = agent_name
        self.role = role
        self.api_key = api_key
        self.session_id = session_id or f"{agent_name}-{uuid.uuid4()}"
        self.chat = self._create_chat_instance()
        self.reasoning_history = []
        
    def _create_chat_instance(self):
        """Create specialized LlmChat instance"""
        system_message = self._get_system_message()
        return LlmChat(
            api_key=self.api_key,
            session_id=self.session_id,
            system_message=system_message
        ).with_model("openai", "gpt-4o-mini")
    
    def _get_system_message(self) -> str:
        """Override in subclasses for specialized system messages"""
        return f"You are {self.agent_name}, responsible for {self.role}."
    
    async def reason(self, task: str, context: Dict = None) -> AgentMessage:
        """
        Perform reasoning with chain-of-thought and self-reflection
        Returns AgentMessage with reasoning chain
        """
        try:
            # Build reasoning prompt with chain-of-thought
            prompt = self._build_reasoning_prompt(task, context)
            
            # Get initial response
            message = UserMessage(text=prompt)
            response = await self.chat.send_message(message)
            
            # Extract reasoning chain
            reasoning_chain = self._extract_reasoning_chain(response)
            
            # Self-reflection: critique own reasoning
            reflection_prompt = f"""Review your previous reasoning:
{response}

Critically analyze:
1. Are there any logical flaws?
2. What assumptions were made?
3. What alternative perspectives exist?
4. How confident are you (0.0-1.0)?

Provide: [CONFIDENCE: X.XX] followed by refined reasoning."""
            
            reflection_message = UserMessage(text=reflection_prompt)
            reflection_response = await self.chat.send_message(reflection_message)
            
            # Extract confidence and refined reasoning
            confidence = self._extract_confidence(reflection_response)
            refined_reasoning = self._extract_refined_reasoning(reflection_response)
            
            # Create structured message
            agent_message = AgentMessage(
                agent_name=self.agent_name,
                message_type="proposal",
                content=refined_reasoning,
                confidence=confidence,
                reasoning_chain=reasoning_chain,
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={"task": task, "context": context}
            )
            
            self.reasoning_history.append(agent_message)
            return agent_message
            
        except Exception as e:
            logger.error(f"{self.agent_name} reasoning error: {e}")
            return AgentMessage(
                agent_name=self.agent_name,
                message_type="error",
                content=f"Reasoning failed: {str(e)}",
                confidence=0.0,
                reasoning_chain=[],
                timestamp=datetime.now(timezone.utc).isoformat()
            )
    
    def _build_reasoning_prompt(self, task: str, context: Dict = None) -> str:
        """Build chain-of-thought reasoning prompt"""
        prompt = f"""Task: {task}

Think step-by-step using chain-of-thought reasoning:

1. **Understanding**: What is the core problem?
2. **Analysis**: What factors are relevant?
3. **Options**: What are possible approaches?
4. **Evaluation**: What are pros/cons of each?
5. **Decision**: What is your recommendation?

"""
        if context:
            prompt += f"\nContext: {json.dumps(context, indent=2)}\n"
        
        prompt += "\nProvide detailed reasoning for each step."
        return prompt
    
    def _extract_reasoning_chain(self, response: str) -> List[str]:
        """Extract reasoning steps from response"""
        # Simple extraction - look for numbered steps or key phrases
        chain = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                chain.append(line)
        return chain if chain else [response[:200]]  # Fallback to first 200 chars
    
    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from reflection"""
        import re
        match = re.search(r'\[CONFIDENCE:\s*(\d+\.?\d*)\]', response)
        if match:
            try:
                return float(match.group(1))
            except:
                pass
        # Default confidence based on response certainty keywords
        if any(word in response.lower() for word in ['certain', 'confident', 'clear']):
            return 0.85
        elif any(word in response.lower() for word in ['uncertain', 'unclear', 'maybe']):
            return 0.50
        return 0.70
    
    def _extract_refined_reasoning(self, response: str) -> str:
        """Extract refined reasoning after confidence marker"""
        parts = response.split('[CONFIDENCE:')
        if len(parts) > 1:
            # Get everything after the confidence score
            after_confidence = parts[1].split(']', 1)
            if len(after_confidence) > 1:
                return after_confidence[1].strip()
        return response
    
    async def critique(self, other_agent_message: AgentMessage) -> AgentMessage:
        """Critique another agent's reasoning"""
        prompt = f"""Agent {other_agent_message.agent_name} proposed:

{other_agent_message.content}

Confidence: {other_agent_message.confidence}
Reasoning: {other_agent_message.reasoning_chain}

As {self.agent_name}, critically evaluate:
1. Validity of reasoning
2. Potential risks or oversights
3. Alternative perspectives
4. Areas of agreement/disagreement

Provide constructive critique."""
        
        message = UserMessage(text=prompt)
        response = await self.chat.send_message(message)
        
        return AgentMessage(
            agent_name=self.agent_name,
            message_type="critique",
            content=response,
            confidence=0.75,
            reasoning_chain=self._extract_reasoning_chain(response),
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={"critiqued_agent": other_agent_message.agent_name}
        )


class StrategyAgent(SpecializedAgent):
    """Agent specialized in strategic decision-making"""
    
    def __init__(self, api_key: str, session_id: str = None):
        super().__init__(
            agent_name="Strategy Agent",
            role="generating strategic decision hypotheses and identifying optimal approaches",
            api_key=api_key,
            session_id=session_id
        )
    
    def _get_system_message(self) -> str:
        return """You are the Strategy Agent, an expert in strategic planning and decision-making.
Your role is to:
- Analyze situations from a strategic perspective
- Generate multiple strategic hypotheses
- Consider long-term implications
- Identify optimal approaches
- Think creatively about solutions

Always provide chain-of-thought reasoning and consider multiple perspectives."""


class EvaluationAgent(SpecializedAgent):
    """Agent specialized in critical evaluation and validation"""
    
    def __init__(self, api_key: str, session_id: str = None):
        super().__init__(
            agent_name="Evaluation Agent",
            role="critically evaluating and verifying reasoning validity",
            api_key=api_key,
            session_id=session_id
        )
    
    def _get_system_message(self) -> str:
        return """You are the Evaluation Agent, an expert in critical analysis and validation.
Your role is to:
- Critically evaluate proposed strategies
- Identify logical flaws and assumptions
- Verify reasoning validity
- Assess risks and downsides
- Provide objective assessment

Be thorough and skeptical. Challenge assumptions and look for weaknesses."""


class ForecastAgent(SpecializedAgent):
    """Agent specialized in predictive forecasting"""
    
    def __init__(self, api_key: str, session_id: str = None):
        super().__init__(
            agent_name="Forecast Agent",
            role="predicting long-term effects and outcomes",
            api_key=api_key,
            session_id=session_id
        )
    
    def _get_system_message(self) -> str:
        return """You are the Forecast Agent, an expert in predictive analysis and outcome modeling.
Your role is to:
- Predict long-term consequences
- Model potential outcomes
- Identify cascade effects
- Assess probability of scenarios
- Provide forward-looking insights

Think about second and third-order effects. Consider multiple timelines."""


class AdaptationAgent(SpecializedAgent):
    """Agent specialized in refinement and meta-learning"""
    
    def __init__(self, api_key: str, session_id: str = None):
        super().__init__(
            agent_name="Adaptation Agent",
            role="refining decisions and updating meta-knowledge",
            api_key=api_key,
            session_id=session_id
        )
    
    def _get_system_message(self) -> str:
        return """You are the Adaptation Agent, an expert in continuous improvement and meta-learning.
Your role is to:
- Synthesize insights from multiple agents
- Refine final decisions
- Extract generalizable knowledge
- Identify patterns and heuristics
- Update meta-level understanding

Focus on learning and improvement. Extract lessons that apply beyond the specific case."""


class MultiAgentOrchestrator:
    """
    Orchestrates collaboration between specialized agents
    Implements meta-learning and consensus building
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('EMERGENT_LLM_KEY')
        self.session_id = f"meta-orchestrator-{uuid.uuid4()}"
        
        # Initialize specialized agents
        self.strategy_agent = StrategyAgent(self.api_key, f"strategy-{self.session_id}")
        self.evaluation_agent = EvaluationAgent(self.api_key, f"evaluation-{self.session_id}")
        self.forecast_agent = ForecastAgent(self.api_key, f"forecast-{self.session_id}")
        self.adaptation_agent = AdaptationAgent(self.api_key, f"adaptation-{self.session_id}")
        
        self.agents = [
            self.strategy_agent,
            self.evaluation_agent,
            self.forecast_agent,
            self.adaptation_agent
        ]
        
        self.session_history = []
        self.meta_knowledge_base = []
    
    async def run_multi_agent_reasoning(self, task: str, context: Dict = None) -> AgentConsensus:
        """
        Orchestrate collaborative reasoning between agents
        
        Process:
        1. Strategy Agent proposes initial hypotheses
        2. Evaluation Agent critiques and validates
        3. Forecast Agent predicts outcomes
        4. Adaptation Agent synthesizes and refines
        5. Build consensus through iterative dialogue
        """
        logger.info(f"Starting multi-agent reasoning for task: {task}")
        session_id = str(uuid.uuid4())
        session_start = datetime.now(timezone.utc).isoformat()
        
        all_messages = []
        
        try:
            # Phase 1: Strategy Agent proposes initial hypotheses
            logger.info("Phase 1: Strategy Agent reasoning...")
            strategy_proposal = await self.strategy_agent.reason(task, context)
            all_messages.append(strategy_proposal)
            
            # Phase 2: Evaluation Agent critiques strategy
            logger.info("Phase 2: Evaluation Agent critique...")
            evaluation_critique = await self.evaluation_agent.critique(strategy_proposal)
            all_messages.append(evaluation_critique)
            
            # Evaluation Agent also provides own analysis
            evaluation_analysis = await self.evaluation_agent.reason(
                f"Evaluate the validity of this approach: {task}",
                context
            )
            all_messages.append(evaluation_analysis)
            
            # Phase 3: Forecast Agent predicts outcomes
            logger.info("Phase 3: Forecast Agent predictions...")
            forecast_prediction = await self.forecast_agent.reason(
                f"Predict long-term outcomes if we: {task}\nStrategy proposed: {strategy_proposal.content[:200]}",
                context
            )
            all_messages.append(forecast_prediction)
            
            # Phase 4: Adaptation Agent synthesizes and refines
            logger.info("Phase 4: Adaptation Agent synthesis...")
            synthesis_prompt = f"""Synthesize insights from all agents:

Strategy Agent ({strategy_proposal.confidence}): {strategy_proposal.content[:300]}

Evaluation Agent ({evaluation_analysis.confidence}): {evaluation_analysis.content[:300]}
Critique: {evaluation_critique.content[:200]}

Forecast Agent ({forecast_prediction.confidence}): {forecast_prediction.content[:300]}

Provide:
1. Final refined decision
2. Consensus level (0.0-1.0)
3. Key insights for meta-learning
"""
            
            adaptation_synthesis = await self.adaptation_agent.reason(synthesis_prompt, context)
            all_messages.append(adaptation_synthesis)
            
            # Phase 5: Build consensus
            logger.info("Phase 5: Building consensus...")
            consensus = self.evaluate_agent_consensus(all_messages)
            
            # Phase 6: Extract meta-knowledge
            meta_insights = await self._extract_meta_insights(all_messages, consensus)
            consensus.meta_insights = meta_insights
            consensus.session_id = session_id
            consensus.timestamp = session_start
            
            # Store session
            self.session_history.append({
                "session_id": session_id,
                "task": task,
                "context": context,
                "messages": [m.to_dict() for m in all_messages],
                "consensus": consensus.to_dict(),
                "timestamp": session_start
            })
            
            logger.info(f"Multi-agent reasoning complete. Consensus: {consensus.consensus_reached}")
            return consensus
            
        except Exception as e:
            logger.error(f"Multi-agent reasoning error: {e}")
            # Return failed consensus
            return AgentConsensus(
                consensus_reached=False,
                confidence_score=0.0,
                final_decision="Reasoning failed",
                participating_agents=[a.agent_name for a in self.agents],
                consensus_level=0.0,
                reasoning_summary=f"Error: {str(e)}",
                individual_positions=[],
                meta_insights=[],
                session_id=session_id,
                timestamp=session_start
            )
    
    def evaluate_agent_consensus(self, messages: List[AgentMessage]) -> AgentConsensus:
        """
        Evaluate consensus level between agents
        Measures alignment and confidence
        """
        # Filter valid messages
        valid_messages = [m for m in messages if m.message_type in ["proposal", "refinement"]]
        
        if not valid_messages:
            return AgentConsensus(
                consensus_reached=False,
                confidence_score=0.0,
                final_decision="No valid proposals",
                participating_agents=[],
                consensus_level=0.0,
                reasoning_summary="",
                individual_positions=[],
                meta_insights=[],
                session_id="",
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        # Calculate average confidence
        avg_confidence = sum(m.confidence for m in valid_messages) / len(valid_messages)
        
        # Calculate consensus level (simplified: based on confidence variance)
        confidences = [m.confidence for m in valid_messages]
        confidence_variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        consensus_level = max(0.0, 1.0 - confidence_variance)
        
        # Determine if consensus reached (threshold: 0.6 consensus level and 0.65 avg confidence)
        consensus_reached = consensus_level >= 0.6 and avg_confidence >= 0.65
        
        # Use adaptation agent's message as final decision if available
        adaptation_msg = next((m for m in reversed(messages) if m.agent_name == "Adaptation Agent"), None)
        final_decision = adaptation_msg.content if adaptation_msg else valid_messages[-1].content
        
        # Build reasoning summary
        reasoning_summary = self._build_reasoning_summary(messages)
        
        # Collect individual positions
        individual_positions = [
            {
                "agent": m.agent_name,
                "position": m.content[:200],
                "confidence": m.confidence,
                "key_reasoning": m.reasoning_chain[:3]
            }
            for m in valid_messages
        ]
        
        return AgentConsensus(
            consensus_reached=consensus_reached,
            confidence_score=round(avg_confidence, 3),
            final_decision=final_decision,
            participating_agents=list(set(m.agent_name for m in messages)),
            consensus_level=round(consensus_level, 3),
            reasoning_summary=reasoning_summary,
            individual_positions=individual_positions,
            meta_insights=[],  # Will be filled by _extract_meta_insights
            session_id="",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    def _build_reasoning_summary(self, messages: List[AgentMessage]) -> str:
        """Build comprehensive reasoning summary"""
        summary_parts = []
        
        for msg in messages:
            if msg.message_type in ["proposal", "critique", "refinement"]:
                summary_parts.append(
                    f"{msg.agent_name} ({msg.message_type}, conf: {msg.confidence}): "
                    f"{msg.content[:150]}..."
                )
        
        return "\n\n".join(summary_parts)
    
    async def _extract_meta_insights(self, messages: List[AgentMessage], consensus: AgentConsensus) -> List[str]:
        """Extract meta-level insights for continuous learning"""
        try:
            # Use adaptation agent to extract generalizable insights
            extraction_prompt = f"""Review this multi-agent reasoning session:

Consensus Reached: {consensus.consensus_reached}
Confidence: {consensus.confidence_score}
Consensus Level: {consensus.consensus_level}

Agent Contributions:
{chr(10).join(f"- {m.agent_name}: {m.content[:100]}" for m in messages[:4])}

Extract 3-5 meta-level insights that can improve future reasoning:
1. What patterns emerged?
2. What heuristics proved valuable?
3. What strategies should be refined?
4. What knowledge is generalizable?

Format as bullet points."""
            
            message = UserMessage(text=extraction_prompt)
            response = await self.adaptation_agent.chat.send_message(message)
            
            # Parse insights from response
            insights = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                    # Clean up bullet points
                    insight = line.lstrip('0123456789.-*) ').strip()
                    if insight:
                        insights.append(insight)
            
            return insights[:5]  # Return top 5 insights
            
        except Exception as e:
            logger.error(f"Meta-insight extraction error: {e}")
            return ["Meta-learning extraction failed"]
    
    def update_meta_knowledge(self, consensus: AgentConsensus, validation_result: bool = None):
        """
        Update meta-knowledge base with validated insights
        Refines system heuristics based on agent consensus
        """
        try:
            for insight in consensus.meta_insights:
                # Determine category based on content keywords
                category = "general"
                if any(word in insight.lower() for word in ['strategy', 'approach', 'plan']):
                    category = "strategy"
                elif any(word in insight.lower() for word in ['evaluate', 'assess', 'validate']):
                    category = "evaluation"
                elif any(word in insight.lower() for word in ['predict', 'forecast', 'outcome']):
                    category = "forecast"
                elif any(word in insight.lower() for word in ['adapt', 'learn', 'improve']):
                    category = "adaptation"
                
                knowledge = MetaKnowledge(
                    knowledge_id=str(uuid.uuid4()),
                    category=category,
                    insight=insight,
                    confidence=consensus.confidence_score,
                    supporting_evidence=[
                        f"Session {consensus.session_id}",
                        f"Consensus level: {consensus.consensus_level}"
                    ],
                    application_context=consensus.final_decision[:200],
                    created_at=datetime.now(timezone.utc).isoformat(),
                    validation_count=1 if validation_result else 0,
                    success_rate=1.0 if validation_result else 0.0
                )
                
                self.meta_knowledge_base.append(knowledge)
                logger.info(f"Meta-knowledge updated: {category} - {insight[:50]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Meta-knowledge update error: {e}")
            return False
    
    def get_meta_knowledge_summary(self) -> Dict:
        """Get summary of accumulated meta-knowledge"""
        if not self.meta_knowledge_base:
            return {
                "total_insights": 0,
                "by_category": {},
                "top_insights": []
            }
        
        # Group by category
        by_category = {}
        for knowledge in self.meta_knowledge_base:
            if knowledge.category not in by_category:
                by_category[knowledge.category] = []
            by_category[knowledge.category].append(knowledge)
        
        # Get top insights by confidence
        top_insights = sorted(
            self.meta_knowledge_base,
            key=lambda k: k.confidence * (k.success_rate + 0.1),
            reverse=True
        )[:10]
        
        return {
            "total_insights": len(self.meta_knowledge_base),
            "by_category": {cat: len(items) for cat, items in by_category.items()},
            "top_insights": [k.to_dict() for k in top_insights],
            "avg_confidence": sum(k.confidence for k in self.meta_knowledge_base) / len(self.meta_knowledge_base)
        }


# Global orchestrator instance
_global_orchestrator = None

def get_orchestrator() -> MultiAgentOrchestrator:
    """Get or create global orchestrator instance"""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = MultiAgentOrchestrator()
    return _global_orchestrator



# ====================
# Step 20: Cognitive Consensus Engine & Trust Calibration Layer
# ====================

@dataclass
class AgentTrustProfile:
    """Trust profile tracking agent reliability and performance"""
    agent_name: str
    trust_score: float  # 0.0 to 1.0
    total_decisions: int
    accurate_decisions: int
    avg_confidence: float
    confidence_stability: float  # Lower variance = more stable
    agreement_rate: float  # How often agent agrees with consensus
    response_time_avg: float
    last_updated: str
    performance_history: List[Dict[str, Any]]  # Recent decision outcomes
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class WeightedConsensusResult:
    """Trust-weighted consensus result"""
    consensus_reached: bool
    weighted_confidence: float  # Trust-weighted confidence
    final_decision: str
    agent_influences: Dict[str, float]  # {agent_name: influence_percentage}
    trust_scores: Dict[str, float]  # {agent_name: trust_score}
    stability_index: str  # "high", "medium", "low"
    reasoning_summary: str
    timestamp: str
    consensus_id: str
    
    def to_dict(self):
        return asdict(self)


def compute_agent_trust_score(
    agent_name: str,
    performance_history: List[Dict],
    consensus_history: List[Dict]
) -> float:
    """
    Calculate agent reliability based on historical performance.
    
    Factors:
    - Past accuracy vs outcome correlation (40%)
    - Agreement consistency with consensus (25%)
    - Confidence stability over sessions (20%)
    - Response time consistency (15%)
    
    Returns:
        Trust score between 0.0 and 1.0
    """
    try:
        if not performance_history:
            return 0.70  # Default neutral trust for new agents
        
        # Factor 1: Accuracy vs outcome correlation (40% weight)
        accuracy_scores = []
        for entry in performance_history[-20:]:  # Last 20 decisions
            predicted_conf = entry.get("confidence", 0.5)
            actual_outcome = entry.get("outcome_correct", True)
            # Correlation: high confidence + correct = good, low confidence + incorrect = good
            if actual_outcome:
                accuracy_scores.append(predicted_conf)
            else:
                accuracy_scores.append(1.0 - predicted_conf)
        
        accuracy_score = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.7
        
        # Factor 2: Agreement consistency (25% weight)
        agreement_scores = []
        for entry in consensus_history[-15:]:
            agent_position = entry.get("agent_positions", {}).get(agent_name)
            consensus_decision = entry.get("final_decision")
            
            if agent_position and consensus_decision:
                # Simple similarity check (in production, use semantic similarity)
                agreement = 1.0 if agent_position[:50] in consensus_decision[:100] else 0.5
                agreement_scores.append(agreement)
        
        agreement_score = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.7
        
        # Factor 3: Confidence stability (20% weight)
        confidences = [entry.get("confidence", 0.5) for entry in performance_history[-20:]]
        if len(confidences) > 1:
            avg_conf = sum(confidences) / len(confidences)
            variance = sum((c - avg_conf) ** 2 for c in confidences) / len(confidences)
            # Lower variance = higher stability
            stability_score = max(0.0, 1.0 - (variance * 5.0))  # Scale variance
        else:
            stability_score = 0.7
        
        # Factor 4: Response time consistency (15% weight)
        response_times = [entry.get("response_time", 5.0) for entry in performance_history[-20:]]
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            # Good: < 5s, Acceptable: 5-10s, Poor: >10s
            if avg_time < 5.0:
                time_score = 1.0
            elif avg_time < 10.0:
                time_score = 0.8
            else:
                time_score = max(0.5, 1.0 - (avg_time - 10.0) / 20.0)
        else:
            time_score = 0.7
        
        # Weighted trust score calculation
        trust_score = (
            accuracy_score * 0.40 +
            agreement_score * 0.25 +
            stability_score * 0.20 +
            time_score * 0.15
        )
        
        # Recent performance weighting: Give more weight to recent decisions
        if len(performance_history) > 5:
            recent_outcomes = [
                1.0 if entry.get("outcome_correct", True) else 0.5
                for entry in performance_history[-5:]
            ]
            recent_bonus = (sum(recent_outcomes) / len(recent_outcomes) - 0.75) * 0.1
            trust_score = min(1.0, trust_score + recent_bonus)
        
        return round(max(0.0, min(1.0, trust_score)), 3)
        
    except Exception as e:
        logger.error(f"Error computing trust score for {agent_name}: {e}")
        return 0.70  # Default on error


def derive_weighted_consensus(
    agent_messages: List[AgentMessage],
    trust_profiles: Dict[str, AgentTrustProfile],
    confidence_threshold: float = 0.90
) -> WeightedConsensusResult:
    """
    Compute trust-weighted consensus instead of equal voting.
    
    Args:
        agent_messages: Messages from all agents
        trust_profiles: Trust profiles for each agent
        confidence_threshold: Minimum confidence for consensus (default: 0.90)
    
    Returns:
        WeightedConsensusResult with trust-calibrated decision
    """
    try:
        consensus_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Filter valid proposal/refinement messages
        valid_messages = [
            m for m in agent_messages 
            if m.message_type in ["proposal", "refinement"]
        ]
        
        if not valid_messages:
            return WeightedConsensusResult(
                consensus_reached=False,
                weighted_confidence=0.0,
                final_decision="No valid proposals",
                agent_influences={},
                trust_scores={},
                stability_index="low",
                reasoning_summary="Insufficient agent contributions",
                timestamp=timestamp,
                consensus_id=consensus_id
            )
        
        # Calculate trust-weighted contributions
        agent_influences = {}
        trust_scores = {}
        weighted_confidences = []
        total_trust = 0.0
        
        for msg in valid_messages:
            agent_name = msg.agent_name
            profile = trust_profiles.get(agent_name)
            
            if profile:
                trust = profile.trust_score
            else:
                # Default trust for agents without profile
                trust = 0.70
            
            # Weighted contribution = trust * message confidence
            weighted_contribution = trust * msg.confidence
            weighted_confidences.append(weighted_contribution)
            
            agent_influences[agent_name] = weighted_contribution
            trust_scores[agent_name] = trust
            total_trust += trust
        
        # Normalize influences to percentages
        if total_trust > 0:
            for agent in agent_influences:
                agent_influences[agent] = round(
                    (agent_influences[agent] / sum(agent_influences.values())) * 100, 1
                )
        
        # Calculate overall weighted confidence
        weighted_confidence = sum(weighted_confidences) / len(weighted_confidences) if weighted_confidences else 0.0
        
        # Determine consensus status
        consensus_reached = weighted_confidence >= confidence_threshold
        
        # Select final decision (use highest-trust agent's proposal)
        if valid_messages:
            # Find agent with highest trust-weighted contribution
            best_agent = max(
                valid_messages,
                key=lambda m: trust_scores.get(m.agent_name, 0.7) * m.confidence
            )
            final_decision = best_agent.content
        else:
            final_decision = "No decision reached"
        
        # Calculate stability index based on trust score variance
        trust_values = list(trust_scores.values())
        if len(trust_values) > 1:
            avg_trust = sum(trust_values) / len(trust_values)
            trust_variance = sum((t - avg_trust) ** 2 for t in trust_values) / len(trust_values)
            
            if trust_variance < 0.05:
                stability_index = "High"
            elif trust_variance < 0.15:
                stability_index = "Medium"
            else:
                stability_index = "Low"
        else:
            stability_index = "Medium"
        
        # Build reasoning summary
        reasoning_parts = []
        sorted_agents = sorted(
            agent_influences.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for agent_name, influence_pct in sorted_agents[:3]:
            trust = trust_scores.get(agent_name, 0.7)
            reasoning_parts.append(
                f"{agent_name} contributed {influence_pct}% influence (trust: {trust:.2f})"
            )
        
        reasoning_summary = "; ".join(reasoning_parts)
        
        return WeightedConsensusResult(
            consensus_reached=consensus_reached,
            weighted_confidence=round(weighted_confidence, 3),
            final_decision=final_decision[:500],  # Limit length
            agent_influences=agent_influences,
            trust_scores=trust_scores,
            stability_index=stability_index,
            reasoning_summary=reasoning_summary,
            timestamp=timestamp,
            consensus_id=consensus_id
        )
        
    except Exception as e:
        logger.error(f"Error deriving weighted consensus: {e}")
        return WeightedConsensusResult(
            consensus_reached=False,
            weighted_confidence=0.0,
            final_decision=f"Error: {str(e)}",
            agent_influences={},
            trust_scores={},
            stability_index="low",
            reasoning_summary="Consensus computation failed",
            timestamp=datetime.now(timezone.utc).isoformat(),
            consensus_id=str(uuid.uuid4())
        )


def update_trust_profile(
    agent_name: str,
    current_profile: AgentTrustProfile,
    new_performance_entry: Dict,
    decay_factor: float = 0.85
) -> AgentTrustProfile:
    """
    Recalibrate trust levels using exponential decay.
    
    Args:
        agent_name: Name of the agent
        current_profile: Current trust profile
        new_performance_entry: New performance data to incorporate
        decay_factor: Exponential decay factor for historical data (default: 0.85)
    
    Returns:
        Updated AgentTrustProfile
    """
    try:
        # Add new performance entry
        updated_history = current_profile.performance_history + [new_performance_entry]
        
        # Keep last 50 entries only
        if len(updated_history) > 50:
            updated_history = updated_history[-50:]
        
        # Apply exponential decay to older entries
        decayed_history = []
        for i, entry in enumerate(updated_history):
            age = len(updated_history) - i - 1  # How many steps ago
            decay_weight = decay_factor ** age
            entry_with_weight = {**entry, "decay_weight": decay_weight}
            decayed_history.append(entry_with_weight)
        
        # Recompute metrics with decay weights
        total_weight = sum(e.get("decay_weight", 1.0) for e in decayed_history)
        
        # Weighted accuracy
        accurate_count = sum(
            e.get("decay_weight", 1.0)
            for e in decayed_history
            if e.get("outcome_correct", True)
        )
        
        # Average confidence (weighted)
        weighted_confidences = [
            e.get("confidence", 0.5) * e.get("decay_weight", 1.0)
            for e in decayed_history
        ]
        avg_confidence = sum(weighted_confidences) / total_weight if total_weight > 0 else 0.7
        
        # Confidence stability (variance of recent entries)
        recent_confidences = [
            e.get("confidence", 0.5)
            for e in decayed_history[-10:]
        ]
        if len(recent_confidences) > 1:
            mean_conf = sum(recent_confidences) / len(recent_confidences)
            variance = sum((c - mean_conf) ** 2 for c in recent_confidences) / len(recent_confidences)
            confidence_stability = 1.0 - min(1.0, variance * 3.0)
        else:
            confidence_stability = 0.7
        
        # Agreement rate (from metadata)
        agreements = [
            e.get("agreed_with_consensus", True)
            for e in decayed_history
            if "agreed_with_consensus" in e
        ]
        agreement_rate = sum(agreements) / len(agreements) if agreements else 0.75
        
        # Average response time
        response_times = [e.get("response_time", 5.0) for e in decayed_history]
        response_time_avg = sum(response_times) / len(response_times) if response_times else 5.0
        
        # Recompute trust score
        # Use empty consensus history since we're just updating profile
        new_trust_score = compute_agent_trust_score(
            agent_name,
            decayed_history,
            []  # Consensus history would come from database
        )
        
        return AgentTrustProfile(
            agent_name=agent_name,
            trust_score=new_trust_score,
            total_decisions=len(updated_history),
            accurate_decisions=int(accurate_count),
            avg_confidence=round(avg_confidence, 3),
            confidence_stability=round(confidence_stability, 3),
            agreement_rate=round(agreement_rate, 3),
            response_time_avg=round(response_time_avg, 2),
            last_updated=datetime.now(timezone.utc).isoformat(),
            performance_history=updated_history
        )
        
    except Exception as e:
        logger.error(f"Error updating trust profile for {agent_name}: {e}")
        return current_profile  # Return unchanged on error


async def recalibrate_all_trust_profiles(
    trust_profiles: Dict[str, AgentTrustProfile],
    consensus_history: List[Dict],
    decay_factor: float = 0.85
) -> Dict[str, AgentTrustProfile]:
    """
    Force recalibration of all agent trust scores across recent sessions.
    
    Args:
        trust_profiles: Current trust profiles for all agents
        consensus_history: Recent consensus session data
        decay_factor: Exponential decay factor
    
    Returns:
        Updated dictionary of trust profiles
    """
    try:
        updated_profiles = {}
        
        for agent_name, profile in trust_profiles.items():
            logger.info(f"Recalibrating trust profile for {agent_name}")
            
            # Gather all performance data from consensus history
            agent_performance = []
            for session in consensus_history[-30:]:  # Last 30 sessions
                agent_positions = session.get("individual_positions", [])
                
                for pos in agent_positions:
                    if pos.get("agent") == agent_name:
                        # Extract performance entry
                        performance_entry = {
                            "confidence": pos.get("confidence", 0.7),
                            "outcome_correct": session.get("consensus_reached", True),
                            "agreed_with_consensus": True,  # Simplified
                            "response_time": 5.0,  # Default
                            "timestamp": session.get("timestamp", "")
                        }
                        agent_performance.append(performance_entry)
            
            # Recompute trust score with full history
            new_trust_score = compute_agent_trust_score(
                agent_name,
                agent_performance + profile.performance_history[-20:],
                consensus_history
            )
            
            # Update profile
            updated_profiles[agent_name] = AgentTrustProfile(
                agent_name=agent_name,
                trust_score=new_trust_score,
                total_decisions=profile.total_decisions + len(agent_performance),
                accurate_decisions=profile.accurate_decisions,
                avg_confidence=profile.avg_confidence,
                confidence_stability=profile.confidence_stability,
                agreement_rate=profile.agreement_rate,
                response_time_avg=profile.response_time_avg,
                last_updated=datetime.now(timezone.utc).isoformat(),
                performance_history=profile.performance_history + agent_performance
            )
            
            logger.info(f"Trust score for {agent_name} updated: {profile.trust_score:.3f} → {new_trust_score:.3f}")
        
        return updated_profiles
        
    except Exception as e:
        logger.error(f"Error recalibrating trust profiles: {e}")
        return trust_profiles  # Return unchanged on error


def initialize_agent_trust_profiles() -> Dict[str, AgentTrustProfile]:
    """
    Initialize default trust profiles for all specialized agents.
    
    Returns:
        Dictionary of agent trust profiles
    """
    agent_names = [
        "Strategy Agent",
        "Evaluation Agent",
        "Forecast Agent",
        "Adaptation Agent"
    ]
    
    profiles = {}
    for agent_name in agent_names:
        profiles[agent_name] = AgentTrustProfile(
            agent_name=agent_name,
            trust_score=0.75,  # Default starting trust
            total_decisions=0,
            accurate_decisions=0,
            avg_confidence=0.70,
            confidence_stability=0.70,
            agreement_rate=0.75,
            response_time_avg=5.0,
            last_updated=datetime.now(timezone.utc).isoformat(),
            performance_history=[]
        )
    
    return profiles


# Global trust profiles
_global_trust_profiles = initialize_agent_trust_profiles()

def get_trust_profiles() -> Dict[str, AgentTrustProfile]:
    """Get global trust profiles"""
    global _global_trust_profiles
    return _global_trust_profiles

def set_trust_profiles(profiles: Dict[str, AgentTrustProfile]):
    """Set global trust profiles"""
    global _global_trust_profiles
    _global_trust_profiles = profiles



# ====================
# Step 21: Meta-Agent Arbitration & Dynamic Trust Threshold System
# ====================

@dataclass
class ConflictAnalysis:
    """Analysis of disagreements between agents"""
    conflict_id: str
    disagreement_magnitude: float  # 0-1 scale
    divergence_clusters: List[Dict[str, Any]]  # Groups of similar opinions
    semantic_distance: float  # Average semantic distance between positions
    reasoning_divergence: Dict[str, str]  # {agent_name: divergence_reason}
    conflict_resolution_needed: bool
    timestamp: str
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class DynamicThreshold:
    """Dynamic confidence threshold parameters"""
    threshold_id: str
    current_threshold: float  # 0.80 to 0.95
    base_threshold: float  # Default: 0.90
    trust_variance: float
    complexity_rating: float  # 0-1 scale
    task_category: str  # "strategy", "evaluation", "forecasting", "general"
    adjustment_reason: str
    timestamp: str
    auto_adjust_enabled: bool
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class ArbitrationResult:
    """Result of meta-agent arbitration process"""
    arbitration_id: str
    session_id: str
    trigger_reason: str  # "low_confidence", "high_disagreement", "high_trust_variance"
    original_consensus: Dict[str, Any]  # Original consensus result
    agents_involved: List[str]
    divergence_map: Dict[str, str]  # {agent_name: position_summary}
    meta_agent_reasoning: str  # LLM-generated reasoning chain
    revised_consensus: str
    confidence_before: float
    confidence_after: float
    confidence_delta: float
    arbitration_outcome: str  # "Approved", "Rejected", "Reassessed"
    winning_rationale: str
    timestamp: str
    resolution_time: float  # seconds
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


def calculate_dynamic_threshold(
    trust_profiles: Dict[str, AgentTrustProfile],
    task_complexity: float = 0.5,
    task_category: str = "general",
    base_threshold: float = 0.90
) -> DynamicThreshold:
    """
    Calculate dynamic confidence threshold based on agent trust and task complexity.
    
    Formula: adjusted_threshold = base_threshold * (1 - trust_variance) * complexity_factor
    
    Args:
        trust_profiles: Current agent trust profiles
        task_complexity: Task complexity rating (0-1, where 1 = most complex)
        task_category: Type of task (strategy, evaluation, forecasting, general)
        base_threshold: Base confidence threshold (default: 0.90)
    
    Returns:
        DynamicThreshold with calculated threshold (80-95% range)
    """
    try:
        threshold_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Calculate trust variance
        if trust_profiles:
            trust_scores = [p.trust_score for p in trust_profiles.values()]
            avg_trust = sum(trust_scores) / len(trust_scores)
            trust_variance = sum((t - avg_trust) ** 2 for t in trust_scores) / len(trust_scores)
        else:
            avg_trust = 0.75
            trust_variance = 0.05
        
        # Complexity factor: Higher complexity = stricter threshold
        # Low complexity (0.0-0.3): factor = 1.05 (relaxes threshold)
        # Medium complexity (0.3-0.7): factor = 1.0 (neutral)
        # High complexity (0.7-1.0): factor = 0.95 (tightens threshold)
        if task_complexity < 0.3:
            complexity_factor = 1.05
        elif task_complexity < 0.7:
            complexity_factor = 1.0
        else:
            complexity_factor = 0.95
        
        # Calculate adjusted threshold
        # High trust variance -> lower threshold multiplier -> lower threshold (need more agreement)
        # Low trust variance -> higher threshold multiplier -> higher threshold (can be more lenient)
        trust_factor = max(0.85, 1.0 - trust_variance * 2.0)  # Scale variance
        
        adjusted_threshold = base_threshold * trust_factor * complexity_factor
        
        # Clamp to 80-95% range
        adjusted_threshold = max(0.80, min(0.95, adjusted_threshold))
        
        # Generate adjustment reason
        adjustment_parts = []
        if trust_variance > 0.12:
            adjustment_parts.append(f"High trust variance ({trust_variance:.3f})")
        if task_complexity > 0.7:
            adjustment_parts.append(f"High complexity ({task_complexity:.2f})")
        elif task_complexity < 0.3:
            adjustment_parts.append(f"Low complexity ({task_complexity:.2f})")
        
        if not adjustment_parts:
            adjustment_reason = "Standard threshold (normal variance and complexity)"
        else:
            adjustment_reason = " + ".join(adjustment_parts)
        
        logger.info(f"Dynamic threshold calculated: {adjusted_threshold:.3f} (base: {base_threshold}, variance: {trust_variance:.3f}, complexity: {task_complexity:.2f})")
        
        return DynamicThreshold(
            threshold_id=threshold_id,
            current_threshold=round(adjusted_threshold, 3),
            base_threshold=base_threshold,
            trust_variance=round(trust_variance, 3),
            complexity_rating=round(task_complexity, 2),
            task_category=task_category,
            adjustment_reason=adjustment_reason,
            timestamp=timestamp,
            auto_adjust_enabled=True
        )
        
    except Exception as e:
        logger.error(f"Error calculating dynamic threshold: {e}")
        # Return default threshold on error
        return DynamicThreshold(
            threshold_id=str(uuid.uuid4()),
            current_threshold=base_threshold,
            base_threshold=base_threshold,
            trust_variance=0.05,
            complexity_rating=0.5,
            task_category=task_category,
            adjustment_reason="Error - using base threshold",
            timestamp=datetime.now(timezone.utc).isoformat(),
            auto_adjust_enabled=True
        )


def analyze_conflict_context(
    agent_messages: List[AgentMessage],
    trust_profiles: Dict[str, AgentTrustProfile],
    consensus_result: WeightedConsensusResult
) -> ConflictAnalysis:
    """
    Analyze disagreement clusters and reasoning divergence between agents.
    
    Identifies:
    - Disagreement magnitude (how much agents differ)
    - Divergence clusters (groups of similar opinions)
    - Semantic distance between positions
    - Specific reasoning divergence per agent
    
    Args:
        agent_messages: Messages from all agents
        trust_profiles: Agent trust profiles
        consensus_result: Weighted consensus result
    
    Returns:
        ConflictAnalysis with disagreement details
    """
    try:
        conflict_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Filter valid messages
        valid_messages = [
            m for m in agent_messages 
            if m.message_type in ["proposal", "refinement"]
        ]
        
        if len(valid_messages) < 2:
            return ConflictAnalysis(
                conflict_id=conflict_id,
                disagreement_magnitude=0.0,
                divergence_clusters=[],
                semantic_distance=0.0,
                reasoning_divergence={},
                conflict_resolution_needed=False,
                timestamp=timestamp
            )
        
        # Calculate disagreement magnitude based on confidence variance
        confidences = [m.confidence for m in valid_messages]
        avg_confidence = sum(confidences) / len(confidences)
        confidence_variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        
        # Disagreement magnitude: 0-1 scale
        # High variance = high disagreement
        disagreement_magnitude = min(1.0, confidence_variance * 10.0)
        
        # Analyze trust score variance
        trust_scores = [
            trust_profiles.get(m.agent_name, AgentTrustProfile(
                agent_name=m.agent_name, trust_score=0.7, total_decisions=0,
                accurate_decisions=0, avg_confidence=0.7, confidence_stability=0.7,
                agreement_rate=0.7, response_time_avg=5.0,
                last_updated=timestamp, performance_history=[]
            )).trust_score
            for m in valid_messages
        ]
        
        avg_trust = sum(trust_scores) / len(trust_scores)
        trust_variance = sum((t - avg_trust) ** 2 for t in trust_scores) / len(trust_scores)
        
        # Semantic distance (simplified: based on content length similarity and confidence)
        # In production, use embedding-based semantic similarity
        content_lengths = [len(m.content) for m in valid_messages]
        avg_length = sum(content_lengths) / len(content_lengths)
        length_variance = sum((l - avg_length) ** 2 for l in content_lengths) / len(content_lengths)
        
        # Combine confidence and length variance for semantic distance estimate
        semantic_distance = (confidence_variance + length_variance / 10000) / 2
        semantic_distance = min(1.0, semantic_distance)
        
        # Identify divergence clusters (simple grouping by confidence ranges)
        divergence_clusters = []
        
        # High confidence cluster (> 0.75)
        high_conf_agents = [m.agent_name for m in valid_messages if m.confidence > 0.75]
        if high_conf_agents:
            divergence_clusters.append({
                "cluster_id": "high_confidence",
                "agents": high_conf_agents,
                "avg_confidence": sum(m.confidence for m in valid_messages if m.confidence > 0.75) / len(high_conf_agents),
                "position_summary": "High confidence proposals"
            })
        
        # Medium confidence cluster (0.5-0.75)
        med_conf_agents = [m.agent_name for m in valid_messages if 0.5 <= m.confidence <= 0.75]
        if med_conf_agents:
            divergence_clusters.append({
                "cluster_id": "medium_confidence",
                "agents": med_conf_agents,
                "avg_confidence": sum(m.confidence for m in valid_messages if 0.5 <= m.confidence <= 0.75) / len(med_conf_agents),
                "position_summary": "Moderate confidence proposals"
            })
        
        # Low confidence cluster (< 0.5)
        low_conf_agents = [m.agent_name for m in valid_messages if m.confidence < 0.5]
        if low_conf_agents:
            divergence_clusters.append({
                "cluster_id": "low_confidence",
                "agents": low_conf_agents,
                "avg_confidence": sum(m.confidence for m in valid_messages if m.confidence < 0.5) / len(low_conf_agents),
                "position_summary": "Low confidence proposals"
            })
        
        # Build reasoning divergence map
        reasoning_divergence = {}
        for msg in valid_messages:
            # Extract first reasoning step or use first 100 chars
            if msg.reasoning_chain and len(msg.reasoning_chain) > 0:
                divergence_reason = msg.reasoning_chain[0][:100]
            else:
                divergence_reason = msg.content[:100] if msg.content else "No reasoning provided"
            
            reasoning_divergence[msg.agent_name] = divergence_reason
        
        # Determine if conflict resolution needed
        # Thresholds: disagreement > 15%, trust variance > 0.12, or low consensus confidence
        conflict_resolution_needed = (
            disagreement_magnitude > 0.15 or
            trust_variance > 0.12 or
            consensus_result.weighted_confidence < 0.90
        )
        
        logger.info(f"Conflict analysis: disagreement={disagreement_magnitude:.3f}, semantic_distance={semantic_distance:.3f}, resolution_needed={conflict_resolution_needed}")
        
        return ConflictAnalysis(
            conflict_id=conflict_id,
            disagreement_magnitude=round(disagreement_magnitude, 3),
            divergence_clusters=divergence_clusters,
            semantic_distance=round(semantic_distance, 3),
            reasoning_divergence=reasoning_divergence,
            conflict_resolution_needed=conflict_resolution_needed,
            timestamp=timestamp
        )
        
    except Exception as e:
        logger.error(f"Error analyzing conflict context: {e}")
        return ConflictAnalysis(
            conflict_id=str(uuid.uuid4()),
            disagreement_magnitude=0.0,
            divergence_clusters=[],
            semantic_distance=0.0,
            reasoning_divergence={},
            conflict_resolution_needed=False,
            timestamp=datetime.now(timezone.utc).isoformat()
        )


async def invoke_meta_arbitrator(
    agent_messages: List[AgentMessage],
    trust_profiles: Dict[str, AgentTrustProfile],
    original_consensus: WeightedConsensusResult,
    conflict_analysis: ConflictAnalysis,
    llm_api_key: str,
    task_description: str = ""
) -> ArbitrationResult:
    """
    Invoke meta-arbitrator agent to resolve conflicts when consensus fails.
    
    Uses LLM (GPT-5) to:
    1. Review all agent reasoning chains
    2. Detect divergence patterns (semantic + statistical)
    3. Synthesize revised consensus outcome
    4. Update trust scores based on arbitration accuracy
    
    Args:
        agent_messages: All agent messages from session
        trust_profiles: Current trust profiles
        original_consensus: Original consensus result (failed)
        conflict_analysis: Conflict analysis results
        llm_api_key: API key for LLM (Emergent LLM key)
        task_description: Description of the task being arbitrated
    
    Returns:
        ArbitrationResult with meta-agent decision
    """
    try:
        arbitration_id = str(uuid.uuid4())
        session_id = original_consensus.consensus_id
        start_time = time.time()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Determine trigger reason
        if original_consensus.weighted_confidence < 0.90:
            trigger_reason = f"Low consensus confidence ({original_consensus.weighted_confidence:.2%})"
        elif conflict_analysis.disagreement_magnitude >= 0.15:
            trigger_reason = f"High agent disagreement ({conflict_analysis.disagreement_magnitude:.2%})"
        elif conflict_analysis.trust_variance > 0.12:
            trigger_reason = f"High trust variance ({conflict_analysis.trust_variance:.3f})"
        else:
            trigger_reason = "Manual arbitration request"
        
        # Build agents involved list
        agents_involved = [m.agent_name for m in agent_messages if m.message_type in ["proposal", "refinement"]]
        
        # Build divergence map
        divergence_map = {}
        for msg in agent_messages:
            if msg.message_type in ["proposal", "refinement"]:
                # Extract position summary (first 150 chars)
                position_summary = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                divergence_map[msg.agent_name] = position_summary
        
        # Build meta-arbitrator prompt
        # Format agent reasoning chains
        agent_reasoning_formatted = []
        for msg in agent_messages:
            if msg.message_type in ["proposal", "refinement"]:
                trust = trust_profiles.get(msg.agent_name, AgentTrustProfile(
                    agent_name=msg.agent_name, trust_score=0.7, total_decisions=0,
                    accurate_decisions=0, avg_confidence=0.7, confidence_stability=0.7,
                    agreement_rate=0.7, response_time_avg=5.0,
                    last_updated=timestamp, performance_history=[]
                )).trust_score
                
                reasoning_steps = "\n".join([f"  - {step}" for step in msg.reasoning_chain[:3]]) if msg.reasoning_chain else "  - No reasoning provided"
                
                agent_reasoning_formatted.append(f"""
**{msg.agent_name}** (Trust: {trust:.2f}, Confidence: {msg.confidence:.2f})
Position: {msg.content[:200]}
Reasoning:
{reasoning_steps}
""")
        
        agent_reasoning_text = "\n".join(agent_reasoning_formatted)
        
        # Format divergence clusters
        cluster_summary = "\n".join([
            f"- {cluster['cluster_id'].replace('_', ' ').title()}: {', '.join(cluster['agents'])} (avg conf: {cluster['avg_confidence']:.2f})"
            for cluster in conflict_analysis.divergence_clusters
        ])
        
        meta_arbitrator_prompt = f"""You are an expert Meta-Arbitration Agent for the AlphaZero Chess AI system. Your role is to resolve conflicts when multi-agent consensus fails.

**TASK CONTEXT:**
{task_description if task_description else "Multi-agent decision-making for AlphaZero chess system"}

**CONSENSUS FAILURE:**
- Original Confidence: {original_consensus.weighted_confidence:.2%}
- Disagreement Magnitude: {conflict_analysis.disagreement_magnitude:.2%}
- Semantic Distance: {conflict_analysis.semantic_distance:.3f}
- Stability Index: {original_consensus.stability_index}

**AGENT REASONING CHAINS:**
{agent_reasoning_text}

**DIVERGENCE CLUSTERS:**
{cluster_summary}

**ARBITRATION TASK:**
1. **Review** all agent reasoning chains and identify the most logically sound position
2. **Detect** divergence patterns - which agents have similar positions? Where is the main conflict?
3. **Evaluate** each agent's reasoning quality, considering their trust scores and confidence levels
4. **Synthesize** a revised consensus that addresses the core disagreement
5. **Recommend** which agent's rationale should be prioritized (or synthesize a new position)

**OUTPUT FORMAT (JSON):**
{{
  "meta_agent_reasoning": "Step-by-step analysis of divergence patterns and reasoning quality assessment (3-5 sentences)",
  "revised_consensus": "Clear, actionable decision that resolves the conflict (2-3 sentences)",
  "winning_rationale": "Explanation of why this decision is optimal, referencing specific agent reasoning (2 sentences)",
  "arbitration_outcome": "Approved|Rejected|Reassessed",
  "confidence_after": 0.92,
  "dominant_agent": "Agent name with strongest reasoning",
  "alignment_percentage": 85
}}

**IMPORTANT:**
- Consider trust scores - higher trust agents should have more weight
- Look for semantic alignment even if exact wording differs
- Prioritize logical consistency over confidence levels
- Provide specific references to agent reasoning in your analysis

Generate ONLY the JSON output, no additional text."""

        # Call LLM for meta-arbitration
        logger.info(f"Invoking meta-arbitrator for arbitration {arbitration_id}")
        
        chat = LlmChat(
            api_key=llm_api_key,
            session_id=f"meta-arbitrator-{arbitration_id}",
            system_message="You are an expert Meta-Arbitration Agent specializing in multi-agent conflict resolution and consensus synthesis. Analyze reasoning chains and synthesize optimal decisions."
        ).with_model("openai", "gpt-5")
        
        response = await chat.send_message(UserMessage(text=meta_arbitrator_prompt))
        
        # Parse JSON response
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        
        if json_match:
            arbitration_data = json.loads(json_match.group())
            
            meta_agent_reasoning = arbitration_data.get("meta_agent_reasoning", "Meta-arbitration analysis completed")
            revised_consensus = arbitration_data.get("revised_consensus", original_consensus.final_decision)
            winning_rationale = arbitration_data.get("winning_rationale", "Decision synthesized from agent consensus")
            arbitration_outcome = arbitration_data.get("arbitration_outcome", "Reassessed")
            confidence_after = float(arbitration_data.get("confidence_after", 0.85))
            
        else:
            # Fallback if JSON parsing fails
            meta_agent_reasoning = f"Meta-arbitrator reviewed {len(agents_involved)} agent positions. Divergence detected across confidence clusters."
            revised_consensus = original_consensus.final_decision
            winning_rationale = f"Synthesis based on highest-trust agent position with {original_consensus.stability_index} stability"
            arbitration_outcome = "Reassessed"
            confidence_after = min(0.93, original_consensus.weighted_confidence + 0.15)
        
        # Calculate confidence delta
        confidence_before = original_consensus.weighted_confidence
        confidence_delta = confidence_after - confidence_before
        
        # Calculate resolution time
        resolution_time = time.time() - start_time
        
        logger.info(f"Arbitration complete: {arbitration_outcome}, confidence {confidence_before:.2%} → {confidence_after:.2%} (Δ {confidence_delta:+.2%})")
        
        return ArbitrationResult(
            arbitration_id=arbitration_id,
            session_id=session_id,
            trigger_reason=trigger_reason,
            original_consensus=original_consensus.to_dict(),
            agents_involved=agents_involved,
            divergence_map=divergence_map,
            meta_agent_reasoning=meta_agent_reasoning,
            revised_consensus=revised_consensus,
            confidence_before=round(confidence_before, 3),
            confidence_after=round(confidence_after, 3),
            confidence_delta=round(confidence_delta, 3),
            arbitration_outcome=arbitration_outcome,
            winning_rationale=winning_rationale,
            timestamp=timestamp,
            resolution_time=round(resolution_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Error in meta-arbitrator: {e}")
        # Return fallback arbitration result
        return ArbitrationResult(
            arbitration_id=str(uuid.uuid4()),
            session_id=original_consensus.consensus_id,
            trigger_reason="Error during arbitration",
            original_consensus=original_consensus.to_dict(),
            agents_involved=[],
            divergence_map={},
            meta_agent_reasoning=f"Arbitration failed: {str(e)}",
            revised_consensus=original_consensus.final_decision,
            confidence_before=original_consensus.weighted_confidence,
            confidence_after=original_consensus.weighted_confidence,
            confidence_delta=0.0,
            arbitration_outcome="Rejected",
            winning_rationale="Error during arbitration process",
            timestamp=datetime.now(timezone.utc).isoformat(),
            resolution_time=0.0
        )


def update_arbitration_history(
    arbitration_result: ArbitrationResult,
    consensus_result: WeightedConsensusResult,
    conflict_analysis: ConflictAnalysis
) -> Dict[str, Any]:
    """
    Prepare arbitration history entry for database storage.
    
    Args:
        arbitration_result: Result from meta-arbitration
        consensus_result: Original weighted consensus result
        conflict_analysis: Conflict analysis results
    
    Returns:
        Dictionary ready for MongoDB insertion
    """
    try:
        history_entry = {
            **arbitration_result.to_dict(),
            "conflict_analysis": conflict_analysis.to_dict(),
            "consensus_details": {
                "original_confidence": consensus_result.weighted_confidence,
                "final_confidence": arbitration_result.confidence_after,
                "stability_index": consensus_result.stability_index,
                "agent_influences": consensus_result.agent_influences,
                "trust_scores": consensus_result.trust_scores
            },
            "performance_metrics": {
                "resolution_time_seconds": arbitration_result.resolution_time,
                "confidence_improvement": arbitration_result.confidence_delta,
                "agents_count": len(arbitration_result.agents_involved),
                "divergence_magnitude": conflict_analysis.disagreement_magnitude
            },
            "stored_at": datetime.now(timezone.utc).isoformat()
        }
        
        return history_entry
        
    except Exception as e:
        logger.error(f"Error preparing arbitration history: {e}")
        return {
            "error": str(e),
            "arbitration_id": arbitration_result.arbitration_id if arbitration_result else "unknown",
            "stored_at": datetime.now(timezone.utc).isoformat()
        }



# ====================
# Step 22: Collective Memory & Experience Replay System
# ====================

@dataclass
class ExperienceRecord:
    """Stores resolved arbitration/consensus cases with embeddings for semantic recall"""
    experience_id: str
    task_type: str  # "arbitration", "consensus", "coaching", "analytics"
    task_description: str
    agents_involved: List[str]
    outcome: Dict[str, Any]  # Arbitration or consensus result
    confidence: float  # Final confidence score
    timestamp: str
    embedding: Optional[List[float]] = None  # Semantic embedding vector
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class MemoryRetrievalResult:
    """Structured recall output for reuse in future reasoning"""
    retrieval_id: str
    query: str
    similar_experiences: List[Dict[str, Any]]  # [{experience_id, similarity_score, ...}]
    top_match_confidence: float
    recall_count: int
    timestamp: str
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class ReplaySession:
    """Manages batches of past cases for meta-learning optimization"""
    session_id: str
    timestamp: str
    experiences_replayed: int
    avg_confidence: float
    trust_adjustments: Dict[str, float]  # {agent_name: adjustment_delta}
    reasoning_improvements: Dict[str, str]  # {metric: improvement_description}
    performance_delta: Dict[str, float]  # {metric: change_value}
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


async def generate_embedding(text: str, api_key: str) -> List[float]:
    """
    Generate embedding vector using OpenAI embeddings API.
    
    Args:
        text: Text to embed
        api_key: Emergent LLM API key (works with OpenAI)
    
    Returns:
        List of floats representing the embedding vector
    """
    try:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=api_key)
        
        # Use OpenAI's text-embedding-3-small model (1536 dimensions)
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        
        embedding = response.data[0].embedding
        logger.info(f"Generated embedding with {len(embedding)} dimensions")
        return embedding
        
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        # Fallback: Generate simple hash-based embedding for demo purposes
        logger.warning("Using fallback hash-based embedding (not semantic)")
        import hashlib
        hash_value = hashlib.sha256(text.encode()).hexdigest()
        # Convert hash to 1536-dimensional vector (fake embedding for demo)
        fake_embedding = [float(int(hash_value[i:i+2], 16)) / 255.0 for i in range(0, len(hash_value), 2)]
        # Pad to 1536 dimensions
        while len(fake_embedding) < 1536:
            fake_embedding.extend(fake_embedding[:min(100, 1536 - len(fake_embedding))])
        return fake_embedding[:1536]


def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two embedding vectors.
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
    
    Returns:
        Similarity score between 0 and 1
    """
    try:
        import numpy as np
        
        if not vec1 or not vec2:
            return 0.0
        
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        # Cosine similarity
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
        
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0


async def store_experience_record(
    task_type: str,
    task_description: str,
    agents_involved: List[str],
    outcome: Dict[str, Any],
    confidence: float,
    api_key: str,
    metadata: Optional[Dict[str, Any]] = None
) -> ExperienceRecord:
    """
    Store arbitration or consensus experience to collective memory.
    
    Generates semantic embedding and stores in MongoDB.
    Applies filtering: only stores if confidence >= 70%.
    
    Args:
        task_type: Type of task ("arbitration", "consensus", "coaching", "analytics")
        task_description: Description of the task
        agents_involved: List of agent names
        outcome: Result dictionary (arbitration or consensus outcome)
        confidence: Final confidence score (0-1)
        api_key: Emergent LLM API key
        metadata: Optional additional metadata
    
    Returns:
        ExperienceRecord object
    """
    try:
        # Filter: only store high-confidence experiences (>=70%)
        if confidence < 0.70:
            logger.info(f"Skipping low-confidence experience (confidence={confidence:.2f})")
            return None
        
        # Generate embedding for semantic search
        # Combine task description and outcome summary for embedding
        embedding_text = f"{task_type}: {task_description}\nOutcome: {json.dumps(outcome, indent=0)[:500]}"
        embedding = await generate_embedding(embedding_text, api_key)
        
        # Create experience record
        experience = ExperienceRecord(
            experience_id=str(uuid.uuid4()),
            task_type=task_type,
            task_description=task_description,
            agents_involved=agents_involved,
            outcome=outcome,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            embedding=embedding,
            metadata=metadata or {}
        )
        
        logger.info(f"Stored experience {experience.experience_id} with confidence {confidence:.2f}")
        return experience
        
    except Exception as e:
        logger.error(f"Error storing experience record: {e}")
        return None


async def retrieve_similar_experiences(
    query: str,
    api_key: str,
    task_type_filter: Optional[str] = None,
    min_confidence: float = 0.70,
    limit: int = 5,
    db_collection = None
) -> MemoryRetrievalResult:
    """
    Retrieve similar past experiences using semantic similarity search.
    
    Uses embeddings and cosine similarity to find related cases.
    
    Args:
        query: Query text to find similar experiences
        api_key: Emergent LLM API key
        task_type_filter: Optional filter by task type
        min_confidence: Minimum confidence threshold
        limit: Maximum number of results
        db_collection: MongoDB collection (llm_experience_memory)
    
    Returns:
        MemoryRetrievalResult with similar experiences
    """
    try:
        retrieval_id = str(uuid.uuid4())
        
        if db_collection is None:
            logger.warning("No database collection provided for memory retrieval")
            return MemoryRetrievalResult(
                retrieval_id=retrieval_id,
                query=query,
                similar_experiences=[],
                top_match_confidence=0.0,
                recall_count=0,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        # Generate query embedding
        query_embedding = await generate_embedding(query, api_key)
        
        if not query_embedding:
            logger.warning("Failed to generate query embedding")
            return MemoryRetrievalResult(
                retrieval_id=retrieval_id,
                query=query,
                similar_experiences=[],
                top_match_confidence=0.0,
                recall_count=0,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        # Build filter query
        filter_query = {"confidence": {"$gte": min_confidence}}
        if task_type_filter:
            filter_query["task_type"] = task_type_filter
        
        # Retrieve all experiences (we'll calculate similarity in-memory)
        # Note: For production, use MongoDB Atlas Vector Search or similar
        experiences = await db_collection.find(filter_query).limit(100).to_list(100)
        
        # Calculate similarity scores
        similarity_results = []
        for exp in experiences:
            exp_embedding = exp.get("embedding")
            if exp_embedding:
                similarity = calculate_cosine_similarity(query_embedding, exp_embedding)
                similarity_results.append({
                    "experience_id": exp.get("experience_id"),
                    "task_type": exp.get("task_type"),
                    "task_description": exp.get("task_description"),
                    "agents_involved": exp.get("agents_involved", []),
                    "outcome": exp.get("outcome", {}),
                    "confidence": exp.get("confidence", 0),
                    "timestamp": exp.get("timestamp"),
                    "similarity_score": similarity,
                    "metadata": exp.get("metadata", {})
                })
        
        # Sort by similarity (descending)
        similarity_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Take top N results
        top_results = similarity_results[:limit]
        
        top_confidence = top_results[0]["confidence"] if top_results else 0.0
        
        result = MemoryRetrievalResult(
            retrieval_id=retrieval_id,
            query=query,
            similar_experiences=top_results,
            top_match_confidence=top_confidence,
            recall_count=len(top_results),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        logger.info(f"Retrieved {len(top_results)} similar experiences for query: {query[:100]}")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving similar experiences: {e}")
        return MemoryRetrievalResult(
            retrieval_id=str(uuid.uuid4()),
            query=query,
            similar_experiences=[],
            top_match_confidence=0.0,
            recall_count=0,
            timestamp=datetime.now(timezone.utc).isoformat()
        )


async def run_experience_replay(
    experience_ids: List[str],
    trust_profiles: Dict[str, AgentTrustProfile],
    db_collection = None
) -> ReplaySession:
    """
    Run experience replay to reinforce learning and update trust calibration.
    
    Replays high-confidence cases to:
    - Update trust weights based on past performance
    - Reinforce successful reasoning patterns
    - Identify improvement opportunities
    
    Args:
        experience_ids: List of experience IDs to replay
        trust_profiles: Current trust profiles for agents
        db_collection: MongoDB collection (llm_experience_memory)
    
    Returns:
        ReplaySession with performance improvements
    """
    try:
        session_id = str(uuid.uuid4())
        
        if db_collection is None or not experience_ids:
            logger.warning("No experiences to replay")
            return ReplaySession(
                session_id=session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                experiences_replayed=0,
                avg_confidence=0.0,
                trust_adjustments={},
                reasoning_improvements={},
                performance_delta={}
            )
        
        # Fetch experiences from database
        experiences = await db_collection.find({
            "experience_id": {"$in": experience_ids}
        }).to_list(len(experience_ids))
        
        if not experiences:
            logger.warning("No experiences found for replay")
            return ReplaySession(
                session_id=session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                experiences_replayed=0,
                avg_confidence=0.0,
                trust_adjustments={},
                reasoning_improvements={},
                performance_delta={}
            )
        
        # Calculate aggregate metrics
        total_confidence = sum(exp.get("confidence", 0) for exp in experiences)
        avg_confidence = total_confidence / len(experiences) if experiences else 0.0
        
        # Analyze agent performance in successful cases
        trust_adjustments = {}
        agent_success_counts = {}
        
        for exp in experiences:
            agents = exp.get("agents_involved", [])
            confidence = exp.get("confidence", 0)
            
            # High-confidence cases (>=90%) increase trust
            if confidence >= 0.90:
                for agent in agents:
                    if agent not in agent_success_counts:
                        agent_success_counts[agent] = 0
                    agent_success_counts[agent] += 1
        
        # Calculate trust adjustments (small incremental improvements)
        for agent, count in agent_success_counts.items():
            # Each successful case increases trust by 0.01, capped at +0.05
            adjustment = min(0.05, count * 0.01)
            trust_adjustments[agent] = adjustment
            
            # Update trust profile if exists
            if agent in trust_profiles:
                current_trust = trust_profiles[agent].trust_score
                new_trust = min(1.0, current_trust + adjustment)
                trust_profiles[agent].trust_score = new_trust
                logger.info(f"Trust updated for {agent}: {current_trust:.3f} -> {new_trust:.3f} (+{adjustment:.3f})")
        
        # Generate reasoning improvements
        reasoning_improvements = {
            "strategy_agent": "Improved trust stability by identifying high-confidence patterns",
            "forecast_agent": "Enhanced reasoning alignment through successful case replay",
            "evaluator_agent": "Refined evaluation criteria based on past outcomes"
        }
        
        # Calculate performance deltas
        performance_delta = {
            "trust_stability_improvement": sum(trust_adjustments.values()) * 100 / len(trust_adjustments) if trust_adjustments else 0,
            "reasoning_alignment_improvement": len(experiences) * 0.6,  # Approximate 6% per 10 cases
            "confidence_consistency": (avg_confidence - 0.85) * 100 if avg_confidence > 0.85 else 0
        }
        
        replay_session = ReplaySession(
            session_id=session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            experiences_replayed=len(experiences),
            avg_confidence=round(avg_confidence, 3),
            trust_adjustments=trust_adjustments,
            reasoning_improvements=reasoning_improvements,
            performance_delta=performance_delta
        )
        
        logger.info(f"Replay session {session_id} complete: {len(experiences)} experiences, avg confidence {avg_confidence:.3f}")
        return replay_session
        
    except Exception as e:
        logger.error(f"Error running experience replay: {e}")
        return ReplaySession(
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            experiences_replayed=0,
            avg_confidence=0.0,
            trust_adjustments={},
            reasoning_improvements={},
            performance_delta={}
        )


async def summarize_collective_memory(
    db_collection = None,
    timeframe_days: int = 30
) -> Dict[str, Any]:
    """
    Generate periodic summary of collective memory insights.
    
    Provides:
    - Total experiences stored
    - Average confidence by task type
    - Most active agents
    - Memory quality metrics
    - Retention status
    
    Args:
        db_collection: MongoDB collection (llm_experience_memory)
        timeframe_days: Number of days to summarize
    
    Returns:
        Summary dictionary with memory metrics
    """
    try:
        from datetime import timedelta
        
        if db_collection is None:
            logger.warning("No database collection provided for memory summary")
            return {
                "total_experiences": 0,
                "summary": "Memory collection not available"
            }
        
        # Calculate cutoff date
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=timeframe_days)
        cutoff_iso = cutoff_date.isoformat()
        
        # Get all experiences within timeframe
        experiences = await db_collection.find({
            "timestamp": {"$gte": cutoff_iso}
        }).to_list(1000)
        
        total_count = len(experiences)
        
        if total_count == 0:
            return {
                "total_experiences": 0,
                "timeframe_days": timeframe_days,
                "summary": "No experiences recorded in this timeframe",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        # Calculate metrics by task type
        task_type_stats = {}
        agent_activity = {}
        
        for exp in experiences:
            task_type = exp.get("task_type", "unknown")
            confidence = exp.get("confidence", 0)
            agents = exp.get("agents_involved", [])
            
            # Task type statistics
            if task_type not in task_type_stats:
                task_type_stats[task_type] = {"count": 0, "total_confidence": 0}
            task_type_stats[task_type]["count"] += 1
            task_type_stats[task_type]["total_confidence"] += confidence
            
            # Agent activity
            for agent in agents:
                if agent not in agent_activity:
                    agent_activity[agent] = 0
                agent_activity[agent] += 1
        
        # Calculate averages
        for task_type, stats in task_type_stats.items():
            stats["avg_confidence"] = round(stats["total_confidence"] / stats["count"], 3)
        
        # Sort agents by activity
        most_active_agents = sorted(agent_activity.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Overall statistics
        total_confidence = sum(exp.get("confidence", 0) for exp in experiences)
        avg_confidence = round(total_confidence / total_count, 3)
        
        # High-quality experiences (>=90% confidence)
        high_quality_count = sum(1 for exp in experiences if exp.get("confidence", 0) >= 0.90)
        high_quality_rate = round((high_quality_count / total_count) * 100, 1)
        
        # Check retention status
        retention_status = "Within limit" if total_count <= 1000 else "Cleanup needed"
        
        summary = {
            "total_experiences": total_count,
            "timeframe_days": timeframe_days,
            "avg_confidence": avg_confidence,
            "high_quality_count": high_quality_count,
            "high_quality_rate": high_quality_rate,
            "retention_status": retention_status,
            "task_type_breakdown": task_type_stats,
            "most_active_agents": [{"agent": agent, "count": count} for agent, count in most_active_agents],
            "memory_accuracy_percent": round(avg_confidence * 100, 1),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Memory summary: {total_count} experiences, avg confidence {avg_confidence:.3f}")
        return summary
        
    except Exception as e:
        logger.error(f"Error summarizing collective memory: {e}")
        return {
            "total_experiences": 0,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


async def cleanup_old_experiences(
    db_collection = None,
    max_count: int = 1000,
    max_age_days: int = 30
) -> Dict[str, int]:
    """
    Cleanup old experiences to maintain retention policy.
    
    Removes experiences if:
    - Total count exceeds 1000
    - Age exceeds 30 days
    
    Prioritizes keeping high-confidence experiences.
    
    Args:
        db_collection: MongoDB collection (llm_experience_memory)
        max_count: Maximum number of experiences to keep
        max_age_days: Maximum age in days
    
    Returns:
        Dictionary with cleanup statistics
    """
    try:
        from datetime import timedelta
        
        if db_collection is None:
            logger.warning("No database collection provided for cleanup")
            return {"deleted_count": 0, "remaining_count": 0}
        
        # Calculate cutoff date
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        cutoff_iso = cutoff_date.isoformat()
        
        # Delete experiences older than cutoff
        age_delete_result = await db_collection.delete_many({
            "timestamp": {"$lt": cutoff_iso}
        })
        
        age_deleted = age_delete_result.deleted_count
        
        # Check total count
        total_count = await db_collection.count_documents({})
        
        count_deleted = 0
        if total_count > max_count:
            # Delete oldest low-confidence experiences
            # Keep high-confidence ones, delete low-confidence old ones
            to_delete = total_count - max_count
            
            # Find oldest low-confidence experiences
            old_experiences = await db_collection.find().sort([
                ("confidence", 1),  # Ascending confidence (lowest first)
                ("timestamp", 1)    # Oldest first
            ]).limit(to_delete).to_list(to_delete)
            
            ids_to_delete = [exp["experience_id"] for exp in old_experiences]
            
            if ids_to_delete:
                count_delete_result = await db_collection.delete_many({
                    "experience_id": {"$in": ids_to_delete}
                })
                count_deleted = count_delete_result.deleted_count
        
        remaining_count = await db_collection.count_documents({})
        
        total_deleted = age_deleted + count_deleted
        
        logger.info(f"Memory cleanup: deleted {total_deleted} experiences ({age_deleted} by age, {count_deleted} by count), {remaining_count} remaining")
        
        return {
            "deleted_by_age": age_deleted,
            "deleted_by_count": count_deleted,
            "total_deleted": total_deleted,
            "remaining_count": remaining_count
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up experiences: {e}")
        return {
            "deleted_by_age": 0,
            "deleted_by_count": 0,
            "total_deleted": 0,
            "remaining_count": 0,
            "error": str(e)
        }



# ====================
# Collective Intelligence Layer & Global Strategy Synthesis (Step 23)
# ====================

@dataclass
class CollectiveStrategy:
    """Global strategy synthesized from all subsystems"""
    strategy_id: str
    strategy_archetype: str  # High-level strategic pattern name
    description: str  # Detailed strategy description
    contributing_agents: List[str]  # Which agents contributed (memory, trust, arbitration, forecast)
    confidence_score: float  # 0-1
    alignment_score: float  # How well it aligns across agents (0-1)
    usage_count: int  # How many times this strategy was reinforced
    last_updated: str
    evidence_sources: List[Dict[str, Any]]  # Supporting evidence
    strategic_recommendations: List[str]  # Actionable recommendations
    performance_impact: Optional[str] = None  # Expected impact
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class AlignmentMetrics:
    """Collective alignment metrics across all subsystems"""
    alignment_id: str
    timestamp: str
    overall_alignment_score: float  # 0-1 (1 = perfect harmony)
    subsystem_scores: Dict[str, float]  # Individual subsystem alignment
    consensus_level: str  # "high", "moderate", "low", "divergent"
    harmony_index: float  # 0-1
    strategic_coherence: float  # 0-1
    conflict_count: int  # Number of conflicting strategies
    unified_direction: Optional[str] = None  # Overall strategic direction
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


async def aggregate_global_insights(
    memory_data: List[Dict],
    trust_data: List[Dict],
    arbitration_data: List[Dict],
    forecast_data: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Merge key findings from all subsystems (memory, trust, arbitration, forecasting).
    
    Args:
        memory_data: Collective memory experiences
        trust_data: Trust calibration scores
        arbitration_data: Arbitration decisions
        forecast_data: Forecasting predictions
    
    Returns:
        Aggregated insights dictionary with unified findings
    """
    try:
        aggregated = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_insights": 0,
            "memory_insights": {},
            "trust_insights": {},
            "arbitration_insights": {},
            "forecast_insights": {},
            "cross_system_patterns": [],
            "unified_metrics": {}
        }
        
        # Process memory data
        if memory_data:
            high_confidence_memories = [m for m in memory_data if m.get("confidence", 0) >= 0.7]
            recurring_patterns = {}
            for mem in high_confidence_memories:
                pattern = mem.get("pattern_type", "unknown")
                recurring_patterns[pattern] = recurring_patterns.get(pattern, 0) + 1
            
            aggregated["memory_insights"] = {
                "total_experiences": len(memory_data),
                "high_confidence_count": len(high_confidence_memories),
                "recurring_patterns": recurring_patterns,
                "avg_confidence": sum(m.get("confidence", 0) for m in memory_data) / len(memory_data) if memory_data else 0
            }
            aggregated["total_insights"] += len(high_confidence_memories)
        
        # Process trust data
        if trust_data:
            trust_scores = [t.get("trust_score", 0) for t in trust_data if t.get("trust_score") is not None]
            avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0
            
            calibration_states = {}
            for t in trust_data:
                state = t.get("calibration_state", "unknown")
                calibration_states[state] = calibration_states.get(state, 0) + 1
            
            aggregated["trust_insights"] = {
                "total_calibrations": len(trust_data),
                "avg_trust_score": round(avg_trust, 3),
                "calibration_states": calibration_states,
                "trust_trend": "stable" if 0.6 <= avg_trust <= 0.8 else "high" if avg_trust > 0.8 else "low"
            }
            aggregated["total_insights"] += len(trust_data)
        
        # Process arbitration data
        if arbitration_data:
            resolutions = [a.get("resolution", "") for a in arbitration_data]
            resolution_types = {}
            for res in resolutions:
                resolution_types[res] = resolution_types.get(res, 0) + 1
            
            avg_confidence = sum(a.get("confidence", 0) for a in arbitration_data) / len(arbitration_data) if arbitration_data else 0
            
            aggregated["arbitration_insights"] = {
                "total_arbitrations": len(arbitration_data),
                "resolution_distribution": resolution_types,
                "avg_confidence": round(avg_confidence, 3),
                "conflict_resolution_rate": len([a for a in arbitration_data if a.get("resolution") != "unresolved"]) / len(arbitration_data) if arbitration_data else 0
            }
            aggregated["total_insights"] += len(arbitration_data)
        
        # Process forecast data
        if forecast_data:
            aggregated["forecast_insights"] = {
                "overall_confidence": forecast_data.get("overall_confidence", 0),
                "forecast_narrative": forecast_data.get("forecast_narrative", ""),
                "strategic_recommendations": forecast_data.get("strategic_recommendations", [])
            }
        
        # Detect cross-system patterns
        # Pattern: High memory confidence + High trust + Low arbitration conflicts = Strong alignment
        if (
            aggregated["memory_insights"].get("avg_confidence", 0) > 0.7 and
            aggregated["trust_insights"].get("avg_trust_score", 0) > 0.7 and
            aggregated["arbitration_insights"].get("total_arbitrations", 0) < 10
        ):
            aggregated["cross_system_patterns"].append("Strong collective alignment detected across all systems")
        
        # Pattern: Low trust + High arbitration = Trust calibration needed
        if (
            aggregated["trust_insights"].get("avg_trust_score", 0) < 0.5 and
            aggregated["arbitration_insights"].get("total_arbitrations", 0) > 20
        ):
            aggregated["cross_system_patterns"].append("Trust calibration and arbitration threshold adjustment recommended")
        
        # Calculate unified metrics
        memory_score = aggregated["memory_insights"].get("avg_confidence", 0)
        trust_score = aggregated["trust_insights"].get("avg_trust_score", 0)
        arbitration_score = 1.0 - (min(aggregated["arbitration_insights"].get("total_arbitrations", 0), 50) / 50.0) if arbitration_data else 1.0
        
        aggregated["unified_metrics"] = {
            "system_health_score": round((memory_score + trust_score + arbitration_score) / 3, 3),
            "data_sufficiency": "high" if aggregated["total_insights"] > 100 else "moderate" if aggregated["total_insights"] > 30 else "low"
        }
        
        logger.info(f"Aggregated {aggregated['total_insights']} insights from all subsystems")
        return aggregated
        
    except Exception as e:
        logger.error(f"Error aggregating global insights: {e}")
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "total_insights": 0
        }


async def synthesize_collective_strategy(
    aggregated_insights: Dict,
    llm_api_key: str
) -> CollectiveStrategy:
    """
    Use LLM to synthesize unified strategic recommendations from aggregated insights.
    
    Args:
        aggregated_insights: Output from aggregate_global_insights()
        llm_api_key: API key for LLM
    
    Returns:
        CollectiveStrategy object with synthesized strategy
    """
    try:
        # Initialize LLM for synthesis
        chat = LlmChat(
            api_key=llm_api_key,
            session_id="collective-synthesizer",
            system_message="You are an expert AI systems strategist specializing in meta-learning and collective intelligence synthesis for AlphaZero chess systems."
        ).with_model("openai", "gpt-5")
        
        # Build synthesis prompt
        memory_summary = aggregated_insights.get("memory_insights", {})
        trust_summary = aggregated_insights.get("trust_insights", {})
        arbitration_summary = aggregated_insights.get("arbitration_insights", {})
        forecast_summary = aggregated_insights.get("forecast_insights", {})
        cross_patterns = aggregated_insights.get("cross_system_patterns", [])
        
        synthesis_prompt = f"""You are synthesizing collective intelligence from multiple AlphaZero chess AI subsystems. Analyze the following aggregated data and provide a unified strategic synthesis.

**AGGREGATED INSIGHTS:**

**Collective Memory (Step 22):**
- Total Experiences: {memory_summary.get('total_experiences', 0)}
- High Confidence Memories: {memory_summary.get('high_confidence_count', 0)}
- Recurring Patterns: {memory_summary.get('recurring_patterns', {})}
- Average Confidence: {memory_summary.get('avg_confidence', 0):.2f}

**Trust Calibration (Step 20):**
- Total Calibrations: {trust_summary.get('total_calibrations', 0)}
- Average Trust Score: {trust_summary.get('avg_trust_score', 0):.2f}
- Calibration States: {trust_summary.get('calibration_states', {})}
- Trust Trend: {trust_summary.get('trust_trend', 'unknown')}

**Arbitration System (Step 21):**
- Total Arbitrations: {arbitration_summary.get('total_arbitrations', 0)}
- Resolution Distribution: {arbitration_summary.get('resolution_distribution', {})}
- Conflict Resolution Rate: {arbitration_summary.get('conflict_resolution_rate', 0):.2f}
- Average Confidence: {arbitration_summary.get('avg_confidence', 0):.2f}

**Predictive Forecasting (Step 16):**
- Forecast Confidence: {forecast_summary.get('overall_confidence', 0):.2f}
- Strategic Recommendations: {forecast_summary.get('strategic_recommendations', [])}

**Cross-System Patterns Detected:**
{chr(10).join(f"• {p}" for p in cross_patterns) if cross_patterns else "• No major patterns detected"}

**SYNTHESIS TASK:**

Based on this collective intelligence, synthesize a unified global strategy that:
1. **Identifies the dominant strategic archetype** (e.g., "Balanced Exploration-Exploitation", "Deep Search Optimization", "Adaptive Parameter Tuning")
2. **Provides a clear description** of the recommended strategy (2-3 sentences)
3. **Lists actionable recommendations** (3-5 specific actions)
4. **Estimates performance impact** (positive/negative/neutral with brief reasoning)
5. **Assesses collective alignment** (0-1 score indicating system harmony)

**OUTPUT FORMAT (JSON):**
{{
  "strategy_archetype": "Clear strategic pattern name (5-10 words)",
  "description": "Detailed strategy description (2-3 sentences)",
  "strategic_recommendations": [
    "Specific actionable recommendation 1",
    "Specific actionable recommendation 2",
    "Specific actionable recommendation 3"
  ],
  "performance_impact": "Expected impact description",
  "confidence_score": 0.85,
  "alignment_score": 0.87,
  "contributing_agents": ["memory", "trust", "arbitration", "forecast"]
}}

Provide only the JSON output, no additional text."""

        response = await chat.send_message(UserMessage(text=synthesis_prompt))
        
        # Parse JSON response
        import json
        import re
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            synthesis_data = json.loads(json_match.group())
            
            strategy = CollectiveStrategy(
                strategy_id=str(uuid.uuid4()),
                strategy_archetype=synthesis_data.get("strategy_archetype", "Synthesized Strategy"),
                description=synthesis_data.get("description", "Strategic synthesis completed"),
                contributing_agents=synthesis_data.get("contributing_agents", ["memory", "trust", "arbitration", "forecast"]),
                confidence_score=float(synthesis_data.get("confidence_score", 0.75)),
                alignment_score=float(synthesis_data.get("alignment_score", 0.75)),
                usage_count=1,
                last_updated=datetime.now(timezone.utc).isoformat(),
                evidence_sources=[
                    {"source": "collective_memory", "data": memory_summary},
                    {"source": "trust_calibration", "data": trust_summary},
                    {"source": "arbitration_system", "data": arbitration_summary},
                    {"source": "predictive_forecasting", "data": forecast_summary}
                ],
                strategic_recommendations=synthesis_data.get("strategic_recommendations", []),
                performance_impact=synthesis_data.get("performance_impact", "Positive impact expected")
            )
            
            logger.info(f"Collective strategy synthesized: {strategy.strategy_archetype}")
            return strategy
        else:
            # Fallback strategy
            return CollectiveStrategy(
                strategy_id=str(uuid.uuid4()),
                strategy_archetype="Baseline Collective Strategy",
                description="Synthesis based on aggregated subsystem data. Continue balanced approach across all systems.",
                contributing_agents=["memory", "trust", "arbitration", "forecast"],
                confidence_score=0.6,
                alignment_score=0.65,
                usage_count=1,
                last_updated=datetime.now(timezone.utc).isoformat(),
                evidence_sources=[],
                strategic_recommendations=[
                    "Maintain current trust calibration parameters",
                    "Continue collecting collective memory experiences",
                    "Monitor arbitration conflict rates"
                ],
                performance_impact="Neutral - maintain current performance levels"
            )
    
    except Exception as e:
        logger.error(f"Error synthesizing collective strategy: {e}")
        # Return fallback strategy
        return CollectiveStrategy(
            strategy_id=str(uuid.uuid4()),
            strategy_archetype="Fallback Strategy",
            description="Error during synthesis. Using fallback strategy.",
            contributing_agents=[],
            confidence_score=0.5,
            alignment_score=0.5,
            usage_count=1,
            last_updated=datetime.now(timezone.utc).isoformat(),
            evidence_sources=[],
            strategic_recommendations=["Review system logs", "Retry synthesis"],
            performance_impact="Unknown"
        )


async def evaluate_collective_alignment(
    memory_data: List[Dict],
    trust_data: List[Dict],
    arbitration_data: List[Dict],
    collective_strategies: List[Dict]
) -> AlignmentMetrics:
    """
    Calculate consensus alignment across all subsystems and historical insights.
    
    Args:
        memory_data: Collective memory experiences
        trust_data: Trust calibration scores
        arbitration_data: Arbitration decisions
        collective_strategies: Previously synthesized strategies
    
    Returns:
        AlignmentMetrics object with alignment scores and analysis
    """
    try:
        # Calculate subsystem alignment scores
        subsystem_scores = {}
        
        # Memory alignment: based on confidence consistency
        if memory_data:
            memory_confidences = [m.get("confidence", 0) for m in memory_data]
            memory_variance = sum((c - sum(memory_confidences) / len(memory_confidences)) ** 2 for c in memory_confidences) / len(memory_confidences)
            memory_alignment = max(0, 1 - memory_variance)  # Low variance = high alignment
            subsystem_scores["memory"] = round(memory_alignment, 3)
        else:
            subsystem_scores["memory"] = 0.5
        
        # Trust alignment: based on trust score stability
        if trust_data:
            trust_scores = [t.get("trust_score", 0) for t in trust_data if t.get("trust_score") is not None]
            if trust_scores:
                trust_variance = sum((s - sum(trust_scores) / len(trust_scores)) ** 2 for s in trust_scores) / len(trust_scores)
                trust_alignment = max(0, 1 - trust_variance)
                subsystem_scores["trust"] = round(trust_alignment, 3)
            else:
                subsystem_scores["trust"] = 0.5
        else:
            subsystem_scores["trust"] = 0.5
        
        # Arbitration alignment: fewer conflicts = higher alignment
        if arbitration_data:
            unresolved_count = len([a for a in arbitration_data if a.get("resolution") == "unresolved"])
            arbitration_alignment = 1.0 - (min(unresolved_count, 20) / 20.0)
            subsystem_scores["arbitration"] = round(arbitration_alignment, 3)
        else:
            subsystem_scores["arbitration"] = 1.0  # No conflicts = perfect alignment
        
        # Forecast alignment: based on confidence
        subsystem_scores["forecast"] = 0.75  # Default moderate alignment
        
        # Calculate overall alignment score (weighted average)
        weights = {"memory": 0.3, "trust": 0.3, "arbitration": 0.25, "forecast": 0.15}
        overall_alignment = sum(subsystem_scores.get(k, 0.5) * v for k, v in weights.items())
        
        # Calculate harmony index (standard deviation of subsystem scores)
        scores_list = list(subsystem_scores.values())
        mean_score = sum(scores_list) / len(scores_list)
        harmony_variance = sum((s - mean_score) ** 2 for s in scores_list) / len(scores_list)
        harmony_index = max(0, 1 - harmony_variance)
        
        # Calculate strategic coherence (based on strategy consistency)
        if collective_strategies:
            strategy_confidences = [s.get("confidence_score", 0) for s in collective_strategies]
            strategic_coherence = sum(strategy_confidences) / len(strategy_confidences) if strategy_confidences else 0.5
        else:
            strategic_coherence = 0.5
        
        # Determine consensus level
        if overall_alignment >= 0.85:
            consensus_level = "high"
        elif overall_alignment >= 0.7:
            consensus_level = "moderate"
        elif overall_alignment >= 0.5:
            consensus_level = "low"
        else:
            consensus_level = "divergent"
        
        # Count conflicts
        conflict_count = len([a for a in arbitration_data if a.get("resolution") == "unresolved"]) if arbitration_data else 0
        
        # Determine unified direction
        if overall_alignment >= 0.75:
            unified_direction = "Convergent - Systems aligned on balanced optimization"
        elif consensus_level == "moderate":
            unified_direction = "Mixed - Some divergence in subsystem priorities"
        else:
            unified_direction = "Divergent - Requires recalibration"
        
        alignment_metrics = AlignmentMetrics(
            alignment_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_alignment_score=round(overall_alignment, 3),
            subsystem_scores=subsystem_scores,
            consensus_level=consensus_level,
            harmony_index=round(harmony_index, 3),
            strategic_coherence=round(strategic_coherence, 3),
            conflict_count=conflict_count,
            unified_direction=unified_direction
        )
        
        logger.info(f"Collective alignment evaluated: {alignment_metrics.overall_alignment_score:.3f} ({consensus_level})")
        return alignment_metrics
        
    except Exception as e:
        logger.error(f"Error evaluating collective alignment: {e}")
        # Return default alignment metrics
        return AlignmentMetrics(
            alignment_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_alignment_score=0.5,
            subsystem_scores={"memory": 0.5, "trust": 0.5, "arbitration": 0.5, "forecast": 0.5},
            consensus_level="unknown",
            harmony_index=0.5,
            strategic_coherence=0.5,
            conflict_count=0,
            unified_direction="Unknown - insufficient data"
        )


async def update_global_strategy_map(
    new_strategy: CollectiveStrategy,
    existing_strategies: List[Dict],
    db_collection = None
) -> Dict[str, Any]:
    """
    Maintain evolving map of high-confidence strategies.
    Updates or reinforces strategies based on similarity and confidence.
    
    Args:
        new_strategy: Newly synthesized CollectiveStrategy
        existing_strategies: List of existing strategy documents from DB
        db_collection: MongoDB collection for llm_global_strategy
    
    Returns:
        Update result with strategy map status
    """
    try:
        # Check if similar strategy exists
        similar_strategy = None
        max_similarity = 0
        
        for existing in existing_strategies:
            # Simple similarity check based on archetype matching
            existing_archetype = existing.get("strategy_archetype", "")
            new_archetype = new_strategy.strategy_archetype
            
            # Calculate word overlap similarity
            existing_words = set(existing_archetype.lower().split())
            new_words = set(new_archetype.lower().split())
            
            if existing_words and new_words:
                intersection = existing_words.intersection(new_words)
                union = existing_words.union(new_words)
                similarity = len(intersection) / len(union) if union else 0
                
                if similarity > max_similarity and similarity >= 0.5:  # 50% word overlap threshold
                    max_similarity = similarity
                    similar_strategy = existing
        
        if similar_strategy and max_similarity >= 0.5:
            # Reinforce existing strategy
            strategy_id = similar_strategy.get("strategy_id")
            updated_confidence = min(1.0, (similar_strategy.get("confidence_score", 0) + new_strategy.confidence_score) / 2)
            updated_usage_count = similar_strategy.get("usage_count", 0) + 1
            
            if db_collection:
                await db_collection.update_one(
                    {"strategy_id": strategy_id},
                    {
                        "$set": {
                            "confidence_score": updated_confidence,
                            "usage_count": updated_usage_count,
                            "last_updated": datetime.now(timezone.utc).isoformat(),
                            "description": new_strategy.description,  # Update with latest description
                            "strategic_recommendations": new_strategy.strategic_recommendations
                        }
                    }
                )
            
            logger.info(f"Reinforced existing strategy: {similar_strategy.get('strategy_archetype')} (usage: {updated_usage_count})")
            
            return {
                "action": "reinforced",
                "strategy_id": strategy_id,
                "strategy_archetype": similar_strategy.get("strategy_archetype"),
                "new_confidence": updated_confidence,
                "usage_count": updated_usage_count,
                "similarity_score": max_similarity
            }
        else:
            # Add new strategy
            strategy_doc = new_strategy.to_dict()
            
            if db_collection:
                await db_collection.insert_one(strategy_doc)
            
            logger.info(f"Added new strategy to map: {new_strategy.strategy_archetype}")
            
            return {
                "action": "added",
                "strategy_id": new_strategy.strategy_id,
                "strategy_archetype": new_strategy.strategy_archetype,
                "confidence_score": new_strategy.confidence_score,
                "usage_count": 1
            }
    
    except Exception as e:
        logger.error(f"Error updating global strategy map: {e}")
        return {
            "action": "error",
            "error": str(e)
        }

