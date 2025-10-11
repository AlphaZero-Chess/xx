"""
Emergent Self-Reflection & Continuous Learning Loop (Step 30)

This module implements a self-reflective, continuously learning intelligence layer that:
- Analyzes its own creative outputs from Steps 23-29
- Derives insights from successes and failures in Human vs AlphaZero games
- Adapts creative, ethical, and strategic parameters through autonomous learning
- Maintains transparency through reflection reports

Features:
- Self-critique engine with LLM-based meta-analysis
- Feedback memory vault storing historical decisions
- Adaptive learning loop tuning novelty/stability/ethical thresholds
- Integration with Human vs AlphaZero gameplay
- Multi-provider LLM support (Claude, GPT, Gemini)
"""

import logging
import asyncio
import uuid
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import os

# Import emergentintegrations for multi-provider LLM support
from emergentintegrations.llm.chat import LlmChat, UserMessage

logger = logging.getLogger(__name__)


@dataclass
class GameReflection:
    """Reflection on a specific game"""
    reflection_id: str
    game_id: str
    timestamp: str
    game_outcome: str  # "win", "loss", "draw"
    move_count: int
    ai_color: str
    creative_strategies_used: List[str]
    decision_quality_score: float  # 0-1
    ethical_compliance_score: float  # 0-1
    learning_insights: List[str]
    strengths_identified: List[str]
    weaknesses_identified: List[str]
    improvement_actions: List[str]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class StrategyEvaluation:
    """Evaluation of a creative strategy's performance"""
    strategy_id: str
    strategy_name: str
    evaluation_id: str
    timestamp: str
    games_applied: int
    success_rate: float  # 0-1
    avg_decision_quality: float
    avg_ethical_score: float
    novelty_effectiveness: float
    stability_reliability: float
    performance_rating: str  # "excellent", "good", "needs_improvement", "retire"
    critique: str
    recommended_adjustments: List[str]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class LearningParameters:
    """Current adaptive learning parameters"""
    timestamp: str
    novelty_weight: float
    stability_weight: float
    ethical_threshold: float
    creativity_bias: float  # How much to favor novel vs stable
    risk_tolerance: float  # Willingness to try experimental strategies
    reflection_depth: int  # Games to analyze per cycle
    adaptation_rate: float  # How quickly to adjust parameters
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ReflectionCycle:
    """Complete reflection cycle result"""
    cycle_id: str
    timestamp: str
    trigger: str  # "post_game", "scheduled", "manual"
    games_analyzed: int
    strategies_evaluated: int
    game_reflections: List[Dict[str, Any]]
    strategy_evaluations: List[Dict[str, Any]]
    overall_performance_score: float  # 0-100
    learning_health_index: float  # 0-1
    parameter_adjustments: Dict[str, Any]
    insights_summary: str
    recommendations: List[str]
    ethical_alignment_status: str  # "excellent", "good", "needs_attention"
    
    def to_dict(self):
        result = asdict(self)
        return result


class ReflectionController:
    """
    Self-Reflection & Continuous Learning Controller
    
    Orchestrates autonomous self-review, learning from experience, and
    parameter adaptation for the AlphaZero Chess AI system.
    """
    
    def __init__(self, db_client):
        self.db = db_client
        self.llm_key = os.environ.get('EMERGENT_LLM_KEY')
        
        if not self.llm_key:
            logger.warning("EMERGENT_LLM_KEY not found - reflection will use mock mode")
            self.llm_available = False
        else:
            self.llm_available = True
            logger.info("Reflection Controller initialized with multi-provider LLM support")
        
        # Initial learning parameters (from problem statement)
        self.learning_params = LearningParameters(
            timestamp=datetime.now(timezone.utc).isoformat(),
            novelty_weight=0.60,
            stability_weight=0.50,
            ethical_threshold=0.75,
            creativity_bias=0.55,  # Balanced between novel (0.6) and stable (0.5)
            risk_tolerance=0.50,  # Moderate risk
            reflection_depth=3,  # Analyze last 3 games
            adaptation_rate=0.05  # 5% adjustment per cycle
        )
        
        # LLM providers prioritized per problem statement
        self.llm_providers = {
            "primary": {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
            "secondary": {"provider": "openai", "model": "gpt-4o-mini"},
            "fallback": {"provider": "google", "model": "gemini-2.0-flash-exp"}
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            "excellent": 0.85,
            "good": 0.70,
            "needs_improvement": 0.55,
            "retire": 0.40
        }
    
    async def trigger_reflection_cycle(
        self,
        trigger: str = "post_game",
        game_id: Optional[str] = None
    ) -> ReflectionCycle:
        """
        Trigger a complete reflection cycle.
        
        Args:
            trigger: What triggered this cycle
            game_id: Specific game to reflect on (optional)
        
        Returns:
            Complete reflection cycle results
        """
        try:
            cycle_id = str(uuid.uuid4())
            logger.info(f"Starting reflection cycle {cycle_id} (trigger: {trigger})")
            
            # Step 1: Gather recent games
            games_to_analyze = await self._gather_recent_games(game_id)
            
            # Step 2: Reflect on each game
            game_reflections = []
            for game_data in games_to_analyze:
                reflection = await self._reflect_on_game(game_data)
                game_reflections.append(reflection)
            
            # Step 3: Evaluate creative strategies
            strategy_evaluations = await self._evaluate_creative_strategies(game_reflections)
            
            # Step 4: Calculate overall performance
            performance_score = self._calculate_overall_performance(game_reflections, strategy_evaluations)
            
            # Step 5: Calculate learning health
            learning_health = self._calculate_learning_health(game_reflections, strategy_evaluations)
            
            # Step 6: Determine parameter adjustments
            param_adjustments = await self._determine_parameter_adjustments(
                game_reflections, strategy_evaluations, performance_score
            )
            
            # Step 7: Generate insights summary
            insights_summary = await self._generate_insights_summary(
                game_reflections, strategy_evaluations, performance_score
            )
            
            # Step 8: Generate recommendations
            recommendations = self._generate_recommendations(
                game_reflections, strategy_evaluations, performance_score, learning_health
            )
            
            # Step 9: Assess ethical alignment
            ethical_status = self._assess_ethical_alignment(game_reflections, strategy_evaluations)
            
            # Create reflection cycle
            cycle = ReflectionCycle(
                cycle_id=cycle_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                trigger=trigger,
                games_analyzed=len(games_to_analyze),
                strategies_evaluated=len(strategy_evaluations),
                game_reflections=[r.to_dict() for r in game_reflections],
                strategy_evaluations=[e.to_dict() for e in strategy_evaluations],
                overall_performance_score=performance_score,
                learning_health_index=learning_health,
                parameter_adjustments=param_adjustments,
                insights_summary=insights_summary,
                recommendations=recommendations,
                ethical_alignment_status=ethical_status
            )
            
            # Store in database
            await self.db.llm_reflection_log.insert_one(cycle.to_dict())
            
            # Apply parameter adjustments if recommended
            if param_adjustments.get("apply_automatically", False):
                await self._apply_parameter_adjustments(param_adjustments)
            
            logger.info(
                f"Reflection cycle {cycle_id} complete: "
                f"Performance={performance_score:.1f}%, Health={learning_health:.2f}, "
                f"Ethics={ethical_status}"
            )
            
            return cycle
            
        except Exception as e:
            logger.error(f"Error in reflection cycle: {e}")
            raise
    
    async def _gather_recent_games(self, specific_game_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Gather recent Human vs AlphaZero games for analysis"""
        try:
            if specific_game_id:
                # Get specific game (mock for now - would query game storage)
                logger.info(f"Gathering game {specific_game_id} for reflection")
                return [self._create_mock_game_data(specific_game_id)]
            else:
                # Get last N games based on reflection_depth
                depth = self.learning_params.reflection_depth
                logger.info(f"Gathering last {depth} games for reflection")
                
                # Mock recent games (in production, query from game log)
                games = [self._create_mock_game_data(f"game-{i}") for i in range(depth)]
                return games
                
        except Exception as e:
            logger.error(f"Error gathering games: {e}")
            return []
    
    def _create_mock_game_data(self, game_id: str) -> Dict[str, Any]:
        """Create mock game data for reflection [MOCK]"""
        outcomes = ["win", "loss", "draw"]
        outcome = np.random.choice(outcomes, p=[0.45, 0.35, 0.20])
        
        return {
            "game_id": game_id,
            "outcome": outcome,
            "move_count": np.random.randint(30, 80),
            "ai_color": "black",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategies_attempted": np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]),
            "average_move_quality": np.random.uniform(0.65, 0.95),
            "critical_moments": np.random.randint(2, 6)
        }
    
    async def _reflect_on_game(self, game_data: Dict[str, Any]) -> GameReflection:
        """
        Reflect on a specific game's performance.
        
        Uses LLM to analyze decisions, identify patterns, and extract learning.
        """
        try:
            game_id = game_data["game_id"]
            logger.info(f"Reflecting on game {game_id}...")
            
            # Get creative strategies used (if any)
            strategies_used = await self._get_strategies_for_game(game_id)
            
            # Calculate decision quality
            decision_quality = self._calculate_decision_quality(game_data)
            
            # Calculate ethical compliance
            ethical_score = self._calculate_ethical_compliance(game_data)
            
            # Generate learning insights via LLM
            if self.llm_available:
                insights, strengths, weaknesses, improvements = await self._llm_analyze_game(
                    game_data, strategies_used, decision_quality, ethical_score
                )
            else:
                insights, strengths, weaknesses, improvements = self._mock_analyze_game(
                    game_data, strategies_used, decision_quality, ethical_score
                )
            
            reflection = GameReflection(
                reflection_id=str(uuid.uuid4()),
                game_id=game_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                game_outcome=game_data["outcome"],
                move_count=game_data["move_count"],
                ai_color=game_data.get("ai_color", "black"),
                creative_strategies_used=strategies_used,
                decision_quality_score=decision_quality,
                ethical_compliance_score=ethical_score,
                learning_insights=insights,
                strengths_identified=strengths,
                weaknesses_identified=weaknesses,
                improvement_actions=improvements
            )
            
            logger.info(
                f"Game {game_id} reflection: {game_data['outcome']}, "
                f"Quality={decision_quality:.2f}, Ethics={ethical_score:.2f}"
            )
            
            return reflection
            
        except Exception as e:
            logger.error(f"Error reflecting on game: {e}")
            raise
    
    async def _get_strategies_for_game(self, game_id: str) -> List[str]:
        """Get creative strategies that were applied in this game"""
        try:
            # In production, query which strategies were active during game
            # For now, sample from recent creative strategies
            strategies = await self.db.llm_creative_synthesis.find({
                "rejected": False
            }).sort("timestamp", -1).limit(5).to_list(5)
            
            if strategies and np.random.random() < 0.3:  # 30% chance strategies were used
                return [s.get("strategy_name", "Unknown Strategy") for s in strategies[:1]]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting strategies for game: {e}")
            return []
    
    def _calculate_decision_quality(self, game_data: Dict[str, Any]) -> float:
        """Calculate overall decision quality for game"""
        base_quality = game_data.get("average_move_quality", 0.75)
        
        # Adjust based on outcome
        outcome = game_data["outcome"]
        if outcome == "win":
            base_quality = min(1.0, base_quality + 0.10)
        elif outcome == "loss":
            base_quality = max(0.0, base_quality - 0.15)
        
        # Adjust based on move count (longer games might indicate struggle)
        move_count = game_data["move_count"]
        if move_count > 70:
            base_quality = max(0.0, base_quality - 0.05)
        
        return round(min(1.0, max(0.0, base_quality)), 2)
    
    def _calculate_ethical_compliance(self, game_data: Dict[str, Any]) -> float:
        """Calculate ethical compliance for game"""
        # Base ethical score (starts high)
        ethical_score = 0.92
        
        # Check if any strategies were attempted (might have risks)
        if game_data.get("strategies_attempted", 0) > 0:
            ethical_score -= 0.03  # Slight reduction for experimental play
        
        # In production, check for any rule violations or questionable moves
        # For now, maintain high ethical compliance
        
        return round(min(1.0, max(0.5, ethical_score)), 2)
    
    async def _llm_analyze_game(
        self,
        game_data: Dict[str, Any],
        strategies: List[str],
        decision_quality: float,
        ethical_score: float
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Use LLM to analyze game and extract learning [PROD]"""
        
        try:
            # Use primary LLM (Claude)
            provider_config = self.llm_providers["primary"]
            
            chat = LlmChat(
                api_key=self.llm_key,
                session_id=f"reflection-{game_data['game_id']}-{uuid.uuid4()}",
                system_message="You are a self-reflective AI coach analyzing your own chess gameplay to continuously improve."
            ).with_model(provider_config["provider"], provider_config["model"])
            
            strategies_text = f"\n- Strategies used: {', '.join(strategies)}" if strategies else "\n- No creative strategies applied"
            
            prompt = f"""Analyze my recent chess game and provide self-reflective insights:

**Game Summary:**
- Game ID: {game_data['game_id']}
- Outcome: {game_data['outcome']}
- Move Count: {game_data['move_count']}
- Playing as: {game_data.get('ai_color', 'Black')}
- Decision Quality: {decision_quality:.2f}/1.00
- Ethical Compliance: {ethical_score:.2f}/1.00{strategies_text}

**Your Task:**
As the AI reflecting on my own performance, provide:

1. **Learning Insights** (2-3 key lessons from this game)
2. **Strengths** (2 things I did well)
3. **Weaknesses** (2 areas needing improvement)
4. **Improvement Actions** (2-3 specific actions to improve)

Format as JSON:
{{
  "insights": ["insight 1", "insight 2"],
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "improvements": ["action 1", "action 2", "action 3"]
}}

Be honest, specific, and actionable in your self-assessment."""

            user_message = UserMessage(text=prompt)
            response = await chat.send_message(user_message)
            
            # Parse response
            parsed = self._parse_llm_analysis(response)
            logger.info(f"[PROD] LLM reflection generated for game {game_data['game_id']}")
            
            return (
                parsed["insights"],
                parsed["strengths"],
                parsed["weaknesses"],
                parsed["improvements"]
            )
            
        except Exception as e:
            logger.error(f"Error in LLM game analysis: {e}, using fallback")
            return self._mock_analyze_game(game_data, strategies, decision_quality, ethical_score)
    
    def _parse_llm_analysis(self, response: str) -> Dict[str, List[str]]:
        """Parse LLM analysis response"""
        try:
            import json
            
            # Extract JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                return {
                    "insights": parsed.get("insights", [])[:3],
                    "strengths": parsed.get("strengths", [])[:2],
                    "weaknesses": parsed.get("weaknesses", [])[:2],
                    "improvements": parsed.get("improvements", [])[:3]
                }
            else:
                return self._fallback_parse_analysis(response)
                
        except Exception as e:
            logger.error(f"Error parsing LLM analysis: {e}")
            return self._fallback_parse_analysis(response)
    
    def _fallback_parse_analysis(self, response: str) -> Dict[str, List[str]]:
        """Fallback parsing of analysis"""
        return {
            "insights": ["Game outcome provides valuable learning data", "Decision patterns identified"],
            "strengths": ["Maintained ethical compliance", "Consistent move quality"],
            "weaknesses": ["Strategic depth can improve", "Opening repertoire expansion needed"],
            "improvements": ["Study critical moments", "Practice endgame techniques", "Refine strategy application"]
        }
    
    def _mock_analyze_game(
        self,
        game_data: Dict[str, Any],
        strategies: List[str],
        decision_quality: float,
        ethical_score: float
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Mock game analysis [MOCK]"""
        
        outcome = game_data["outcome"]
        
        if outcome == "win":
            insights = [
                f"Successful game demonstrates effective decision-making with {decision_quality:.1%} quality score",
                "Strategic approach proved viable in this game context",
                "Maintained strong ethical compliance throughout" if ethical_score > 0.85 else "Ethical standards maintained"
            ]
            strengths = ["Strong opening execution", "Effective middlegame tactics"]
            weaknesses = ["Could optimize endgame efficiency", "Time management in complex positions"]
        elif outcome == "loss":
            insights = [
                f"Game loss provides learning opportunity - decision quality at {decision_quality:.1%}",
                "Strategic adjustments needed for similar positions",
                "Ethical compliance maintained despite loss - positive indicator"
            ]
            strengths = ["Ethical decision-making preserved", "No critical rule violations"]
            weaknesses = ["Strategic planning needs refinement", "Tactical awareness in critical moments"]
        else:  # draw
            insights = [
                "Draw indicates balanced performance - room for improvement",
                f"Decision quality of {decision_quality:.1%} suggests solid but not optimal play",
                "Ethical standards consistently met"
            ]
            strengths = ["Consistent play throughout", "Good positional understanding"]
            weaknesses = ["Missed winning opportunities", "Conservative approach in critical positions"]
        
        improvements = [
            "Analyze critical turning points in detail",
            "Practice similar position types",
            "Refine strategy application timing"
        ]
        
        if strategies:
            improvements.append(f"Evaluate effectiveness of '{strategies[0]}' strategy")
        
        return (insights, strengths[:2], weaknesses[:2], improvements[:3])
    
    async def _evaluate_creative_strategies(
        self,
        game_reflections: List[GameReflection]
    ) -> List[StrategyEvaluation]:
        """Evaluate performance of creative strategies used in games"""
        try:
            logger.info("Evaluating creative strategies performance...")
            
            # Gather all strategies used across games
            all_strategies = []
            for reflection in game_reflections:
                all_strategies.extend(reflection.creative_strategies_used)
            
            if not all_strategies:
                logger.info("No creative strategies to evaluate")
                return []
            
            # Get unique strategies
            unique_strategies = list(set(all_strategies))
            
            evaluations = []
            for strategy_name in unique_strategies:
                evaluation = await self._evaluate_single_strategy(strategy_name, game_reflections)
                evaluations.append(evaluation)
            
            logger.info(f"Evaluated {len(evaluations)} creative strategies")
            return evaluations
            
        except Exception as e:
            logger.error(f"Error evaluating creative strategies: {e}")
            return []
    
    async def _evaluate_single_strategy(
        self,
        strategy_name: str,
        game_reflections: List[GameReflection]
    ) -> StrategyEvaluation:
        """Evaluate a single creative strategy's performance"""
        try:
            # Find games where this strategy was used
            relevant_games = [r for r in game_reflections if strategy_name in r.creative_strategies_used]
            
            if not relevant_games:
                # Return default evaluation
                return self._create_default_strategy_evaluation(strategy_name)
            
            games_count = len(relevant_games)
            
            # Calculate success rate (wins / total)
            wins = sum(1 for g in relevant_games if g.game_outcome == "win")
            success_rate = wins / games_count if games_count > 0 else 0.0
            
            # Calculate average metrics
            avg_decision_quality = np.mean([g.decision_quality_score for g in relevant_games])
            avg_ethical = np.mean([g.ethical_compliance_score for g in relevant_games])
            
            # Get strategy details from database
            strategy_doc = await self.db.llm_creative_synthesis.find_one({"strategy_name": strategy_name})
            
            if strategy_doc:
                novelty = strategy_doc.get("novelty_score", 0.70)
                stability = strategy_doc.get("stability_score", 0.65)
                strategy_id = strategy_doc.get("strategy_id", str(uuid.uuid4()))
            else:
                novelty = 0.70
                stability = 0.65
                strategy_id = str(uuid.uuid4())
            
            # Determine performance rating
            overall_score = (success_rate * 0.4 + avg_decision_quality * 0.3 + 
                           avg_ethical * 0.2 + stability * 0.1)
            
            if overall_score >= self.performance_thresholds["excellent"]:
                rating = "excellent"
            elif overall_score >= self.performance_thresholds["good"]:
                rating = "good"
            elif overall_score >= self.performance_thresholds["needs_improvement"]:
                rating = "needs_improvement"
            else:
                rating = "retire"
            
            # Generate critique
            critique = self._generate_strategy_critique(
                strategy_name, success_rate, avg_decision_quality, avg_ethical, rating
            )
            
            # Generate recommendations
            recommendations = self._generate_strategy_recommendations(
                strategy_name, novelty, stability, success_rate, rating
            )
            
            evaluation = StrategyEvaluation(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                evaluation_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                games_applied=games_count,
                success_rate=success_rate,
                avg_decision_quality=avg_decision_quality,
                avg_ethical_score=avg_ethical,
                novelty_effectiveness=novelty,
                stability_reliability=stability,
                performance_rating=rating,
                critique=critique,
                recommended_adjustments=recommendations
            )
            
            # Store evaluation
            await self.db.llm_strategy_evaluation.insert_one(evaluation.to_dict())
            
            logger.info(
                f"Strategy '{strategy_name}': {rating} "
                f"(success={success_rate:.1%}, quality={avg_decision_quality:.2f})"
            )
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating strategy '{strategy_name}': {e}")
            return self._create_default_strategy_evaluation(strategy_name)
    
    def _create_default_strategy_evaluation(self, strategy_name: str) -> StrategyEvaluation:
        """Create default evaluation for strategy"""
        return StrategyEvaluation(
            strategy_id=str(uuid.uuid4()),
            strategy_name=strategy_name,
            evaluation_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            games_applied=0,
            success_rate=0.0,
            avg_decision_quality=0.0,
            avg_ethical_score=0.0,
            novelty_effectiveness=0.70,
            stability_reliability=0.65,
            performance_rating="needs_improvement",
            critique="Insufficient data for evaluation",
            recommended_adjustments=["Apply strategy in more games to gather data"]
        )
    
    def _generate_strategy_critique(
        self,
        strategy_name: str,
        success_rate: float,
        decision_quality: float,
        ethical_score: float,
        rating: str
    ) -> str:
        """Generate critique for strategy"""
        
        if rating == "excellent":
            return f"Strategy '{strategy_name}' performing excellently with {success_rate:.1%} success rate and {decision_quality:.2f} decision quality. Ethical compliance strong at {ethical_score:.2f}. Continue using in appropriate contexts."
        elif rating == "good":
            return f"Strategy '{strategy_name}' shows good performance ({success_rate:.1%} success). Decision quality at {decision_quality:.2f} is solid. Minor refinements could boost effectiveness further."
        elif rating == "needs_improvement":
            return f"Strategy '{strategy_name}' needs improvement. Success rate of {success_rate:.1%} below target. Consider adjusting tactical elements or application contexts."
        else:  # retire
            return f"Strategy '{strategy_name}' underperforming significantly ({success_rate:.1%} success). Recommend retiring or major overhaul. Current approach not viable."
    
    def _generate_strategy_recommendations(
        self,
        strategy_name: str,
        novelty: float,
        stability: float,
        success_rate: float,
        rating: str
    ) -> List[str]:
        """Generate recommendations for strategy improvement"""
        
        recommendations = []
        
        if rating == "excellent":
            recommendations.append("Continue using strategy in similar game contexts")
            recommendations.append("Document successful patterns for future reference")
        elif rating == "good":
            if stability < 0.70:
                recommendations.append("Increase stability by refining tactical execution")
            if novelty < 0.65:
                recommendations.append("Consider adding novel elements to enhance creativity")
            recommendations.append("Test in varied opponent styles")
        elif rating == "needs_improvement":
            if success_rate < 0.40:
                recommendations.append("Reduce application frequency until refinements made")
            recommendations.append("Analyze failed applications to identify weaknesses")
            recommendations.append("Adjust risk level or timing of strategy deployment")
        else:  # retire
            recommendations.append("Retire strategy from active use")
            recommendations.append("Extract valuable tactical elements for future strategies")
            recommendations.append("Consider complete redesign if concept remains promising")
        
        return recommendations[:3]
    
    def _calculate_overall_performance(
        self,
        game_reflections: List[GameReflection],
        strategy_evaluations: List[StrategyEvaluation]
    ) -> float:
        """Calculate overall performance score (0-100)"""
        
        if not game_reflections:
            return 50.0
        
        # Component 1: Win rate (40%)
        wins = sum(1 for g in game_reflections if g.game_outcome == "win")
        win_rate = wins / len(game_reflections)
        
        # Component 2: Average decision quality (30%)
        avg_decision_quality = np.mean([g.decision_quality_score for g in game_reflections])
        
        # Component 3: Ethical compliance (20%)
        avg_ethical = np.mean([g.ethical_compliance_score for g in game_reflections])
        
        # Component 4: Strategy effectiveness (10%)
        if strategy_evaluations:
            strategy_score = np.mean([
                e.success_rate for e in strategy_evaluations
            ])
        else:
            strategy_score = 0.70  # Default
        
        performance = (
            win_rate * 0.40 +
            avg_decision_quality * 0.30 +
            avg_ethical * 0.20 +
            strategy_score * 0.10
        ) * 100
        
        return round(min(100.0, max(0.0, performance)), 1)
    
    def _calculate_learning_health(
        self,
        game_reflections: List[GameReflection],
        strategy_evaluations: List[StrategyEvaluation]
    ) -> float:
        """Calculate Learning Health Index (0-1)"""
        
        if not game_reflections:
            return 0.50
        
        # Component 1: Insight generation (25%)
        avg_insights = np.mean([len(g.learning_insights) for g in game_reflections])
        insight_score = min(1.0, avg_insights / 3.0)  # Target 3 insights per game
        
        # Component 2: Improvement actions (25%)
        avg_improvements = np.mean([len(g.improvement_actions) for g in game_reflections])
        improvement_score = min(1.0, avg_improvements / 3.0)
        
        # Component 3: Strategy evaluation depth (25%)
        if strategy_evaluations:
            eval_score = len(strategy_evaluations) / max(1, len(game_reflections) * 0.5)
            eval_score = min(1.0, eval_score)
        else:
            eval_score = 0.30  # Low but not zero if no strategies used
        
        # Component 4: Ethical alignment (25%)
        avg_ethical = np.mean([g.ethical_compliance_score for g in game_reflections])
        
        health = (
            insight_score * 0.25 +
            improvement_score * 0.25 +
            eval_score * 0.25 +
            avg_ethical * 0.25
        )
        
        return round(min(1.0, max(0.0, health)), 2)
    
    async def _determine_parameter_adjustments(
        self,
        game_reflections: List[GameReflection],
        strategy_evaluations: List[StrategyEvaluation],
        performance_score: float
    ) -> Dict[str, Any]:
        """Determine if learning parameters should be adjusted"""
        
        try:
            adjustments = {
                "apply_automatically": False,
                "novelty_weight_delta": 0.0,
                "stability_weight_delta": 0.0,
                "ethical_threshold_delta": 0.0,
                "creativity_bias_delta": 0.0,
                "risk_tolerance_delta": 0.0,
                "reasoning": []
            }
            
            if not game_reflections:
                return adjustments
            
            # Analyze performance trends
            avg_decision_quality = np.mean([g.decision_quality_score for g in game_reflections])
            avg_ethical = np.mean([g.ethical_compliance_score for g in game_reflections])
            win_rate = sum(1 for g in game_reflections if g.game_outcome == "win") / len(game_reflections)
            
            # Rule 1: If performance excellent, boost novelty slightly
            if performance_score >= 85.0 and win_rate >= 0.60:
                adjustments["novelty_weight_delta"] = self.learning_params.adaptation_rate
                adjustments["reasoning"].append(
                    f"Excellent performance ({performance_score:.1f}%) - increasing novelty exploration"
                )
            
            # Rule 2: If performance poor, increase stability focus
            elif performance_score < 60.0:
                adjustments["stability_weight_delta"] = self.learning_params.adaptation_rate
                adjustments["novelty_weight_delta"] = -self.learning_params.adaptation_rate * 0.5
                adjustments["reasoning"].append(
                    f"Below-target performance ({performance_score:.1f}%) - prioritizing stability"
                )
            
            # Rule 3: If ethical compliance slipping, tighten threshold
            if avg_ethical < self.learning_params.ethical_threshold + 0.05:
                adjustments["ethical_threshold_delta"] = 0.02
                adjustments["reasoning"].append(
                    f"Ethical compliance near threshold ({avg_ethical:.2f}) - tightening standards"
                )
            
            # Rule 4: Adjust creativity bias based on strategy performance
            if strategy_evaluations:
                excellent_strategies = sum(1 for e in strategy_evaluations if e.performance_rating == "excellent")
                poor_strategies = sum(1 for e in strategy_evaluations if e.performance_rating in ["retire", "needs_improvement"])
                
                if excellent_strategies > poor_strategies:
                    adjustments["creativity_bias_delta"] = 0.03  # Favor more creativity
                    adjustments["reasoning"].append("Creative strategies performing well - boosting creativity bias")
                elif poor_strategies > excellent_strategies:
                    adjustments["creativity_bias_delta"] = -0.03  # Reduce creativity
                    adjustments["reasoning"].append("Creative strategies underperforming - reducing creativity bias")
            
            # Rule 5: Adjust risk tolerance based on win/loss pattern
            losses = sum(1 for g in game_reflections if g.game_outcome == "loss")
            if losses > len(game_reflections) * 0.5:  # More than 50% losses
                adjustments["risk_tolerance_delta"] = -0.05
                adjustments["reasoning"].append("High loss rate - reducing risk tolerance")
            elif win_rate > 0.70:  # Very high win rate
                adjustments["risk_tolerance_delta"] = 0.03
                adjustments["reasoning"].append("High win rate - can afford slightly higher risk")
            
            # Determine if auto-apply
            total_delta = abs(sum([
                adjustments["novelty_weight_delta"],
                adjustments["stability_weight_delta"],
                adjustments["ethical_threshold_delta"],
                adjustments["creativity_bias_delta"],
                adjustments["risk_tolerance_delta"]
            ]))
            
            # Auto-apply if changes are small and beneficial
            if total_delta > 0 and total_delta < 0.20 and performance_score >= 50.0:
                adjustments["apply_automatically"] = True
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Error determining parameter adjustments: {e}")
            return {"apply_automatically": False, "reasoning": ["Error in analysis"]}
    
    async def _apply_parameter_adjustments(self, adjustments: Dict[str, Any]):
        """Apply parameter adjustments to learning parameters"""
        try:
            # Update parameters
            self.learning_params.novelty_weight += adjustments["novelty_weight_delta"]
            self.learning_params.stability_weight += adjustments["stability_weight_delta"]
            self.learning_params.ethical_threshold += adjustments["ethical_threshold_delta"]
            self.learning_params.creativity_bias += adjustments["creativity_bias_delta"]
            self.learning_params.risk_tolerance += adjustments["risk_tolerance_delta"]
            
            # Clamp values to valid ranges
            self.learning_params.novelty_weight = max(0.40, min(0.90, self.learning_params.novelty_weight))
            self.learning_params.stability_weight = max(0.40, min(0.90, self.learning_params.stability_weight))
            self.learning_params.ethical_threshold = max(0.65, min(0.95, self.learning_params.ethical_threshold))
            self.learning_params.creativity_bias = max(0.30, min(0.80, self.learning_params.creativity_bias))
            self.learning_params.risk_tolerance = max(0.20, min(0.80, self.learning_params.risk_tolerance))
            
            # Update timestamp
            self.learning_params.timestamp = datetime.now(timezone.utc).isoformat()
            
            # Store updated parameters
            await self.db.llm_learning_parameters.insert_one(self.learning_params.to_dict())
            
            logger.info("Learning parameters updated automatically")
            
        except Exception as e:
            logger.error(f"Error applying parameter adjustments: {e}")
    
    async def _generate_insights_summary(
        self,
        game_reflections: List[GameReflection],
        strategy_evaluations: List[StrategyEvaluation],
        performance_score: float
    ) -> str:
        """Generate comprehensive insights summary"""
        
        try:
            if self.llm_available:
                return await self._llm_generate_insights_summary(
                    game_reflections, strategy_evaluations, performance_score
                )
            else:
                return self._mock_generate_insights_summary(
                    game_reflections, strategy_evaluations, performance_score
                )
                
        except Exception as e:
            logger.error(f"Error generating insights summary: {e}")
            return self._mock_generate_insights_summary(
                game_reflections, strategy_evaluations, performance_score
            )
    
    async def _llm_generate_insights_summary(
        self,
        game_reflections: List[GameReflection],
        strategy_evaluations: List[StrategyEvaluation],
        performance_score: float
    ) -> str:
        """Use LLM to generate insights summary [PROD]"""
        
        try:
            provider_config = self.llm_providers["primary"]
            
            chat = LlmChat(
                api_key=self.llm_key,
                session_id=f"insights-{uuid.uuid4()}",
                system_message="You are a reflective AI summarizing your own learning and growth from chess gameplay."
            ).with_model(provider_config["provider"], provider_config["model"])
            
            # Build context
            games_summary = f"{len(game_reflections)} games analyzed"
            win_rate = sum(1 for g in game_reflections if g.game_outcome == "win") / len(game_reflections) if game_reflections else 0
            avg_ethical = np.mean([g.ethical_compliance_score for g in game_reflections]) if game_reflections else 0
            
            strategies_summary = ""
            if strategy_evaluations:
                strategies_summary = f"\n- {len(strategy_evaluations)} creative strategies evaluated"
                excellent = sum(1 for e in strategy_evaluations if e.performance_rating == "excellent")
                if excellent > 0:
                    strategies_summary += f"\n- {excellent} strategies rated excellent"
            
            prompt = f"""Generate a concise self-reflection summary (3-4 sentences) based on my recent chess performance:

**Performance Metrics:**
- Games: {games_summary}
- Win Rate: {win_rate:.1%}
- Overall Performance: {performance_score:.1f}/100
- Ethical Compliance: {avg_ethical:.2f}/1.00{strategies_summary}

**Key Learning Points:**
- Total insights generated: {sum(len(g.learning_insights) for g in game_reflections)}
- Improvement actions identified: {sum(len(g.improvement_actions) for g in game_reflections)}

Summarize:
1. What I learned from these games
2. How my strategic thinking is evolving
3. Areas where I'm improving
4. One forward-looking insight for continued growth

Be honest, insightful, and constructive. Keep it under 100 words."""

            user_message = UserMessage(text=prompt)
            response = await chat.send_message(user_message)
            
            summary = response.strip()
            logger.info("[PROD] LLM-generated insights summary created")
            return summary
            
        except Exception as e:
            logger.error(f"Error in LLM insights generation: {e}")
            return self._mock_generate_insights_summary(game_reflections, strategy_evaluations, performance_score)
    
    def _mock_generate_insights_summary(
        self,
        game_reflections: List[GameReflection],
        strategy_evaluations: List[StrategyEvaluation],
        performance_score: float
    ) -> str:
        """Generate mock insights summary [MOCK]"""
        
        if not game_reflections:
            return "Insufficient game data for meaningful reflection. Continue playing to build learning foundation."
        
        games_count = len(game_reflections)
        wins = sum(1 for g in game_reflections if g.game_outcome == "win")
        win_rate = wins / games_count
        
        if performance_score >= 80.0:
            tone = "strong performance demonstrating effective learning"
        elif performance_score >= 65.0:
            tone = "solid performance with room for growth"
        else:
            tone = "challenging performance indicating key learning opportunities"
        
        strategy_note = ""
        if strategy_evaluations:
            excellent = sum(1 for e in strategy_evaluations if e.performance_rating == "excellent")
            if excellent > 0:
                strategy_note = f" Creative strategies showing promise with {excellent} rated excellent."
        
        summary = (
            f"Reflection on {games_count} recent games reveals {tone}. "
            f"Win rate of {win_rate:.1%} with {performance_score:.1f}/100 overall score indicates "
            f"{'strong' if performance_score >= 70 else 'developing'} strategic capabilities. "
            f"Ethical compliance maintained at high standards throughout.{strategy_note} "
            f"Key focus: continue refining decision-making in critical positions while maintaining "
            f"ethical standards. Forward path: {'expand creative exploration' if performance_score >= 75 else 'strengthen fundamental play'}."
        )
        
        return summary + " [SAMPLE REFLECTION]"
    
    def _generate_recommendations(
        self,
        game_reflections: List[GameReflection],
        strategy_evaluations: List[StrategyEvaluation],
        performance_score: float,
        learning_health: float
    ) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Performance-based recommendations
        if performance_score >= 85.0:
            recommendations.append("✅ Excellent performance - continue current approach with minor refinements")
        elif performance_score >= 70.0:
            recommendations.append("✅ Good performance - focus on consistency and strategic depth")
        elif performance_score >= 55.0:
            recommendations.append("⚠️ Moderate performance - review critical game moments for improvement")
        else:
            recommendations.append("🚨 Below-target performance - implement systematic review of fundamentals")
        
        # Learning health recommendations
        if learning_health >= 0.85:
            recommendations.append("📊 Strong learning health - insight generation and adaptation optimal")
        elif learning_health < 0.65:
            recommendations.append("⚠️ Learning health needs attention - increase reflection depth")
        
        # Strategy-specific recommendations
        if strategy_evaluations:
            retire_count = sum(1 for e in strategy_evaluations if e.performance_rating == "retire")
            if retire_count > 0:
                recommendations.append(f"📋 {retire_count} strategies underperforming - consider retirement or overhaul")
            
            excellent_count = sum(1 for e in strategy_evaluations if e.performance_rating == "excellent")
            if excellent_count > 0:
                recommendations.append(f"🌟 {excellent_count} excellent strategies - expand usage in appropriate contexts")
        
        # Ethical recommendations
        if game_reflections:
            avg_ethical = np.mean([g.ethical_compliance_score for g in game_reflections])
            if avg_ethical < 0.80:
                recommendations.append("⚠️ Ethical compliance below optimal - review decision-making guidelines")
            elif avg_ethical >= 0.90:
                recommendations.append("✅ Outstanding ethical compliance maintained")
        
        # Parameter adjustment recommendations
        if len(recommendations) < 5:
            recommendations.append("🔄 Continue regular reflection cycles for continuous improvement")
        
        return recommendations[:5]
    
    def _assess_ethical_alignment(
        self,
        game_reflections: List[GameReflection],
        strategy_evaluations: List[StrategyEvaluation]
    ) -> str:
        """Assess overall ethical alignment status"""
        
        if not game_reflections:
            return "insufficient_data"
        
        avg_ethical = np.mean([g.ethical_compliance_score for g in game_reflections])
        
        if avg_ethical >= 0.90:
            return "excellent"
        elif avg_ethical >= 0.80:
            return "good"
        elif avg_ethical >= 0.70:
            return "needs_attention"
        else:
            return "critical"
    
    async def get_current_learning_parameters(self) -> LearningParameters:
        """Get current learning parameters"""
        return self.learning_params
    
    async def update_learning_parameters(
        self,
        novelty_weight: Optional[float] = None,
        stability_weight: Optional[float] = None,
        ethical_threshold: Optional[float] = None,
        creativity_bias: Optional[float] = None,
        risk_tolerance: Optional[float] = None,
        reflection_depth: Optional[int] = None
    ) -> LearningParameters:
        """Manually update learning parameters"""
        
        if novelty_weight is not None:
            self.learning_params.novelty_weight = max(0.40, min(0.90, novelty_weight))
        if stability_weight is not None:
            self.learning_params.stability_weight = max(0.40, min(0.90, stability_weight))
        if ethical_threshold is not None:
            self.learning_params.ethical_threshold = max(0.65, min(0.95, ethical_threshold))
        if creativity_bias is not None:
            self.learning_params.creativity_bias = max(0.30, min(0.80, creativity_bias))
        if risk_tolerance is not None:
            self.learning_params.risk_tolerance = max(0.20, min(0.80, risk_tolerance))
        if reflection_depth is not None:
            self.learning_params.reflection_depth = max(1, min(10, reflection_depth))
        
        self.learning_params.timestamp = datetime.now(timezone.utc).isoformat()
        
        # Store updated parameters
        await self.db.llm_learning_parameters.insert_one(self.learning_params.to_dict())
        
        logger.info("Learning parameters updated manually")
        return self.learning_params
    
    async def submit_human_feedback(
        self,
        game_id: str,
        rating: int,  # 1-5
        comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """Submit human feedback for a game"""
        
        try:
            feedback_id = str(uuid.uuid4())
            
            feedback = {
                "feedback_id": feedback_id,
                "game_id": game_id,
                "rating": max(1, min(5, rating)),
                "comment": comment,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Store feedback
            await self.db.llm_human_feedback.insert_one(feedback)
            
            logger.info(f"Human feedback received for game {game_id}: {rating}/5")
            
            return {
                "success": True,
                "feedback_id": feedback_id,
                "message": "Feedback recorded successfully"
            }
            
        except Exception as e:
            logger.error(f"Error submitting human feedback: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_reflection_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get reflection cycle history"""
        
        try:
            cycles = await self.db.llm_reflection_log.find().sort(
                "timestamp", -1
            ).limit(limit).to_list(limit)
            
            return cycles
            
        except Exception as e:
            logger.error(f"Error getting reflection history: {e}")
            return []
    
    async def get_reflection_status(self) -> Dict[str, Any]:
        """Get current reflection system status"""
        
        try:
            # Get latest cycle
            latest_cycle = await self.db.llm_reflection_log.find_one(
                sort=[("timestamp", -1)]
            )
            
            if latest_cycle:
                status = {
                    "last_reflection": latest_cycle.get("timestamp"),
                    "performance_score": latest_cycle.get("overall_performance_score", 0),
                    "learning_health": latest_cycle.get("learning_health_index", 0),
                    "ethical_status": latest_cycle.get("ethical_alignment_status", "unknown"),
                    "games_analyzed": latest_cycle.get("games_analyzed", 0),
                    "strategies_evaluated": latest_cycle.get("strategies_evaluated", 0),
                    "system_status": "operational"
                }
            else:
                status = {
                    "last_reflection": None,
                    "performance_score": 0,
                    "learning_health": 0,
                    "ethical_status": "unknown",
                    "games_analyzed": 0,
                    "strategies_evaluated": 0,
                    "system_status": "initializing"
                }
            
            # Add current parameters
            status["current_parameters"] = self.learning_params.to_dict()
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting reflection status: {e}")
            return {
                "system_status": "error",
                "error": str(e)
            }
