"""
Test script to generate sample governance data for demonstration
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from motor.motor_asyncio import AsyncIOMotorClient
from goal_governor import AdaptiveGoalController
import os

async def generate_demo_data():
    """Generate sample goals for demonstration"""
    
    # Connect to MongoDB
    mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
    client = AsyncIOMotorClient(mongo_url)
    db = client[os.environ.get('DB_NAME', 'test_database')]
    
    # Initialize controller
    controller = AdaptiveGoalController(db)
    
    print("🎯 Generating demo governance data...")
    
    # Create sample performance trends
    performance_trends = {
        "lookback_hours": 24,
        "subsystem_metrics": {
            "llm": {"alignment_pct": 78.5, "trust_variance": 0.12, "win_rate": 0.0},
            "trust": {"alignment_pct": 82.0, "trust_variance": 0.08, "win_rate": 0.0},
            "training": {"alignment_pct": 72.0, "trust_variance": 0.05, "win_rate": 0.58}
        }
    }
    
    # Create emergent signals showing some issues
    emergent_signals = {
        "overall_alignment": 0.77,  # Below 0.80 target
        "trust_variance": 0.18,  # Above 0.15 threshold
        "system_health": 77.0
    }
    
    # Generate goals
    goals = await controller.generate_adaptive_goals(
        performance_trends=performance_trends,
        emergent_signals=emergent_signals
    )
    
    print(f"✅ Generated {len(goals)} adaptive goals")
    
    # Evaluate and store each goal
    for i, goal in enumerate(goals, 1):
        print(f"\n📋 Goal {i}: {goal.description[:60]}...")
        
        # Evaluate alignment
        evaluation = await controller.evaluate_goal_alignment(goal)
        print(f"   Alignment: {evaluation.overall_alignment:.2f}")
        print(f"   Status: {evaluation.approval_status}")
        
        # Check governance rules
        can_execute, reason = await controller.apply_governance_rules(goal, evaluation)
        print(f"   Can execute: {can_execute}")
        
        # Store goal
        goal_dict = goal.to_dict()
        goal_dict["evaluation"] = evaluation.to_dict()
        goal_dict["can_auto_execute"] = can_execute
        goal_dict["governance_reason"] = reason
        
        await db.llm_adaptive_goals.insert_one(goal_dict)
        
        # Record in governance log
        await controller.record_goal_outcome(
            goal=goal,
            evaluation=evaluation,
            executed=False,
            outcome={"status": "proposed", "can_auto_execute": can_execute}
        )
    
    # Generate report
    print("\n📊 Generating governance report...")
    report = await controller.generate_governance_report()
    
    print(f"\n✅ Demo data generation complete!")
    print(f"   Total goals: {len(goals)}")
    print(f"   System health: {report['system_health']['health_score']}%")
    print(f"   Governance logs: {report['total_governance_logs']}")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(generate_demo_data())
