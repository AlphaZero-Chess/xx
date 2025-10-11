"""
Test script for Ethical Consensus System (Step 26)

This script demonstrates the multi-agent consensus system without requiring
full goal generation infrastructure.
"""

import asyncio
import os
import sys
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone
from dotenv import load_dotenv

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Load environment
load_dotenv(backend_dir / '.env')

# Import consensus engine
from ethical_consensus import EthicalConsensusEngine


async def test_consensus():
    """Test the ethical consensus system"""
    
    print("\n" + "="*80)
    print("ETHICAL CONSENSUS SYSTEM TEST - STEP 26")
    print("="*80 + "\n")
    
    # Connect to MongoDB
    mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
    client = AsyncIOMotorClient(mongo_url)
    db = client[os.environ.get('DB_NAME', 'test_database')]
    
    print(f"✅ Connected to MongoDB: {mongo_url}\n")
    
    # Initialize consensus engine
    engine = EthicalConsensusEngine(db)
    
    if not engine.agents_available:
        print("❌ LLM agents not available - check EMERGENT_LLM_KEY")
        return
    
    print(f"✅ Consensus Engine initialized with {len(engine.agent_configs)} agents")
    print(f"   - Base consensus threshold: {engine.base_threshold:.0%}\n")
    
    # Print agent configuration
    print("🤖 Agent Configuration:")
    for agent in engine.agent_configs:
        print(f"   - {agent['name']} ({agent['provider']}/{agent['model']})")
        print(f"     Weight: {agent['weight']}, Specialty: {agent['specialty']}")
    print()
    
    # Create a test goal
    test_goal = {
        "goal_id": "test-consensus-001",
        "goal_type": "chess_performance",
        "description": "Increase MCTS simulations from 800 to 1000 for improved tactical accuracy",
        "rationale": "Analysis of recent games shows that 15% of tactical positions require deeper search. Increasing simulations will improve evaluation quality in complex middlegame positions.",
        "expected_outcomes": [
            "Improved position evaluation accuracy by 8-12%",
            "Better endgame performance in complex positions",
            "Slight increase in computation time (200ms avg per move)"
        ],
        "risks": [
            "Increased latency per move may affect user experience",
            "Higher computational resource usage"
        ],
        "is_critical": False,
        "stability_impact": 0.03,
        "strategic_alignment": 0.85,
        "ethical_alignment": 0.80,
        "confidence": 0.78,
        "auto_apply": True,
        "status": "proposed",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    print("📝 Test Goal:")
    print(f"   Type: {test_goal['goal_type']}")
    print(f"   Description: {test_goal['description']}")
    print(f"   Stability Impact: {test_goal['stability_impact']:.1%}")
    print(f"   Critical: {test_goal['is_critical']}\n")
    
    # Governance context
    governance_context = {
        "active_rules": [
            {
                "rule_id": "rule_transparency_001",
                "name": "Transparency Preservation",
                "description": "Decisions must include an explainable summary",
                "constraint_type": "transparency",
                "threshold": 0.0
            },
            {
                "rule_id": "rule_fairness_001",
                "name": "Fairness Parity",
                "description": "No biased evaluations (parity checks required)",
                "constraint_type": "fairness",
                "threshold": 0.15
            },
            {
                "rule_id": "rule_safety_001",
                "name": "Interpretability Safety",
                "description": "Do not trade interpretability for >10% raw performance gain",
                "constraint_type": "safety",
                "threshold": 0.10
            }
        ],
        "avg_alignment": 0.85,
        "recent_violations": 0
    }
    
    print("⚖️ Governance Context:")
    print(f"   Active Rules: {len(governance_context['active_rules'])}")
    print(f"   Average Alignment: {governance_context['avg_alignment']:.1%}")
    print(f"   Recent Violations: {governance_context['recent_violations']}\n")
    
    # Run consensus pipeline
    print("🚀 Initiating Multi-Agent Ethical Deliberation...")
    print("-" * 80 + "\n")
    
    try:
        # Step 1: Aggregate agent ethics
        print("📊 Step 1: Aggregating agent opinions (parallel queries)...")
        agent_opinions = await engine.aggregate_agent_ethics(test_goal, governance_context)
        
        if not agent_opinions:
            print("❌ Failed to gather agent opinions\n")
            return
        
        print(f"✅ Received {len(agent_opinions)} agent opinions\n")
        
        # Display agent opinions
        print("🗳️ Agent Opinions:")
        for opinion in agent_opinions:
            vote_emoji = "✅" if opinion.vote == "approve" else "❌" if opinion.vote == "reject" else "⚠️"
            print(f"\n   {vote_emoji} {opinion.agent_name} ({opinion.provider})")
            print(f"      Vote: {opinion.vote.upper()}")
            print(f"      Alignment: {opinion.alignment_score:.2f} | Confidence: {opinion.confidence:.2f}")
            print(f"      Opinion: {opinion.opinion[:150]}...")
            print(f"      Response Time: {opinion.response_time:.2f}s")
        
        print("\n" + "-" * 80 + "\n")
        
        # Step 2: Run consensus voting
        print("🔢 Step 2: Running consensus voting algorithm...")
        consensus_result = await engine.run_consensus_voting(agent_opinions, test_goal)
        
        print(f"✅ Consensus voting complete\n")
        
        # Display consensus results
        print("📈 Consensus Results:")
        print(f"   Final Decision: {consensus_result.final_decision.upper()}")
        print(f"   Consensus Reached: {'Yes ✅' if consensus_result.consensus_reached else 'No ❌'}")
        print(f"   Ethical Alignment Index (EAI): {consensus_result.agreement_score:.3f}")
        print(f"   Agreement Variance (σ): {consensus_result.agreement_variance:.3f}")
        print(f"   Consensus Threshold: {consensus_result.consensus_threshold:.1%}")
        print(f"\n   Vote Distribution:")
        print(f"      - Approve: {consensus_result.vote_distribution.get('approve', 0)}")
        print(f"      - Reject: {consensus_result.vote_distribution.get('reject', 0)}")
        print(f"      - Abstain: {consensus_result.vote_distribution.get('abstain', 0)}")
        
        if consensus_result.conflicts_detected:
            print(f"\n   ⚠️ Conflicts Detected:")
            for conflict in consensus_result.conflicts_detected:
                print(f"      - {conflict}")
        
        print(f"\n   Reasoning Summary:")
        print(f"      {consensus_result.reasoning_summary}")
        
        print("\n" + "-" * 80 + "\n")
        
        # Step 3: Resolve conflicts if any
        if consensus_result.conflicts_detected:
            print("🔧 Step 3: Resolving conflicts...")
            consensus_result, resolution = await engine.resolve_conflicts(consensus_result)
            print(f"✅ Conflict resolution: {resolution}\n")
        else:
            print("✅ Step 3: No conflicts to resolve\n")
        
        # Step 4: Store in database
        print("💾 Step 4: Storing consensus in database...")
        await db.llm_ethics_consensus_log.insert_one(consensus_result.to_dict())
        print(f"✅ Stored consensus: {consensus_result.consensus_id}\n")
        
        # Step 5: Refine governance rules
        print("⚙️ Step 5: Analyzing rule refinement opportunities...")
        refinements = await engine.refine_governance_rules(
            consensus_result, 
            governance_context['active_rules']
        )
        
        if refinements:
            print(f"✅ Generated {len(refinements)} rule refinements:")
            for refinement in refinements:
                print(f"\n   📊 {refinement.rule_name}")
                print(f"      Previous: {refinement.previous_weight:.3f} → New: {refinement.new_weight:.3f}")
                print(f"      Delta: {refinement.weight_delta:+.3f}")
                print(f"      Reason: {refinement.refinement_reason}")
                print(f"      Requires Approval: {'Yes' if refinement.requires_approval else 'No'}")
                
                # Store refinement
                await db.llm_rule_refinement.insert_one(refinement.to_dict())
        else:
            print("✅ No rule refinements needed at this time")
        
        print("\n" + "-" * 80 + "\n")
        
        # Step 6: Generate ethics report
        print("📋 Step 6: Generating ethics report...")
        report = await engine.generate_ethics_report(
            consensus_id=consensus_result.consensus_id,
            lookback_hours=24
        )
        
        print(f"✅ Ethics Report Generated\n")
        
        print("📊 System Health:")
        if 'system_health' in report:
            health = report['system_health']
            print(f"   Health Score: {health['health_score']:.1f}/100")
            print(f"   Status: {health['status'].upper()}")
            print(f"   Description: {health['description']}")
        
        if 'recommendations' in report and report['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"   - {rec}")
        
        print("\n" + "="*80)
        print("TEST COMPLETE ✅")
        print("="*80 + "\n")
        
        print("🌐 View Results:")
        print(f"   - API Status: http://localhost:8001/api/llm/ethics/status")
        print(f"   - Consensus History: http://localhost:8001/api/llm/ethics/history")
        print(f"   - Agent Details: http://localhost:8001/api/llm/ethics/agent-details/{consensus_result.consensus_id}")
        print(f"   - Ethics Report: http://localhost:8001/api/llm/ethics/report")
        print("\n   Frontend Dashboard: Navigate to 'Ethical Consensus' tab\n")
        
    except Exception as e:
        print(f"\n❌ Error during consensus: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        client.close()


if __name__ == "__main__":
    print("\n🔬 Testing Ethical Consensus System (Step 26)")
    print("   Multi-Agent Deliberation with GPT-5, Claude 4, Gemini 2.5\n")
    
    asyncio.run(test_consensus())
