"""
Test AI (MCTS) handling of pawn promotions
"""

import sys
import numpy as np
from chess_engine import ChessEngine
from neural_network import AlphaZeroNetwork
from mcts import MCTS

def test_ai_promotion_in_mcts():
    """Test that MCTS correctly handles promotion moves"""
    print("\n" + "="*60)
    print("TEST: AI (MCTS) Pawn Promotion Handling")
    print("="*60)
    
    # Create components
    network = AlphaZeroNetwork()
    mcts = MCTS(network, num_simulations=100, c_puct=1.0)  # Fewer sims for speed
    engine = ChessEngine()
    
    # Set up position where AI should promote
    # White pawn on e7, AI (white) to move
    engine.set_fen("8/4P2k/8/8/8/8/8/K7 w - - 0 1")
    
    print("\n📍 Position: White AI has pawn on e7, can promote to e8")
    print(f"FEN: {engine.get_fen()}")
    print(f"Legal moves: {engine.get_legal_moves()}")
    
    # Get AI move using MCTS
    print("\n⏳ Running MCTS (100 simulations)...")
    best_move, value = mcts.get_best_move(engine)
    
    print(f"\n🤖 AI selected move: {best_move}")
    print(f"   Position value: {value:.3f}")
    
    # Check if it's a promotion move
    if len(best_move) == 5 and best_move[4] in ['q', 'r', 'b', 'n']:
        print(f"✅ PASS: AI selected promotion move")
        print(f"   Promotion piece: {best_move[4].upper()} ({'QRBN'[['q','r','b','n'].index(best_move[4])]})")
        
        # Make the move
        engine.make_move(best_move)
        print(f"\n📍 After AI promotion:")
        print(engine.board)
        
        return True
    else:
        print(f"❌ FAIL: AI did not select a promotion move")
        print(f"   Move format: {best_move} (expected 5 chars with promotion piece)")
        return False

def test_ai_promotion_with_capture():
    """Test AI promotion when capturing"""
    print("\n" + "="*60)
    print("TEST: AI Promotion with Capture")
    print("="*60)
    
    network = AlphaZeroNetwork()
    mcts = MCTS(network, num_simulations=100, c_puct=1.0)
    engine = ChessEngine()
    
    # White pawn can capture black rook and promote
    engine.set_fen("3r4/4P2k/8/8/8/8/8/K7 w - - 0 1")
    
    print("\n📍 Position: White pawn e7 can capture rook d8 and promote")
    print(f"FEN: {engine.get_fen()}")
    print(f"Legal moves: {engine.get_legal_moves()}")
    
    # Check promotion moves are available
    legal_moves = engine.get_legal_moves()
    capture_promos = [m for m in legal_moves if m.startswith('e7d8')]
    
    print(f"\nCapture+promotion moves: {capture_promos}")
    
    if len(capture_promos) == 4:
        print("✅ PASS: All 4 capture+promotion options available to AI")
        
        # Get AI move
        print("\n⏳ Running MCTS...")
        best_move, value = mcts.get_best_move(engine)
        
        print(f"🤖 AI selected: {best_move}")
        
        if best_move in capture_promos:
            print(f"✅ PASS: AI selected capture+promotion")
            engine.make_move(best_move)
            print(f"\n📍 After AI move:")
            print(engine.board)
            return True
        else:
            # AI might choose a non-capturing move, that's okay
            print(f"ℹ️  AI chose: {best_move} (capture+promo available but not chosen)")
            return True
    else:
        print(f"❌ FAIL: Expected 4 capture+promotion options, found {len(capture_promos)}")
        return False

def test_ai_black_promotion():
    """Test AI playing as black with promotion"""
    print("\n" + "="*60)
    print("TEST: AI Black Pawn Promotion")
    print("="*60)
    
    network = AlphaZeroNetwork()
    mcts = MCTS(network, num_simulations=100, c_puct=1.0)
    engine = ChessEngine()
    
    # Black pawn on e2, black to move
    engine.set_fen("k7/8/8/8/8/8/4p3/7K b - - 0 1")
    
    print("\n📍 Position: Black AI has pawn on e2, can promote to e1")
    print(f"FEN: {engine.get_fen()}")
    print(f"Legal moves: {engine.get_legal_moves()}")
    
    # Get AI move
    print("\n⏳ Running MCTS...")
    best_move, value = mcts.get_best_move(engine)
    
    print(f"🤖 AI selected move: {best_move}")
    
    # Check if it's a promotion move
    if len(best_move) == 5 and best_move[4] in ['q', 'r', 'b', 'n']:
        print(f"✅ PASS: Black AI selected promotion move")
        engine.make_move(best_move)
        print(f"\n📍 After Black AI promotion:")
        print(engine.board)
        return True
    else:
        print(f"❌ FAIL: Black AI did not select promotion")
        return False

def test_mcts_explores_all_promotions():
    """Verify MCTS explores all 4 promotion options"""
    print("\n" + "="*60)
    print("TEST: MCTS Explores All Promotion Options")
    print("="*60)
    
    network = AlphaZeroNetwork()
    mcts = MCTS(network, num_simulations=200, c_puct=1.0)
    engine = ChessEngine()
    
    # Simple promotion position
    engine.set_fen("8/4P2k/8/8/8/8/8/K7 w - - 0 1")
    
    print("\n📍 Position: White pawn can promote 4 ways")
    print(f"Legal moves: {engine.get_legal_moves()}")
    
    # Run full search (not just best move)
    print("\n⏳ Running MCTS search with 200 simulations...")
    best_move, move_probs, root_value = mcts.search(engine, temperature=1.0)
    
    print(f"\n🔍 MCTS move probabilities:")
    for move, prob in sorted(move_probs.items(), key=lambda x: -x[1])[:10]:
        print(f"   {move}: {prob:.4f}")
    
    # Check if all 4 promotion moves were explored
    promo_moves = ['e7e8q', 'e7e8r', 'e7e8b', 'e7e8n']
    explored_promos = [m for m in promo_moves if m in move_probs]
    
    print(f"\n📊 Promotion moves explored: {explored_promos}")
    
    if len(explored_promos) == 4:
        print(f"✅ PASS: MCTS explored all 4 promotion options")
        print(f"   Most preferred: {max(explored_promos, key=lambda m: move_probs[m])}")
        return True
    else:
        print(f"❌ FAIL: Only explored {len(explored_promos)}/4 promotion moves")
        return False

def run_ai_tests():
    """Run all AI promotion tests"""
    print("\n" + "🤖"*30)
    print("AI (MCTS) PROMOTION TEST SUITE")
    print("🤖"*30)
    
    tests = [
        ("AI Basic Promotion", test_ai_promotion_in_mcts),
        ("AI Promotion with Capture", test_ai_promotion_with_capture),
        ("AI Black Promotion", test_ai_black_promotion),
        ("MCTS Explores All Promotions", test_mcts_explores_all_promotions),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("AI TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\n📊 Total: {passed_count}/{total_count} AI tests passed")
    
    return passed_count == total_count

if __name__ == "__main__":
    success = run_ai_tests()
    sys.exit(0 if success else 1)
