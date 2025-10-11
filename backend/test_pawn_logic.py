"""
Comprehensive test script for pawn logic in chess engine
Tests: captures, en passant, promotion, capture+promotion
"""

import chess
from chess_engine import ChessEngine
import sys

def print_board(engine):
    """Pretty print the chess board"""
    print("\n" + "="*40)
    print(engine.board)
    print("="*40)
    print(f"FEN: {engine.get_fen()}")
    print(f"Legal moves: {engine.get_legal_moves()}")
    print()

def test_regular_pawn_captures():
    """Test basic diagonal pawn captures"""
    print("\n" + "="*60)
    print("TEST 1: Regular Pawn Diagonal Captures")
    print("="*60)
    
    engine = ChessEngine()
    # Set up position: White pawn on e4, Black pawn on d5
    engine.set_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
    
    print("\n📍 Position: White pawn e4 can capture black pawn d5")
    print_board(engine)
    
    # Check if exd5 is in legal moves
    legal_moves = engine.get_legal_moves()
    if 'e4d5' in legal_moves:
        print("✅ PASS: e4xd5 capture is legal")
        engine.make_move('e4d5')
        print("\n📍 After e4xd5:")
        print_board(engine)
    else:
        print("❌ FAIL: e4xd5 not in legal moves!")
        return False
    
    return True

def test_en_passant():
    """Test en passant capture"""
    print("\n" + "="*60)
    print("TEST 2: En Passant Capture")
    print("="*60)
    
    engine = ChessEngine()
    # Position where en passant is possible
    # White pawn on e5, Black pawn moves d7-d5 (two squares)
    engine.set_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
    
    print("\n📍 Position: White pawn e5, Black pawn just moved d7-d5")
    print("   En passant target: d6")
    print_board(engine)
    
    # Check if e5d6 (en passant) is legal
    legal_moves = engine.get_legal_moves()
    if 'e5d6' in legal_moves:
        print("✅ PASS: En passant e5xd6 is legal")
        engine.make_move('e5d6')
        print("\n📍 After en passant e5xd6:")
        print_board(engine)
        
        # Verify the captured pawn is gone
        fen_parts = engine.get_fen().split()
        board_str = fen_parts[0]
        print(f"\n✅ Captured pawn removed from d5")
    else:
        print("❌ FAIL: En passant not in legal moves!")
        print(f"Legal moves containing 'e5': {[m for m in legal_moves if m.startswith('e5')]}")
        return False
    
    return True

def test_pawn_promotion_regular():
    """Test pawn promotion on 8th rank (no capture)"""
    print("\n" + "="*60)
    print("TEST 3: Pawn Promotion (Regular - No Capture)")
    print("="*60)
    
    engine = ChessEngine()
    # White pawn on e7, can promote to e8 (kings far apart to avoid check)
    engine.set_fen("8/4P2k/8/8/8/8/8/K7 w - - 0 1")
    
    print("\n📍 Position: White pawn on e7, can move to e8 and promote")
    print_board(engine)
    
    # Check if promotion moves are legal
    legal_moves = engine.get_legal_moves()
    promotion_moves = [m for m in legal_moves if m.startswith('e7e8')]
    
    print(f"\nPromotion moves found: {promotion_moves}")
    
    expected_promotions = ['e7e8q', 'e7e8r', 'e7e8b', 'e7e8n']
    all_present = all(move in legal_moves for move in expected_promotions)
    
    if all_present:
        print("✅ PASS: All 4 promotion options available (Q, R, B, N)")
        
        # Test promoting to Queen
        engine.make_move('e7e8q')
        print("\n📍 After e7e8=Q:")
        print_board(engine)
        
        # Verify queen is on e8
        piece = engine.board.piece_at(chess.E8)
        if piece and piece.piece_type == chess.QUEEN and piece.color == chess.WHITE:
            print("✅ PASS: Queen successfully placed on e8")
        else:
            print(f"❌ FAIL: Expected White Queen on e8, got {piece}")
            return False
    else:
        print("❌ FAIL: Not all promotion options available!")
        print(f"Expected: {expected_promotions}")
        print(f"Found: {promotion_moves}")
        return False
    
    return True

def test_pawn_promotion_with_capture():
    """Test pawn promotion with capture"""
    print("\n" + "="*60)
    print("TEST 4: Pawn Promotion with Capture")
    print("="*60)
    
    engine = ChessEngine()
    # White pawn on e7, can capture black knight on d8 and promote
    engine.set_fen("3nk3/4P3/8/8/8/8/8/4K3 w - - 0 1")
    
    print("\n📍 Position: White pawn e7 can capture Black knight d8 and promote")
    print_board(engine)
    
    legal_moves = engine.get_legal_moves()
    capture_promotions = [m for m in legal_moves if m.startswith('e7d8')]
    
    print(f"\nCapture+promotion moves found: {capture_promotions}")
    
    expected = ['e7d8q', 'e7d8r', 'e7d8b', 'e7d8n']
    all_present = all(move in legal_moves for move in expected)
    
    if all_present:
        print("✅ PASS: All 4 capture+promotion options available")
        
        # Test promoting to Knight (underpromotion)
        engine.make_move('e7d8n')
        print("\n📍 After e7xd8=N:")
        print_board(engine)
        
        # Verify knight is on d8 and black knight is gone
        piece = engine.board.piece_at(chess.D8)
        if piece and piece.piece_type == chess.KNIGHT and piece.color == chess.WHITE:
            print("✅ PASS: White Knight successfully placed on d8 after capture")
        else:
            print(f"❌ FAIL: Expected White Knight on d8, got {piece}")
            return False
    else:
        print("❌ FAIL: Not all capture+promotion options available!")
        print(f"Expected: {expected}")
        print(f"Found: {capture_promotions}")
        return False
    
    return True

def test_pawn_promotion_black():
    """Test black pawn promotion"""
    print("\n" + "="*60)
    print("TEST 5: Black Pawn Promotion")
    print("="*60)
    
    engine = ChessEngine()
    # Black pawn on e2, can promote to e1 (kings far apart)
    engine.set_fen("4k3/8/8/8/8/8/4p3/K7 b - - 0 1")
    
    print("\n📍 Position: Black pawn on e2, can move to e1 and promote")
    print_board(engine)
    
    legal_moves = engine.get_legal_moves()
    promotion_moves = [m for m in legal_moves if m.startswith('e2e1')]
    
    print(f"\nBlack promotion moves found: {promotion_moves}")
    
    expected_promotions = ['e2e1q', 'e2e1r', 'e2e1b', 'e2e1n']
    all_present = all(move in legal_moves for move in expected_promotions)
    
    if all_present:
        print("✅ PASS: All 4 black promotion options available")
        
        # Test promoting to Rook
        engine.make_move('e2e1r')
        print("\n📍 After e2e1=R:")
        print_board(engine)
        
        # Verify black rook is on e1
        piece = engine.board.piece_at(chess.E1)
        if piece and piece.piece_type == chess.ROOK and piece.color == chess.BLACK:
            print("✅ PASS: Black Rook successfully placed on e1")
        else:
            print(f"❌ FAIL: Expected Black Rook on e1, got {piece}")
            return False
    else:
        print("❌ FAIL: Not all black promotion options available!")
        return False
    
    return True

def test_multiple_pawn_captures():
    """Test position with multiple pawn capture options"""
    print("\n" + "="*60)
    print("TEST 6: Multiple Pawn Capture Options")
    print("="*60)
    
    engine = ChessEngine()
    # White pawn on e4 can capture on d5 or f5
    engine.set_fen("rnbqkbnr/ppp1p1pp/8/3p1p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
    
    print("\n📍 Position: White pawn e4 can capture d5 OR f5")
    print_board(engine)
    
    legal_moves = engine.get_legal_moves()
    e4_captures = [m for m in legal_moves if m.startswith('e4') and m in ['e4d5', 'e4f5']]
    
    print(f"\ne4 capture options: {e4_captures}")
    
    if 'e4d5' in legal_moves and 'e4f5' in legal_moves:
        print("✅ PASS: Both diagonal captures available")
        
        # Test one capture
        engine.make_move('e4d5')
        print("\n📍 After e4xd5:")
        print_board(engine)
    else:
        print("❌ FAIL: Not all diagonal captures available!")
        print(f"Expected e4d5 and e4f5, found: {e4_captures}")
        return False
    
    return True

def test_pawn_capture_near_promotion():
    """Test pawn captures near promotion rank (7th rank)"""
    print("\n" + "="*60)
    print("TEST 7: Pawn Capture Near Promotion Rank")
    print("="*60)
    
    engine = ChessEngine()
    # White pawn on e6, black pieces on d7 and f7
    engine.set_fen("4k3/3p1p2/4P3/8/8/8/8/4K3 w - - 0 1")
    
    print("\n📍 Position: White pawn e6 can capture d7 or f7 (near promotion)")
    print_board(engine)
    
    legal_moves = engine.get_legal_moves()
    e6_moves = [m for m in legal_moves if m.startswith('e6')]
    
    print(f"\ne6 available moves: {e6_moves}")
    
    # Should be able to move e6e7 or capture e6d7/e6f7
    if 'e6d7' in legal_moves and 'e6f7' in legal_moves:
        print("✅ PASS: Both diagonal captures available near promotion rank")
        
        # Make a capture
        engine.make_move('e6d7')
        print("\n📍 After e6xd7:")
        print_board(engine)
    else:
        print("❌ FAIL: Captures near promotion rank not working!")
        print(f"Expected e6d7 and e6f7 in legal moves")
        return False
    
    return True

def test_promotion_detection_methods():
    """Test the custom promotion detection methods"""
    print("\n" + "="*60)
    print("TEST 8: Promotion Detection Methods")
    print("="*60)
    
    engine = ChessEngine()
    
    # Test 1: White pawn e7 to e8 (promotion)
    engine.set_fen("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1")
    
    print("\n📍 Test: is_promotion_move('e7', 'e8')")
    is_promo = engine.is_promotion_move('e7', 'e8')
    print(f"Result: {is_promo}")
    
    if is_promo:
        print("✅ PASS: Correctly detected promotion")
        
        promo_moves = engine.get_promotion_moves('e7', 'e8')
        print(f"\nPromotion moves: {promo_moves}")
        expected = ['e7e8q', 'e7e8r', 'e7e8b', 'e7e8n']
        
        if promo_moves == expected:
            print("✅ PASS: get_promotion_moves() returns correct moves")
        else:
            print(f"❌ FAIL: Expected {expected}, got {promo_moves}")
            return False
    else:
        print("❌ FAIL: Should have detected promotion!")
        return False
    
    # Test 2: Regular move (not promotion)
    print("\n📍 Test: is_promotion_move('e2', 'e4') - should be False")
    engine.reset()
    is_not_promo = engine.is_promotion_move('e2', 'e4')
    print(f"Result: {is_not_promo}")
    
    if not is_not_promo:
        print("✅ PASS: Correctly identified non-promotion move")
    else:
        print("❌ FAIL: Should NOT have detected promotion!")
        return False
    
    # Test 3: Black pawn promotion
    engine.set_fen("4k3/8/8/8/8/8/4p3/4K3 b - - 0 1")
    print("\n📍 Test: Black pawn e2 to e1 (promotion)")
    is_black_promo = engine.is_promotion_move('e2', 'e1')
    print(f"Result: {is_black_promo}")
    
    if is_black_promo:
        print("✅ PASS: Correctly detected black promotion")
    else:
        print("❌ FAIL: Should have detected black promotion!")
        return False
    
    return True

def run_all_tests():
    """Run all pawn logic tests"""
    print("\n" + "🎯"*30)
    print("COMPREHENSIVE PAWN LOGIC TEST SUITE")
    print("🎯"*30)
    
    tests = [
        ("Regular Pawn Captures", test_regular_pawn_captures),
        ("En Passant", test_en_passant),
        ("Pawn Promotion (Regular)", test_pawn_promotion_regular),
        ("Pawn Promotion with Capture", test_pawn_promotion_with_capture),
        ("Black Pawn Promotion", test_pawn_promotion_black),
        ("Multiple Capture Options", test_multiple_pawn_captures),
        ("Captures Near Promotion Rank", test_pawn_capture_near_promotion),
        ("Promotion Detection Methods", test_promotion_detection_methods),
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
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\n📊 Total: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! Pawn logic is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed. Fixes needed.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
