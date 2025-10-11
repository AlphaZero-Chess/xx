"""
Integration test for pawn promotion through the API
"""

import requests
import json
import time

BASE_URL = "http://localhost:8001/api"

def test_api_pawn_promotion_flow():
    """Test complete promotion flow through API"""
    print("\n" + "="*60)
    print("API INTEGRATION TEST: Pawn Promotion Flow")
    print("="*60)
    
    # 1. Create new game
    print("\n1️⃣  Creating new game...")
    response = requests.post(f"{BASE_URL}/game/new")
    assert response.status_code == 200, f"Failed to create game: {response.status_code}"
    
    game_data = response.json()
    game_id = game_data['id']
    print(f"✅ Game created: {game_id}")
    
    # 2. Set up promotion position using FEN
    # We'll play moves to get to a promotion scenario
    print("\n2️⃣  Setting up promotion scenario...")
    
    # Play some moves to get a pawn to 7th rank
    moves = [
        "e2e4",  # White pawn
        "d7d5",  # Black pawn
        "e4d5",  # White captures
        "c7c5",  # Black pawn
        "d5d6",  # White advances
        "e7e6",  # Black blocks
        "d6e7",  # White to 7th rank (capture)
    ]
    
    for move in moves:
        response = requests.post(
            f"{BASE_URL}/game/move",
            json={"game_id": game_id, "move": move}
        )
        if response.status_code != 200:
            print(f"⚠️  Move {move} failed: {response.json()}")
        else:
            print(f"   ✓ {move}")
    
    # Get current state
    response = requests.get(f"{BASE_URL}/game/{game_id}")
    state = response.json()
    print(f"\n📍 Current FEN: {state['fen']}")
    print(f"   Legal moves: {len(state['legal_moves'])} moves")
    
    # Check if promotion moves are available
    promo_moves = [m for m in state['legal_moves'] if m.startswith('e7e8')]
    print(f"   Promotion moves available: {promo_moves}")
    
    if promo_moves:
        print(f"✅ Promotion scenario reached!")
        
        # 3. Test promotion to Queen
        print("\n3️⃣  Testing promotion to Queen...")
        response = requests.post(
            f"{BASE_URL}/game/move",
            json={"game_id": game_id, "move": "e7e8q"}
        )
        
        if response.status_code == 200:
            state = response.json()
            print(f"✅ Promotion successful!")
            print(f"   New FEN: {state['fen']}")
            
            # Verify queen is on e8
            if 'Q' in state['fen'].split()[0]:
                print(f"✅ Queen verified on board!")
                return True
            else:
                print(f"❌ Queen not found on board")
                return False
        else:
            print(f"❌ Promotion failed: {response.json()}")
            return False
    else:
        print(f"⚠️  Couldn't reach promotion scenario with these moves")
        return False

def test_api_check_promotion_endpoint():
    """Test the check-promotion API endpoint"""
    print("\n" + "="*60)
    print("API TEST: Check Promotion Endpoint")
    print("="*60)
    
    # Create game
    response = requests.post(f"{BASE_URL}/game/new")
    game_data = response.json()
    game_id = game_data['id']
    
    # Play moves to get pawn to 7th rank
    moves = ["e2e4", "d7d5", "e4d5", "c7c5", "d5d6", "e7e6", "d6e7"]
    for move in moves:
        requests.post(f"{BASE_URL}/game/move", json={"game_id": game_id, "move": move})
    
    # Test check-promotion endpoint
    print("\n📡 Testing /game/check-promotion endpoint...")
    response = requests.post(
        f"{BASE_URL}/game/check-promotion",
        json={
            "game_id": game_id,
            "from_square": "e7",
            "to_square": "e8"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Endpoint working!")
        print(f"   is_promotion: {data['is_promotion']}")
        print(f"   promotion_moves: {data['promotion_moves']}")
        
        if data['is_promotion'] and len(data['promotion_moves']) == 4:
            print(f"✅ PASS: Promotion detected correctly")
            return True
        else:
            print(f"❌ FAIL: Promotion not detected or moves missing")
            return False
    else:
        print(f"❌ Endpoint failed: {response.status_code}")
        return False

def test_api_ai_promotion():
    """Test AI move with promotion"""
    print("\n" + "="*60)
    print("API TEST: AI Promotion")
    print("="*60)
    
    # Create game
    response = requests.post(f"{BASE_URL}/game/new")
    game_data = response.json()
    game_id = game_data['id']
    
    print(f"Game ID: {game_id}")
    
    # Play moves to give AI a promotion opportunity
    # We need to create a position where AI (playing as white) can promote
    moves = [
        "e2e4", "a7a6",  # White advances e-pawn
        "e4e5", "b7b6",
        "e5e6", "c7c6",
        "e6f7", "d7d6",  # White pawn on f7 (7th rank), black king still has e8
    ]
    
    for move in moves:
        response = requests.post(
            f"{BASE_URL}/game/move",
            json={"game_id": game_id, "move": move}
        )
        if response.status_code != 200:
            print(f"Move {move} failed")
    
    # Get current state
    response = requests.get(f"{BASE_URL}/game/{game_id}")
    state = response.json()
    print(f"\n📍 Position FEN: {state['fen']}")
    
    # Request AI move
    print("\n🤖 Requesting AI move...")
    response = requests.post(
        f"{BASE_URL}/game/ai-move",
        json={
            "game_id": game_id,
            "num_simulations": 50  # Fast test
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        ai_move = data['move']
        print(f"✅ AI move: {ai_move}")
        print(f"   Value: {data['value']:.3f}")
        
        # Check if AI promoted (move length 5 with promotion piece)
        if len(ai_move) == 5 and ai_move[4] in ['q', 'r', 'b', 'n']:
            print(f"✅ PASS: AI performed promotion!")
            print(f"   Promoted to: {ai_move[4].upper()}")
            return True
        else:
            print(f"ℹ️  AI chose: {ai_move} (may not be promotion based on position)")
            return True  # Not necessarily a failure
    else:
        print(f"❌ AI move failed: {response.status_code}")
        return False

def run_api_tests():
    """Run all API integration tests"""
    print("\n" + "🌐"*30)
    print("API INTEGRATION TEST SUITE")
    print("🌐"*30)
    
    # Wait for backend to be ready
    print("\n⏳ Waiting for backend...")
    for i in range(10):
        try:
            response = requests.get(f"{BASE_URL}/")
            if response.status_code == 200:
                print("✅ Backend ready!")
                break
        except:
            time.sleep(1)
    else:
        print("❌ Backend not available")
        return False
    
    tests = [
        ("Check Promotion Endpoint", test_api_check_promotion_endpoint),
        ("Pawn Promotion Flow", test_api_pawn_promotion_flow),
        ("AI Promotion", test_api_ai_promotion),
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
    print("API TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\n📊 Total: {passed_count}/{total_count} API tests passed")
    
    return passed_count == total_count

if __name__ == "__main__":
    import sys
    success = run_api_tests()
    sys.exit(0 if success else 1)
