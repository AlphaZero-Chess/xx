"""
Test script to verify LLM plugin offline/online behavior
"""

import asyncio
from llm_plugin import llm_plugin


async def test_plugin_functionality():
    """Test plugin in different scenarios"""
    
    print("=" * 60)
    print("LLM Plugin Offline/Online Test")
    print("=" * 60)
    
    # Test 1: Check plugin status
    print("\n1. Checking plugin availability...")
    available, message = llm_plugin.is_available()
    print(f"   Available: {available}")
    print(f"   Message: {message}")
    
    # Test 2: Evaluate a position
    print("\n2. Testing position evaluation...")
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    result = await llm_plugin.evaluate_position(fen, "Starting position")
    print(f"   Success: {result['success']}")
    print(f"   Evaluation: {result['evaluation'][:100]}...")
    print(f"   Plugin Status: {result['plugin_status']}")
    print(f"   Offline: {result['offline']}")
    
    # Test 3: Test internet connectivity check
    print("\n3. Testing internet connectivity check...")
    is_online = llm_plugin.check_internet_connectivity()
    print(f"   Internet Available: {is_online}")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    
    # Summary
    print("\n📊 Summary:")
    print(f"   - Plugin initialized: {'✅' if llm_plugin.chat else '❌'}")
    print(f"   - Internet connection: {'✅' if is_online else '❌'}")
    print(f"   - Plugin available: {'✅' if available else '❌'}")
    print(f"   - Evaluation working: {'✅' if result['success'] else '❌'}")
    
    if not available:
        print("\n⚠️  LLM plugin is offline. This is expected behavior.")
        print("   Core AlphaZero functionality will work without LLM.")
    else:
        print("\n✨ LLM plugin is fully operational!")


if __name__ == "__main__":
    asyncio.run(test_plugin_functionality())
