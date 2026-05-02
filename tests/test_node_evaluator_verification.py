"""
Test script for Node Evaluators with Kimina Server integration.

This script tests:
1. REPLEvaluator - Basic proof verification
2. JudgeEvaluator - Advanced evaluation with infotree
"""

import sys

from kimina_client import KiminaClient
from kimina_client.models import Infotree

# Mock proofs for testing
VALID_PROOFS = [
    """import Mathlib

theorem test1 (a b : Nat) : a + b = b + a := by
  rw [Nat.add_comm]""",
    """import Mathlib

theorem test2 (n : Nat) : n + 0 = n := by
  rw [Nat.add_zero]""",
    """import Mathlib

theorem test3 (a b c : Nat) : a + (b + c) = (a + b) + c := by
  rw [Nat.add_assoc]""",
]

INVALID_PROOFS = [
    """import Mathlib

theorem test_invalid (a b : Nat) : a + b = a * b := by
  rw [Nat.add_comm]  -- This won't prove the theorem""",
    """import Mathlib

theorem test_sorry (n : Nat) : n + 1 = 1 + n := by
  sorry""",
]


class MockMethod:
    """Mock method for testing node evaluators."""

    def traverse_to_root(self, node: Node, include_root: bool = True) -> str:
        """Return the node's answer (proof) directly."""
        return node.answer


def create_mock_node(proof: str, level: int = 1) -> Node:
    """Create a mock node with a proof."""
    node = Node(answer=proof, level=level)
    return node


def test_repl_evaluator():
    """Test REPLEvaluator with valid and invalid proofs."""
    print("\n" + "=" * 80)
    print("Testing REPLEvaluator")
    print("=" * 80)

    # Initialize evaluator
    repl_args = LeanREPLArgs(
        lean_server_url="http://localhost:8000", timeout=30
    )
    evaluator = REPLEvaluator(repl_args=repl_args)
    method = MockMethod()

    print("\n📝 Testing VALID proofs...")
    print("-" * 80)

    for i, proof in enumerate(VALID_PROOFS):
        node = create_mock_node(proof)
        score = evaluator(node, method)

        print(f"\nProof {i + 1}:")
        print(f"  Code: {proof[:80]}...")
        print(f"  Score: {score}")
        print("  Expected: 1.0 (valid)")
        print(f"  Status: {'✅ PASS' if score == 1.0 else '❌ FAIL'}")

    print("\n" + "-" * 80)
    print("📝 Testing INVALID proofs...")
    print("-" * 80)

    for i, proof in enumerate(INVALID_PROOFS):
        node = create_mock_node(proof)
        score = evaluator(node, method)

        print(f"\nProof {i + 1}:")
        print(f"  Code: {proof[:80]}...")
        print(f"  Score: {score}")
        print("  Expected: 0.0 (invalid)")
        print(f"  Status: {'✅ PASS' if score == 0.0 else '❌ FAIL'}")


def test_kimina_client_direct():
    """Test KiminaClient directly to ensure infotree is returned."""
    print("\n" + "=" * 80)
    print("Testing KiminaClient Direct Integration")
    print("=" * 80)

    client = KiminaClient()

    # Test with a simple valid proof
    snips = [VALID_PROOFS[0]]

    print("\n📊 Checking proof with infotree...")
    print("-" * 80)

    response = client.check(
        snips=snips, timeout=30, infotree=Infotree.original, show_progress=False
    )

    result = response.results[0]

    print(f"\nProof: {snips[0][:100]}...")
    print("\nResult Analysis:")
    analysis = result.analyze()
    print(f"  Status: {analysis.status.value}")
    print(f"  Time: {result.time}s")

    if result.error:
        print(f"  Error: {result.error}")

    # Check infotree
    if result.response and "infotree" in result.response:
        infotree = result.response["infotree"]
        print("\n✅ Infotree retrieved successfully!")
        print(f"  Type: {type(infotree)}")

        if isinstance(infotree, dict):
            print(f"  Keys: {list(infotree.keys())}")
        elif isinstance(infotree, list):
            print(f"  Length: {len(infotree)}")
            if len(infotree) > 0 and isinstance(infotree[0], dict):
                print(f"  First item keys: {list(infotree[0].keys())}")
    else:
        print("\n❌ No infotree in response")

    # Check messages
    if result.response and "messages" in result.response:
        messages = result.response.get("messages", [])
        if messages:
            print(f"\n💬 Messages ({len(messages)}):")
            for msg in messages[:3]:
                severity = msg.get("severity", "unknown")
                print(f"  - {severity}: {msg.get('data', 'N/A')[:100]}")


def test_batch_verification():
    """Test batch verification with multiple proofs."""
    print("\n" + "=" * 80)
    print("Testing Batch Verification")
    print("=" * 80)

    client = KiminaClient()

    # Mix of valid and invalid proofs
    all_proofs = VALID_PROOFS + INVALID_PROOFS

    print(f"\n📦 Verifying {len(all_proofs)} proofs in batch...")
    print("-" * 80)

    response = client.check(
        snips=all_proofs,
        timeout=30,
        batch_size=4,
        max_workers=2,
        infotree=Infotree.original,
        show_progress=True,
    )

    print("\n📊 Results Summary:")
    print("-" * 80)

    valid_count = 0
    invalid_count = 0

    for i, result in enumerate(response.results):
        analysis = result.analyze()
        status = analysis.status.value

        is_valid = result.error is None and not any(
            msg.get("severity") == "error"
            for msg in result.response.get("messages", [])
        )

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1

        emoji = "✅" if is_valid else "❌"
        proof_type = "VALID" if i < len(VALID_PROOFS) else "INVALID"

        print(
            f"{emoji} Proof {i + 1} ({proof_type}): {status} - {result.time:.2f}s"
        )

    print("\n📈 Summary:")
    print(f"  Valid: {valid_count}/{len(all_proofs)}")
    print(f"  Invalid: {invalid_count}/{len(all_proofs)}")
    print(
        f"  Expected: {len(VALID_PROOFS)} valid, {len(INVALID_PROOFS)} invalid"
    )

    success = valid_count == len(VALID_PROOFS) and invalid_count == len(
        INVALID_PROOFS
    )
    print(f"\n  Overall: {'✅ PASS' if success else '❌ FAIL'}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Node Evaluator Verification Tests")
    print("Testing Kimina Server Integration")
    print("=" * 80)

    try:
        # Test 1: Direct KiminaClient integration
        test_kimina_client_direct()

        # Test 2: Batch verification
        test_batch_verification()

        # Test 3: REPLEvaluator
        test_repl_evaluator()

        print("\n" + "=" * 80)
        print("✅ All tests completed!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
