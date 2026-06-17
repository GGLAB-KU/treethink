"""
Test script for Kimina Lean Server with infotree information.

Infotree Types:
- Infotree.original: Original infotree from Lean (most detailed)
- Infotree.full: Full infotree with all information
- Infotree.tactics: Only tactic-related information
- Infotree.substantive: Substantive information only
"""

from kimina_client import KiminaClient
from kimina_client.models import Infotree, SnippetStatus

# Simple and working proofs
mock_proofs = [  # 10. Commutative property
    """theorem test9 (a b : Nat) : a + b = b + a := by
  rw [Nat.add_comm]""",
]

client = KiminaClient()

response = client.check(
    snips=mock_proofs,
    timeout=60,
    batch_size=4,  # Send in groups of 4
    max_workers=2,  # 2 parallel workers
    show_progress=True,
    infotree=Infotree.original,  # Get Infotree information (options: full, tactics, original, substantive)
)

# Print results in detail
print(f"\n{'=' * 80}")
print(f"Total proofs checked: {len(response.results)}")
print(f"{'=' * 80}\n")

for i, result in enumerate(response.results):
    print(f"\n{'=' * 80}")
    print(f"Proof {i + 1}:")
    print(f"{'-' * 80}")
    print(
        f"Code: {mock_proofs[i][:100]}{'...' if len(mock_proofs[i]) > 100 else ''}"
    )

    # Analyze the result
    analysis = result.analyze()
    print(f"Status: {analysis.status.value}")
    print(f"Time: {result.time}s")

    # Check if there's a server/repl error
    if result.error:
        print(f"\n❌ Error: {result.error}")

    # Show Infotree information (only if response exists)
    if result.response and "infotree" in result.response:
        infotree = result.response["infotree"]
        print("\n📊 Infotree Information:")
        print(f"  - Type: {type(infotree)}")
        print("  - Available: Yes")
        if isinstance(infotree, dict):
            print(f"  - Keys: {list(infotree.keys())}")
        elif isinstance(infotree, list):
            print(f"  - Length: {len(infotree)}")
            if len(infotree) > 0:
                print(
                    f"  - First item keys: {list(infotree[0].keys()) if isinstance(infotree[0], dict) else 'N/A'}"
                )

    # Show messages (errors, warnings, etc.)
    if result.response and "messages" in result.response:
        messages = result.response.get("messages", [])
        if messages:
            print("\n💬 Messages:")
            for msg in messages:
                severity = msg.get("severity", "unknown")
                emoji = (
                    "❌"
                    if severity == "error"
                    else "⚠️"
                    if severity == "warning"
                    else "ℹ️"
                )
                print(f"  {emoji} Severity: {severity}")
                print(
                    f"    Position: line {msg.get('pos', {}).get('line', '?')}, col {msg.get('pos', {}).get('column', '?')}"
                )
                print(f"    Message: {msg.get('data', 'No message')[:200]}")

    # Show sorries if any
    if result.response and "sorries" in result.response:
        sorries = result.response.get("sorries", [])
        if sorries:
            print(f"\n⚠️  Sorries found: {len(sorries)}")
            for sorry in sorries[:3]:  # Show first 3
                print(
                    f"  - Line {sorry.get('pos', {}).get('line', '?')}: {sorry.get('goal', 'No goal')[:100]}"
                )

    # Success message
    if analysis.status == SnippetStatus.valid:
        print("\n✅ Proof verified successfully!")
    elif analysis.status == SnippetStatus.sorry:
        print("\n⚠️  Proof contains sorry")
    elif analysis.status == SnippetStatus.lean_error:
        print("\n❌ Lean error in proof")
    elif analysis.status == SnippetStatus.timeout_error:
        print("\n⏱️  Timeout error")
    else:
        print(f"\n❌ Error: {analysis.status.value}")

print(f"\n{'=' * 80}\n")
