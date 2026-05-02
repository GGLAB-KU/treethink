"""
Test script for Kimina Lean Server with infotree information.

Infotree Types:
- Infotree.original: Original infotree from Lean (most detailed)
- Infotree.full: Full infotree with all information
- Infotree.tactics: Only tactic-related information
- Infotree.substantive: Substantive information only
"""

import json
from kimina_client import KiminaClient
from kimina_client.models import Infotree

# Load dataset
with open(
    "deepseek-ai_DeepSeek-Prover-V2-7B_LukeBailey181_putnambench-dataset_outputs.json",
    "r",
) as f:
    data = json.load(f)
proofs = [item["proof"] for item in data]

client = KiminaClient()

response = client.check(
    snips=proofs,
    timeout=60,
    batch_size=8,
    max_workers=10,
    show_progress=True,
    infotree=Infotree.original,
)

# Prepare JSON-serializable results
results_list = []
for i, result in enumerate(response.results):
    analysis = result.analyze()
    result_dict = {
        "proof_index": i,
        "proof": proofs[i] if i < len(proofs) else "",
        "status": analysis.status.value,
        "time": result.time,
        "error": result.error,
        "response": result.response,  # should be already serializable dict
    }
    results_list.append(result_dict)

output_json = {
    "input_file": "deepseek-ai_DeepSeek-Prover-V2-7B_LukeBailey181_putnambench-dataset_outputs.json",
    "total_proofs": len(response.results),
    "results": results_list,
}

with open("kimina_benchmark_results.json", "w") as out_f:
    json.dump(output_json, out_f, indent=2)

print(f"Results saved to kimina_benchmark_results.json")
