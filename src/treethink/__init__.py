from treethink import (
    child_finders,
    grading,
    inference_time_methods,
    methods,
    node_evaluators,
    utils,
)
from treethink.child_finders import (
    CHILD_FINDER_TYPE,
    CHILD_FINDERS,
    IMPLEMENTED_CF,
    BaseFinder,
    get_child_finder,
    get_child_finder_from_config,
    VLLMFinder,
)
from treethink.grading import (
    Lean4Client,
    analyze,
    batch_verify_proof,
    extract_data,
    has_error_response,
    process_batches,
    split_proof_header,
)
from treethink.inference_time_methods import (
    TreeThink,
    TreeThinkOutputs,
)
from treethink.methods import (
    BFTS,
    MCTS,
    BaseMethod,
    BeamSearch,
    Node,
    base_method,
    beam,
    bfts,
    get_inference_time_method,
    get_total_child_num,
    mcts,
    node,
)
from treethink.node_evaluators import (
    IMPLEMENTED_ND,
    NODE_EVALUATORS,
    BaseEvaluator,
    LogprobEvaluator,
    JudgeEvaluator,
    REPLEvaluator,
    get_node_evaluator,
    get_node_evaluator_from_config,
)
from treethink.utils import (
    BaseArgs,
    FinderArgs,
    InferenceTimeArgs,
    LeanREPLArgs,
    ModelArgs,
    EvaluatorArgs,
    SamplingArgs,
    extract_result,
)

__all__ = [
    "BFTS",
    "BaseArgs",
    "BaseFinder",
    "BaseMethod",
    "BaseEvaluator",
    "BeamSearch",
    "CHILD_FINDERS",
    "CHILD_FINDER_TYPE",
    "FinderArgs",
    "LogprobEvaluator",
    "IMPLEMENTED_CF",
    "IMPLEMENTED_ND",
    "TreeThinkOutputs",
    "InferenceTimeArgs",
    "TreeThink",
    "JudgeEvaluator",
    "Lean4Client",
    "LeanREPLArgs",
    "VLLMFinder",
    "MCTS",
    "ModelArgs",
    "NODE_EVALUATORS",
    "Node",
    "EvaluatorArgs",
    "REPLEvaluator",
    "SamplingArgs",
    "analyze",
    "base_method",
    "batch_verify_proof",
    "beam",
    "bfts",
    "child_finders",
    "extract_data",
    "extract_result",
    "get_child_finder",
    "get_child_finder_from_config",
    "get_node_evaluator",
    "get_node_evaluator_from_config",
    "get_inference_time_method",
    "get_total_child_num",
    "grading",
    "has_error_response",
    "inference_time_methods",
    "mcts",
    "methods",
    "node",
    "node_evaluators",
    "process_batches",
    "split_proof_header",
    "utils",
]

# Change recursion depth to avoid RecursionError
import resource
import sys

# TODO(burak): We can dynamically change recursion depth with expansion_count too but to we need actually need it?
resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
sys.setrecursionlimit(10**6)
