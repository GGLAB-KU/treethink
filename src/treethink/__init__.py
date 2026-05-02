from treethink import (
    evaluators,
    finders,
    grading,
    inference_time_methods,
    methods,
    utils,
)
from treethink.evaluators import (
    IMPLEMENTED_ND,
    NODE_EVALUATORS,
    BaseEvaluator,
    JudgeEvaluator,
    LogprobEvaluator,
    REPLEvaluator,
    get_evaluator,
    get_evaluator_from_config,
)
from treethink.finders import (
    CHILD_FINDER_TYPE,
    CHILD_FINDERS,
    IMPLEMENTED_CF,
    BaseFinder,
    VLLMFinder,
    get_finder,
    get_finder_from_config,
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
from treethink.utils import (
    BaseArgs,
    EvaluatorArgs,
    FinderArgs,
    InferenceTimeArgs,
    LeanREPLArgs,
    ModelArgs,
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
    "finders",
    "extract_data",
    "extract_result",
    "get_finder",
    "get_finder_from_config",
    "get_evaluator",
    "get_evaluator_from_config",
    "get_inference_time_method",
    "get_total_child_num",
    "grading",
    "has_error_response",
    "inference_time_methods",
    "mcts",
    "methods",
    "node",
    "evaluators",
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
