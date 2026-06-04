import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from loguru import logger

from .graph import save_tree_to_txt
from .methods import METHOD_TYPE, Node
from .utils import TreeThinkArgs
from .utils.enums import BestAnswerReason


@dataclass
class TreeThinkOutputs:
    """TreeThink-native generation result.

    This object intentionally stays small and serialization-friendly. It holds
    the generated solution text(s) plus TreeThink-specific metadata, without
    inheriting from vLLM request types.
    """

    solution_text: str = ""
    outputs: List[str] = field(default_factory=list)
    method: Any = None
    graph_stats: Dict[str, Any] = field(default_factory=dict)
    checked_and_true: bool = False
    raw_output: Any = None
    generation_meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.solution_text and not self.outputs:
            self.outputs = [self.solution_text]
        elif self.outputs and not self.solution_text:
            self.solution_text = self.outputs[0]

    def solution_to_outputs(self, solution: Union[str, List[str]]):
        """Store solution text in the result object."""

        if isinstance(solution, str):
            solution = [solution]

        self.outputs = list(solution)
        self.solution_text = self.outputs[0] if self.outputs else ""

    @property
    def solution(self):
        return self.solution_text

    @property
    def has_solution(self) -> bool:
        return bool(self.solution_text)

    @property
    def is_checked_and_true(self) -> bool:
        return self.checked_and_true

    def to_dict(
        self,
        *,
        include_method: bool = False,
        include_raw_output: bool = False,
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "solution_text": self.solution_text,
            "outputs": list(self.outputs),
            "graph_stats": dict(self.graph_stats),
            "checked_and_true": self.checked_and_true,
            "generation_meta": dict(self.generation_meta),
        }
        if include_method:
            data["method"] = self.method
        if include_raw_output:
            data["raw_output"] = self.raw_output
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreeThinkOutputs":
        return cls(**data)

    def summary(self) -> Dict[str, Any]:
        return {
            "has_solution": self.has_solution,
            "solution_length": len(self.solution_text),
            "output_count": len(self.outputs),
            "checked_and_true": self.checked_and_true,
            "graph_stats_keys": list(self.graph_stats.keys()),
        }


class TreeThink:
    def __init__(self, method: METHOD_TYPE, treethink_args: TreeThinkArgs):
        logger.debug(
            f"Initializing TreeThink with method: {method} and treethink_args: {treethink_args}"
        )
        self.method = method
        self.treethink_args = treethink_args

        self.repl_runtime = self.treethink_args.build_repl_runtime()
        if self.repl_runtime.needs_repl:
            logger.debug("REPL runtime initialized.")
        else:
            logger.debug("REPL runtime not initialized.")

        logger.info("TreeThink initialized.")

    def generate(
        self, prompts: Union[str, List[str]], problem_id=None, *args, **kwargs
    ):
        # if a list of prompts, process them sequentially
        if not isinstance(prompts, str):
            all_request_outputs = []

            for p in prompts:
                # gather outputs
                all_request_outputs.append(
                    self.generate(prompts=p, problem_id=problem_id)
                )
                # reset method to make it ready for the next generation
                self.method.reset()

            return all_request_outputs

        # Set the root node based on settings
        self._set_root_node(prompts)

        _termination_fn = self.repl_runtime.build_termination_callback(
            self.method
        )

        # Simulate the search method
        self.method.simulate(
            expansion_count=self.treethink_args.expansion_count,
            timeout=self.treethink_args.timeout,
            remove_duplicate_children=self.treethink_args.remove_duplicate_children,
            termination_encountered_fn=_termination_fn,
        )

        # Create the final output
        generation_result = TreeThinkOutputs()

        # Check all terminated leaves via REPL
        _solution_found = False
        if (
            self.repl_runtime.paths_config.enabled
            and self.method.best_answer_reason
            != BestAnswerReason.CHECKED_AND_TRUE
        ):
            logger.trace("Checking terminated paths with REPL...")
            solution = self.repl_runtime.check_terminated_paths(self.method)

            if solution:
                _solution_found = True
                generation_result.checked_and_true = True

        # Fallback to best_answer whose reason may checked_and_true or compute
        if not _solution_found:
            solution = self.method.best_answer

        # Could be from terminination encountered or repl_terminated_paths
        if self.method.best_answer_reason == BestAnswerReason.CHECKED_AND_TRUE:
            generation_result.checked_and_true = True

        # Save the graph if specified
        if self.treethink_args.graph_path:
            save_tree_to_txt(
                self.method.root_node,
                self.treethink_args.graph_path,
                solution,
                problem_id=problem_id,
            )

        if self.treethink_args.store_method_class:
            generation_result.method = self.method

        if self.treethink_args.store_graph_stats:
            generation_result.graph_stats = self.method.get_stat_dict()

        generation_result.solution_to_outputs(solution)
        logger.success(f"Inference ended, result: {generation_result}")

        return generation_result

    async def async_generate(
        self, prompts: Union[str, List[str]], problem_id=None, *args, **kwargs
    ):
        """
        Async version of generate.
        """
        if not isinstance(prompts, str):
            # For lists, we should probably run concurrently?
            # For simplicity, sequential await loop
            all_request_outputs = []
            for p in prompts:
                all_request_outputs.append(
                    await self.async_generate(prompts=p, problem_id=problem_id)
                )
                self.method.reset()
            return all_request_outputs

        # Set root
        self._set_root_node(prompts)

        # Termination function for early stopping when solution is found
        _termination_fn = self.repl_runtime.build_termination_callback(
            self.method,
            async_mode=True,
        )

        # Async Simulate
        # Check for async_simulate method first (preferred for MCTS)
        if hasattr(self.method, "async_simulate"):
            logger.debug("Using async_simulate method")
            await self.method.async_simulate(
                expansion_count=self.treethink_args.expansion_count,
                timeout=self.treethink_args.timeout,
                remove_duplicate_children=self.treethink_args.remove_duplicate_children,
                termination_encountered_fn=_termination_fn,
            )
        elif inspect.iscoroutinefunction(self.method.simulate):
            # Fallback to async simulate if available
            await self.method.simulate(
                expansion_count=self.treethink_args.expansion_count,
                timeout=self.treethink_args.timeout,
                remove_duplicate_children=self.treethink_args.remove_duplicate_children,
                termination_encountered_fn=_termination_fn,
            )
        else:
            # Run sync simulate in executor
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self.method.simulate(
                    expansion_count=self.treethink_args.expansion_count,
                    timeout=self.treethink_args.timeout,
                    remove_duplicate_children=self.treethink_args.remove_duplicate_children,
                    termination_encountered_fn=_termination_fn,
                ),
            )

        # Create Output
        generation_result = TreeThinkOutputs()

        # Check terminated paths with REPL
        _solution_found = False
        if (
            self.repl_runtime.paths_config.enabled
            and self.method.best_answer_reason
            != BestAnswerReason.CHECKED_AND_TRUE
        ):
            solution = await self.repl_runtime.async_check_terminated_paths(
                self.method
            )

            if solution:
                _solution_found = True
                generation_result.checked_and_true = True

        if not _solution_found:
            solution = self.method.best_answer

        if self.method.best_answer_reason == BestAnswerReason.CHECKED_AND_TRUE:
            generation_result.checked_and_true = True

        # Save graph (Sync I/O - run in executor)
        if self.treethink_args.graph_path:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: save_tree_to_txt(
                    self.method.root_node,
                    self.treethink_args.graph_path,
                    solution,
                    problem_id=problem_id,
                ),
            )

        if self.treethink_args.store_method_class:
            generation_result.method = self.method

        if self.treethink_args.store_graph_stats:
            generation_result.graph_stats = self.method.get_stat_dict()

        generation_result.solution_to_outputs(solution)
        # logger.success(f"Async Inference ended: {problem_id}")

        return generation_result

    def __str__(self) -> str:
        return f"TreeThink(model={self.treethink_args.model_name if hasattr(self.treethink_args, 'model_name') else 'unknown'}, method={self.method})"

    def __repr__(self) -> str:
        return str(self)

    def _set_root_node(self, text: str):
        _root_node = Node(
            text=text,
            max_children=self.treethink_args.max_children,
            exploration_weight=self.treethink_args.exploration_weight,
            termination_str=self.treethink_args.termination_str,
            win_value=0.0,
        )
        self.method.set_root_node(_root_node)
