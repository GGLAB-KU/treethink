import asyncio
import inspect
from typing import List, Union

import vllm
from loguru import logger

from .graph import save_tree_to_txt
from .methods import METHOD_TYPE, Node
from .utils import TreeThinkArgs
from .utils.enums import BestAnswerReason


class TreeThinkOutputs(vllm.RequestOutput):
    """vllm.RequestOutput subclassed version augmented for TreeThink.

    Other than holding the generation output, this result may hold `method` for
    further playing with the method's inner variables. See
    `treethink_args.py:TreeThinkArgs:store_method_class`.
    """

    def __init__(
        self,
        request_id=None,
        prompt=None,
        prompt_token_ids=None,
        prompt_logprobs=None,
        outputs=None,
        finished=None,
        metrics=None,
        lora_request=None,
        encoder_prompt=None,
        encoder_prompt_token_ids=None,
        num_cached_tokens=None,
        *,
        multi_modal_placeholders=None,
        method=None,
        graph_stats=None,
        checked_and_true=False,
    ):
        self.method = method
        self.graph_stats = graph_stats
        self.checked_and_true = checked_and_true

        super().__init__(
            request_id,
            prompt,
            prompt_token_ids,
            prompt_logprobs,
            outputs,
            finished,
            metrics,
            lora_request,
            encoder_prompt,
            encoder_prompt_token_ids,
            num_cached_tokens,
            multi_modal_placeholders=multi_modal_placeholders,
        )

    def solution_to_outputs(self, solution: Union[str, List[str]]):
        """Insert solution to self.outputs[i].text

        Args:
            solution (str): Found solution(s).

        Returns:
            None. Updates self.outputs with the solution.
        """

        if isinstance(solution, str):
            solution = [solution]

        outputs = []
        for i, s in enumerate(solution):
            outputs.append(
                vllm.CompletionOutput(
                    index=i,
                    text=s,
                    token_ids=None,
                    cumulative_logprob=None,
                    logprobs=None,
                )
            )

        self.outputs = outputs

    @property
    def solution(self):
        try:
            return self.outputs[0].text
        except IndexError:
            logger.error(f"Solution not found. self.outputs: {self.outputs}")


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
            self.repl_runtime.terminated_paths.enabled
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
            self.repl_runtime.terminated_paths.enabled
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
