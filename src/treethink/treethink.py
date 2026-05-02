from typing import List, Optional, Union
import asyncio
from functools import partial

import vllm

# from treethink.grading import Lean4Client
from kimina_client import KiminaClient, AsyncKiminaClient
from loguru import logger

from .graph import save_tree_to_txt
from .methods import METHOD_TYPE, Node
from .utils import InferenceTimeArgs


class TreeThinkOutputs(vllm.RequestOutput):
    """vllm.RequestOutput subclassed version augmented for TreeThink.

    Other than holding the generation output, this result may hold `method` for
    further playing with the method's inner variables. See
    `inftime_args.py:InferenceTimeArgs:store_method_class`.
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
    def __init__(self, method: METHOD_TYPE, inftime_args: InferenceTimeArgs):
        logger.debug(
            f"Initializing TreeThink with method: {method} and inftime_args: {inftime_args}"
        )
        self.method = method
        self.inftime_args = inftime_args

        # Frequent checks
        self._must_repl_paths = (
            self.inftime_args.termination_str
            and self.inftime_args.repl_args
            and self.inftime_args.repl_terminated_paths
        )
        self._must_repl_encountered = (
            self.inftime_args.termination_str
            and self.inftime_args.repl_args
            and self.inftime_args.repl_encountered_termination
        )

        # If REPL client is needed, start it with given REPL args
        if self._must_repl_encountered or self._must_repl_paths:
            self.client = KiminaClient(
                self.inftime_args.repl_args.lean_server_url
            )
            self.async_client = AsyncKiminaClient(
                self.inftime_args.repl_args.lean_server_url
            )
            logger.debug("REPL clients initialized.")
        else:
            self.client = None
            self.async_client = None
            logger.debug("REPL clients not initialized.")

        logger.info(f"TreeThink initialized.")

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

        _termination_fn = None
        if self._must_repl_encountered:
            _termination_fn = partial(
                self.method.repl_encountered_termination,
                client=self.client,
                timeout=self.inftime_args.repl_args.timeout,
                num_proc=self.inftime_args.repl_args.num_proc,
            )

        # Simulate the search method
        self.method.simulate(
            expansion_count=self.inftime_args.expansion_count,
            timeout=self.inftime_args.timeout,
            remove_duplicate_children=self.inftime_args.remove_duplicate_children,
            termination_encountered_fn=_termination_fn,
        )

        # Create the final output
        generation_result = TreeThinkOutputs()

        # Check all terminated leaves via REPL
        _solution_found = False
        if (
            self._must_repl_paths
            and self.method.best_answer_reason != "checked_and_true"
        ):
            logger.trace("Checking terminated paths with REPL...")
            solution = self.method.repl_terminated_paths(
                client=self.client,
                timeout=self.inftime_args.repl_args.timeout,
                num_proc=self.inftime_args.repl_args.num_proc,
                batch_size=self.inftime_args.repl_args.batch_size,
                max_repl=self.inftime_args.max_repl,
            )

            if solution:
                _solution_found = True
                generation_result.checked_and_true = True

        # Fallback to best_answer whose reason may checked_and_true or compute
        if not _solution_found:
            solution = self.method.best_answer

        # Could be from terminination encountered or repl_terminated_paths
        if self.method.best_answer_reason == "checked_and_true":
            generation_result.checked_and_true = True

        # Save the graph if specified
        if self.inftime_args.graph_path:
            save_tree_to_txt(
                self.method.root_node,
                self.inftime_args.graph_path,
                solution,
                problem_id=problem_id,
            )

        if self.inftime_args.store_method_class:
            generation_result.method = self.method

        if self.inftime_args.store_graph_stats:
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
        _termination_fn = None
        if self._must_repl_encountered:
            # Use async version if available
            if self.async_client and hasattr(
                self.method, "async_repl_encountered_termination"
            ):
                _termination_fn = partial(
                    self.method.async_repl_encountered_termination,
                    client=self.async_client,
                    timeout=self.inftime_args.repl_args.timeout,
                    num_proc=self.inftime_args.repl_args.num_proc,
                )
            else:
                # Fallback to sync version
                _termination_fn = partial(
                    self.method.repl_encountered_termination,
                    client=self.client,
                    timeout=self.inftime_args.repl_args.timeout,
                    num_proc=self.inftime_args.repl_args.num_proc,
                )

        # Async Simulate
        # Check for async_simulate method first (preferred for MCTS)
        if hasattr(self.method, "async_simulate"):
            logger.debug("Using async_simulate method")
            await self.method.async_simulate(
                expansion_count=self.inftime_args.expansion_count,
                timeout=self.inftime_args.timeout,
                remove_duplicate_children=self.inftime_args.remove_duplicate_children,
                termination_encountered_fn=_termination_fn,
            )
        elif asyncio.iscoroutinefunction(self.method.simulate):
            # Fallback to async simulate if available
            await self.method.simulate(
                expansion_count=self.inftime_args.expansion_count,
                timeout=self.inftime_args.timeout,
                remove_duplicate_children=self.inftime_args.remove_duplicate_children,
                termination_encountered_fn=_termination_fn,
            )
        else:
            # Run sync simulate in executor
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self.method.simulate(
                    expansion_count=self.inftime_args.expansion_count,
                    timeout=self.inftime_args.timeout,
                    remove_duplicate_children=self.inftime_args.remove_duplicate_children,
                    termination_encountered_fn=_termination_fn,
                ),
            )

        # Create Output
        generation_result = TreeThinkOutputs()

        # Check terminated paths with REPL
        _solution_found = False
        if (
            self._must_repl_paths
            and self.method.best_answer_reason != "checked_and_true"
        ):
            # Use async client if available, otherwise run sync client in executor
            if self.async_client:
                # Check if async_repl_terminated_paths exists, else fallback to sync in executor
                if hasattr(self.method, "async_repl_terminated_paths"):
                    solution = await self.method.async_repl_terminated_paths(
                        client=self.async_client,
                        timeout=self.inftime_args.repl_args.timeout,
                        num_proc=self.inftime_args.repl_args.num_proc,
                        batch_size=self.inftime_args.repl_args.batch_size,
                        max_repl=self.inftime_args.max_repl,
                    )
                else:
                    # Fallback: Run sync method with async_client in executor
                    loop = asyncio.get_running_loop()
                    solution = await loop.run_in_executor(
                        None,
                        lambda: self.method.repl_terminated_paths(
                            client=self.async_client,
                            timeout=self.inftime_args.repl_args.timeout,
                            num_proc=self.inftime_args.repl_args.num_proc,
                            batch_size=self.inftime_args.repl_args.batch_size,
                            max_repl=self.inftime_args.max_repl,
                        ),
                    )
            else:
                # No async client, run sync in executor
                loop = asyncio.get_running_loop()
                solution = await loop.run_in_executor(
                    None,
                    lambda: self.method.repl_terminated_paths(
                        client=self.client,
                        timeout=self.inftime_args.repl_args.timeout,
                        num_proc=self.inftime_args.repl_args.num_proc,
                        batch_size=self.inftime_args.repl_args.batch_size,
                        max_repl=self.inftime_args.max_repl,
                    ),
                )

            if solution:
                _solution_found = True
                generation_result.checked_and_true = True

        if not _solution_found:
            solution = self.method.best_answer

        if self.method.best_answer_reason == "checked_and_true":
            generation_result.checked_and_true = True

        # Save graph (Sync I/O - run in executor)
        if self.inftime_args.graph_path:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: save_tree_to_txt(
                    self.method.root_node,
                    self.inftime_args.graph_path,
                    solution,
                    problem_id=problem_id,
                ),
            )

        if self.inftime_args.store_method_class:
            generation_result.method = self.method

        if self.inftime_args.store_graph_stats:
            generation_result.graph_stats = self.method.get_stat_dict()

        generation_result.solution_to_outputs(solution)
        # logger.success(f"Async Inference ended: {problem_id}")

        return generation_result

    def __str__(self) -> str:
        return f"TreeThink(model={self.inftime_args.model_name if hasattr(self.inftime_args, 'model_name') else 'unknown'}, method={self.method})"

    def __repr__(self) -> str:
        return str(self)

    def _set_root_node(self, text: str):
        _root_node = Node(
            text=text,
            max_children=self.inftime_args.max_children,
            exploration_weight=self.inftime_args.exploration_weight,
            termination_str=self.inftime_args.termination_str,
            win_value=0.0,
        )
        self.method.set_root_node(_root_node)
