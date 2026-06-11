import math
import os
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import vllm
from loguru import logger
from vllm.lora.request import LoRARequest

from treethink.clients.cache import CachedClient, ProofCache
from treethink.clients.coq.rocq import RocqBatchClient
from treethink.clients.lean import (
    extract_data,
    split_proof_header,
)
from treethink.clients.lean.adapter import LeanClientAdapter
from treethink.methods import BaseMethod, Node
from treethink.utils import (
    ClientArgs,
    EvaluatorArgs,
    ModelArgs,
    SamplingArgs,
    calculate_logprobs,
    extract_result,
)

LLM_AS_JUDGE_SYSTEM_PROMPT = """
You are a LLM judge who assesses a student's solution. The given solution is not complete,
it requires your assessment of whether if the student is on the right track or not. 
You score the solution out of 20 and put your final score inside \\boxed{}.
Do NOT attempt to solve the problem, only provide a score out of 20 inside \\boxed{}.
"""
LLM_AS_JUDGE_SYSTEM_PROMPT_PAIRWISE = """
You are a LLM judge who assesses a student's solution. The given solution is not complete,
it requires your assessment of whether if the student is on the right track or not. 
You score the solution out of 20 and put your final score inside \\boxed{}.
Do NOT attempt to solve the problem, only provide a score out of 20 inside \\boxed{}.
"""


class BaseEvaluator(ABC):
    """Abstract base for all node scoring evaluators.

    An evaluator wraps a callable that scores nodes based on proof quality,
    model likelihood, or verification results.  Subclasses must implement
    ``__call__(self, node, method)`` which accepts a node or list of nodes
    and returns a list of float scores.

    To create a custom evaluator, subclass this and implement ``__call__``,
    then register in the ``EvaluatorType`` enum.
    """

    def __init__(self, name: str, *args, **kwargs):
        self.name = name

    def _format_str_value(self, value):
        if callable(value):
            return getattr(value, "__name__", value.__class__.__name__)
        return repr(value)

    def _str_fields(self):
        return [("name", self.name)]

    def __str__(self) -> str:
        fields = ", ".join(
            f"{name}={self._format_str_value(value)}"
            for name, value in self._str_fields()
        )
        return f"{self.__class__.__name__}({fields})"

    __repr__ = __str__

    @abstractmethod
    def __call__(
        self, node: Union[Node, List[Node]], method: BaseMethod
    ) -> List[float]:
        pass


class LogprobEvaluator(BaseEvaluator):
    """Returns cumulative log-probability score from vLLM output.

    Score is in the range (-\u221e, 0] and represents the cumulative likelihood
    of the generated tokens.  Simple likelihood-based selection prefers
    sequences with higher (less negative) cumulative logprobs.

    Extracts from ``node.vllm_output.cumulative_logprob`` or calculates
    from token logprobs if unavailable.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(name="cumulative_logprob_evaluator", *args, **kwargs)

    def __call__(
        self, node: Union[Node, List[Node]], method: BaseMethod
    ) -> List[float]:
        if isinstance(node, Node):
            node = [node]

        if all([n.vllm_output for n in node]):
            # Just return the cumulative logprobs from vllm
            if hasattr(node[0].vllm_output, "cumulative_logprob"):
                return [n.vllm_output.cumulative_logprob for n in node]

            # Otherwise calculate cumulative logprobs by hand
            return calculate_logprobs(node)
        else:
            # if it is root node
            if node[0].parent:
                logger.warning(f"No vllm_output found in node(s): {node}")
            return [0.0] * len(node)

    def _str_fields(self):
        return super()._str_fields()


class ProbEvaluator(BaseEvaluator):
    """Returns cumulative probability score from vLLM output.

    Like :class:`LogprobEvaluator` but exponentiates cumulative logprobs
    to probabilities in the range [0, 1].  Represents the joint probability
    of the generated sequence.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(name="cumulative_prob_evaluator", *args, **kwargs)

    def __call__(
        self, node: Union[Node, List[Node]], method: BaseMethod
    ) -> List[float]:
        if isinstance(node, Node):
            node = [node]

        if all([n.vllm_output for n in node]):
            # math.exp is x2 faster than np.exp for small lists
            if hasattr(node[0].vllm_output, "cumulative_logprob"):
                return [
                    math.exp(n.vllm_output.cumulative_logprob) for n in node
                ]

            # Otherwise calculate cumulative logprobs by hand and exponentiate
            _logprobs = calculate_logprobs(node)
            return [math.exp(lp) for lp in _logprobs]

        else:
            # if it is root node
            if node[0].parent:
                logger.warning(f"No vllm_output found in node(s): {node}")
            return [0.0] * len(node)

    def _str_fields(self):
        return super()._str_fields()


class LeanREPLEvaluator(BaseEvaluator):
    """Evaluate proof snippets via a Lean 4 REPL (Kimina server).

    Uses :class:`LeanClientAdapter` to wrap the external ``KiminaClient``.
    If *cache* is provided the client is wrapped with :class:`CachedClient`.
    """

    def __init__(
        self,
        client_args: Optional[ClientArgs] = None,
        cache: Optional[ProofCache] = None,
        *args,
        **kwargs,
    ):
        super().__init__(name="lean_repl_evaluator", *args, **kwargs)
        if client_args is None:
            client_args = ClientArgs()
        self.client_args = client_args
        self.lean_client = LeanClientAdapter(
            lean_server_url=(
                client_args.lean_server_url or "http://localhost:8000"
            ),
        )
        if cache is not None:
            self.lean_client = CachedClient(self.lean_client, cache=cache)

    def __call__(self, node: Union[Node, List[Node]], method: BaseMethod):
        """Evaluate node(s) — returns 1.0 for verified, 0.0 otherwise."""
        if isinstance(node, Node):
            nodes = [node]
        else:
            nodes = node

        snips = [n.answer for n in nodes]

        try:
            response = self.lean_client.check(
                snips=snips,
                timeout=self.client_args.timeout,
                show_progress=False,
                batch_size=self.client_args.batch_size,
                max_workers=self.client_args.num_proc,
            )
        except Exception as e:
            logger.error(f"Lean REPL evaluator failed: {e}")
            return [0.0] * len(nodes)

        results = []
        for idx, _n in enumerate(nodes):
            try:
                if response and getattr(response, "results", None):
                    entry = response.results[idx].response
                else:
                    logger.warning("No results in Lean server response")
                    results.append(0.0)
                    continue

                results.append(
                    1.0 if self.lean_client.is_success_response(entry) else 0.0
                )
            except Exception as e:
                logger.error(f"Lean verification error: {e}")
                results.append(0.0)

        return results if isinstance(node, list) else results[0]

    def _str_fields(self):
        return super()._str_fields() + [
            ("client_args", self.client_args),
        ]


class JudgeEvaluator(BaseEvaluator):
    """Uses a secondary LLM judge to score proof quality.

    Queries the Lean 4 REPL to extract proof state information (applied
    tactics, open goals, solved goals), then uses an LLM judge to assign
    a score out of 20 based on proof progress and quality.  Supports LoRA
    adapters for task-specific fine-tuning.

    Scores are normalized to [0, 1] for use in tree search.

    Typical YAML config::

        evaluator:
          func_name: "llm_as_judge_evaluator"
          llm_as_judge_model:
            model: "meta-llama/Llama-2-7b-hf"
          llm_as_judge_sampling:
            temperature: 0.7
            max_tokens: 256
    """

    def __init__(
        self,
        llm_as_judge_model: Union[vllm.LLM, ModelArgs],
        llm_as_judge_sampling: Union[vllm.SamplingParams, SamplingArgs],
        client_args: Optional[ClientArgs] = None,
        cache: Optional[ProofCache] = None,
        llm_as_judge_system_prompt: str = LLM_AS_JUDGE_SYSTEM_PROMPT,
        prompter: Optional[Callable] = None,
        lora_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(name="llm_as_judge_evaluator", *args, **kwargs)
        if client_args is None:
            client_args = ClientArgs()
        self.client_args = client_args

        if isinstance(llm_as_judge_model, ModelArgs):
            self.model = self.init_model(
                llm_as_judge_model,
                kwargs.get("llm_as_judge_visible_devices", "1"),
            )
            self.enable_lora = llm_as_judge_model.enable_lora
        else:
            self.model = llm_as_judge_model
            self.enable_lora = bool(kwargs.get("enable_lora", False))

        self.lora_path = lora_path
        if self.lora_path and not self.enable_lora:
            logger.warning(
                "lora_path provided but enable_lora is False. Ignoring lora_path."
            )
            self.lora_path = None

        self.sampling_params = (
            self.set_sampling_params(llm_as_judge_sampling)
            if llm_as_judge_sampling
            else vllm.SamplingParams()
        )

        self.system_prompt = llm_as_judge_system_prompt

        # Sync Lean Client
        raw = LeanClientAdapter(
            lean_server_url=(
                client_args.lean_server_url or "http://localhost:8000"
            ),
        )
        self.lean_client = (
            CachedClient(raw, cache=cache) if cache is not None else raw
        )

        if isinstance(prompter, Callable):
            self.prompter = prompter
        else:
            self.prompter = partial(
                self.model.get_tokenizer().apply_chat_template,
                tokenize=False,
                add_generation_prompt=True,
                continue_final_message=False,
                enable_thinking=False,
            )

    def _get_lora_request(self) -> Optional[LoRARequest]:
        if self.enable_lora and self.lora_path:
            return LoRARequest("lora_adapter", 1, self.lora_path)
        return None

    def set_sampling_params(
        self, sampling_params: Union[SamplingArgs, vllm.SamplingParams]
    ):
        if isinstance(sampling_params, SamplingArgs):
            sampling_params = vllm.SamplingParams(**sampling_params)
        self.sampling_params = sampling_params
        return self.sampling_params

    def init_model(self, model_args: ModelArgs, visible_devices: str = "1"):
        logger.info(f"Instantiating model:{model_args.model} for {self.name}.")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
        return vllm.LLM(**model_args)

    def _prepare_judge_messages(
        self,
        proof_so_far: str,
        current_goals: Optional[str] = None,
        applied_tactic: Optional[str] = None,
        solved_goals: Optional[str] = None,
        error_message_from_lean: Optional[str] = None,
    ):
        """Create a prompt for LLM to judge the response."""
        prompt = f"Here you can see the proof so far, the applied tactic, the open goals, and the solved goals.  Depends on these information, you should judge the quality of the tactic application. # Proof So Far:\n{proof_so_far}\n"
        prompt += f"# Applied Tactic:\n{applied_tactic}\n"
        prompt += f"# Open Goals:\n{current_goals}\n"
        prompt += f"# Solved Goals:\n{solved_goals}\n"
        prompt += "If goals are not provided, you should judge the quality of the tactic application based on the proof so far. Put your score in \\boxed{}. "
        if error_message_from_lean:
            prompt += f"# Error Message from Lean:\n{error_message_from_lean}\n"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        messages = self.prompter(messages)

        return messages

    def generate_judge_answer(
        self, messages: Union[str, List[str]]
    ) -> List[str]:
        try:
            lora_request = self._get_lora_request()
            if lora_request:
                output = self.model.generate(
                    messages,
                    self.sampling_params,
                    use_tqdm=False,
                    lora_request=lora_request,
                )
            else:
                output = self.model.generate(
                    messages, self.sampling_params, use_tqdm=False
                )
            num_messages = len(messages) if isinstance(messages, list) else 1
            num_outputs = len(output)
            if num_outputs != num_messages:
                logger.error(
                    f"vLLM generation mismatch: sent {num_messages} messages, "
                    f"got {num_outputs} outputs"
                )
                answers = []
                for i in range(num_messages):
                    if i < num_outputs:
                        try:
                            answers.append(output[i].outputs[0].text)
                        except (IndexError, AttributeError) as e:
                            logger.error(
                                f"Failed to extract text from output {i}: {e}"
                            )
                            answers.append(
                                "ERROR: Failed to generate judge answer"
                            )
                    else:
                        logger.warning(
                            f"Missing output for message {i}, using placeholder"
                        )
                        answers.append("ERROR: Missing judge answer")
                return answers
            answers = []
            for i in range(len(output)):
                try:
                    text = output[i].outputs[0].text
                    answers.append(text)
                except (IndexError, AttributeError) as e:
                    logger.error(f"Failed to extract text from output {i}: {e}")
                    answers.append("ERROR: Failed to extract text")
            return answers
        except Exception as e:
            logger.error(f"Judge answer generation failed: {e}")
            logger.exception(e)
            num_messages = len(messages) if isinstance(messages, list) else 1
            return [f"ERROR: Generation failed - {str(e)}"] * num_messages

    def __call__(
        self, node: Union[Node, List[Node]], method: BaseMethod
    ) -> List[float]:
        if isinstance(node, Node):
            node = [node]

        snips = []
        for i in range(len(node)):
            proof_so_far = method.traverse_to_root(node[i], include_root=True)
            proof_so_far = self.parse_proof(proof=proof_so_far)
            snips.append(proof_so_far)

        try:
            response = self.lean_client.check(
                snips=snips,
                timeout=self.client_args.timeout,
                show_progress=False,
            )
        except Exception as e:
            logger.error(f"KiminaClient failed: {e}")
            # Allow downstream code to handle fallback: all infotree=None
            response = None

        messages = []

        for i in range(len(node)):
            # If response is available and valid, extract infotree, else fallback
            if response and getattr(response, "results", None):
                # Each result is an object with .response; .response is dict
                result_obj = response.results[i]
                infotree = (
                    result_obj.response.get("infotree", None)
                    if result_obj.response
                    else None
                )
            else:
                infotree = None

            if infotree:
                header, body = split_proof_header(snips[i])
                intervals = extract_data(infotree, body)
                current_goals = intervals[-1]["goalsAfter"]
                applied_tactic = intervals[-1]["tactic"]
                solved_goals = intervals[-1]["goalsBefore"]
            else:
                logger.warning(
                    "Failed to get infotree from REPL, giving only the proof itself to judge."
                )
                current_goals = None
                applied_tactic = None
                solved_goals = None
                error_message_from_lean = (
                    result_obj.response.get("error", None)
                    if response
                    and getattr(response, "results", None)
                    and result_obj.response
                    else None
                )
                if error_message_from_lean:
                    logger.warning(
                        f"Error message from Lean: {error_message_from_lean}"
                    )
                    judge_message = self._prepare_judge_messages(
                        snips[i],
                        current_goals,
                        applied_tactic,
                        solved_goals,
                        error_message_from_lean,
                    )
                    messages.append(judge_message)
                    continue

            judge_message = self._prepare_judge_messages(
                snips[i],
                current_goals,
                applied_tactic,
                solved_goals,
            )
            messages.append(judge_message)

        judge_answers = self.generate_judge_answer(messages)

        if len(judge_answers) != len(node):
            logger.error(
                f"Critical mismatch: expected {len(node)} answers, got {len(judge_answers)}. "
                f"This should not happen after generate_judge_answer refactoring!"
            )
            if len(judge_answers) < len(node):
                judge_answers.extend(
                    ["ERROR: Missing answer"] * (len(node) - len(judge_answers))
                )
            else:
                judge_answers = judge_answers[: len(node)]
        scores = []
        for i in range(len(node)):
            if judge_answers[i].startswith("ERROR:"):
                logger.warning(
                    f"Judge answer {i} is an error placeholder: {judge_answers[i]}. "
                    f"Assigning neutral score (10.0)."
                )
                scores.append(10.0)
                continue

            extracted_score = extract_result(judge_answers[i])
            if extracted_score == "NO_BOXED_STRING_FOUND":
                logger.warning(
                    f"Failed to parse Judge Answer for node {i}. "
                    f"Raw answer: {judge_answers[i][:200]}... "
                    f"Assigning neutral score (10.0)."
                )
                scores.append(10.0)
                continue

            if extracted_score.isnumeric():
                score = float(extracted_score)
                if not (0.0 <= score <= 20.0):
                    logger.warning(
                        f"Judge score {score} out of range [0,20], clamping."
                    )
                    score = max(0.0, min(20.0, score))
                scores.append(score)
                continue

            logger.warning(
                f"Judge score is not valid. score: {extracted_score}, "
                f"Raw answer: {judge_answers[i][:200]}... "
                f"Assigning neutral score (10.0)."
            )
            scores.append(10.0)
        return [score / 20.0 for score in scores]

    def parse_proof(self, proof: str, pattern=None):
        start = proof.find("import Mathlib")
        end = proof.find("```", start)
        if end == -1 and start == -1:
            return proof
        elif end == -1:
            return proof[start:]
        return proof[start:end]

    def _str_fields(self):
        return super()._str_fields() + [
            ("model", self.model.__class__.__name__),
            ("enable_lora", self.enable_lora),
            ("lora_path", self.lora_path),
            ("sampling_params", self.sampling_params),
            ("client_args", self.client_args),
            ("system_prompt", self.system_prompt),
        ]


class TournamentEvaluator(BaseEvaluator):
    """Single-elimination tournament evaluation between sibling nodes.

    Runs a bracket-style tournament where an LLM judge picks winners
    pairwise among siblings.  Losers are scored by their elimination round
    (first round loss = 1, second = 2, etc.), and the winner receives the
    highest score.  All scores are normalized to [0, 1].

    Supports optional bracket shuffling for varied comparisons.

    Typical YAML config::

        evaluator:
          func_name: "pairwise_tournament_evaluator"
          llm_as_judge_model:
            model: "meta-llama/Llama-2-7b-hf"
          shuffle_bracket: true
    """

    def __init__(
        self,
        llm_as_judge_model: Union[vllm.LLM, ModelArgs],
        llm_as_judge_sampling: Union[vllm.SamplingParams, SamplingArgs],
        client_args: Optional[ClientArgs] = None,
        cache: Optional[ProofCache] = None,
        llm_as_judge_system_prompt: str = LLM_AS_JUDGE_SYSTEM_PROMPT_PAIRWISE,
        prompter: Optional[Callable] = None,
        shuffle_bracket: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(name="pairwise_tournament_evaluator", *args, **kwargs)
        if client_args is None:
            client_args = ClientArgs()
        self.client_args = client_args
        self.shuffle_bracket = shuffle_bracket

        if isinstance(llm_as_judge_model, ModelArgs):
            self.model = self.init_model(
                llm_as_judge_model,
                kwargs.get("llm_as_judge_visible_devices", "1"),
            )
        else:
            self.model = llm_as_judge_model

        self.sampling_params = (
            self.set_sampling_params(llm_as_judge_sampling)
            if llm_as_judge_sampling
            else vllm.SamplingParams()
        )

        self.system_prompt = llm_as_judge_system_prompt

        # Sync Lean Client (optionally cached)
        raw_lean = LeanClientAdapter(
            lean_server_url=(
                client_args.lean_server_url or "http://localhost:8000"
            ),
        )
        self.lean_client = (
            CachedClient(raw_lean, cache=cache)
            if cache is not None
            else raw_lean
        )

        if isinstance(prompter, Callable):
            self.prompter = prompter
        else:
            self.prompter = partial(
                self.model.get_tokenizer().apply_chat_template,
                tokenize=False,
                add_generation_prompt=True,
                continue_final_message=False,
                enable_thinking=False,
            )

    def set_sampling_params(
        self, sampling_params: Union[SamplingArgs, vllm.SamplingParams]
    ):
        if isinstance(sampling_params, SamplingArgs):
            sampling_params = vllm.SamplingParams(**sampling_params)
        self.sampling_params = sampling_params
        return self.sampling_params

    def init_model(self, model_args: ModelArgs, visible_devices: str = "1"):
        logger.info(f"Instantiating model:{model_args.model} for {self.name}.")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
        return vllm.LLM(**model_args)

    def _prepare_pairwise_judge_messages(
        self,
        proof_a: str,
        proof_b: str,
        info_a: Optional[dict] = None,
        info_b: Optional[dict] = None,
    ):
        """Create a prompt for LLM to judge between two proofs."""
        prompt = "You are comparing two proof attempts. Choose which one is better.\n\n"

        prompt += "# Proof A:\n"
        prompt += f"```lean\n{proof_a}\n```\n"
        if info_a:
            if info_a.get("applied_tactic"):
                prompt += f"Applied Tactic: {info_a['applied_tactic']}\n"
            if info_a.get("current_goals"):
                prompt += f"Open Goals: {info_a['current_goals']}\n"
            if info_a.get("solved_goals"):
                prompt += f"Solved Goals: {info_a['solved_goals']}\n"
            if info_a.get("error_message"):
                prompt += f"Error: {info_a['error_message']}\n"

        prompt += "\n# Proof B:\n"
        prompt += f"```lean\n{proof_b}\n```\n"
        if info_b:
            if info_b.get("applied_tactic"):
                prompt += f"Applied Tactic: {info_b['applied_tactic']}\n"
            if info_b.get("current_goals"):
                prompt += f"Open Goals: {info_b['current_goals']}\n"
            if info_b.get("solved_goals"):
                prompt += f"Solved Goals: {info_b['solved_goals']}\n"
            if info_b.get("error_message"):
                prompt += f"Error: {info_b['error_message']}\n"

        prompt += "\nWhich proof is better? Answer with either 'A' or 'B' in \\boxed{}."

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        return self.prompter(messages)

    def _extract_lean_info(
        self, snip: str, response, result_idx: int
    ) -> Optional[dict]:
        """Extract tactic and goal information from Lean REPL response."""
        if not response or not getattr(response, "results", None):
            return None

        result_obj = response.results[result_idx]
        infotree = (
            result_obj.response.get("infotree", None)
            if result_obj.response
            else None
        )

        if not infotree:
            error_message = (
                result_obj.response.get("error", None)
                if result_obj.response
                else None
            )
            return {"error_message": error_message} if error_message else None

        try:
            header, body = split_proof_header(snip)
            intervals = extract_data(infotree, body)
            return {
                "applied_tactic": intervals[-1]["tactic"],
                "current_goals": intervals[-1]["goalsAfter"],
                "solved_goals": intervals[-1]["goalsBefore"],
            }
        except Exception as e:
            logger.warning(f"Failed to extract info from infotree: {e}")
            return None

    def _batch_compare_pairs(
        self,
        pairs: List[Tuple[int, int]],
        nodes: List[Node],
        method: BaseMethod,
        snips: List[str],
        lean_infos: List[Optional[dict]],
    ) -> List[int]:
        """Compare pairs of nodes in batch and return winner indices."""
        if not pairs:
            return []

        # Prepare messages for all pairs
        messages = []
        for idx_a, idx_b in pairs:
            message = self._prepare_pairwise_judge_messages(
                snips[idx_a],
                snips[idx_b],
                lean_infos[idx_a],
                lean_infos[idx_b],
            )
            messages.append(message)

        # Batch generate
        try:
            output = self.model.generate(
                messages, self.sampling_params, use_tqdm=False
            )

            if len(output) != len(messages):
                logger.error(
                    f"vLLM generation mismatch: sent {len(messages)} messages, "
                    f"got {len(output)} outputs"
                )

            # Extract winners
            winners = []
            for i, (idx_a, idx_b) in enumerate(pairs):
                try:
                    if i < len(output):
                        answer = output[i].outputs[0].text
                        extracted = extract_result(answer)

                        if extracted == "NO_BOXED_STRING_FOUND":
                            logger.warning(
                                f"No boxed answer found for pair ({idx_a}, {idx_b}), defaulting to A"
                            )
                            winners.append(idx_a)
                        elif extracted.strip().upper() == "A":
                            winners.append(idx_a)
                        elif extracted.strip().upper() == "B":
                            winners.append(idx_b)
                        else:
                            logger.warning(
                                f"Invalid answer '{extracted}' for pair ({idx_a}, {idx_b}), defaulting to A"
                            )
                            winners.append(idx_a)
                    else:
                        logger.warning(
                            f"Missing output for pair ({idx_a}, {idx_b}), defaulting to A"
                        )
                        winners.append(idx_a)
                except Exception as e:
                    logger.error(
                        f"Failed to process pair ({idx_a}, {idx_b}): {e}"
                    )
                    winners.append(idx_a)

            return winners

        except Exception as e:
            logger.error(f"Batch comparison failed: {e}")
            # Fallback: return first element of each pair
            return [pair[0] for pair in pairs]

    def __call__(
        self, node: Union[Node, List[Node]], method: BaseMethod
    ) -> List[float]:
        if isinstance(node, Node):
            node = [node]

        n = len(node)

        # Single node case
        if n == 1:
            return [1.0]

        # Prepare all proofs
        snips = []
        for i in range(n):
            proof_so_far = method.traverse_to_root(node[i], include_root=True)
            proof_so_far = self.parse_proof(proof=proof_so_far)
            snips.append(proof_so_far)

        # Get Lean info for all proofs in batch
        try:
            response = self.lean_client.check(
                snips=snips,
                timeout=self.client_args.timeout,
                show_progress=False,
            )
        except Exception as e:
            logger.error(f"KiminaClient failed: {e}")
            response = None

        # Extract info for all nodes
        lean_infos = []
        for i in range(n):
            info = self._extract_lean_info(snips[i], response, i)
            lean_infos.append(info)

        # Initialize bracket with shuffled or sequential indices
        bracket_indices = list(range(n))
        if self.shuffle_bracket:
            import random

            random.shuffle(bracket_indices)
            logger.info(f"Shuffled bracket order: {bracket_indices}")

        # Pad to next power of 2 if needed
        n_padded = 2 ** math.ceil(math.log2(n))

        # Add dummy indices for padding (they will lose immediately)
        while len(bracket_indices) < n_padded:
            bracket_indices.append(-1)  # -1 represents dummy/bye

        # Track scores: initially all zeros
        scores = [0.0] * n

        # Tournament rounds
        round_num = 1
        current_bracket = bracket_indices.copy()

        while len(current_bracket) > 1:
            # Create pairs
            pairs = []
            valid_pairs = []  # pairs without dummies
            pair_to_valid_idx = {}

            for i in range(0, len(current_bracket), 2):
                idx_a = current_bracket[i]
                idx_b = current_bracket[i + 1]

                # Handle dummy nodes (auto-advance real node)
                if idx_a == -1 and idx_b == -1:
                    pairs.append((-1, -1))
                elif idx_a == -1:
                    pairs.append((idx_a, idx_b))
                elif idx_b == -1:
                    pairs.append((idx_a, idx_b))
                else:
                    pair_to_valid_idx[len(pairs)] = len(valid_pairs)
                    valid_pairs.append((idx_a, idx_b))
                    pairs.append((idx_a, idx_b))

            # Batch compare only valid pairs
            if valid_pairs:
                winners_from_comparison = self._batch_compare_pairs(
                    valid_pairs, node, method, snips, lean_infos
                )
            else:
                winners_from_comparison = []

            # Process all pairs to get winners and assign scores to losers
            winners = []
            comparison_idx = 0

            for pair_idx, (idx_a, idx_b) in enumerate(pairs):
                if idx_a == -1 and idx_b == -1:
                    winners.append(-1)
                elif idx_a == -1:
                    winners.append(idx_b)
                elif idx_b == -1:
                    winners.append(idx_a)
                else:
                    # Real comparison
                    winner_idx = winners_from_comparison[comparison_idx]
                    loser_idx = idx_b if winner_idx == idx_a else idx_a

                    # Assign score to loser based on round
                    # Round 1: score = 1, Round 2: score = 2, etc.
                    scores[loser_idx] = float(round_num)

                    winners.append(winner_idx)
                    comparison_idx += 1

            current_bracket = winners
            round_num += 1

        # Winner gets the highest score (number of rounds)
        winner_idx = current_bracket[0]
        if winner_idx != -1:
            scores[winner_idx] = float(round_num)

        # Normalize scores to [0, 1]
        max_score = float(round_num)
        normalized_scores = [s / max_score for s in scores]

        logger.info(
            f"Tournament complete. Final scores: {scores} -> normalized: {normalized_scores}"
        )

        return normalized_scores

    def parse_proof(self, proof: str, pattern=None):
        start = proof.find("import Mathlib")
        end = proof.find("```", start)
        if end == -1 and start == -1:
            return proof
        elif end == -1:
            return proof[start:]
        return proof[start:end]

    def _str_fields(self):
        return super()._str_fields() + [
            ("model", self.model.__class__.__name__),
            ("sampling_params", self.sampling_params),
            ("client_args", self.client_args),
            ("shuffle_bracket", self.shuffle_bracket),
            ("system_prompt", self.system_prompt),
        ]


class NormLenEvaluator(BaseEvaluator):
    """Normalized Lengths node evaluation: cumulative_logprob / L^alpha.

    BFS-Prover style scoring (https://arxiv.org/pdf/2502.03438) that
    penalizes depth by dividing cumulative logprobability by path length
    raised to the *length_norm* (alpha) power.

    Applicable to any tree search method, not just Best First Search.
    """

    def __init__(self, length_norm: float = 0.5, *args, **kwargs):
        """
        Args:
            length_norm (float): tunable alpha parameter that is used in L^alpha
        """
        self.length_norm = length_norm
        super().__init__(name="normalized_lengths_evaluator", *args, **kwargs)

    def __call__(
        self, node: Union[Node, List[Node]], method: BaseMethod
    ) -> List[float]:
        """
        1. Collect cumulative_logprobs of the whole path.
        2. Take count of the total length passed.
        3. Divide the result with L^alpha and return.
        """
        if isinstance(node, list):
            evaluations = []
            for n in node:
                evaluations.extend(self.__call__(node=n, method=method))
            return evaluations

        # We already know the path length from node.level
        L = node.level

        whole_path_cumulative_logprobs = 0.0

        if not node.parent:
            return [0.0]

        # Traverse back and sum the cumulative log probs.
        if hasattr(node.vllm_output, "cumulative_logprob"):
            while node.parent:
                whole_path_cumulative_logprobs += (
                    node.vllm_output.cumulative_logprob
                )
                node = node.parent
        # If cumulative_logprob is not available, calculate logprobs by hand.
        else:
            while node.parent:
                whole_path_cumulative_logprobs += calculate_logprobs([node])[0]
                node = node.parent

        return [whole_path_cumulative_logprobs / (L**self.length_norm)]

    def _str_fields(self):
        return super()._str_fields() + [("length_norm", self.length_norm)]


class NormLenProbEvaluator(BaseEvaluator):
    """Normalized Lengths node evaluation: cumulative_prob / L^alpha.

    Like :class:`NormLenEvaluator` but uses probabilities (exponentiated
    logprobs) instead of cumulative logprobs.  Same depth penalty strategy
    with tunable *length_norm* (alpha) parameter.

    BFS-Prover variant (https://arxiv.org/pdf/2502.03438) applicable to
    any tree search method.
    """

    def __init__(self, length_norm: float = 0.5, *args, **kwargs):
        """
        Args:
            length_norm (float): tunable alpha parameter that is used in L^alpha
        """
        self.length_norm = length_norm
        super().__init__(
            name="normalized_lengths_probs_evaluator", *args, **kwargs
        )

    def __call__(
        self, node: Union[Node, List[Node]], method: BaseMethod
    ) -> List[float]:
        """
        1. Collect cumulative_logprobs of the whole path.
        2. Take count of the total length passed.
        3. Divide the result with L^alpha and return.
        """
        if isinstance(node, list):
            evaluations = []
            for n in node:
                evaluations.extend(self.__call__(node=n, method=method))
            return evaluations

        # We already know the path length from node.level
        L = node.level

        # By whole path, we mean for every node,
        # by "cumulative_logprob" we mean that tactic's logprobs as a sentence.
        whole_path_cumulative_probs = 0.0

        # If it is already root, just return a value of 0.0 as it will be
        # processed first no matter what.
        if not node.parent:
            return [0.0]

        # Traverse back and sum the cumulative log probs.
        if hasattr(node.vllm_output, "cumulative_logprob"):
            while node.parent:
                whole_path_cumulative_probs += math.exp(
                    node.vllm_output.cumulative_logprob
                )
                node = node.parent
        # If cumulative_logprob is not available, calculate logprobs by hand.
        else:
            while node.parent:
                whole_path_cumulative_probs += math.exp(
                    calculate_logprobs([node])[0]
                )
                node = node.parent

        return [whole_path_cumulative_probs / (L**self.length_norm)]

    def _str_fields(self):
        return super()._str_fields() + [("length_norm", self.length_norm)]


class RocqEvaluator(BaseEvaluator):
    """Evaluates Rocq proof snippets via rocq-ml-server.

    Wraps a snippet in a theorem statement and executes commands via the
    ``rocq-ml-server`` (typically on ``localhost:5000``).  Returns 1.0 if
    the proof closes, 0.0 otherwise.

    Supports custom prelude code, theorem statements, and workspace
    configuration.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5000,
        workspace_dir: Optional[str] = ".",
        timeout: Optional[float] = 5.0,
        statement: str = "True",
        theorem_name: str = "__eval",
        prelude: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.workspace_dir = workspace_dir
        self.timeout = timeout
        self.statement = statement
        self.theorem_name = theorem_name
        self.prelude = prelude
        self._client = RocqBatchClient(
            host=self.host,
            port=self.port,
            workspace_dir=self.workspace_dir,
            theorem_name=self.theorem_name,
            statement=self.statement,
            prelude=self.prelude,
        )

        super().__init__(name="rocq_evaluator")

    def __call__(
        self, code: Union[str, List[str]]
    ) -> Union[float, List[float]]:
        snippets = [code] if isinstance(code, str) else code
        results: List[float] = []

        for snippet in snippets:
            response = self._client.verify_snippet(snippet)
            results.append(1.0 if response.get("proof_finished") else 0.0)

        return results[0] if isinstance(code, str) else results

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "RocqEvaluator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _build_file_text(self) -> str:
        parts: List[str] = []
        if self.prelude:
            parts.append(self.prelude.rstrip())
        parts.append(f"Theorem {self.theorem_name} : {self.statement}.")
        return "\n".join(parts) + "\n"


class EvaluatorType(Enum):
    """Enum mapping evaluator config names to their implementation classes.

    Members are accessed via ``from_str()`` which normalises the config
    ``func_name`` (e.g. ``"cumulative_logprob_evaluator"`` →
    ``EvaluatorType.CUMULATIVE_LOGPROB``).

    To add a new evaluator, add a member here and ensure the value is a
    :class:`BaseEvaluator` subclass.
    """

    CUMULATIVE_LOGPROB = LogprobEvaluator
    LEAN_REPL = LeanREPLEvaluator
    LLM_AS_JUDGE = JudgeEvaluator
    TOURNAMENT = TournamentEvaluator
    NORMALIZED_LENGTHS = NormLenEvaluator
    NORMALIZED_LENGTHS_PROBS = NormLenProbEvaluator
    ROCQ = RocqEvaluator

    @classmethod
    def from_str(cls, name: str) -> "EvaluatorType":
        normalized = name.strip().lower().replace("-", "_")
        for suffix in ("_evaluator", "_policy"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
        for member in cls:
            if normalized == member.name.lower():
                return member
        valid_keys = [member.name.lower() for member in cls]
        raise ValueError(
            f"Unknown evaluator '{name}'. Valid options: {valid_keys}"
        )

    def initialize(self, *args, **kwargs) -> Callable:
        return self.value(*args, **kwargs)


IMPLEMENTED_EVALUATORS = EvaluatorType
EVALUATORS = [member.name.lower() for member in EvaluatorType]


def get_evaluator(func_name, *args, **kwargs) -> Callable:
    return EvaluatorType.from_str(func_name).initialize(*args, **kwargs)


def get_evaluator_from_config(
    config: EvaluatorArgs, *args, **kwargs
) -> Callable:
    logger.info(f"Instantiating node evaluator from config: {config.func_name}")
    return EvaluatorType.from_str(config.func_name).initialize(
        *args, **dict(vars(config)), **kwargs
    )
