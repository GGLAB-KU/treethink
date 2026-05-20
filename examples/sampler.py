import json
from typing import Callable, Optional, Union

import vllm
from loguru import logger
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest

from treethink import (
    EvaluatorArgs,
    ModelArgs,
    PolicyArgs,
    SamplingArgs,
    TreeThink,
    TreeThinkArgs,
    get_evaluator_from_config,
    get_method,
    get_policy_from_config,
)


class SamplerBase:
    def __init__(
        self,
        sample_params: Optional[
            Union[vllm.SamplingParams, SamplingArgs]
        ] = None,
        prompter: Optional[Callable] = None,
        task_name="generate",
        model_name: Optional[str] = None,
    ):
        self.task_name = task_name
        self.model_name = model_name

        if isinstance(sample_params, SamplingArgs):
            sample_params = vllm.SamplingParams(**sample_params)
        self.sample_params = sample_params

        if self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )

        if prompter:
            self.prompter = prompter
        else:
            self.prompter = self._prompter

    def _format_text(self, text):
        text = text[text.find("<|end_header_id|>") :]
        return text

    def _prompter(self, messages: dict):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return prompt

    def _save_outputs(self, outputs, k, path="data/results"):
        model_name = (
            self.model_name.replace("-", "_").replace("/", "_")
            if self.model_name
            else "unknown_model"
        )
        file_path = path + f"/{self.task_name}_{model_name}_{k}.jsonl"
        with open(file_path, "w") as f:
            for output in outputs:
                f.write(json.dumps(output) + "\n")

    def _message_creator(
        self, system_prompt, datapoint, data_key, prompt_format
    ):
        try:
            values = [datapoint[k] for k in data_key]
            if prompt_format:
                content = prompt_format.format(*values)
            else:
                content = " ".join(str(v) for v in values)

            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
        except KeyError as e:
            available_keys = list(datapoint.keys())
            raise KeyError(
                f"Key {e} not found in datapoint. Available keys: {available_keys}. Looking for keys: {data_key}"
            ) from e
        except IndexError:
            raise ValueError("data_key must be a non-empty list")

    def _generate_batch(self, batch_data, batch_messages):
        raise NotImplementedError("Subclasses must implement _generate_batch")

    def batched_inference(
        self,
        data: list,
        batch_size: int = None,
        data_key: list = ["problem"],
        system_prompt: str = None,
        start_from_kth_batch: int = 0,
        save_after_k_batches: int = None,
        save_final_outputs: bool = False,
        prompt_format: str = None,
        lora_path: Optional[str] = None,
    ) -> list:
        if batch_size is None or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if start_from_kth_batch * batch_size >= len(data):
            raise IndexError("start_from_kth_batch is out of range")

        processed_data = []
        try:
            for i in range(
                batch_size * start_from_kth_batch, len(data), batch_size
            ):
                batch_data = data[i : i + batch_size]
                batch_messages = []

                for datapoint in batch_data:
                    messages = self._message_creator(
                        system_prompt, datapoint, data_key, prompt_format
                    )
                    datapoint["messages"] = messages
                    messages = self.prompter(messages)
                    batch_messages.append(messages)
                    datapoint["model_input"] = messages

                batch_responses = self._generate_batch(
                    batch_data, batch_messages
                )

                for j, resp in enumerate(batch_responses):
                    datapoint = batch_data[j].copy()

                    # For vllm outputs
                    if hasattr(resp, "outputs"):
                        datapoint["output"] = [
                            resp.outputs[idx].text
                            for idx in range(len(resp.outputs))
                        ]
                    elif hasattr(resp, "solution"):
                        # TreeThinkOutputs or direct string outputs
                        datapoint["output"] = [resp.solution]
                    else:
                        datapoint["output"] = [resp]

                    if (
                        hasattr(self, "treethink_args")
                        and self.treethink_args
                        and self.treethink_args.store_graph_stats
                    ):
                        if hasattr(resp, "graph_stats"):
                            datapoint["graph_stats"] = resp.graph_stats

                    processed_data.append(datapoint)

                if (
                    save_after_k_batches is not None
                    and (i + batch_size) % (save_after_k_batches * batch_size)
                    == 0
                ):
                    self._save_outputs(
                        processed_data,
                        k=(i + batch_size)
                        // (save_after_k_batches * batch_size),
                    )
        except Exception as e:
            logger.error(f"Error occurred while processing batch: {e}")
            if save_final_outputs:
                self._save_outputs(processed_data, "error")

        return processed_data


class VLLMSampler(SamplerBase):
    def __init__(
        self,
        model_args: ModelArgs,
        sample_params: Optional[
            Union[vllm.SamplingParams, SamplingArgs]
        ] = None,
        prompter: Optional[Callable] = None,
        task_name="generate",
        lora_path: Optional[str] = None,
    ):
        super().__init__(
            sample_params=sample_params,
            prompter=prompter,
            task_name=task_name,
            model_name=model_args.model,
        )
        self.model = vllm.LLM(
            **model_args, max_lora_rank=32, trust_remote_code=True
        )
        self.enable_lora = model_args.enable_lora
        self.lora_path = lora_path

    def _generate_batch(self, batch_data, batch_messages):
        if self.enable_lora and self.lora_path:
            return self.model.generate(
                prompts=batch_messages,
                sampling_params=self.sample_params,
                lora_request=LoRARequest("lora_adapter", 1, self.lora_path),
            )
        else:
            return self.model.generate(
                prompts=batch_messages,
                sampling_params=self.sample_params,
            )

    def batched_inference(self, *args, **kwargs):
        # Override to support lora_path parameter specific to VLLMSampler if needed
        self.lora_path = kwargs.pop("lora_path", self.lora_path)
        return super().batched_inference(*args, **kwargs)


class TreeThinkSampler(SamplerBase):
    def __init__(
        self,
        policy_args: PolicyArgs,
        evaluator_args: EvaluatorArgs,
        treethink_args: TreeThinkArgs,
        sample_params: Optional[
            Union[vllm.SamplingParams, SamplingArgs]
        ] = None,
        prompter: Optional[Callable] = None,
        task_name="generate",
        lora_path: Optional[str] = None,
    ):
        super().__init__(
            sample_params=sample_params,
            prompter=prompter,
            task_name=task_name,
            model_name=policy_args.model.model,
        )
        self.policy_args = policy_args
        self.evaluator_args = evaluator_args
        self.treethink_args = treethink_args
        self.enable_lora = policy_args.model.enable_lora
        self.lora_path = lora_path

        self._init_treethink_method()

    def _init_treethink_method(self):
        logger.info(
            f"Instantiating selected method: {self.treethink_args.method_name}"
        )
        self.policy = get_policy_from_config(
            self.policy_args,
            prompter=self.prompter,
            lora_path=self.lora_path,
        )
        self.evaluator = get_evaluator_from_config(
            self.evaluator_args,
            prompter=self.prompter,
            lora_path=self.lora_path,
        )
        self.method = get_method(
            treethink_config=self.treethink_args,
            root_node=None,
            policy=self.policy,
            evaluator=self.evaluator,
        )
        self.model = TreeThink(self.method, self.treethink_args)

    def _set_lora_path(self, lora_path: Optional[str]):
        if lora_path is None:
            return
        self.lora_path = lora_path
        if hasattr(self, "policy"):
            self.policy.lora_path = lora_path
        if hasattr(self, "evaluator"):
            self.evaluator.lora_path = lora_path

    def _generate_batch(self, batch_data, batch_messages):
        batch_responses = []
        for datapoint, prompt in zip(batch_data, batch_messages):
            problem_id = (
                datapoint.get("id")
                or datapoint.get("problem_id")
                or datapoint.get("custom_id")
            )
            response = self.model.generate(
                prompts=prompt, problem_id=problem_id
            )
            batch_responses.append(response)
        return batch_responses

    def batched_inference(self, *args, **kwargs):
        self._set_lora_path(kwargs.pop("lora_path", None))
        return super().batched_inference(*args, **kwargs)
