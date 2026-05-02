import json
from typing import Callable, Optional, Union

import vllm
from loguru import logger
from transformers import AutoTokenizer
from utils import ModelArgs, SamplingArgs
from vllm.lora.request import LoRARequest

from treethink import (
    ChildFinderArgs,
    InferenceTimeArgs,
    InferenceTimeMethods,
    NodeEvaluatorArgs,
    get_child_finder_from_config,
    get_inference_time_method,
    get_node_evaluator_from_config,
)


class Sampler:
    def __init__(
        self,
        model_args: Optional[ModelArgs] = None,
        sample_params: Optional[
            Union[vllm.SamplingParams, SamplingArgs]
        ] = None,
        child_finder_args: Optional[ChildFinderArgs] = None,
        node_evaluator_args: Optional[NodeEvaluatorArgs] = None,
        inference_time_args: Optional[InferenceTimeArgs] = None,
        prompter: Optional[Callable] = None,
        task_name="generate",
    ):
        # Can't rename model_args.model to model_args.model_name as it would
        # mess up LLM(*model_args) statement
        if model_args:
            self.model_name = model_args.model
            self.model = vllm.LLM(
                **model_args, max_lora_rank=32, trust_remote_code=True
            )
            self.enable_lora = model_args.enable_lora
        elif child_finder_args:
            self.model_name = child_finder_args.model.model
            self.model = None
            self.enable_lora = False
        else:
            logger.warning("You have not provided model_args or child_finder_args, so model is not initialized.")

        self.child_finder_args = child_finder_args
        self.node_evaluator_args = node_evaluator_args
        self.inference_time_args = inference_time_args
        self.task_name = task_name

        # Conversion on the fly
        if isinstance(sample_params, SamplingArgs):
            sample_params = vllm.SamplingParams(**sample_params)
        self.sample_params = sample_params

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # Opening endpoint for other prompters
        if prompter:
            self.prompter = prompter
        else:  # use the default prompter
            self.prompter = self._prompter

        self._init_inference_time_method()

    def _init_inference_time_method(self):
        """Initiates method by using child_finder_args, node_evaluator_args, inference_time_args."""
        if (
            self.inference_time_args
            and self.child_finder_args
            and self.node_evaluator_args
        ):
            logger.info(
                f"Instantiating selected method: {self.inference_time_args.method_name}"
            )
        else:
            return

        self.child_finder = get_child_finder_from_config(
            self.child_finder_args, prompter=self.prompter
        )
        self.node_evaluator = get_node_evaluator_from_config(
            self.node_evaluator_args, prompter=self.prompter
        )
        self.method = get_inference_time_method(
            inference_time_config=self.inference_time_args,
            root_node=None,
            child_finder=self.child_finder,
            node_evaluator=self.node_evaluator,
        )

        # change the model and therefore the generation
        self.model = InferenceTimeMethods(self.method, self.inference_time_args)

    def _format_text(self, text):
        # find the text starting from <|end_header_id|> to the end of the text
        text = text[text.find("<|end_header_id|>") :]

        return text

    def _prompter(self, messages: dict):
        """Prompt the model before getting the generations. If you would like to
        change

        Args:
            messages (dict): OpenAI style messages.

        Returns:
            dict: chat template applied prompts.
        """
        # Template prompt, includes being not harmful etc.
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        return prompt

    def _save_outputs(self, outputs, k, path="data/results"):
        model_name = self.model_name.replace("-", "_").replace("/", "_")
        file_path = path + f"/{self.task_name}_{model_name}_{k}.jsonl"
        with open(file_path, "w") as f:
            for output in outputs:
                f.write(json.dumps(output) + "\n")

    def _message_creator(
        self, system_prompt, datapoint, data_key, prompt_format
    ):
        """
        Creates a message structure for the model input based on the provided parameters.

        Args:
            system_prompt (str): The system prompt to be used.
            datapoint (dict): The data point containing the input information.
            data_key (list): A list of keys to access the relevant data in the datapoint.
            prompt_format (str, optional): A format string to be applied to the data.

        Returns:
            list: A list of dictionaries representing the message structure.

        Raises:
            ValueError: If the length of data_key is not 1 or 2.
            KeyError: If a specified key is not found in the datapoint.
        """
        try:
            # Gather values for all keys
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
                f"Key {e} not found in datapoint. "
                f"Available keys: {available_keys}. "
                f"Looking for keys: {data_key}"
            ) from e
        except IndexError:
            raise ValueError("data_key must be a non-empty list")

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
        """
        Perform batched inference on the given data.

        Args:
            data (list): List of data points to process.
            batch_size (int, optional): Size of each batch. If None, processes all data at once.
            data_key (list): List of keys to access data in each data point. Must contain 1 or 2 elements.
            system_prompt (str, optional): System prompt for the model. Defaults to "You are a helpful assistant.".
            start_from_kth_batch (int, optional): Index of the batch to start processing from. Defaults to 0.
            save_after_k_batches (int, optional): Number of batches to process before saving intermediate results. If None, doesn't save intermediate results.
            save_final_outputs (bool, optional): If True, saves the final outputs after processing all batches. Defaults to False.
            prompt_format (str, optional): Format string to be applied to the data. If None, no formatting is applied.
            lora_path (str, optional): Works if model_args.enable_lora is True.

        Returns:
            list: List of processed data points with added 'messages' and 'output' fields.

        Raises:
            ValueError: If batch_size is not a positive integer when specified.
            IndexError: If start_from_kth_batch is out of range of the data.
            KeyError: If a specified key in data_key is not found in a datapoint.
            ValueError: If data_key contains neither 1 nor 2 elements.
        """
        # Validate batch_size
        if batch_size is None or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        # Check if start_from_kth_batch is within range
        if start_from_kth_batch * batch_size >= len(data):
            raise IndexError("start_from_kth_batch is out of range")

        processed_data = []
        try:
            # Iterate over the data in batches
            for i in range(
                batch_size * start_from_kth_batch, len(data), batch_size
            ):
                # Get the current batch of data
                batch_data = data[i : i + batch_size]
                batch_messages = []

            # Prepare messages for each datapoint in the batch
            for datapoint in batch_data:
                messages = self._message_creator(
                    system_prompt, datapoint, data_key, prompt_format
                )
                datapoint["messages"] = messages
                messages = self.prompter(messages)
                batch_messages.append(messages)
                datapoint["model_input"] = messages

            # Check if using inference time methods (InferenceTimeMethods has generate with problem_id)
            if hasattr(self.model, "generate") and hasattr(
                self.model, "inftime_args"
            ):
                # Inference time methods - pass problem_id to each
                batch_responses = []
                for idx, (datapoint, prompt) in enumerate(
                    zip(batch_data, batch_messages)
                ):
                    problem_id = (
                        datapoint.get("id")
                        or datapoint.get("problem_id")
                        or datapoint.get("custom_id")
                    )
                    response = self.model.generate(
                        prompts=prompt, problem_id=problem_id
                    )
                    batch_responses.append(response)
            elif self.enable_lora:
                # Normal vLLM with LoRA
                batch_responses = self.model.generate(
                    prompts=batch_messages,
                    sampling_params=self.sample_params,
                    lora_request=LoRARequest("lora_adapter", 1, lora_path),
                )
            else:
                # Normal vLLM
                batch_responses = self.model.generate(
                    prompts=batch_messages,
                    sampling_params=self.sample_params,
                )

                # Add outputs to datapoints and append to processed_data
                for j, response in enumerate(batch_responses):
                    datapoint = batch_data[j].copy()
                    datapoint["output"] = [
                        response.outputs[i].text
                        for i in range(len(response.outputs))
                    ]
                    if (
                        self.inference_time_args
                        and self.inference_time_args.store_graph_stats
                    ):
                        datapoint["graph_stats"] = response.graph_stats

                    processed_data.append(datapoint)

                # Save intermediate results if specified
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
            # Save final outputs if specified
            if save_final_outputs:
                self._save_outputs(processed_data, "error")

        return processed_data
