"""vLLM-based batched-inference sampler."""

from typing import Callable, Optional, Union

import vllm
from vllm.lora.request import LoRARequest

from treethink import ModelArgs, SamplingArgs
from treethink.sampler import SamplerBase


class VLLMSampler(SamplerBase):
    """Batched vLLM inference sampler.

    Uses synchronous :class:`vllm.LLM` for high-throughput generation.
    Supports LoRA adapters.
    """

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
