from typing import List

from loguru import logger


def calculate_logprobs(nodes: List) -> List[float]:

    # NOTE(burak): OpenAI does not store/calculate cumulative logprobs
    # so we need to calculate it ourselves by summing token logprobs.
    calculated_logprobs = []
    for n in nodes:
        if n.vllm_output.logprobs and n.vllm_output.logprobs.content:
            cumulative_logprob = sum(
                token.logprob for token in n.vllm_output.logprobs.content
            )
            calculated_logprobs.append(cumulative_logprob)
        else:
            logger.warning(f"No token logprobs found for nodes: {n}")
            calculated_logprobs.append(0.0)

    return calculated_logprobs
