import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        metrics = {}
        request_id = uuid4().hex
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            ),
        )

        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
            )
        response_mask = [1] * len(output.token_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            multi_modal_data={},
            num_turns=2,
            metrics=metrics,
        )
        return output
