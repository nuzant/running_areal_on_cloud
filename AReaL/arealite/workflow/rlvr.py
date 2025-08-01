import asyncio
import functools
import os
import uuid
from concurrent.futures import ProcessPoolExecutor

import colorama
import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast

import realhf.base.logging as logging
from arealite.api.cli_args import RLVRWorkflowConfig
from arealite.api.engine_api import InferenceEngine
from arealite.api.io_struct import LLMRequest
from arealite.api.workflow_api import RolloutResult, RolloutWorkflow
from arealite.utils.data import concat_padded_tensors

logger = logging.getLogger("RLVRWorkflow")
REWARD_TIMEOUT_SECONDS = 15


class RLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        config: RLVRWorkflowConfig,
        tokenizer: PreTrainedTokenizerFast,
        dump_dir: str | None = None,
    ):
        self.reward_fn = reward_fn
        self.gconfig = config.gconfig
        self.tokenizer = tokenizer
        self.enable_thinking = config.enable_thinking
        self.success_rate_ub = config.success_rate_ub
        self.success_rate_lb = config.success_rate_lb
        self.max_prompt_len = config.max_prompt_len
        self.dump_dir = dump_dir
        self.rw_executor = ProcessPoolExecutor(max_workers=4)
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    async def arun_episode(self, engine: InferenceEngine, data):
        input_ids = self.tokenizer.apply_chat_template(
            data["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
            max_length=self.max_prompt_len,
            truncation=True,
        )
        n_samples = self.gconfig.n_samples
        req = LLMRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
        )
        resps = await asyncio.gather(*[engine.agenerate(req) for _ in range(n_samples)])

        version = engine.get_version()
        prompt_strs = []
        completions_strs = []
        rewards = []
        seqlens = []

        results = []
        loop = asyncio.get_event_loop()
        for resp in resps:
            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions

            prompt_str = self.tokenizer.decode(input_ids)
            completions_str = self.tokenizer.decode(resp.output_tokens)
            prompt_strs.append(prompt_str)
            completions_strs.append(completions_str)
            seqlens.append(len(seq))
            try:
                reward = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.rw_executor,
                        functools.partial(
                            self.reward_fn,
                            prompt_str,
                            completions_str,
                            resp.input_tokens,
                            resp.output_tokens,
                            **data,
                        ),
                    ),
                    timeout=REWARD_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                reward = 0
            rewards.append(reward)
            res = dict(
                # unsqueeze to add an additional batch dimension
                input_ids=torch.tensor(seq).unsqueeze(0),
                loss_mask=torch.tensor(loss_mask).unsqueeze(0),
                logprobs=torch.tensor(logprobs).unsqueeze(0),
                versions=torch.tensor(versions).unsqueeze(0),
                attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                # reward
                rewards=torch.tensor([float(reward)]),
            )
            results.append(TensorDict(res, batch_size=[1]))

        success_rate = sum(r > 0 for r in rewards) / n_samples

        if self.dump_dir is not None:
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)
            # Get the unique identifier for this prompt
            qid = None
            for key in ["query_id", "id", "qid"]:
                qid = data.get(key, None)
                if qid is not None:
                    break
            qid = qid or uuid.uuid4().hex

            # Dump rollout to file
            with open(
                os.path.join(self.dump_dir, str(version), f"{qid}.txt"), "a"
            ) as f:
                n_samples = self.gconfig.n_samples
                for i, (p, c, r, sl) in enumerate(
                    zip(prompt_strs, completions_strs, rewards, seqlens)
                ):
                    info = "\n".join(
                        [
                            f"idx: {i + 1} / {n_samples}, seqlen: {sl}, reward: {r}, success rate: {success_rate:.2f}.",
                            f"prompt is \n{colorama.Fore.YELLOW + colorama.Style.DIM}{p}{colorama.Style.RESET_ALL}",
                            f"sequence is: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{c}{colorama.Style.RESET_ALL}",
                        ]
                    )
                    f.write(info + "\n")

        if success_rate > self.success_rate_ub or success_rate < self.success_rate_lb:
            logger.info(
                f"Success rate {success_rate:.2f} is out of bounds [{self.success_rate_lb}, {self.success_rate_ub}]. Rejected."
            )
            return RolloutResult(data=None, index=data["index"])
        return RolloutResult(data=concat_padded_tensors(results), index=data["index"])
