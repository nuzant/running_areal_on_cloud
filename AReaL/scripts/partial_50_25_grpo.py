import os
import sys

import torch
import torch.distributed as dist
from datasets.distributed import split_dataset_by_node
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import GRPOConfig, load_expr_config
from arealite.api.io_struct import AllocationMode, FinetuneSpec, WeightUpdateMeta
from arealite.engine.ppo.actor import FSDPPPOActor
from arealite.engine.sglang_remote import RemoteSGLangEngine
from arealite.utils.device import log_gpu_stats
from arealite.utils.evaluator import Evaluator
from arealite.utils.recover import RecoverHandler, StepInfo
from arealite.utils.saver import Saver
from arealite.utils.stats_logger import StatsLogger
from arealite.workflow.rlvr import RLVRWorkflow
from datasets import Dataset, load_dataset, load_from_disk
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import logging, seeding, stats_tracker

logger = logging.getLogger("Partial-50-25 grpo")


def process_gsm8k_rl_dataset(dataset: Dataset):
    def process(sample):
        messages = [{"role": "user", "content": sample["question"]}]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["question"])
    return dataset


def get_gsm8k_dataset(split, rank, world_size):
    dataset = load_from_disk("/storage/running_areal_on_cloud/AReaL/datasets/openr1_25/openr1_25")[split]
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return process_gsm8k_rl_dataset(dataset)


# def gsm8k_reward_fn(
#     prompt, completions, prompt_ids, completion_ids, answer, **kwargs
# ):
#     from pebble import ProcessExpired, ProcessPool

#     from realhf.impl.dataset.math_parser import process_results

#     jobs = []
#     with ProcessPool(max_workers=1) as executor:
#         job = executor.schedule(
#             process_results, args=[completions, answer], timeout=15
#         )
#         jobs.append(job)

#     label = 0
#     for job in jobs:
#         try:
#             x = job.result()
#         except TimeoutError:
#             logger.warning(f"Timeout occurred while justifying the math answer.")
#             x = (0, "timeout", "timeout")
#         except ProcessExpired as e:
#             logger.warning(f"Process terminated abnormally: {e}")
#             x = (0, "error", "error")
#         except Exception as e:
#             logger.warning(f"Other error occurred: {e.__class__.__name__}, {e}")
#             x = (0, "error", "error")
#         label = label or x[0]
#     return label


def gsm8k_reward_fn(
    prompt, completions, prompt_ids, completion_ids, query_id, solutions, **kwargs
):
    from realhf.impl.dataset.math_parser import process_results

    label = 0
    for sol in solutions:
        x = process_results(completions, sol)
        label = label or x[0]
    return label


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    config: GRPOConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")

    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        get_gsm8k_dataset("train", rank, world_size),
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )
    valid_dataloader = StatefulDataLoader(
        get_gsm8k_dataset("test", rank, world_size),
        batch_size=config.valid_dataset.batch_size // world_size,
        shuffle=config.valid_dataset.shuffle,
        num_workers=config.valid_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.valid_dataset.drop_last,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, ft_spec)
    eval_rollout = RemoteSGLangEngine(config.rollout)
    eval_rollout.initialize(None, ft_spec)
    # NOTE: set a large version such that eval does not have any offpolicyness control
    eval_rollout.set_version(int(1e12))

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.initialize(None, ft_spec)
    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.initialize(None, ft_spec)

    # NOTE: Weight update meta only requires address and free port of rank 0,
    # but `WeightUpdateMeta.from_fsdp_nccl` has to be executed on all ranks
    # due to `engine.get_param_specs()`.
    # Therefore, we create weight update meta on all ranks, then broadcast the one on rank 0.
    weight_update_meta = [
        WeightUpdateMeta.from_fsdp_nccl(
            AllocationMode.from_str(config.allocation_mode), actor
        )
        # WeightUpdateMeta.from_disk(
        #     config.experiment_name, config.trial_name, config.cluster.fileroot
        # )
    ]
    dist.broadcast_object_list(weight_update_meta, src=0)
    weight_update_meta = weight_update_meta[0]

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.rlvr.gconfig.stop_token_ids:
        config.rlvr.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.rlvr.gconfig.stop_token_ids:
        config.rlvr.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        config=config.rlvr,
        tokenizer=tokenizer,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )

    workflow_test = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        config=config.rlvr_test,
        tokenizer=tokenizer,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated_test"
        ),
    )

    # Run training.
    saver = Saver(config.saver, ft_spec, for_recover=False)
    logger = StatsLogger(config.stats_logger, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)
    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        actor,
        saver,
        evaluator,
        logger,
        train_dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
        workflow=workflow,
    )
    start_step = (
        recover_info.recover_start.global_step if recover_info is not None else 0
    )

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    logger.info(f"total_epochs={total_epochs} step_per_epoch={steps_per_epoch}")
    data_generator = iter(train_dataloader)
    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        with stats_tracker.record_timing("rollout"):
            if config.async_training:
                batch = rollout.prepare_batch(
                    train_dataloader,
                    workflow=workflow,
                    should_accept=lambda x: x is not None,
                )
            else:
                try:
                    data = next(data_generator)
                except StopIteration:
                    data_generator = iter(train_dataloader)
                    data = next(data_generator)
                batch = rollout.rollout_batch(data, workflow=workflow)

        batch = batch.to(actor.device)
        # Create barrier to synchronize all rollout processes.
        dist.barrier(device_ids=[actor.device.index])
        torch.cuda.synchronize()

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

        with stats_tracker.record_timing("update_weights"):
            rollout.pause()
            if dist.get_rank() == 0:
                future = rollout.update_weights(weight_update_meta)
            actor.upload_weights(weight_update_meta)
            if dist.get_rank() == 0:
                future.result()
            dist.barrier(device_ids=[actor.device.index])
            torch.cuda.synchronize()
            rollout.resume()
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step)

        # with stats_tracker.record_timing("eval"):

        #     def evaluate_fn():
        #         rollout.pause()
        #         cnt = 0
        #         for data in valid_dataloader:
        #             for item in data:
        #                 eval_rollout.submit(item, workflow_test)
        #                 cnt += 1
        #         batch = eval_rollout.wait(cnt, timeout=None)
        #         rewards = batch["rewards"].float().to(actor.device)
        #         with stats_tracker.scope("grpo-eval"):
        #             stats_tracker.denominator(
        #                 n_seqs=torch.ones(
        #                     rewards.shape[0],
        #                     device=rewards.device,
        #                     dtype=torch.bool,
        #                 )
        #             )
        #             stats_tracker.stat(task_reward=rewards, denominator="n_seqs")
        #         rollout.resume()

        #     evaluator.evaluate(
        #         evaluate_fn,
        #         epoch,
        #         step,
        #         global_step,
        #     )

        with stats_tracker.record_timing("recover"):
            # recover_handler.dump_recover_info(
            #     step_info, saver, evaluator, logger, train_dataloader
            # )
            # recover_handler.save_checkpoint(actor, step_info)
            recover_handler.dump(
                actor, step_info, saver, evaluator, logger, train_dataloader, rollout
            )

        logger.commit(epoch, step, global_step, stats)

        logger.commit(epoch, step, global_step, stats)

    logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
