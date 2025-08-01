import asyncio
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import torch.distributed as dist
import uvloop
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import InferenceEngineConfig
from arealite.api.engine_api import InferenceEngine
from arealite.api.io_struct import RolloutStat
from arealite.utils.data import concat_padded_tensors
from realhf.base import logging

if TYPE_CHECKING:
    from arealite.api.engine_api import InferenceEngine

logger = logging.getLogger("arealite.workflow_api")


ROLLOUT_POLL_WAIT_TIME = 0.05


@dataclass
class RolloutResult:
    data: TensorDict
    index: int | None = None


class RolloutWorkflow:

    async def arun_episode(
        self, engine: "InferenceEngine", data: Dict[str, Any]
    ) -> RolloutResult:
        """Run a single episode of the workflow.

        See concrete example implementations under the `arealite/workflow` directory.
        """
        raise NotImplementedError()


class WorkflowExecutor:

    def __init__(
        self,
        config: InferenceEngineConfig,
        inference_engine: "InferenceEngine",
    ):
        config.max_concurrent_rollouts = (
            config.max_concurrent_rollouts or config.consumer_batch_size
        )
        self.config = config
        self.exiting = threading.Event()
        self.paused = threading.Event()
        self.lock = threading.Lock()

        self.inference_engine = inference_engine

        qsize = config.queue_size or config.max_concurrent_rollouts * 16
        self.input_queue = queue.Queue(maxsize=qsize)
        self.output_queue = queue.Queue(maxsize=qsize)
        self.result_cache: List[TensorDict] = []

        self.inflight_data_indices: List[int] = []
        self.rollout_stat = RolloutStat()

    def initialize(self):
        self.rollout_tasks: Dict[str, asyncio.Task] = {}
        self.rollout_thread = threading.Thread(target=self._rollout_thread, daemon=True)
        self.rollout_thread.start()

    def destroy(self):
        self.exiting.set()
        self.rollout_thread.join()

    def get_capacity(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1

        max_concurrent_rollouts = max(
            1, self.config.max_concurrent_rollouts // world_size
        )
        capacity = max_concurrent_rollouts - len(self.rollout_tasks)
        # Staleness control
        with self.lock:
            version = self.inference_engine.get_version()
        ofp = self.config.max_head_offpolicyness
        with self.lock:
            sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running
        consumer_bs = max(1, self.config.consumer_batch_size // world_size)
        capacity = min(capacity, (ofp + version + 1) * consumer_bs - sample_cnt)
        return capacity

    def _rollout_thread(self):
        """Thread that runs the rollout loop."""
        try:
            uvloop.run(self._rollout_thread_async())
        except Exception:
            traceback.print_exc()

    async def _rollout_thread_async(self):
        rollout_tasks = self.rollout_tasks
        rid = 0
        try:
            while not self.exiting.is_set():
                # Check capacity
                capacity = self.get_capacity()
                # Create new rollout task
                while (
                    capacity > 0
                    and not self.paused.is_set()
                    and self.input_queue.qsize() > 0
                ):
                    data, workflow = self.input_queue.get_nowait()
                    logger.debug(f"Get data from puller: {data}")
                    task = asyncio.create_task(
                        workflow.arun_episode(self.inference_engine, data),
                        name=str(rid),
                    )
                    with self.lock:
                        rollout_tasks[str(rid)] = task
                        self.rollout_stat.submitted += 1
                        self.rollout_stat.running += 1
                        if self.config.enable_rollout_tracing:
                            logger.info(
                                f"Submit rollout rid {rid}. "
                                f"Submit: {self.rollout_stat.submitted}, "
                                f"running: {self.rollout_stat.running}, "
                                f"accepted: {self.rollout_stat.accepted}."
                            )
                    capacity -= 1
                    rid += 1
                # Wait for rollout completion
                with self.lock:
                    tasks = list(rollout_tasks.values())
                done = []
                if tasks:
                    done, _ = await asyncio.wait(
                        tasks,
                        timeout=ROLLOUT_POLL_WAIT_TIME,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                # Collect done results
                for task in done:
                    traj = await task
                    traj: RolloutResult
                    task_rid = task.get_name()
                    with self.lock:
                        rollout_tasks.pop(task_rid)
                        self.rollout_stat.accepted += 1

                    try:
                        self.output_queue.put_nowait(traj)
                    except queue.Full:
                        raise RuntimeError(
                            "Output queue full. Please increase queue_size."
                        )

                    with self.lock:
                        self.rollout_stat.running -= 1
                        if self.config.enable_rollout_tracing:
                            logger.info(
                                f"Finish rollout {task_rid}. "
                                f"Submit: {self.rollout_stat.submitted}, "
                                f"running: {self.rollout_stat.running}, "
                                f"accepted: {self.rollout_stat.accepted}."
                            )
                await asyncio.sleep(1)
        except Exception:
            traceback.print_exc()
        finally:
            # Cancel remaining tasks
            with self.lock:
                for task in rollout_tasks.values():
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

    def submit(self, data: Dict[str, Any], workflow: "RolloutWorkflow") -> None:
        try:
            self.input_queue.put_nowait((data, workflow))
            if "index" in data:
                self.inflight_data_indices.append(data["index"])
        except queue.Full:
            raise RuntimeError("Input queue full. Please increase queue_size.")

    def wait(
        self,
        count: int,
        timeout: float | None = None,
        should_accept: Callable | None = None,
    ) -> TensorDict:
        tik = time.perf_counter()
        accepted = len(self.result_cache)
        timeout = timeout or float(7 * 24 * 3600)
        while (
            accepted < count
            and not self.exiting.is_set()
            and time.perf_counter() - tik < timeout
        ):
            try:
                result: RolloutResult = self.output_queue.get(
                    timeout=ROLLOUT_POLL_WAIT_TIME
                )
                if result.index:
                    # logger.info(f"Remove inflight index {result.index}")
                    self.inflight_data_indices.remove(result.index)
                result = result.data
                if should_accept is None or should_accept(result):
                    self.result_cache.append(result)
                    accepted += 1
                else:
                    with self.lock:
                        self.rollout_stat.accepted -= 1
            except queue.Empty:
                pass
        if self.exiting.is_set():
            raise RuntimeError("Rollout engine is exiting, cannot wait for results.")
        if accepted < count:
            raise TimeoutError(
                f"Timed out waiting for {count} rollouts, " f"only received {accepted}."
            )
        results, self.result_cache = (
            self.result_cache[:count],
            self.result_cache[count:],
        )
        return concat_padded_tensors(results)

    def rollout_batch(
        self, data: List[Dict[str, Any]], workflow: "RolloutWorkflow"
    ) -> TensorDict:
        """Submit a batch of requests to the inference engine and wait for the results."""
        for item in data:
            self.submit(item, workflow)
        return self.wait(count=len(data))

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: "RolloutWorkflow",
        should_accept: Callable | None = None,
    ):
        if not hasattr(self, "data_generator"):
            self.data_generator = iter(dataloader)
        assert dataloader.batch_size is not None
        while True:
            # Submit at least two batches to allow maximum overlap
            if (
                self.get_capacity() + dataloader.batch_size > 0
                and self.input_queue.qsize() + dataloader.batch_size
                < self.input_queue.maxsize
            ):
                try:
                    data = next(self.data_generator)
                except StopIteration:
                    self.data_generator = iter(dataloader)
                    data = next(self.data_generator)
                for item in data:
                    self.submit(item, workflow=workflow)
            try:
                return self.wait(
                    dataloader.batch_size, timeout=1, should_accept=should_accept
                )
            except TimeoutError:
                pass

    def pause(self):
        self.paused.set()

    def resume(self):
        self.paused.clear()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "inflight_data_indices": self.inflight_data_indices,
        }

    def recover(
        self,
        state_dict: Dict[str, Any],
        recover_dataloader: StatefulDataLoader,
        workflow: RolloutWorkflow,
    ):
        # TODO: only temporary solution, should be revised.
        generator = iter(recover_dataloader)
        recover_indices = state_dict.get("inflight_data_indices", [])
        logger.info(
            f"Rank {dist.get_rank()} Submitting {len(recover_indices)} items for recovery."
        )
        while True:
            try:
                data = next(generator)
            except StopIteration:
                break

            for item in data:
                if "index" in item and item["index"] in recover_indices:
                    # logger.info(f"Submitting item for recovery: index={item['index']}")
                    self.submit(item, workflow=workflow)
