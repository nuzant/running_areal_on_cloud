#!/bin/bash

cmd="python3 -m arealite.launcher.ray scripts/partial_50_25_grpo.py --config scripts/partial_50_25_grpo.yaml \
     experiment_name=mzy-test trial_name=run0 cluster.n_nodes=4 cluster.n_gpus_per_node=8 allocation_mode=sglang.d16p1t1+d16p1t1"

docker exec raycluster-root ray status
# docker exec raycluster-root bash -c "cd /AReaL; $cmd"