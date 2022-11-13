# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A script to run multinode training with submitit.
"""

import argparse
import os
import uuid
import numpy as np
from pathlib import Path
import time
import shutil
import itertools
from distutils import dir_util

import train as classification
import submitit

FOLDER_NAME = "paper"

def parse_args():
    classification_parser = classification.get_parser()
    parser = argparse.ArgumentParser(
        "Submitit for recur", parents=[classification_parser]
    )
    parser.add_argument(
        "--ngpus", default=8, type=int, help="Number of gpus to request on each node"
    )
    parser.add_argument(
        "--nodes", default=2, type=int, help="Number of nodes to request"
    )
    parser.add_argument("--timeout", default=4000, type=int, help="Duration of the job")
    parser.add_argument(
        "--job_dir", default="", type=str, help="Job dir. Leave empty for automatic."
    )

    parser.add_argument(
        "--partition",
        default="devlab,learnlab",
        type=str,
        help="Partition where to submit",
    )
    parser.add_argument(
        "--use_volta32", action="store_true", help="Big models? Use this"
    )
    parser.add_argument(
        "--comment",
        default="icml",
        type=str,
        help="Comment to pass to scheduler, e.g. priority message",
    )
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path("/checkpoint/{}/symbolicregression".format(user))
        # p = p / str(int(time.time()))
        p = p / FOLDER_NAME
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import train as classification

        self._setup_gpu_args()
        classification.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        checkpoint_file = os.path.join(self.args.job_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.load_checkpoint = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.job_dir = Path(
            str(self.args.job_dir).replace("%j", str(job_env.job_id))
        )
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():

    args = parse_args()
    shared_folder = get_shared_folder()

    grid = {
        'use_skeleton':[False],
        "tokens_per_batch":[10000, 20000],
        'lr': [0.0002, 0.0004],
        'emb_emb_dim':[64,128],
    }

    def dict_product(d):
        keys = d.keys()
        for element in itertools.product(*d.values()):
            yield dict(zip(keys, element))

    for params in dict_product(grid):

        args.master_port = np.random.randint(10001, 20000)
        args.float_constants = True
        args.prediction_sigmas="1,2,4,8,16,32"
        args.max_input_dimension = 10
        args.n_steps_per_epoch = 3000
        args.use_volta32 = True
        args.eval_size = 2000
        args.batch_size_eval = 64
        args.lr = 0.0002

        name = "_".join(["{}_{}".format(k, v) for k, v in params.items()])
        args.job_dir = shared_folder / name
        Path(args.job_dir).mkdir(exist_ok=True)

        for f in os.listdir():
            if f.endswith(".py"):
                shutil.copy2(f, args.job_dir)
        dir_util.copy_tree("symbolicregression", os.path.join(args.job_dir, "symbolicregression"))
        os.chdir(args.job_dir)

        args.exp_id = args.job_dir.name
        args.exp_name = args.job_dir.parent.name
        args.dump_path = args.job_dir.parent.parent

        # Note that the folder will depend on the job_id, to easily track experiments
        executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

        kwargs = {}
        if args.use_volta32:
            kwargs["slurm_constraint"] = "volta32gb"
        if args.comment:
            kwargs["slurm_comment"] = args.comment
        executor.update_parameters(
            mem_gb=320,
            gpus_per_node=args.ngpus,
            tasks_per_node=args.ngpus,
            cpus_per_task=10,
            nodes=args.nodes,
            timeout_min=args.timeout,  # max is 60 * 72
            slurm_partition=args.partition,
            **kwargs,
        )

        executor.update_parameters(name=name)

        for k, v in params.items():
            setattr(args, k, v)

        trainer = Trainer(args)
        job = executor.submit(trainer)

        print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()