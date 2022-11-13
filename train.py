# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import random
import argparse
import numpy as np
import torch
import os
import pickle
from pathlib import Path

import symbolicregression
from symbolicregression.slurm import init_signal_handler, init_distributed_mode
from symbolicregression.utils import bool_flag, initialize_exp
from symbolicregression.model import check_model_params, build_modules
from symbolicregression.envs import build_env
from symbolicregression.trainer import Trainer
from evaluate import Evaluator
from parsers import get_parser

np.seterr(all="raise")


def main(params):

    # initialize the multi-GPU / multi-node training
    # initialize experiment / SLURM signal handler for time limit / pre-emption
    init_distributed_mode(params)
    logger = initialize_exp(params)
    if params.is_slurm_job:
        init_signal_handler()

    # CPU / CUDA
    if not params.cpu:
        assert torch.cuda.is_available()
    symbolicregression.utils.CUDA = not params.cpu

    # build environment / modules / trainer / evaluator
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)
    if params.eval_dump_path is None:
        params.eval_dump_path = Path(params.dump_path) / "evals_all"
        if not os.path.isdir(params.eval_dump_path):
            os.makedirs(params.eval_dump_path)
    env = build_env(params)

    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    evaluator = Evaluator(trainer)

    # training
    if params.reload_data != "":
        data_types = [
            "valid{}".format(i) for i in range(1, len(trainer.data_path["functions"]))
        ]
    else:
        data_types = ["valid1"]
    evaluator.set_env_copies(data_types)

    # evaluation
    if params.eval_only:
        if params.eval_in_domain:
            scores = evaluator.evaluate_in_domain(
                "valid1",
                "functions",
                logger=logger,
                save=params.save_results,
                ablation_to_keep=params.ablation_to_keep,
            )
            logger.info("__log__:%s" % json.dumps(scores))

        if params.eval_on_pmlb:
            feynman_scores = evaluator.evaluate_pmlb(
                filter_fn=lambda x: x["dataset"].str.contains("feynman")
            )
            logger.info("__feynman__:%s" % json.dumps(feynman_scores))

            filter_fn = lambda x: ~(
                x["dataset"].str.contains("strogatz")
                | x["dataset"].str.contains("feynman")
            )
            black_box_scores = evaluator.evaluate_pmlb(filter_fn=filter_fn)
            logger.info("__black_box__:%s" % json.dumps(black_box_scores))
        exit()

    trainer.n_equations = 0
    for _ in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.inner_epoch = 0
        while trainer.inner_epoch < trainer.n_steps_per_epoch:

            # training steps
            for task_id in np.random.permutation(len(params.tasks)):
                task = params.tasks[task_id]
                if params.export_data:
                    trainer.export_data(task)
                else:
                    trainer.enc_dec_step(task)
                trainer.iter()

        logger.info("============ End of epoch %i ============" % trainer.epoch)
        if params.debug_train_statistics:
            for task in params.tasks:
                trainer.get_generation_statistics(task)

        trainer.save_periodic()

        if params.eval_in_domain:
            scores = evaluator.evaluate_in_domain(
                "valid1",
                "functions",
                logger=logger,
                save=params.save_results,
                ablation_to_keep=params.ablation_to_keep,
            )
            logger.info("__log__:%s" % json.dumps(scores))

        if params.eval_on_pmlb:
            feynman_scores = evaluator.evaluate_pmlb(
                filter_fn=lambda x: x["dataset"].str.contains("feynman")
            )
            logger.info("__feynman__:%s" % json.dumps(feynman_scores))

            filter_fn = lambda x: ~(
                x["dataset"].str.contains("strogatz")
                | x["dataset"].str.contains("feynman")
            )
            black_box_scores = evaluator.evaluate_pmlb(filter_fn=filter_fn)
            logger.info("__black_box__:%s" % json.dumps(black_box_scores))

        trainer.save_best_model(scores, prefix="functions", suffix="fit")
        # end of epoch
        trainer.end_epoch(scores)


if __name__ == "__main__":

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    if params.eval_only and params.eval_from_exp != "":
        if os.path.exists(
            params.eval_from_exp + "/best-" + params.validation_metrics + ".pth"
        ):
            params.reload_model = (
                params.eval_from_exp + "/best-" + params.validation_metrics + ".pth"
            )
        elif os.path.exists(params.eval_from_exp + "/checkpoint.pth"):
            params.reload_model = params.eval_from_exp + "/checkpoint.pth"
        else:
            raise NotImplementedError

        eval_data = params.eval_data

        # read params from pickle
        pickle_file = params.eval_from_exp + "/params.pkl"
        assert os.path.isfile(pickle_file)
        pk = pickle.load(open(pickle_file, "rb"))
        pickled_args = pk.__dict__
        del pickled_args["exp_id"]
        for p in params.__dict__:
            if p in pickled_args:
                params.__dict__[p] = pickled_args[p]

        params.eval_size = None
        if params.reload_data or params.eval_data:
            params.reload_data = (
                params.tasks + "," + eval_data + "," + eval_data + "," + eval_data
            )
        params.is_slurm_job = False
        params.local_rank = -1
        params.master_port = -1
    # params.num_workers = 1

    # debug mode
    if params.debug:
        params.exp_name = "debug"
        if params.exp_id == "":
            params.exp_id = "debug_%08i" % random.randint(0, 100000000)
        params.debug_slurm = True

    # check parameters
    check_model_params(params)

    # run experiment
    main(params)
