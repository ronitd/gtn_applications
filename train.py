import argparse
import editdistance
import itertools
import json
import logging
import os
import sys
import time
import torch

import datasets
import models
import utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a handwriting recognition model."
    )
    parser.add_argument(
        "--config", type=str, help="A json configuration file for experiment."
    )
    parser.add_argument("--disable_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--checkpoint_path",
        default="/tmp/",
        type=str,
        help="Checkpoint path for saving models",
    )
    parser.add_argument(
        "--world_size", default=1, type=int, help="world size for distributed training"
    )
    parser.add_argument(
        "--dist_url",
        default="tcp://localhost:23146",
        type=str,
        help="url used to set up distributed training. This should be"
        "the IP address and open port number of the master node",
    )
    parser.add_argument(
        "--dist_backend", default="nccl", type=str, help="distributed backend"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    use_cpu = args.disable_cuda or not torch.cuda.is_available()
    if args.world_size > 1 and use_cpu:
        logging.fatal("CPU distributed training not supported.")
        sys.exit(1)

    logging.info("World size is : " + str(args.world_size))

    if not use_cpu and torch.cuda.device_count() < args.world_size:
        logging.fatal(
            "At least {} cuda devices required. {} found".format(
                args.world_size, torch.cuda.device_count()
            )
        )
        sys.exit(1)

    if getattr(sys.flags, "nogil", False) and sys.flags.nogil:
        logging.info("Running without GIL")
    else:
        logging.info("Running with GIL")

    return args


def compute_edit_distance(predictions, targets, preprocessor):
    dist = 0
    n_tokens = 0
    for p, t in zip(predictions, targets):
        p, t = preprocessor.tokens_to_text(p), preprocessor.to_text(t)
        dist += editdistance.eval(p, t)
        n_tokens += len(t)
    return dist, n_tokens


@torch.no_grad()
def test(model, criterion, data_loader, preprocessor, device, world_size):
    model.eval()
    criterion.eval()
    meters = utils.Meters()
    for inputs, targets in data_loader:
        outputs = model(inputs.to(device))
        meters.loss += criterion(outputs, targets).item() * len(targets)
        meters.num_samples += len(targets)
        dist, toks = compute_edit_distance(
            criterion.viterbi(outputs), targets, preprocessor
        )
        meters.edit_distance += dist
        meters.num_tokens += toks
    if world_size > 1:
        meters.sync()
    return meters.avg_loss, meters.cer


def checkpoint(model, criterion, checkpoint_path, save_best=False):
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    model_checkpoint = os.path.join(checkpoint_path, "model.checkpoint")
    criterion_checkpoint = os.path.join(checkpoint_path, "criterion.checkpoint")
    torch.save(model.state_dict(), model_checkpoint)
    torch.save(criterion.state_dict(), criterion_checkpoint)
    if save_best:
        torch.save(model.state_dict(), model_checkpoint + ".best")
        torch.save(criterion.state_dict(), criterion_checkpoint + ".best")


def train(world_rank, args):
    # setup logging
    level = logging.INFO
    if world_rank != 0:
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)

    with open(args.config, "r") as fid:
        config = json.load(fid)
        logging.info("Using the config \n{}".format(json.dumps(config)))

    is_distributed_train = False
    if args.world_size > 1:
        is_distributed_train = True
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=world_rank,
        )

    if not args.disable_cuda:
        device = torch.device("cuda")
        torch.cuda.set_device(world_rank)
    else:
        device = torch.device("cpu")

    # seed everything:
    seed = config.get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)

    # setup data loaders:
    logging.info("Loading dataset ...")
    dataset = config["data"]["dataset"]
    if not (hasattr(datasets, dataset)):
        raise ValueError(f"Unknown dataset {dataset}")
    dataset = getattr(datasets, dataset)

    input_size = config["data"]["img_height"]
    data_path = config["data"]["data_path"]
    preprocessor = dataset.Preprocessor(
        data_path,
        img_height=input_size,
        tokens_path=config["data"].get("tokens", None),
        lexicon_path=config["data"].get("lexicon", None),
    )
    trainset = dataset.Dataset(data_path, preprocessor, split="train", augment=True)
    valset = dataset.Dataset(data_path, preprocessor, split="validation")
    train_loader = utils.data_loader(trainset, config, world_rank, args.world_size)
    val_loader = utils.data_loader(valset, config, world_rank, args.world_size)

    # setup criterion, model:
    logging.info("Loading model ...")
    criterion, output_size = models.load_criterion(
        config.get("criterion_type", "ctc"), preprocessor, config.get("criterion", {}),
    )
    criterion = criterion.to(device)
    model = models.load_model(
        config["model_type"], input_size, output_size, config["model"]
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(
        "Training {} model with {:,} parameters.".format(config["model_type"], n_params)
    )

    # Store base module, criterion for saving checkpoints
    base_model = model
    base_criterion = criterion  # `decode` cannot be called on DDP module
    if is_distributed_train:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[world_rank]
        )

        if len(list(criterion.parameters())) > 0:
            criterion = torch.nn.parallel.DistributedDataParallel(
                criterion, device_ids=[world_rank]
            )

    epochs = config["optim"]["epochs"]
    lr = config["optim"]["learning_rate"]
    step_size = config["optim"]["step_size"]
    max_grad_norm = config["optim"].get("max_grad_norm", None)

    # run training:
    logging.info("Starting training ...")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=0.5
    )
    crit_optimizer = crit_scheduler = None
    if len(list(criterion.parameters())) > 0:
        crit_lr = config["optim"]["crit_learning_rate"]
        crit_optimizer = torch.optim.SGD(criterion.parameters(), lr=crit_lr)
        crit_scheduler = torch.optim.lr_scheduler.StepLR(
            crit_optimizer, step_size=step_size, gamma=0.5
        )

    min_val_loss = float("inf")
    min_val_cer = float("inf")

    Timer = utils.CudaTimer if device.type == "cuda" else utils.Timer
    timers = Timer(
        [
            "ds_fetch",  # dataset sample fetch
            "model_fwd",  # model forward
            "crit_fwd",  # criterion forward
            "bwd",  # backward (model + criterion)
            "optim",  # optimizer step
            "metrics",  # viterbi, cer
            "train_total",  # total training
            "test_total",  # total testing
        ]
    )
    num_updates = 0
    for epoch in range(epochs):
        logging.info("Epoch {} started. ".format(epoch + 1))
        model.train()
        criterion.train()
        start_time = time.time()
        meters = utils.Meters()
        timers.reset()
        timers.start("train_total").start("ds_fetch")
        for inputs, targets in train_loader:
            timers.stop("ds_fetch").start("model_fwd")
            optimizer.zero_grad()
            if crit_optimizer:
                crit_optimizer.zero_grad()
            outputs = model(inputs.to(device))
            timers.stop("model_fwd").start("crit_fwd")
            loss = criterion(outputs, targets)
            timers.stop("crit_fwd").start("bwd")
            loss.backward()
            timers.stop("bwd").start("optim")
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(model.parameters(), criterion.parameters()),
                    max_grad_norm,
                )
            optimizer.step()
            if crit_optimizer:
                crit_optimizer.step()
            num_updates += 1
            timers.stop("optim").start("metrics")
            meters.loss += loss.item() * len(targets)
            meters.num_samples += len(targets)
            dist, toks = compute_edit_distance(
                base_criterion.viterbi(outputs), targets, preprocessor
            )
            meters.edit_distance += dist
            meters.num_tokens += toks
            timers.stop("metrics").start("ds_fetch")
        timers.stop("ds_fetch").stop("train_total")
        epoch_time = time.time() - start_time
        if args.world_size > 1:
            meters.sync()
        logging.info(
            "Epoch {} complete. "
            "nUpdates {}, Loss {:.3f}, CER {:.3f}, Time {:.3f} (s)".format(
                epoch + 1, num_updates, meters.avg_loss, meters.cer, epoch_time
            ),
        )
        logging.info("Evaluating validation set..")
        timers.start("test_total")
        val_loss, val_cer = test(
            model, base_criterion, val_loader, preprocessor, device, args.world_size
        )
        timers.stop("test_total")
        if world_rank == 0:
            checkpoint(
                base_model,
                base_criterion,
                args.checkpoint_path,
                (val_cer < min_val_cer),
            )

            min_val_loss = min(val_loss, min_val_loss)
            min_val_cer = min(val_cer, min_val_cer)
        logging.info(
            "Validation Set: Loss {:.3f}, CER {:.3f}, "
            "Best Loss {:.3f}, Best CER {:.3f}".format(
                val_loss, val_cer, min_val_loss, min_val_cer
            ),
        )
        logging.info(
            "Timing Info: "
            + ", ".join(
                [
                    "{} : {:.2f}ms".format(k, v * 1000.0)
                    for k, v in timers.value().items()
                ]
            )
        )
        scheduler.step()
        if crit_scheduler:
            crit_scheduler.step()
        start_time = time.time()

    if is_distributed_train:
        torch.distributed.destroy_process_group()


def main():
    args = parse_args()
    if args.world_size > 1:
        torch.multiprocessing.spawn(
            train, args=(args,), nprocs=args.world_size, join=True
        )
    else:
        train(0, args)


if __name__ == "__main__":
    main()
