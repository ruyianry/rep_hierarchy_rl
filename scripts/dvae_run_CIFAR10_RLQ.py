#!/usr/bin/env python3

import argparse
import copy
import datetime
import logging
import time
import os
import sys
import gc
from typing import List

from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import rich
import sklearn.metrics
import torch
import torch.utils.data
from torch.utils.data import ConcatDataset, DataLoader, Subset, SubsetRandomSampler

from tqdm import tqdm

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import oodd
import oodd.models
import oodd.datasets
import oodd.variational
import oodd.losses

from oodd.utils import str2bool, get_device, log_sum_exp, set_seed, plot_gallery, reduce_to_batch, \
    reduce_batch
from oodd.evaluators import Evaluator

LOGGER = logging.getLogger(name=__file__)

try:
    import wandb

    wandb_available = True
except ImportError:
    LOGGER.warning("Running without remote tracking!")
    wandb_available = False

parser = argparse.ArgumentParser(description="VAE MNIST Example")
parser.add_argument("--model", default="VAE", help="model type (vae | lvae | biva)")
parser.add_argument("--epochs", type=int, default=1, help="number of epochs to train")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
parser.add_argument("--train_samples", type=int, default=1,
                    help="samples from approximate posterior")
parser.add_argument("--test_samples", type=int, default=1,
                    help="samples from approximate posterior")
parser.add_argument("--train_importance_weighted", type=str2bool, default=False, const=True,
                    nargs="?", help="use iw bound")
parser.add_argument("--test_importance_weighted", type=str2bool, default=False, const=True,
                    nargs="?", help="use iw bound")
parser.add_argument("--warmup_epochs", type=int, default=200, help="epochs to warm up the KL term.")
parser.add_argument("--free_nats_epochs", type=int, default=400,
                    help="epochs to warm up the KL term.")
parser.add_argument("--free_nats", type=float, default=2,
                    help="nats considered free in the KL term")
parser.add_argument("--n_eval_samples", type=int, default=32,
                    help="samples from prior for quality inspection")
parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed")
parser.add_argument("--test_every", type=int, default=10, help="epochs between evaluations")
parser.add_argument("--save_dir", type=str, default="./models", help="directory for saving models")
parser.add_argument("--use_wandb", type=str2bool, default=True, help="use wandb tracking")
parser.add_argument("--name", type=str, default=True, help="wandb tracking name")

CONFIG_deterministic = [
    [
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 5, "stride": 1,
         "weightnorm": True, "gated": False},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 5, "stride": 1,
         "weightnorm": True, "gated": False},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 5, "stride": 2,
         "weightnorm": True, "gated": False}
    ],
    [
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 1,
         "weightnorm": True, "gated": False},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 1,
         "weightnorm": True, "gated": False},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 2,
         "weightnorm": True, "gated": False}
    ],
    [
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 1,
         "weightnorm": True, "gated": False},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 1,
         "weightnorm": True, "gated": False},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 2,
         "weightnorm": True, "gated": False}
    ],
        [
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 1,
         "weightnorm": True, "gated": False},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 1,
         "weightnorm": True, "gated": False},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 2,
         "weightnorm": True, "gated": False}
    ],
        [
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 1,
         "weightnorm": True, "gated": False},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 1,
         "weightnorm": True, "gated": False},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 2,
         "weightnorm": True, "gated": False}
    ]
]
CONFIG_stochastic = [
    {"block": "GaussianConv2d", "latent_features": 128, "weightnorm": True},
    {"block": "GaussianConv2d", "latent_features": 64, "weightnorm": True},
    {"block": "GaussianDense", "latent_features": 256, "weightnorm": True},
    {"block": "GaussianDense", "latent_features": 128, "weightnorm": True},
    {"block": "GaussianDense", "latent_features": 64, "weightnorm": True},
]
LIKELIHOOD = 'DiscretizedLogisticMixLikelihoodConv2d'

parser.add_argument("--config_deterministic", type=list, default=CONFIG_deterministic,
                    help="deterministic config")
parser.add_argument("--config_stochastic", type=list, default=CONFIG_stochastic,
                    help="stochastic config")
parser.add_argument("--likelihood", type=str, default=LIKELIHOOD, help="likelihood")
parser.add_argument("--device", type=int, default=0, help="device")

parser = oodd.datasets.DataModule.get_argparser(parents=[parser])

args, unknown_args = parser.parse_known_args()

## runtime alternating parameters
args.model = 'VAE'
args.epoch = 2000 // 5 # divide this by L if the number of layer is modified
args.batch_size = 128
args.free_nats = 0
args.free_nats_epochs = 0
args.warmup_epochs = 0
args.test_every = 10
args.train_datasets = {"CIFAR10Dequantized": {"dynamic": True, "split": "train"}}
args.val_datasets = {
    "CIFAR10Dequantized": {"dynamic": False, "split": "validation"},
}
args.extra_datasets = {
}
args.test_dataset_name = 'CIFAR10Dequantized'

args.not_verbose = False



### end of finetune parameters

args.start_time = str(datetime.datetime.now()).replace(" ", "-").replace(":", "-")
args.train_sample_reduction = log_sum_exp if args.train_importance_weighted else torch.mean
args.test_sample_reduction = log_sum_exp if args.test_importance_weighted else torch.mean
args.use_wandb = wandb_available and args.use_wandb

set_seed(args.seed)
device = get_device(int(args.device))

# this is the training of HVAE without RL where posterior collapse is a problem
# replace the method call of train_with_RL() with train() to see the effect of posterior collapse
def train(epoch):
    model.train()
    evaluator = Evaluator(primary_metric="log p(x)", logger=LOGGER, use_wandb=args.use_wandb)

    beta = next(deterministic_warmup)
    free_nats = next(free_nats_cooldown)

    iterator = tqdm(enumerate(datamodule.train_loader), smoothing=0.9, total=len(datamodule.train_loader), leave=False)
    for _, (x, _) in iterator:
        x = x.to(device)

        likelihood_data, stage_datas, posteriors = model(x, n_posterior_samples=args.train_samples)
        kl_divergences = [
            stage_data.loss.kl_elementwise for stage_data in stage_datas if stage_data.loss.kl_elementwise is not None
        ]
        loss, elbo, likelihood, kl_divergences = criterion(
            likelihood_data.likelihood,
            kl_divergences,
            samples=args.train_samples,
            free_nats=free_nats,
            beta=beta,
            sample_reduction=args.train_sample_reduction,
            batch_reduction=None,
        )

        l = loss.mean()
        l.backward()

        optimizer.step()
        optimizer.zero_grad()

        evaluator.update("Train", "elbo", {"log p(x)": elbo})
        evaluator.update("Train", "likelihoods", {"loss": -loss, "log p(x)": elbo, "log p(x|z)": likelihood})
        klds = {
            f"KL z{i+1}": kl
            for i, kl in enumerate([sd.loss.kl_samplewise for sd in stage_datas if sd.loss.kl_samplewise is not None])
        }
        klds["KL(q(z|x), p(z))"] = kl_divergences
        evaluator.update("Train", "divergences", klds)

    evaluator.update(
        "Train", "hyperparameters", {"free_nats": [free_nats], "beta": [beta], "learning_rate": [args.learning_rate]}
    )
    evaluator.report(epoch * len(datamodule.train_loader))
    evaluator.log(epoch)

def train_with_RL(epoch, gamma=0.9):
    LOGGER.info("RL (Q-value) training epoch %d", epoch)
    model.train()
    evaluator = Evaluator(primary_metric="log p(x)", logger=LOGGER, use_wandb=args.use_wandb)

    beta = next(deterministic_warmup)
    free_nats = next(free_nats_cooldown)

    iterator = tqdm(enumerate(datamodule.train_loader), smoothing=0.9,
                    total=len(datamodule.train_loader), leave=False, disable=args.not_verbose)

    decode_from_p_combinations = [[True] * n_p + [False] * (model.n_latents - n_p) for n_p in
                                  range(model.n_latents)]

    for batch_id, (x, idx) in iterator:
        ### normal
        anchor = x.to(device)
        # likelihood_data, _, posteriors = model(anchor, n_posterior_samples=args.train_samples, ) # 总的model一个infer+generate
        anchor_x = anchor.repeat(1, *((1,) * (x.ndim - 1)))

        for step in range(model.n_latents):
            likelihood_data_no_skip, stage_datas, posteriors = model(anchor,
                                                                     n_posterior_samples=args.train_samples)
            kl_divergences = [
                stage_data.loss.kl_elementwise for stage_data in stage_datas if
                stage_data.loss.kl_elementwise is not None
            ]
            overall_loss, overall_elbo, overall_likelihood, overall_kl_divergences = criterion(
                likelihood_data_no_skip.likelihood,
                kl_divergences,
                samples=args.train_samples,
                free_nats=free_nats,
                beta=beta,
                sample_reduction=args.train_sample_reduction,
                batch_reduction=None,
            )  # this is for logging only

            Q_value_t = .0  # Q_value for current step

            for i, decode_from_p in enumerate(decode_from_p_combinations[step:]):
                # choose KLs for current step
                selected_kl_divergences = []

                kl_index_to_select = [i for i in range(i + step + 1)]
                for kl_index in kl_index_to_select:
                    selected_kl_divergences.append(kl_divergences[kl_index])

                # get skip likelihood
                likelihood_data_skip_i, stage_datas = model.generate(posteriors=posteriors,
                                                                     x=anchor_x,
                                                                     n_prior_samples=anchor_x.size(0),
                                                                     decode_from_p=decode_from_p,
                                                                     use_mode=True,)
                loss, elbo, likelihood, kl_divergences_i = criterion(likelihood_data_skip_i.likelihood,
                                                                     selected_kl_divergences,
                                                                     samples=args.train_samples,
                                                                     free_nats=free_nats, beta=beta,
                                                                     sample_reduction=args.train_sample_reduction,
                                                                     batch_reduction=None,)

                r_t = loss.mean()
                Q_value_t += np.power(gamma, i) * r_t

                evaluator.update("Train", "elbo", {f"log p(x) [{i}]": elbo})
                evaluator.update("Train", "likelihoods",
                                 {f"log p(x) [{i}]": elbo,
                                  f"log p(x|z) [{i}]": likelihood, })
                klds = {
                    f"KL z{i + 1}": kl
                    for i, kl in enumerate(
                        [sd.loss.kl_samplewise for sd in stage_datas if
                         sd.loss.kl_samplewise is not None])
                }
                klds["KL(q(z|x), p(z))"] = kl_divergences_i
                evaluator.update("Train", f"divergences[{i}]", klds)

            Q_value_t.backward()

            optimizer.step()
            optimizer.zero_grad()

        evaluator.update("Train", "elbo", {"log p(x)": overall_elbo, })
        evaluator.update("Train", "likelihoods",
                         {"loss": -overall_loss, "log p(x)": overall_elbo,
                          "log p(x|z)": overall_likelihood, }
                         )

    evaluator.update(
        "Train", "hyperparameters",
        {"free_nats": [free_nats], "beta": [beta], "learning_rate": [args.learning_rate]}
    )
    evaluator.report(epoch * len(datamodule.train_loader))
    evaluator.log(epoch)



@torch.no_grad()
def test(epoch, dataloader, evaluator, dataset_name="test", max_test_examples=float("inf")):
    LOGGER.info(f"Testing: {dataset_name}")
    model.eval()

    x, _ = next(iter(dataloader))
    x = x.to(device)
    n = min(x.size(0), 8)  # number of examples to show, max 8

    likelihood_data, _, _ = model(x, n_posterior_samples=args.test_samples)
    p_x_mean = likelihood_data.mean[: args.batch_size].view(args.batch_size,
                                                            *in_shape)  # Reshape zeroth "sample"
    p_x_samples = likelihood_data.samples[: args.batch_size].view(args.batch_size,
                                                                  *in_shape)  # Reshape zeroth "sample"
    comparison = torch.cat([x[:n], p_x_mean[:n], p_x_samples[:n]])
    comparison = comparison.permute(0, 2, 3, 1)  # [B, H, W, C]
    fig, ax = plot_gallery(comparison.cpu().numpy(), ncols=n)
    fig.savefig(os.path.join(args.save_dir, f"reconstructions_{dataset_name}_{epoch:03}"))
    plt.close()

    decode_from_p_combinations = [[True] * n_p + [False] * (model.n_latents - n_p) for n_p in
                                  range(model.n_latents)]

    for decode_from_p in tqdm(decode_from_p_combinations, leave=False, disable=args.not_verbose):
        n_skipped_latents = sum(decode_from_p)
        if max_test_examples != float("inf"):  # if we want to limit the number of examples
            iterator = tqdm(
                zip(range(max_test_examples // dataloader.batch_size), dataloader),
                smoothing=0.9,
                total=max_test_examples // dataloader.batch_size,
                leave=False,
                disable=args.not_verbose,
            )
        else:
            iterator = tqdm(enumerate(dataloader), smoothing=0.9, total=len(dataloader),
                            leave=False, disable=args.not_verbose,
                            )

        for _, (x, _) in iterator:  # iterator over batches of test data
            x = x.to(device)

            likelihood_data, stage_datas, zs = model(
                x, n_posterior_samples=args.test_samples, decode_from_p=decode_from_p,
                use_mode=decode_from_p
            )  # returns likelihood_data and stage_datas (prior and posterior)
            kl_divergences = [
                stage_data.loss.kl_elementwise
                for stage_data in stage_datas
                if stage_data.loss.kl_elementwise is not None
            ]  # compute the elementwise kl divergence of each stage
            elbo_raw_loss, elbo, likelihood, kl_divergences = criterion(
                likelihood_data.likelihood,
                kl_divergences,
                samples=args.test_samples,
                free_nats=0,
                beta=1,
                sample_reduction=args.test_sample_reduction,
                batch_reduction=None,
            )

            if n_skipped_latents == model.n_latents - 1:  # Highest
                p_x_mean = likelihood_data.mean[: args.batch_size].view(args.batch_size,
                                                                        *in_shape)  # Reshape zeroth "sample"
                p_x_samples = likelihood_data.samples[: args.batch_size].view(args.batch_size,
                                                                              *in_shape)
                comparison = torch.cat([x[:n], p_x_mean[:n], p_x_samples[:n]])
                comparison = comparison.permute(0, 2, 3, 1)  # [B, H, W, C]
                fig, ax = plot_gallery(comparison.cpu().numpy(), ncols=n)
                fig.savefig(os.path.join(args.save_dir,
                                         f"reconstructions_{dataset_name}_{epoch:03}_fromz{n_skipped_latents + 1}"))
                plt.close()

            loss = elbo_raw_loss
            total_raw_loss = elbo_raw_loss
            if n_skipped_latents == 0:  # Regular ELBO
                evaluator.update(dataset_name, "elbo", {"log p(x)": elbo})
                evaluator.update(
                    dataset_name, "likelihoods",
                    {"loss": -total_raw_loss, "log p(x)": elbo, "log p(x|z)": likelihood,
                     "elbo_loss": elbo_raw_loss,
                     }
                )
                klds = {
                    f"KL z{i + 1}": kl
                    for i, kl in enumerate(
                        [sd.loss.kl_samplewise for sd in stage_datas if
                         sd.loss.kl_samplewise is not None]
                    )
                }
                klds["KL(q(z|x), p(z))"] = kl_divergences
                evaluator.update(dataset_name, "divergences", klds)

            evaluator.update(dataset_name, f"skip-elbo", {f"{n_skipped_latents} log p(x)": elbo})
            evaluator.update(dataset_name, f"skip-elbo-{dataset_name}",
                             {f"{n_skipped_latents} log p(x)": elbo})
            evaluator.update(
                dataset_name,
                f"skip-likelihoods-{dataset_name}",
                {
                    f"{n_skipped_latents} loss": -loss,
                    f"{n_skipped_latents} log p(x)": elbo,
                    f"{n_skipped_latents} log p(x|z)": likelihood,
                },
            )
            klds = {
                f"{n_skipped_latents} KL z{i + 1}": kl
                for i, kl in enumerate(
                    [sd.loss.kl_samplewise for sd in stage_datas if
                     sd.loss.kl_samplewise is not None]
                )
            }
            klds[f"{n_skipped_latents} KL(q(z|x), p(z))"] = kl_divergences
            evaluator.update(dataset_name, f"skip-divergences-{dataset_name}", klds)



if __name__ == "__main__":
    # Data
    datamodule = oodd.datasets.DataModule(
        batch_size=args.batch_size,
        test_batch_size=250,
        data_workers=args.data_workers,
        train_datasets=args.train_datasets,
        val_datasets=args.val_datasets,
        test_datasets=args.test_datasets,
    )

    # get the last folder name of this file
    folder_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args.save_dir = os.path.join(args.save_dir,
                                 list(datamodule.train_datasets.keys())[
                                     0] + "-" + folder_name + "-" + args.start_time)
    os.makedirs(args.save_dir, exist_ok=True)

    fh = logging.FileHandler(os.path.join(args.save_dir, "dvae.log"))
    fh.setLevel(logging.INFO)
    LOGGER.addHandler(fh)

    datamodule.save(args.save_dir)  # Save the config of datamodule so we can load it later

    in_shape = datamodule.train_dataset.datasets[0].size[0]

    # Model
    model = getattr(oodd.models.dvae, args.model)  # get the model from the dvae module
    model_argparser = model.get_argparser()
    model_args, unknown_model_args = model_argparser.parse_known_args()
    model_args.input_shape = in_shape

    model_args.config_deterministic = args.config_deterministic
    model_args.config_stochastic = args.config_stochastic
    model_args.likelihood_module = args.likelihood

    model = model(**vars(model_args)).to(device)  # instantiate the model with the model_args

    p_z_samples = model.prior.sample(torch.Size([args.n_eval_samples])).to(device)
    sample_latents = [None] * (model.n_latents - 1) + [p_z_samples]

    # Optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    criterion = oodd.losses.ELBO()

    deterministic_warmup = oodd.variational.DeterministicWarmup(n=args.warmup_epochs)
    free_nats_cooldown = oodd.variational.FreeNatsCooldown(
        constant_epochs=args.free_nats_epochs // 2,
        cooldown_epochs=args.free_nats_epochs // 2,
        start_val=args.free_nats,
        end_val=0,
    )

    # Logging
    LOGGER.info(f'Using device: {device}')
    LOGGER.info("Experiment config:")
    LOGGER.info(args)
    rich.print(vars(args))
    LOGGER.info("%s", deterministic_warmup)
    LOGGER.info("%s", free_nats_cooldown)
    LOGGER.info("DataModule:\n%s", datamodule)
    LOGGER.info("Model:\n%s", model)

    if args.use_wandb:
        wandb.init(project="hvae-oodd", config=args,
                   name=f"{args.model} {datamodule.primary_val_name} {args.name}")
        wandb.save("*.pt")
        wandb.watch(model, log="all")

    # Run
    test_elbos = [-np.inf]
    test_evaluator = Evaluator(primary_source=datamodule.primary_val_name,
                               primary_metric="log p(x)", logger=LOGGER, use_wandb=args.use_wandb)

    LOGGER.info("Running training...")
    total_num_epoch = args.epoch

    start_epoch = 1
    datamodule.add_datasets(val_datasets=args.extra_datasets)

    classifier = SVC(kernel='linear', gamma='auto', C=1.0,)
    for epoch in range(start_epoch, total_num_epoch + 1):
        train_with_RL(epoch)
        if epoch % args.test_every == 0:
            save_dir = os.path.join(args.save_dir, 'weights', f'E{epoch}')
            model.save(save_dir)
            # Test
            for name, dataloader in datamodule.val_loaders.items():
                test(epoch, dataloader=dataloader, evaluator=test_evaluator, dataset_name=name,
                     max_test_examples=10000)

            ##### Linear Classifier #####
            model.eval()
            start_time = time.time()
            iterator = tqdm(enumerate(datamodule.train_loader), smoothing=0.9,
                            total=len(datamodule.train_loader), leave=False)
            test_iterator = tqdm(enumerate(datamodule.val_loaders[args.test_dataset_name]),
                                 smoothing=0.9,
                                 total=len(datamodule.val_loaders[args.test_dataset_name]),
                                 leave=False)
            train_features = []
            train_labels = []

            highest_latent = model.n_latents - 1

            with torch.no_grad():
                for _, (x, y) in iterator:
                    x = x.to(device)
                    likelihood_data, stage_datas, posterior = model(x, decode_from_p=False,
                                                                    use_mode=False)
                    feature = stage_datas[highest_latent].q.mean
                    train_features.append(feature.detach().cpu().numpy())
                    train_labels.append(y.detach().cpu().numpy())

            train_features = np.concatenate(train_features)
            train_labels = np.concatenate(train_labels)

            test_features = []
            test_labels = []

            with torch.no_grad():
                for _, (x, y) in test_iterator:
                    x = x.to(device)
                    likelihood_data, stage_datas, posterior = model(x, decode_from_p=False,
                                                                    use_mode=False)
                    feature = stage_datas[highest_latent].q.mean
                    test_features.append(feature.detach().cpu().numpy())
                    test_labels.append(y.detach().cpu().numpy())

            test_features = np.concatenate(test_features)
            test_labels = np.concatenate(test_labels)

            start = time.time()
            classifier.fit(train_features, train_labels)
            acc = classifier.score(test_features, test_labels) * 100
            gc.collect()

            end = time.time()
            LOGGER.info(f'E{epoch} Linear classifier accuracy: {acc}, time: {end - start}')
