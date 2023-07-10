import argparse
import itertools
import json
import logging
import pathlib
import sys

logger = logging.getLogger(pathlib.Path(__file__).name)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(handler)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import tqdm

import data
import models


def get_parser(
    parser=argparse.ArgumentParser(
        description="Run a reverse dictionary baseline.\nThe task consists in reconstructing an embedding from the glosses listed in the datasets"
    ),
):
    parser.add_argument(
        "--do_train", action="store_true", help="whether to train a model from scratch"
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="whether to eval a model from scratch"
    )
    parser.add_argument(
        "--do_pred", action="store_true", help="whether to produce predictions"
    )
    parser.add_argument(
        "--train_file", type=pathlib.Path, help="path to the train file"
    )
    parser.add_argument("--dev_file", type=pathlib.Path, help="path to the dev file")
    parser.add_argument("--test_file", type=pathlib.Path, help="path to the test file")
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cpu"),
        help="path to the train file",
    )
    parser.add_argument(
        "--target_arch",
        type=str,
        default="electra",
        choices=("sgns", "electra"),
        help="embedding architecture to use as target",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=256,
        choices=(300, 256),
        help="dimension of embedding",
    )
    parser.add_argument(
        "--summary_logdir",
        type=pathlib.Path,
        default=pathlib.Path("logs") / f"revdict-baseline",
        help="write logs for future analysis",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        default=pathlib.Path("models") / f"revdict-baseline",
        help="where to save model & vocab",
    )
    parser.add_argument(
        "--pred_file",
        type=pathlib.Path,
        default=pathlib.Path("revdict-baseline-preds.json"),
        help="where to save predictions",
    )
    return parser


def rank_cosine(preds, targets):
    assocs = F.normalize(preds) @ F.normalize(targets).T
    refs = torch.diagonal(assocs, 0).unsqueeze(1)
    ranks = (assocs >= refs).sum(1).float()
    assert ranks.numel() == preds.size(0)
    ranks = ranks.mean().item()
    return ranks / preds.size(0)

def train(args):
    assert args.train_file is not None, "Missing dataset for training"
    # 1. get data, vocabulary, summary writer
    logger.debug("Preloading data")
    ## make datasets
    train_dataset = data.JSONDataset(args.train_file, maxlen=args.max_len)
    if args.dev_file:
        dev_dataset = data.JSONDataset(args.dev_file, vocab=train_dataset.vocab)
    ## assert they correspond to the task
    assert train_dataset.has_gloss, "Training dataset contains no gloss."
    if args.target_arch == "electra":
        assert train_dataset.has_electra, "Training datatset contains no vector."
    else:
        assert train_dataset.has_vecs, "Training datatset contains no vector."
    if args.dev_file:
        assert dev_dataset.has_gloss, "Development dataset contains no gloss."
        if args.target_arch == "electra":
            assert dev_dataset.has_electra, "Development dataset contains no vector."
        else:
            assert dev_dataset.has_vecs, "Development dataset contains no vector."
    ## make dataloader
    train_dataloader = data.get_dataloader(train_dataset, batch_size=512)
    if args.dev_file:
        dev_dataloader = data.get_dataloader(dev_dataset, shuffle=False, batch_size=1024)
    ## make summary writer
    summary_writer = SummaryWriter(args.summary_logdir)
    train_step = itertools.count()  # to keep track of the training steps for logging

    # 2. construct model
    ## Hyperparams
    logger.debug("Setting up training environment")
    model = models.RevdictModel(train_dataset.vocab, d_model=args.max_len).to(args.device)
    model.train()

    # 3. declare optimizer & criterion
    ## Hyperparams
    EPOCHS, LEARNING_RATE, BETA1, BETA2, WEIGHT_DECAY = 1, 1.0e-4, 0.9, 0.999, 1.0e-6
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.MSELoss()

    vec_tensor_key = f"{args.target_arch}_tensor"

    # 4. train model
    for epoch in tqdm.trange(EPOCHS, desc="Epochs"):
        ## train loop
        pbar = tqdm.tqdm(
            desc=f"Train {epoch}", total=len(train_dataset), disable=None, leave=False
        )
        for batch in train_dataloader:
            optimizer.zero_grad()
            gls = batch["gloss_tensor"].to(args.device)
            vec = batch[vec_tensor_key].to(args.device)
            pred = model(gls)
            loss = criterion(pred, vec)
            loss.backward()
            # keep track of the train loss for this step
            next_step = next(train_step)
            summary_writer.add_scalar(
                "revdict-train/cos",
                F.cosine_similarity(pred, vec).mean().item(),
                next_step,
            )
            summary_writer.add_scalar("revdict-train/mse", loss.item(), next_step)
            optimizer.step()
            pbar.update(vec.size(0))
        pbar.close()
        ## eval loop
        if args.dev_file:
            model.eval()
            with torch.no_grad():
                sum_dev_loss, sum_cosine, sum_rnk = 0.0, 0.0, 0.0
                pbar = tqdm.tqdm(
                    desc=f"Eval {epoch}",
                    total=len(dev_dataset),
                    disable=None,
                    leave=False,
                )
                for batch in dev_dataloader:
                    gls = batch["gloss_tensor"].to(args.device)
                    vec = batch[vec_tensor_key].to(args.device)
                    pred = model(gls)
                    sum_dev_loss += (
                        F.mse_loss(pred, vec, reduction="none").mean(1).sum().item()
                    )
                    sum_cosine += F.cosine_similarity(pred, vec).sum().item()
                    sum_rnk += rank_cosine(pred, vec)
                    pbar.update(vec.size(0))
                pbar = tqdm.tqdm(
                    desc=f"Eval {epoch} cos: "+str(sum_cosine / len(dev_dataset))+" mse: "+str( sum_dev_loss / len(dev_dataset) )+" rnk: "+str(sum_rnk/ len(dev_dataset))+ " sum_rnk: "+str(sum_rnk)+" len of dev: "+str(len(dev_dataset)) +"\n",
                    total=len(dev_dataset),
                    disable=None,
                    leave=False,
                )
                # keep track of the average loss on dev set for this epoch
                summary_writer.add_scalar(
                    "revdict-dev/cos", sum_cosine / len(dev_dataset), epoch
                )
                summary_writer.add_scalar(
                    "revdict-dev/mse", sum_dev_loss / len(dev_dataset), epoch
                )
                summary_writer.add_scalar(
                    "revdict-dev/rnk", sum_rnk / len(dev_dataset), epoch
                )
                pbar.close()
            model.train()

    # 5. save result
    model.save(args.save_dir / "model.pt")
    train_dataset.save(args.save_dir / "train_dataset.pt")
    if args.dev_file:
        dev_dataset.save(args.save_dir / "dev_dataset.pt")


def eval(args):
    assert args.dev_file is not None, "Missing dataset for eval"
    # 1. get data, vocabulary, summary writer
    ## make datasets
    #train_dataset = data.JSONDataset(args.train_file)
    train_vocab = data.JSONDataset.load(args.save_dir / "train_dataset.pt").vocab
    model = models.RevdictModel.load(args.save_dir / "model.pt")
    dev_dataset = data.JSONDataset(args.dev_file, vocab=train_vocab, freeze_vocab=True, maxlen=model.maxlen)
    ## assert they correspond to the task
    # assert train_dataset.has_gloss, "Training dataset contains no gloss."
    # if args.target_arch == "electra":
    #     assert train_dataset.has_electra, "Training datatset contains no vector."
    # else:
    #     assert train_dataset.has_vecs, "Training datatset contains no vector."
    
    assert dev_dataset.has_gloss, "Development dataset contains no gloss."
    if args.target_arch == "electra":
          assert dev_dataset.has_electra, "Development dataset contains no vector."
    else:
          assert dev_dataset.has_vecs, "Development dataset contains no vector."
    ## make dataloader
    # train_dataloader = data.get_dataloader(train_dataset, batch_size=512)
    dev_dataloader = data.get_dataloader(dev_dataset, shuffle=False, batch_size=1024)
    ## make summary writer
    summary_writer = SummaryWriter(args.summary_logdir)

    # 2. construct model
    ## Hyperparams
    logger.debug("Setting up eval environment")
   
    # 3. declare optimizer & criterion
    ## Hyperparams
    EPOCHS, LEARNING_RATE, BETA1, BETA2, WEIGHT_DECAY = 3, 1.0e-4, 0.9, 0.999, 1.0e-6


    vec_tensor_key = f"{args.target_arch}_tensor"

    # 4. eval model
    for epoch in tqdm.trange(EPOCHS, desc="Epochs"):
        ## eval loop
        if args.dev_file:
            model.eval()
            with torch.no_grad():
                sum_dev_loss, sum_cosine, sum_rnk = 0.0, 0.0, 0.0
                pbar = tqdm.tqdm(
                    desc=f"Eval {epoch}",
                    total=len(dev_dataset),
                    disable=None,
                    leave=False,
                )
                for batch in dev_dataloader:
                    gls = batch["gloss_tensor"].to(args.device)
                    vec = batch[vec_tensor_key].to(args.device)
                    pred = model(gls)
                    sum_dev_loss += (
                        F.mse_loss(pred, vec, reduction="none").mean(1).sum().item()
                    )
                    sum_cosine += F.cosine_similarity(pred, vec).sum().item()
                    sum_rnk += rank_cosine(pred, vec)
                    pbar.update(vec.size(0))
                # keep track of the average loss on dev set for this epoch
                pbar = tqdm.tqdm(
                    desc=f"Eval {epoch} cos: "+str(sum_cosine / len(dev_dataset))+" mse: "+str( sum_dev_loss / len(dev_dataset) )+" rnk: "+str(sum_rnk/ len(dev_dataset))+" sum_rnk: "+str(sum_rnk)+" len of dev: "+str(len(dev_dataset)) +"\n",
                    total=len(dev_dataset),
                    disable=None,
                    leave=False,
                )
                summary_writer.add_scalar(
                    "revdict-dev/cos", sum_cosine / len(dev_dataset), epoch
                )
                summary_writer.add_scalar(
                    "revdict-dev/mse", sum_dev_loss / len(dev_dataset), epoch
                )
                summary_writer.add_scalar(
                    "revdict-dev/rnk", sum_rnk / len(dev_dataset), epoch
                )
                pbar.close()
            #model.train()

    # 5. save result
    # model.save(args.save_dir / "model.pt")
    # train_dataset.save(args.save_dir / "train_dataset.pt")
    dev_dataset.save(args.save_dir / "dev_dataset.pt")


def pred(args):
    assert args.test_file is not None, "Missing dataset for test"
    # 1. retrieve vocab, dataset, model
    model = models.RevdictModel.load(args.save_dir / "model.pt")
    train_vocab = data.JSONDataset.load(args.save_dir / "train_dataset.pt").vocab
    test_dataset = data.JSONDataset(
        args.test_file, vocab=train_vocab, freeze_vocab=True, maxlen=model.maxlen
    )
    test_dataloader = data.get_dataloader(test_dataset, shuffle=False, batch_size=1024)
    model.eval()
    vec_tensor_key = f"{args.target_arch}_tensor"
    assert test_dataset.has_gloss, "File is not usable for the task"
    # 2. make predictions
    predictions = []
    with torch.no_grad():
        pbar = tqdm.tqdm(desc="Pred.", total=len(test_dataset))
        for batch in test_dataloader:
            vecs = model(batch["gloss_tensor"].to(args.device)).cpu()
            for id, vec in zip(batch["id"], vecs.unbind()):
                predictions.append(
                    {"id": id, args.target_arch: vec.view(-1).cpu().tolist()}
                )
            pbar.update(vecs.size(0))
        pbar.close()
    with open(args.pred_file, "w") as ostr:
        json.dump(predictions, ostr)


def main(args):
    if args.do_train:
        logger.debug("Performing revdict training")
        train(args)
    if args.do_pred:
        logger.debug("Performing revdict prediction")
        pred(args)

    if args.do_eval:
        logger.debug("Performing revdict evaluation")
        eval(args)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
