import argparse
import collections
import itertools
import json
import logging
import os
import pathlib
import sys
# import pandas as pd


logger = logging.getLogger(pathlib.Path(__file__).name)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(handler)

os.environ["MOVERSCORE_MODEL"] = "distilbert-base-multilingual-cased"
# import moverscore_v2 as mv_sc

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk import word_tokenize as tokenize

import numpy as np

import torch
import torch.nn.functional as F

import tqdm

import check_output


def get_parser(parser=argparse.ArgumentParser(description="score a submission")):
    parser.add_argument(
        "submission_path",
        type=pathlib.Path,
        help="path to submission file to be scored, or to a directory of submissions to be scored",
    )
    parser.add_argument(
        "--reference_files_dir",
        type=pathlib.Path,
        help="directory containing all reference files",
        default=pathlib.Path("data"),
    )
    parser.add_argument(
        "--output_file",
        type=pathlib.Path,
        help="default path to print output",
        default=pathlib.Path("scores.txt"),
    )
    return parser

def bleu(pred, target, smoothing_function=SmoothingFunction().method4):
    return sentence_bleu([pred], target, smoothing_function=smoothing_function)


def mover_corpus_score(sys_stream, ref_streams, trace=0):
    """Adapted from the MoverScore github"""

    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]
    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]
    fhs = [sys_stream] + ref_streams
    corpus_score = 0
    pbar = tqdm.tqdm(desc="MvSc.", disable=None, total=len(sys_stream))
    for lines in itertools.zip_longest(*fhs):
        if None in lines:
            raise EOFError("Source and reference streams have different lengths!")
        hypo, *refs = lines
        idf_dict_hyp = collections.defaultdict(lambda: 1.0)
        idf_dict_ref = collections.defaultdict(lambda: 1.0)
        corpus_score += mv_sc.word_mover_score(
            refs,
            [hypo],
            idf_dict_ref,
            idf_dict_hyp,
            stop_words=[],
            n_gram=1,
            remove_subwords=False,
        )[0]
        pbar.update()
    pbar.close()
    corpus_score /= len(sys_stream)
    return corpus_score



def rank_cosine(preds, targets):
    assocs = F.normalize(preds) @ F.normalize(targets).T
    refs = torch.diagonal(assocs, 0).unsqueeze(1)
    ranks = (assocs >= refs).sum(1).float()
    assert ranks.numel() == preds.size(0)
    ranks = ranks.mean().item()
    
    return ranks / preds.size(0)


def closest_ranks(preds, targets,  targetsword=[] ):
    # Append train tensor to targets
    mixed_targets = torch.cat((targets, ), dim=0)
    mixed_targetsword = targetsword #+ trainword

    # Calculate cosine similarity between preds and mixed_targets
    cosine_similarities = F.normalize(preds) @ F.normalize(mixed_targets).T

    # Find the indices of the top 10 closest ranks for each prediction
    _, topk_indices = torch.topk(cosine_similarities, k=10, dim=1)

    listOftop10=[]
    for i in range(len(preds)):
      listOftop10.append([])
      for x in topk_indices[i]:
          listOftop10[i].append(mixed_targets[x])

    # Calculate precision at k=1
    correct_prediction_at1 = 0
    correct_predictions_atk = 0
    for i in range(len(preds)):
      # print("i", i, "topk_indices[i]", topk_indices[i])
      if i in topk_indices[i][0]:
        correct_prediction_at1 += 1
      if i in topk_indices[i]:
        correct_predictions_atk += 1


    precision_at_1 = correct_prediction_at1 / len(preds)
    precision_at_k = correct_predictions_atk / len(preds)

    closest_indices = torch.empty_like(topk_indices)
    for i in range(len(preds)):
        closest_indices[i] = topk_indices[i] - targets.size(0) if i < targets.size(0) else topk_indices[i]
    
    # df = pd.DataFrame(mixed_targetsword, columns=["word"])

    # Convert tensor indices to NumPy array and take absolute values
    # closest_indices_np = torch.abs(topk_indices).numpy()

    
    # Retrieve words corresponding to closest_indices
    # closest_words = df.loc[closest_indices_np.flatten(), "word"].values.reshape(closest_indices.shape).tolist()

    # top10=pd.DataFrame()
    # top10["target"]=targetsword
    # top10["top10preds"]=closest_words

    return [precision_at_1, precision_at_k]

def eval_revdict(args, summary):
    # 1. read contents
    ## read data files
    with open(args.submission_file, "r") as fp:
        submission = sorted(json.load(fp), key=lambda r: r["id"])
    with open(args.reference_file, "r") as fp:
        reference = sorted(json.load(fp), key=lambda r: r["id"])
    # with open('/content/gdrive/MyDrive/sharedTask/lookups/Train.json', "r") as fp:
    #     train = sorted(json.load(fp), key=lambda r: r["id"])
    vec_archs = sorted(
        set(submission[0].keys())
        - {
            "id",
            "gloss",
            "word",
            "pos",
            "concrete",
            "example",
            "f_rnk",
            "counts",
            "polysemous",
        }
    )
    ## define accumulators for rank-cosine
    all_preds = collections.defaultdict(list)
    all_refs = collections.defaultdict(list)
    # all_train = collections.defaultdict(list)

  
    
    assert len(submission) == len(reference), "Missing items in submission!"
    ## retrieve vectors
    for sub, ref in zip(submission, reference):
      assert sub["id"] == ref["id"], "Mismatch in submission and reference files!"
      for arch in vec_archs:
        all_preds[arch].append(sub[arch])
        all_refs[arch].append(ref[arch])
      all_refs["word"].append(ref["word"])
    # for ii in range(len(train)):
    #     all_train[arch].append(train[ii][arch])
    #     all_train["word"].append(train[ii]["word"])

    torch.autograd.set_grad_enabled(False)

    # Convert all_train["word"] to a tensor of word strings
    all_refs_word = all_refs["word"]
    # all_train_word = all_train["word"]

    all_preds = {arch: torch.tensor(all_preds[arch]) for arch in vec_archs}
    all_refs = {arch: torch.tensor(all_refs[arch]) for arch in vec_archs}
    # all_train= {arch: torch.tensor(all_train[arch]) for arch in vec_archs}


    # 2. compute scores
    MSE_scores = {
        arch: F.mse_loss(all_preds[arch], all_refs[arch]).item() for arch in vec_archs
    }
    cos_scores = {
        arch: F.cosine_similarity(all_preds[arch], all_refs[arch]).mean().item()
        for arch in vec_archs
    }
    rnk_scores = {
        arch: rank_cosine(all_preds[arch], all_refs[arch]) for arch in vec_archs
    }
    cos_closest_ranks={
        arch: closest_ranks(all_preds[arch], all_refs[arch], all_refs_word ) for arch in vec_archs

    }



  # Adjust the threshold as needed

    
    # 3. display results
    # logger.debug(f"Submission {args.submission_file}, \n\tMSE: " + \
    #     ", ".join(f"{a}={MSE_scores[a]}" for a in vec_archs) + \
    #     ", \n\tcosine: " + \
    #     ", ".join(f"{a}={cos_scores[a]}" for a in vec_archs) + \
    #     ", \n\tcosine ranks: " + \
    #     ", ".join(f"{a}={rnk_scores[a]}" for a in vec_archs) + \
    #     "."
    # )
    # all_archs = sorted(set(reference[0].keys()) - {"id", "gloss", "word", "pos"})
    with open(args.output_file, "a") as ostr:
        for arch in vec_archs:
            print(f"MSE_{summary.lang}_{arch}:{MSE_scores[arch]}", file=ostr)
            print(f"cos_{summary.lang}_{arch}:{cos_scores[arch]}", file=ostr)
            print(f"rnk_{summary.lang}_{arch}:{rnk_scores[arch]}", file=ostr)
            print(f"precision_at_1_{summary.lang}_{arch}:{cos_closest_ranks[arch][0]}", file=ostr)
            print(f"precision_at_10_{summary.lang}_{arch}:{cos_closest_ranks[arch][1]}", file=ostr)

            

    return (
        args.submission_file,
        *[MSE_scores.get(a, None) for a in vec_archs],
        *[cos_scores.get(a, None) for a in vec_archs],
    )


def main(args):
    def do_score(submission_file, summary):
        args.submission_file = submission_file
        args.reference_file = (
            args.reference_files_dir
        )
        eval_func = eval_revdict
        eval_func(args, summary)

    if args.output_file.is_dir():
        args.output_file = args.output_file / "scores.txt"
    # wipe file if exists
    open(args.output_file, "w").close()
    if args.submission_path.is_dir():
        files = list(args.submission_path.glob("*.json"))
        assert len(files) >= 1, "No data to score!"
        summaries = [check_output.main(f) for f in files]
        assert len(set(summaries)) == len(files), "Ensure files map to unique setups."
        rd_cfg = [
            (s.lang, a) for s in summaries for a in s.vec_archs
        ]
        assert len(set(rd_cfg)) == len(rd_cfg), "Ensure files map to unique setups."
        for summary, submitted_file in zip(summaries, files):
            do_score(submitted_file, summary)
    else:
        summary = check_output.main(args.submission_path)
        do_score(args.submission_path, summary)


if __name__ == "__main__":
    main(get_parser().parse_args())
