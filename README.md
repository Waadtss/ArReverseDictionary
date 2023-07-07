# Comparing Dictionaries and Word Embeddings

This is the repository for the XXX
which forked from ***SemEval 2022 Shared Task #1: Comparing
Dictionaries and Word Embeddings (CODWOE)***.

This repository currently contains: the baseline programs,  a scorer, a
format-checker to help participants get started.

Participants may be interested in the script `revdict_entrypoint.py`. It contains
a number of useful features, such as scoring submissions, a format checker and a
few simple baseline architectures. It is also the exact copy of what is used on
the codalab.

**Datasets are no longer provided directly on this repository. The competition datasets are now available on this page: [https://codwoe.atilf.fr/](https://codwoe.atilf.fr/).**

# What is this task?
The CODWOE shared task invites you to compare two types of semantic
descriptions: dictionary glosses and word embedding representations. Are these
two types of representation equivalent? Can we generate one from the other? To
study this question, we propose two subtracks: a **definition modeling** track
(Noraset et al., 2017), where participants have to generate glosses from
vectors, and a **reverse dictionary** track (Hill et al., 2016), where
participants have to generate vectors from glosses.

These two tracks display a number of interesting characteristics. Definition
modeling is a vector-to-sequence task, the reverse dictionary task is a
sequence-to-vector task—and you know that kind of thing gets NLP people swearing
out loud. These tasks are also useful for explainable AI, since they involve
converting human-readable data into machine-readable data and back.

To get involved: check out the
[codalab competition](https://competitions.codalab.org/competitions/34022).
There is also a participants'
["semeval2022-dictionaries-and-word-embeddings" google group](mailto:semeval2022-dictionaries-and-word-embeddings@googlegroups.com),
as well as a [discord server](https://discord.gg/y8g6qXakNs).
You can reach us organizers through [this email](mailto:tmickus@atilf.fr); make
sure to mention SemEval in your email object.

# How hard is it?

## Official rankings

Below are the official rankings for the SemEval 2022 CODWOE Shared task.
More information about the submissions we received is available in this git (see the `rankings/` sub-directory).


## Baseline results
Here are baseline results on the development set for the two tracks.
We used the code described in `code/baseline_archs` to generate these scores.

For the Reverse Dictionary track results, rows will correspond to different targets.
On the other hand, rows of the Definition Modeling table below correspond to different inputs to the system.
Scores were computed using the scoring script provided in this git (`code/score.py`).

### Reverse Dictionary track

|            | MSE     | Cosine  | Ranking
|------------|--------:|--------:|--------:
| ar SGNS    | 0.91092 | 0.15132 | 0.49030
| ar electra | 1.41287 | 0.84283 | 0.49849




# Using this repository
To install the exact environment used for our scripts, see the
`requirements.txt` file which lists the library we used. Do
note that the exact installation in the competition underwent supplementary
tweaks: in particular, we patch the moverscore library to have it run on CPU.

Another possibility is to use the dockerfile written for the codalab
competition. You can also pull this docker image from dockerhub:
[`linguistickus/codwoe`](https://hub.docker.com/r/linguistickus/codwoe). This
Docker image doesn't contain the code, so you will also need to clone the
repository within it; but this image will also contain our tweaks.

Code useful to participants is stored in the `code/` directory.
To see options a simple baseline on the definition modeling track, use:
```sh
$ python3 code/revdict_entrypoint.py defmod --help
```
To see options for a simple baseline on the reverse dictionary track, use:
```sh
$ python3 code/revdict_entrypoint.py revdict --help
```
To verify the format of a submission, run:
```sh
$ python3 code/revdict_entrypoint.py check-format $PATH_TO_SUBMISSION_FILE
```
To score a submission, use  
```sh
$ python3 code/revdict_entrypoint.py score $PATH_TO_SUBMISSION_FILE --reference_files_dir $PATH_TO_DATA_DIR
```
Note that this requires the gold files, not available at the start of the
competition.

Other useful files to look at include `code/models.py`, where our baseline
architectures are defined, and `code/data.py`, which shows how to use the JSON
datasets with the PyTorch dataset API.

# Using the datasets

**Datasets are no longer provided directly on this repository. The competition datasets are now available on this page: [https://codwoe.atilf.fr/](https://codwoe.atilf.fr/).**

This section details the structure of the JSON dataset file we provide. More information is available on the competition website: [link](https://competitions.codalab.org/competitions/34022#participate-get_data).

### Brief Overview

As an overview, the expected usage of the datasets is as follow:
 + In the Reverse Dictionary track, we expect participants to use the definition ("gloss") to generate any of the associated embeddings ( "sgns", "electra").


### Dataset files structure

Each dataset file correspond to a data split (train/dev/test) for one of the languages.

Dataset files are in the JSON format. A dataset file contains a list of examples. Each example is a JSON dictionary, containing the following keys:
 + "id",
 + "word"
 + "gloss"
 + "sgns"
 + "electra"


As a concrete instance, here is an example from the English training dataset:
```json
    {
        "id": "en.train.2",
        "gloss": "A vocal genre in Hindustani classical music",
        "sgns": [
            -0.0602365807,
           ...
        ],
        "char": [
            -0.3631578386,
           ...
        ],
        "electra": [
            -1.3904430866,
           ...
     ]
    },
```

### Description of contents

The value associated to "id" tracks the language, data split and unique identifier for this example.

The value associated to the "gloss" key is a definition, as you would find in a classical dictionary. It is to be used either the target in the Definition Modeling track, or  asthe source in the Reverse Dictionary track.

All other keys ("sgns", "electra") correspond to embeddings, and the associated values are arrays of floats representing the components. They all can serve as targets for the Reverse Dictionary track.
 + "sgns" corresponds to skip-gram embeddings (word2vec)
 + "electra" corresponds to Transformer-based contextualized embeddings.


### Using the dataset files

Given that the data is in JSON format, it is straightforward to load it in python:

```python
import json
with open(PATH_TO_DATASET, "r") as file_handler:
    dataset = json.load(file_handler)
```

A more complete example for pytorch is available in the git repository (see here: [link](https://git.atilf.fr/tmickus/codwoe/-/blob/master/code/data.py#L18)).

### Expected output format

During the evaluation phase, we will expect submissions to reconstruct the same JSON format.

The test JSON files will contain the "id" key, and either the "gloss" key (in the reverse dictionary track)


In the reverse dictionary, participants should construct JSON files that contain at least the two following keys:
 + the original "id",
 + any of the valid embeddings ("sgns" or "electra" key in AR/EN)

Other keys can be added. More details concerning the evaluation procedure are available here: [link](https://competitions.codalab.org/competitions/34022#learn_the_details-evaluation).


# Using this repository

The code and data of this shared task are provided under a CC-BY-SA license.
If you use this repository in your research, please cite the task paper (to appear):
```
@inproceedings{mickus-etal-2022-semeval,
    title = "{S}emeval-2022 Task 1: {CODWOE} {--} Comparing Dictionaries and Word Embeddings",
    author = "Mickus, Timothee  and
      Van Deemter, Kees  and
      Constant, Mathieu  and
      Paperno, Denis",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.semeval-1.1",
    pages = "1--14",
    abstract = "Word embeddings have advanced the state of the art in NLP across numerous tasks. Understanding the contents of dense neural representations is of utmost interest to the computational semantics community. We propose to focus on relating these opaque word vectors with human-readable definitions, as found in dictionaries This problem naturally divides into two subtasks: converting definitions into embeddings, and converting embeddings into definitions. This task was conducted in a multilingual setting, using comparable sets of embeddings trained homogeneously.",
}
```
