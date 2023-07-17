# Arabic Reverse Dictionary Shared task at WANLP 2023

This is the repository for the Arabic Reverse Dictionary Shared task at WANLP 2023 which forked from SemEval 2022 Shared Task #1: Comparing
Dictionaries and Word Embeddings (CODWOE).

This repository currently contains: the baseline programs,  a scorer, a
format-checker to help participants get started.

Participants may be interested in the script `revdict_entrypoint.py`. It contains
a number of useful features, such as scoring submissions, a format checker and a
few simple baseline architectures. It is also the exact copy of what is used on
the codalab.

**The competition datasets are now available on this page: [https://codwoe.atilf.fr/](https://codwoe.atilf.fr/).**

# Introduction
A Reverse Dictionary (RD) is a type of dictionary that allows users to find words based on their meanings or definitions. Unlike a traditional dictionary, where users search for a word by its spelling, a reverse dictionary allows users to enter a description of a word or a phrase, and the dictionary will generate a list of words that match that description. Reverse dictionaries can be useful for writers, crossword puzzle enthusiasts, and nonnative language learner and anyone looking to expand their vocabulary. Specifically, it addresses the Tip-of-Tongue (TOT) problem (Brown and McNeill, 1966), which refers to the situation where a person is aware of a word they want to say but is unable to express it accurately (Siddique and Sufyan Beg, 2019) .This shared task includes two subtasks: Arabic RD and Cross-lingual Reverse Dictionary (CLRD).

# Tasks
## Task1: Arabic RD
The structure of reverse dictionaries (sequence-to-vector) is the opposite of traditional dictionaries lookup. This task focuses on the learning of how to convert human readable definitions into word embeddings vector in Arabic. The task involves reconstructing the word embedding vector of the defined word, rather than simply finding the target word, which is similar to the approach used by (Mickus et al., 2022; Zanzotto et al., 2010; Hill et al., 2016).
This would enable the users to search for words based on the definition or meanings they anticipate, TOT. The training set of the data points should contain a source word vector representation and its corresponding word definition, as illustrated in Figure 1 (a) and (b). The proposed model should generate new word vector representations for the target unseen readable definitions in the test set. In this task, the input for the model is Arabic word definition (gloss) and the output is Arabic word embeddings. 

## Task2: Cross-lingual Reverse Dictionary (CLRD)
The objective of the cross-lingual reverse dictionaries task (sequence-to-vector) is to acquire the ability to transform readable definitions in English language into a vector representation for an Arabic word. The main objective of this task is to identify the most accurate and suitable Arabic word vector that can efficiently express the identical semantic interpretation as the provided English language definition or gloss, which is commonly known as Arabicization "تَعْرِيب". The task involves reconstructing the word embedding vector that represents the Arabic word to its corresponding English definition. This approach enables users to search for words in other languages based on their anticipated meanings or definitions in English. This task facilitates cross-lingual search, language understanding, and language translation.
In this task, the input for the model is English word definition (gloss) and the output is Arabic word embeddings. 

# Submission and evaluation
The evaluation of shared tasks will be hosted through CODALAB. Here are the CODALAB links for each task:
+ **[CODALAB link for task 1](https://codalab.lisn.upsaclay.fr/competitions/14568).**
+ **[CODALAB link for task 2](https://codalab.lisn.upsaclay.fr/competitions/14569).**


# How hard is it?

## Official rankings

Below are the official rankings for the  Arabic Reverse Dictionary Shared task.
More information about the submissions we received is available in this git (see the `rankings/` sub-directory).


## Baseline results
Here are the baseline results on the development set for the two tracks.
We used the code described in `code/revdict.oy` to generate these scores.

Scores were computed using the scoring script provided in this git (`code/score.py`).

### Reverse Dictionary track (RD)

|            | Cosine  |  MSE    | Ranking
|------------|--------:|--------:|--------:
| ar SGNS    | 0.91092 | 0.15132 | 0.0
| ar electra | 1.41287 | 0.84283 | 0.0


### Cross-lingual Reverse Dictionary track (CLRD)

|            | Cosine  |  MSE    | Ranking
|------------|--------:|--------:|--------:
| ar SGNS    | 0.26226 | 0.04922 | 0.50167
| ar electra | 0.54090 | 0.22105 | 0.36222


# Using this repository
To install the exact environment used for our scripts, see the
`requirements.txt` file which lists the library we used. Do
note that the exact installation in the competition underwent supplementary
tweaks: in particular, we have utilized Colab Pro to run the experiment.



Code useful to participants is stored in the `code/` directory.
To see options a simple baseline on the definition modeling track, use:

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

This section details the structure of the JSON dataset file we provide. 

### Brief Overview

As an overview, the expected usage of the datasets is as follow:
 + In the Reverse Dictionary track, we expect participants to use the definition ("gloss") to generate any of the associated embeddings ( "sgns", "electra").


### Dataset files structure

Each dataset file correspond to a data split (train/dev/test).
The dataset includes three main components:
- Arabic dictionary has (58010) entries that were selected from LMF Contemporary Arabic dictionary after revising and editing by our annotation team **(Task1)**.
- English dictionary from SemEval 2022 reverse dictionary task has (63596) entries **(Task2)**.
- Mapped dictionary between Arabic and English words to be used as a supervision in the second task available here to download.


Dataset files are in the JSON format. A dataset file contains a list of examples. Each example is a JSON dictionary, containing the following keys:
 + "id",
 + "word"
 + "gloss"
 + "sgns"
 + "electra"
 + "enId"



As a concrete instance, here is an example from the training dataset for Arabic dictionary, English dictionary, and Mapped dictionary, respectively:
```json
    {
"id":"ar.45",
"word":"عين",
"gloss":"عضو الإبصار في ...",
"pos":"n",
"electra":[0.4, 0.3, …],
"sgns":[0.2, 0.5, …],
"enId": "en.150"
}


{
"id":"en.150",
"word":"eye",
"gloss":"One of the two ...",
"pos":"n",
"electra":[0.7, 0.1, …],
"sgns":[0.2, 0.8, …]
}

{
"id":"ar.45",
"arword":"عين",
"argloss":"عضو الإبصار في ...",
"arpos":"n",
"electra":[0.4, 0.3, …],
"sgns":[0.2, 0.5, …],
"enId":"en.150",
"word":"eye",
"gloss":"One of the two ...",
"pos":"n",
}
```

### Description of contents

The value associated to "id" tracks the language and unique identifier for this example.

The value associated to the "gloss" key is a definition, as you would find in a classical dictionary. It is to be used either the target in the Definition Modeling track, or  asthe source in the Reverse Dictionary track.

All other keys ("sgns", "electra") correspond to embeddings, and the associated values are arrays of floats representing the components. They all can serve as targets for the Reverse Dictionary track.
 + "sgns" corresponds to skip-gram embeddings (word2vec)
 + "electra" corresponds to Transformer-based contextualized embeddings.
The value associated to "enId" tracks the mapped identifier in the English dictionary.



### Using the dataset files

Given that the data is in JSON format, it is straightforward to load it in python:

```python
import json
with open(PATH_TO_DATASET, "r") as file_handler:
    dataset = json.load(file_handler)
```


### Expected output format

During the evaluation phase, we will expect submissions to reconstruct the same JSON format.

The test JSON files will contain the "id" key, and either the "gloss" key (in the reverse dictionary track)


In the reverse dictionary, participants should construct JSON files that contain at least the two following keys:
 + the original "id",
 + any of the valid embeddings ("sgns" or "electra" key in AR/EN)




