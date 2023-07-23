# Arabic Reverse Dictionary Shared task at WANLP 2023

This is the repository for the Arabic Reverse Dictionary Shared task at WANLP 2023 which forked from SemEval 2022 Shared Task #1: Comparing
Dictionaries and Word Embeddings (CODWOE).

This repository currently contains the baseline programs,  a scorer, and a format-checker to help participants get started.

Participants may be interested in the script revdict_entrypoint.py. It contains a number of useful features, such as scoring submissions, a format checker, and a few simple baseline architectures. It is also an exact copy of what is used on the Codalab platform.

This shared task includes 2 tracks:
- Closed Track: Participants should only use the provided dataset.
- Open Track: Participants can develop their own datasets using the provided English and Arabic dictionaries.

# Introduction
A Reverse Dictionary (RD) is a type of dictionary that allows users to find words based on their meaning or definition. Unlike a traditional dictionary, where users search for a word by its spelling, a reverse dictionary allows users to enter a description of a word or a phrase, and the dictionary will generate a list of words that match that description. Reverse dictionaries can be useful for writers, crossword puzzle enthusiasts, non-native language learners, and anyone looking to expand their vocabulary. Specifically, it addresses the Tip-of-Tongue (TOT) problem, which refers to the situation where a person is aware of a word they want to say but is unable to express it accurately. This shared task includes two subtasks: Arabic RD (Arabic => Arabic) and Cross-lingual Reverse Dictionary (CLRD) (Arabic => English).

# Tasks
## Task1: Arabic RD (Closed Track)
The structure of reverse dictionaries (sequence-to-vector) is the opposite of traditional dictionaries. This task focuses on learning how to convert a definition of the word into word embedding vectors in Arabic. The task involves reconstructing the word embedding vector of the defined word, rather than simply finding the target word. This would enable the users to search for words based on the definition or meanings they anticipate. 
The training set of the data points should contain word vector representation and its corresponding word definition (gloss). 
The proposed model should generate new word vector representations for the target unseen readable definitions in the test set. 

In this task, the input for the model is an Arabic word definition (gloss) and the output is an Arabic word embedding. 

## Task2: Cross-lingual Reverse Dictionary (CLRD) (Open Track)
The objective of the cross-lingual reverse dictionaries task (sequence-to-vector) is to acquire the ability to convert an English definition into word embedding vectors in Arabic. The main objective of this task is to identify the most accurate and suitable Arabic word vector that can efficiently express the identical semantic English definition or gloss, which is referred to as Arabicization "تَعْرِيب". 

The task involves reconstructing the word embedding vector that represents the Arabic word to its corresponding English definition. This approach enables users to search for words in other languages based on their anticipated meanings or definitions in English. This task facilitates cross-lingual search, language understanding, and language translation.


In this task, the input for the model is an English word definition (gloss) and the output is Arabic word embeddings. 

# Datasets

Datasets can be downloaded from [CODALAB-Task1](https://codalab.lisn.upsaclay.fr/competitions/14568) and  [CODALAB-Task2](https://codalab.lisn.upsaclay.fr/competitions/14569). This section details the structure of the JSON dataset files provided. 

### Brief Overview

As an overview, the participants are expected to use the definition ("gloss") to generate any of the associated embeddings ( "sgns", "electra").


### Dataset files structure

Each dataset file corresponds to a data split (train/dev/test).
The datasets include three main components:
-  **Arabic dictionary** has (58010) entries that were selected from [LMF Contemporary Arabic dictionary](https://lindat.cz/repository/xmlui/handle/11372/LRT-1944?locale-attribute=en) after revising and editing by our annotation team **(Task1)**.
-  **Mapped dictionary** between Arabic and English words to be used as supervision in the second task **(Task2)**.
-  **English dictionary** from [SemEval 2022 reverse dictionary](https://aclanthology.org/2022.semeval-1.1/) task has (63596) entries. 


Dataset files are in JSON format. A dataset file contains a list of examples. Each example is a JSON object, containing the following keys:
 + "id"
 + "word"
 + "gloss"
 + "sgns"
 + "electra"
 + "enId"



As a concrete instance, here is an example from the training dataset for the Arabic dictionary:
```json
#Arabic dictionary
{
"id":"ar.45",
"word":"عين",
"gloss":"عضو الإبصار في ...",
"pos":"n",
"electra":[0.4, 0.3, …],
"sgns":[0.2, 0.5, …],
"enId": "en.150"
}

```

The value associated to "id" key tracks the language and unique identifier for this example.

The value associated to the "gloss" key is a definition, as you would find in a classical dictionary. It is to be used as the source in the RD task.

The value associated to "enId" key tracks the mapped identifier in the English dictionary.

All other keys ("sgns", "electra") correspond to embeddings, and the associated values are arrays of floats representing the components. They all can serve as targets for the RD task.
 + "sgns" corresponds to skip-gram embeddings (word2vec)
 + "electra" corresponds to Transformer-based contextualized embeddings.



As a concrete instance, here is an example from the training dataset for the Mapped dictionary:
```json

#Mapped dictionary
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

The value associated to "id" key tracks the Arabic unique identifier in the Arabic dictionary.

The value associated to the "argloss" and "gloss" keys is the Arabic and English definitions, as you would find in an Arabic and English dictionary, respectively. The "gloss" is to be used as the source in the CLRD task.

The value associated to "enId" key tracks the mapped identifier in the English dictionary.

All other keys ("sgns", "electra") correspond to embeddings, and the associated values are arrays of floats representing the components. They all can serve as targets for the CLRD task.
 + "sgns" corresponds to skip-gram embeddings (word2vec)
 + "electra" corresponds to Transformer-based contextualized embeddings.


As a concrete instance, here is an example from the training dataset for the English dictionary:
```json

#English dictionary

{
"id":"en.150",
"word":"eye",
"gloss":"One of the two ...",
"pos":"n",
"electra":[0.7, 0.1, …],
"sgns":[0.2, 0.8, …]
}

```
The English dictionary has the same keys as the Arabic dictionary and can be utilized in the second task

### Using the dataset files

Given that the data is in JSON format, it is straightforward to load it in Python:

```python
import json
with open(PATH_TO_DATASET, "r") as file_handler:
    dataset = json.load(file_handler)
```





# Baseline results
Here are the baseline results on the development set for the two tracks.
The code described in `code/revdict.oy` was used to generate these scores.

Scores were computed using the scoring script provided in this git (`code/score.py`).

### Reverse Dictionary track (RD)

|            | No. of epoch| Cosine  |  MSE    | Ranking
|------------|------------:|--------:|--------:|--------:
| ar SGNS    |     200     | soon    | soon    | soon
| ar electra |     200     | 0.48848 | 0.24941 | 0.31277


### Cross-lingual Reverse Dictionary track (CLRD)

|            | No. of epoch| Cosine  |  MSE    | Ranking
|------------|------------:|--------:|--------:|--------:
| ar SGNS    |     300     | 0.26226 | 0.04922 | 0.50167
| ar electra |     300     | 0.54090 | 0.22105 | 0.36222

# Submission and evaluation
The model evaluation process follows a hierarchy of metrics. The **primary metric is the ranking metric**, which is used to assess how well the model ranks predictions compared to ground truth values. If models have similar rankings, the secondary metric, mean squared error (MSE), is considered. Lastly, if further differentiation is needed, the tertiary metric, cosine similarity, provides additional insights. This approach ensures the selection of a top-performing and well-rounded model.

The evaluation of shared tasks will be hosted through CODALAB. Here are the CODALAB links for each task:
+ **[CODALAB link for task 1](https://codalab.lisn.upsaclay.fr/competitions/14568).**
+ **[CODALAB link for task 2](https://codalab.lisn.upsaclay.fr/competitions/14569).**

### Expected output format

During the evaluation phase, submissions are expected to reconstruct the same JSON format.

The test JSON files will contain the "id" and the gloss keys.


In both RD and CLRD, participants should construct JSON files that contain at least the two following keys:
 + the original "id",
 + any of the valid embeddings ("sgns" or "electra" key in AR/EN)


# Using this repository
To install the exact environment used for the scripts, see the `requirements.txt` file which lists the library used. Do note that the exact installation in the competition underwent supplementary tweaks: in particular, Colab Pro was utilized to run the experiment.

The code useful to participants is stored in the `code/` directory.

+ To see options for a simple baseline on the reverse dictionary track, use:
```sh
$ python3 code/revdict_entrypoint.py revdict --help
```
+ To verify the format of a submission, run:
```sh
$ python3 code/revdict_entrypoint.py check-format $PATH_TO_SUBMISSION_FILE
```
+ To score a submission, use  
```sh
$ python3 code/revdict_entrypoint.py score $PATH_TO_SUBMISSION_FILE --reference_files_dir $PATH_TO_DATA_FILE
```
Note that this requires the gold files, not available at the start of the
competition.

Other useful files to look at include `code/models.py`, where our baseline
architectures are defined, and `code/data.py`, which shows how to use the JSON
datasets with the PyTorch dataset API.


