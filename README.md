# HotpotQA Group D

---

# Content

[Introduction](#introduction)
[Requirements](#requirements)
[Experiments](#experiments)
[Predictions](#predictions)
[Evaluation](#evaluation)
[Plots](#plots)

---

# Introduction

This repository is for the 6CCSANLP Natural Language Processing(25~26 SEM1000001) module coursework.
It tries to solve the open-domain HotpotQA datasets using Mistral LLM and some NLP techniques.

---

# Requirements

1. Clone the repository

`git clone https://github.com/tzinadah/hotpotqa-group-d`

2. Download the HotpotQA dataset using the scripts
   `./scripts/download-data.sh`

3. Create .env file in the root of the project and include the following
   `MISTRAL_KEY="your Mistral API key"`

4. Make sure you have conda installed and install the python environment using

`conda env create -f ./environment.yaml`

5. Activate the environment

`conda activate hotpotqa-group-d`

---

# Experiments

Experiments can be found in [experiments](src/hotpotqa-group-d/experiments)

You can run them just like any python script using while the conda environment is activated
`python ./src/hotpotqa_group_d/experiments/example-experiment.py`

---

# Predictions

The experiments save their predictions in [predictions](predictions/)

---

# Evaluation

You can run the [evaluation script](scripts/evaluate.sh) using the following format
`./scripts/evaluate-result.sh example`
This script will look for predictions/example.json and produce results/example-results.json
which included the measure resulting from the HotpotQA evaluation script

---

# Plots

You can find plot in [here](plots/) they are the result of running [visualisation](src/hotpotqa-group-d/experiments/visualise.py)
