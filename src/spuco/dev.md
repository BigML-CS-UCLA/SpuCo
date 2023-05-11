# SpuCo (Spurious Correlations Datasets and Benchmarks) Pip Package Organization

## Overview 

Modules:
- Datasets
- Evaluate
- Group Inference
- Trainer
- Visualization

The usage will be config file based for convenience.

Group Inference and Evaluatiod Methods will have their hyperparameters loadable from config file. As a result maybe trainer as well?


# Organizing Spurious Correlation Fixing Methods

## Training from Scratch 

### E2E Module

GroupInference Object
Invariant Training Object

Define Standard Validation Split over Base Images

Outputs Final Performance

MixMatchMethods can go here

### Group Inference

These methods will need dataset and a model (same architecture as later?) that can be train with custom hyperparameters. 

They will output group partition

- Normal but Controlled Training:
    - JTT: 
    - Clustering: 
        kMeans on, pass function to get embeddings / logits / last layer gradients or something else
    - LfF: Define a loss function
- EIIL - training to find worst-case group partition given logits 
- Validation Methods
    - SSA: training model on validation data with labels being spurious feature

### Invariant Training 

Input: group partition

- GroupDRO
- Contrastive Learning CNC style
- Yu Method (later)

## Finetuning

- DFR
- Yihao Method (later)


# TODO:

SPURIOUS well-defined
Larger dataset
Can I overcome keyword search