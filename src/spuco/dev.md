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

Outputs Final Performance

MixMatchMethods can go here

- Learning from Failure

### Group Inference

These methods will need dataset and a model (same architecture as later?) that can be train with custom hyperparameters. 

They will output group partition

- Model Outputs:
    - JTT: 1
    - Clustering: 2
- EIIL - training to find worst-case group partition given logits 
- SSA: training model on validation data with labels being spurious feature

### Invariant Training 

Input: group partition

- GroupDRO: done
- Contrastive Learning CNC style
- Yu Method (later)

## Finetuning

- DFR: 3
- Yihao Method (later)

# Dataset Desiderata

SPURIOUS well-defined

- insert some core features outliers in minority group - makes upsampling methods struggle because upsampling noisy examples
- some core features in majority group - makes misclassification methods upsample these methods 

Larger dataset
Can I overcome keyword search

# v0 Baselines

- [x] jtt
- [x] eiil
- [x] clustering 
- [x] groupdro 
- [x] sampling 
- [x] ssa
- [ ] cnc 
- [ ] lff
- [ ] dfr
- [ ] sampling (up and down, class, group) 

General TODO:
- [ ] lr_scheduler support in trainer 
- [ ] control randomness from python random 

# Engineering Goals 

- Can we get some namespace / dictionary to group the methods that are similar
- Can we get some hyperparameters function or object to basically easily see what hyperparameters each method has