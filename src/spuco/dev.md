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

# v0 Baselines Finalized

- [x] erm
- [x] groupdro 
- [x] jtt
- [x] eiil
- [x] clustering 
- [x] sampling
- [x] ssa
- [x] cnc 
- [x] lff
- [x] dfr

# Reproducing Baselines 

## WaterBirds - WILDS

- [x] erm
- [x] groupdro 
- [ ] jtt
- [ ] eiil
- [ ] clustering 
- [ ] sampling
- [ ] ssa
- [ ] cnc 
- [ ] lff
- [ ] dfr

## CivilComments - WILDS

- [ ] erm
- [ ] groupdro 
- [ ] jtt
- [ ] eiil
- [ ] clustering 
- [ ] sampling
- [ ] ssa
- [ ] cnc 
- [ ] lff
- [ ] dfr

# Debugging

@Yu
Debugging Methods (run but don't give expected improvement):
- [ ] CNC
- [ ] LFF

General TODO:

# Dataset

- Spurious Correlation Strength:
    - fully customizable in SpuCoMNIST
    - linear and uniform settings in SpuCoCT and MedNLI 
- Spurious Feature Strength
    - magnitude and variance for SpuCoMNIST and SpuCoCT
    - MedNLI - still deciding
- Label Noise 
- Feature Noise - just randomly corrupt some words for MedNLI?

ToDo:

- [x] modify base class
- [x] modify spuco mnist to allow fully customizabloe spuco tregnth
- [x] label noise for psucomnist
- [x] feature noise for psucomnist
- [x] SpuCoCT
