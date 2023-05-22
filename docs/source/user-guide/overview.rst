Overview
=========

SpuCo makes the process of developing and evaluation methods to tackle the spurious correlation problem 
effortless and reproducible. 

Experiment with SpuCo can be broken into 4 stages:

- Data Preparation
- Group Inference
- Invariant Training / Finetuning 
- Evaluation 

Data Preparation
-----------------

SpuCo provides 3 custom datasets with configurable spurious feature difficulty and strength of spurious correlations. 

Important SpuCoDataset attributes:
- spurious_correlation_strength - The strength of the spurious correlation 
- spurious_feature difficulty - The difficulty of learning the spurious feature. SpuCo characterizes difficulty of features
using two theoretically motivated notions: magnitude and variance. Easy spurious features have larger magnitude or alternatively 
present with lesser variance and vice-a-versa for harder spurious features. 


Group Information about a dataset in SpuCo is encapsulated in the following 3 attributes: 
- spurious - A list specifying the label of the spurious attribute present in an example. E.g. For Waterbirds, 
the spurious label of an examples is either 0 - land background or 1 - water background. 
- group_partition - Specifies how indices of dataset should be partitioned into groups. Dictionary mapping group label = ``(class_label, spurious_label)`` 
to list of indices belonging to that group. 
- group_weights - Specifies the proportion of total data in each group. Dictionary mapping group label = ``(class_label, spurious_label)`` 
to the fraction of examples belonging to that group. 

SpuCo Datasets come with these properties set and we provide dataset wrapper classes that allow other datasets to be 
packaged with group information as needed. 

In particular, SpuCo provides support for all datasets for Waterbids and Civil Comments provided by WILDS through the WILDSDatasetWrapper
that populates these attributes from a WILDS dataset object.


Group Inference 
----------------

Group Inference refers to the stage in methods tackling spurious correlations which infer group membership. 

All group inference methods subclass BaseGroupInference where the ``infer_groups`` function returns the
inferred group_partition. 

For methods relying on group labeled subsets, the GroupLabeledDataset wrapper provides a way to serve group label 
information and class information along with an example. (``__get_item__`` returns ``data, class_label, spurious_label``)

Similarly, for methods relying on spurious attribute labeled subsets, the SpuriousTargetDataset wrapper provides a way to serve
only the example and spurious attribute information. (``__get_item__`` returns ``data, spurious_label``). 

Currently, we support the following group inference methods: 
- Just Train Twice 
- Clustering
- Spread Spurious Attribute 
- Correct-n-Contrast
- Environment Inference for Invariant Learning 

Invariant Training 
------------------

Invariant Training refers to the stage in methods tackling spurious correlation which utilize group information to train
models that achieve high accuracy across all groups. 

These methods typically require group information during training and thus GroupLabeledDatasets can be used to package any 
dataset with group information. 

Moreover, we provide a variety of sampling strategies considered in the spurious correlations literature. 

Upsampling and downsampling strategies increase or decrease the number of examples in each group available for the
the invariant training phase to ensure that equal number of examples are seen per group. These strategies do not have
any guarantees at a batch granularity i.e. it is possible to have group balanced batches but across an entire epoch the 
data trained on will be group balanced. Through custom sampling, we allow the user to experiment with other such sampling
stragies by simply specifiying the indices of the examples (allowing duplicates) to be seen in one epoch. 

Batch Sampling methods differ from the aformentioned sampling methods in that they control the batch and ensure 
that it is class-balanced or group-balanced. 

Currently, we support the following invariant training methods: 
- GroupDRO 
- Sampling Methods:
    - Upsampling
    - Downsampling
    - Custom Sampling
- Class-Balanced Batch Sampling
- Group-Balanced Batch Sampling
- Correct-n-Contrast

Finetuning
----------

Methods that take a model trained using ERM on datasets with spurious correlations and finetune to ensure high accuracy on 
all groups are placed in the finetuning module. 

SpuCo Models are organized as two module structures, namely backbone and classifier, to allow such methods to only finetune
the last layer if that is sufficient. 

Currently, we support the following finetuning methods: 
- Deep Feature Reweighting 


End to End Methods 
-------------------

Methods that do not decouple the solution into a group inference and invariant training / finetuning phase are placed under 
the ``end2end`` module. 

Currently, we support the following end2end methods:
- Learning from Failure 

Evaluation
-----------

Evaluating the success of methods addressing the spurious correlations problem is done by measuring average accuracy and 
worst group accuracy. 

Since, the number of examples in some groups can be very small in the presence of strong spurious correlations, a dynamically 
generated test may not contain examples from every group. As a result, SpuCo Datasets create group balanced test sets (split="test") 
and the evaluator correctly reports average acccuracy by taking the group_weights of the trainset i.e. the fraction of examples of
the entire dataset in each group. 

Additionally, we provide an API for evaluating how good the model is at identifying the spurious attribute presented in examples. 
This allows for validation of whether or not the spurious attribute was truly learnt by the model. 
