Datasets
========

The package currently provides the following datasets:

- `SpuCoMNIST`: A controllable synthetic dataset that explores real-world data properties (spurious feature: colored background, core feature: MNIST digit)
- `SpuCoCXR`: 


SpuCoMNIST 
----------


SpuCoAnimals
--------

Next, we introduce **Animals**, a large-scale vision dataset curated from ImageNet with **two** realistic spurious correlations (Russakovsky et al., 2015). 

**Animals** has 4 classes: 

- **landbirds**
- **waterbirds**
- **small dog** breeds
- **big dog** breeds.

Waterbirds and Landbirds are spuriously correlated with **water** and **land** backgrounds, respectively. Small dogs and big dogs are spuriously correlated with **indoor** and **outdoor** backgrounds, respectively.


SpuCoDogs
--------

Subset of SpuCoAnimals containing only dogs. 

SpuCoBirds
--------

Subset of SpuCoAnimals containing only birds. 