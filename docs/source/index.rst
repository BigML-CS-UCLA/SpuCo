SpuCo
============


**SpuCo** is a Python package that provides complex Vision and NLP datasets
with the spurious correlations problem. 

The SpuCo package is designed to help researchers and practitioners evaluate
the robustness of their machine learning algorithms against spurious
correlations that may exist in real-world data. The package includes datasets
that have been artificially modified to introduce spurious correlations
between features, making it possible to evaluate how well different algorithms
can identify and handle these spurious correlations.

.. note::

   This project is under active development.

About Us
--------
This package is maintained by **Siddharth Joshi** (PhD Student: https://sjoshi804.github.io/) and 
**Yu Yang** (PhD Student: https://sites.google.com/g.ucla.edu/yuyang/home) from the BigML lab headed
by **Professor Baharan Mirzasoleiman** (http://web.cs.ucla.edu/~baharan/group.htm)
at **UCLA**.

Getting Started
---------------

.. toctree::
   :caption: User Guide
   :maxdepth: 2

   user-guide/quickstart
   user-guide/datasets
   user-guide/models
   user-guide/group_inference
   user-guide/invariant_train

.. toctree::
   :caption: API Reference
   :maxdepth: 1
   :glob:

   reference/*