SpuCo
============

**SpuCo** is a Python package that provides complex Vision and NLP datasets
with the *spurious correlations* problem. 

The SpuCo package is designed to help researchers and practitioners evaluate
the robustness of their machine learning algorithms against spurious
correlations that may exist in real-world data. The package includes datasets
that have been artificially modified to introduce spurious correlations
between features, making it possible to evaluate how well different algorithms
can identify and handle these spurious correlations.

Link to GitHub: https://github.com/sjoshi804/SpuCo/

.. note::

   This project is under active development.

About Us
--------
This package is maintained by `Siddharth Joshi <https://sjoshi804.github.io/>`_ and `Yu Yang <https://sites.google.com/g.ucla.edu/yuyang/home>`_ from `Professor Baharan Mirzasoleiman <http://web.cs.ucla.edu/~baharan/group.htm>`_'s lab at **UCLA**.

Getting Started
---------------

.. toctree::
   :caption: User Guide
   :maxdepth: 2

   user-guide/overview
   user-guide/quickstart
   user-guide/datasets

.. toctree::
   :caption: API Reference
   :maxdepth: 1
   :glob:

   reference/*