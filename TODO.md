# TODO
======

This project is extremely early in development.  Below is a summary of tasks required before the first alpha release.

- implement a high-level "experiment" class to tie everything together (design still needs fleshed out)
- add unit tests for each class/function
- add example code using the sample data sets
- add detailed doc strings for each class/function
- add project documentation and usage examples
- add more error checking and in-line comments
- re-factor visualization functions to have a cleaner API
- update seaborn calls to use the 0.6 APIs
- create ensemble classes and make the design more flexible than current functions
- update model definition, training, cross-validation, and parameter search methods to be more generic
- extend support for loading and saving models to handle a much wider variety of use cases
- model.py - XGBoost min eval loss/iteration
- model.py - training history plot function
- cross_validation.py - review sequence cross-validation functionality
- cross_validation.py - make learning curve plots part of cross-validation functions
- param_search.py - calculate and print best params and training/evaluation error
- param_search.py - create a data structure to store and return iteration details
- param_search.py - add random search capability
- param_search.py - find a way to visually display results