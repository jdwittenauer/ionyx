import numpy as np
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _BaseKFold


class TimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default=3
        Number of splits. Must be at least 1.
    min_train_size : int, optional
        Minimum size for a single training set.
    max_train_size : int, optional
        Maximum size for a single training set.

    Notes
    -----
    The training set has size ``i * n_samples // (n_splits + 1)
    + n_samples % (n_splits + 1)`` in the ``i``th split,
    with a test set of size ``n_samples//(n_splits + 1)``,
    where ``n_samples`` is the number of samples.
    """
    def __init__(self, n_splits=3, max_train_size=None, min_train_size=None):
        super(TimeSeriesSplit, self).__init__(n_splits,
                                              shuffle=False,
                                              random_state=None)
        self.max_train_size = max_train_size
        self.min_train_size = min_train_size

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like, with shape (n_samples,), optional
            Always ignored, exists for compatibility.

        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        if (self.min_train_size and self.max_train_size
                and self.min_train_size > self.max_train_size):
            raise ValueError(
                ("Cannot have min_train_size ={0} greater"
                 " than max_train_size: {1}").format(self.min_train_size,
                                                     self.max_train_size))
        indices = np.arange(n_samples)
        if self.min_train_size:
            test_size = ((n_samples - self.min_train_size) // n_splits)
            test_starts = range(self.min_train_size +
                                (n_samples - self.min_train_size) % n_splits,
                                n_samples, test_size)
        else:
            test_size = (n_samples // n_folds)
            test_starts = range(test_size + n_samples % n_folds,
                                n_samples, test_size)
        for test_start in test_starts:
            if self.max_train_size and self.max_train_size < test_start:
                yield (indices[test_start - self.max_train_size:test_start],
                       indices[test_start:test_start + test_size])
            else:
                yield (indices[:test_start],
                       indices[test_start:test_start + test_size])
