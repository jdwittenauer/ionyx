from sklearn.base import (BaseEstimator, TransformerMixin, MetaEstimatorMixin, clone)
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer


class StackingTransformer(BaseEstimator, MetaEstimatorMixin, TransformerMixin):
    """Transformer to turn estimators into meta-estimators for model stacking

    In stacked generalization, meta estimators are combined in layers to
    improve the final result. To prevent data leaks between layers, a procedure
    similar to cross validation is adopted, where the model is trained in one
    part of the set and predicts the other part. In ``StackingTransformer``, it
    happens during ``fit_transform``, as the result of this procedure is what
    should be used by the next layers. Note that this behavior is different
    from ``fit().transform()``. Read more in the
    :ref:`User Guide <stacking_transformer>`.

    Parameters
    ----------
    estimator : predictor
        The estimator to be blended.

    cv : int, cross-validation generator or an iterable, optional (default=3)
        Determines the cross-validation splitting strategy to be used for
        generating features to train the next layer on the stacked ensemble or,
        more specifically, during ``fit_transform``.

        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. In all
        other cases, :class:`sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    method : string, optional (default='auto')
        This method will be called on the estimator to produce the output of
        transform. If the method is ``auto``, will try to invoke, for each
        estimator, ``predict_proba``, ``decision_function`` or ``predict``
        in that order.

    n_jobs : int, optional (default=1)
        Number of jobs to be passed to ``cross_val_predict`` during
        ``fit_transform``.
    """
    def __init__(self, estimator, cv=3, method='auto', n_jobs=1):
        self.estimator = estimator
        self.cv = cv
        self.method = method
        self.n_jobs = n_jobs

    def fit(self, X, y, **fit_params):
        """Fit the estimator.

        This should only be used in special situations. Read more in the
        :ref:`User Guide <stacking_transformer>`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        **fit_params : parameters to be passed to the base estimator.

        Returns
        -------
        self : object
        """
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        return self

    def _method_name(self):
        if self.method == 'auto':
            if getattr(self.estimator_, 'predict_proba', None):
                method = 'predict_proba'
            elif getattr(self.estimator_, 'decision_function', None):
                method = 'decision_function'
            else:
                method = 'predict'
        else:
            method = self.method

        return method

    def transform(self, *args, **kwargs):
        """Transform dataset.

        Note that, unlike ``fit_transform()``, this won't return the cross
        validation predictions. Read more in the
        :ref:`User Guide <stacking_transformer>`.

        Parameters
        ----------

        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Input data to be transformed. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csr_matrix`` for maximum efficiency.

        Returns
        -------
        X_transformed : sparse matrix, shape=(n_samples, n_out)
            Transformed dataset.
        """
        t = getattr(self.estimator_, self._method_name())
        preds = t(*args, **kwargs)

        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        return preds

    def fit_transform(self, X, y, **fit_params):
        """Fit estimator and transform dataset.

        Note that this behavior is different from ``fit().transform()`` as it
        will return the cross validation predictions instead. Read more in the
        :ref:`User Guide <stacking_transformer>`.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Input data used to build forests. Use ``dtype=np.float32`` for
            maximum efficiency.

        y : array-like, shape = [n_samples]
            Target values.

        **fit_params : parameters to be passed to the base estimator.

        Returns
        -------
        X_transformed : sparse matrix, shape=(n_samples, n_out)
            Transformed dataset.
        """
        self.estimator_ = clone(self.estimator).fit(X, y, **fit_params)

        preds = cross_val_predict(clone(self.estimator), X, y, cv=self.cv,
                                  method=self._method_name(),
                                  n_jobs=self.n_jobs, fit_params=fit_params)

        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        return preds


def _identity(x):
    return x


def _identity_transformer():
    """Contructs a transformer that returns its input unchanged"""
    return FunctionTransformer(_identity, accept_sparse=True)


def make_stack_layer(estimators, restack=False, cv=3, method='auto',
                     n_jobs=1, n_cv_jobs=1, transformer_weights=None):
    """ Construct a single layer for model stacking

    Read more in the :ref:`User Guide <stacking>`.

    Parameters
    ----------
    estimators : list
        List of (name, predictor) tuples to be used in stacking.

    restack : bool, optional (default=False)
        Whether input should be concatenated to the transformation.

    cv : int, cross-validation generator or an iterable, optional (default=3)
        Determines the cross-validation splitting strategy to be used for
        generating features to train the next layer on the stacked ensemble or,
        more specifically, during ``fit_transform``.

        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. In all
        other cases, :class:`sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    method : string, optional (default='auto')
        Invokes the passed method name of the estimators. If the method is
        ``auto``, will try to invoke ``predict_proba`` or ``predict`` in that
        order.

    n_jobs : int, optional (default=1)
        Number of jobs to run in parallel. Each job will be assigned to a base
        estimator.

    n_cv_jobs: int, optional (default=1)
        Number of jobs to be passed to each base estimator's
        ``cross_val_predict`` during ``fit_transform``.
        If ``n_jobs != 1``, ``n_cv_jobs`` must be 1 and vice-versa, since
        nested parallelism is not supported.

    transformer_weights : dict, optional (default=None)
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.

    Returns
    -------
    FeatureUnion
    """
    transformer_list = [(name, StackingTransformer(estimator, cv=cv,
                                                   method=method,
                                                   n_jobs=n_cv_jobs))
                        for name, estimator in estimators]
    if restack:
        transformer_list.append(('restacker', _identity_transformer()))

    return FeatureUnion(transformer_list, n_jobs=n_jobs,
                        transformer_weights=transformer_weights)
