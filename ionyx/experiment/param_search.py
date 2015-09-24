import time
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid

from ..utils import fit_transforms, apply_transforms, predict_score


def parameter_grid_search(X, y, model, metric, transform_grid, param_grid, test_split_size=0.2):
    """
    Performs an exhaustive search over the specified model parameters.
    """
    print('Beginning parameter grid search...')
    t0 = time.time()
    params_list = list(ParameterGrid(param_grid))
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=test_split_size)

    for transforms in transform_grid:
        print('Transforms = {0}'.format(transforms))
        print('')
        print('')
        transforms = fit_transforms(X_train, y_train, transforms)
        X_train = apply_transforms(X_train, transforms)
        X_eval = apply_transforms(X_eval, transforms)

        for params in params_list:
            tsub0 = time.time()
            for param, value in params.iteritems():
                print(param + " = " + str(value))
                setattr(model, param, value)

            print('Fitting model...')
            model.fit(X_train, y_train)

            train_score = predict_score(X_train, y_train, model, metric)
            print('Training score ='), train_score

            eval_score = predict_score(X_eval, y_eval, model, metric)
            print('Evaluation score ='), eval_score

            tsub1 = time.time()
            print('Model trained in {0:3f} s.'.format(tsub0 - tsub1))
            print('')
            print('')

    t1 = time.time()
    print('Grid search complete in {0:3f} s.'.format(t0 - t1))
