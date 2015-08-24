def train_averaged_ensemble(X, y, X_test, metric, transforms, n_folds):
    """
    Creates an averaged ensemble of many models together.
    """
    t0 = time.time()
    models = bag_of_models(X.shape[1])
    n_models = len(models)
    n_records = y.shape[0]

    model_train_scores = np.zeros((n_folds, n_models))
    y_models = np.zeros((n_records, n_models))
    y_avg = np.zeros(n_records)
    y_true = np.zeros(n_records)

    folds = list(KFold(n_records, n_folds=n_folds, shuffle=True, random_state=1337))
    for i, (train_index, eval_index) in enumerate(folds):
        print('Starting fold {0}...'.format(i + 1))
        X_train = X[train_index]
        y_train = y[train_index]
        X_eval = X[eval_index]
        y_eval = y[eval_index]

        transforms = fit_transforms(X_train, y_train, transforms)
        X_train = apply_transforms(X_train, transforms)
        X_eval = apply_transforms(X_eval, transforms)

        print('Fitting individual models...')
        for k, model in enumerate(models):
            model.fit(X_train, y_train)

        print('Generating predictions and scoring...')
        for k, model in enumerate(models):
            model_train_scores[i, k] = score(y_train, model.predict(X_train), metric)
            y_models[eval_index, k] = model.predict(X_eval)

        y_avg[eval_index] = y_models[eval_index, :].sum(axis=1) / n_models
        y_true[eval_index] = y_eval

    t1 = time.time()
    print('Ensemble training completed in {0:3f} s.'.format(t1 - t0))

    for k, model in enumerate(models):
        print('Model ' + str(k) + ' average training score ='), model_train_scores[:, k].sum(axis=0) / n_folds
        print('Model ' + str(k) + ' eval score ='), score(y_true, y_models[:, k], metric)
    print('Ensemble eval score ='), score(y_true, y_avg, metric)

    df = pd.DataFrame(y_models, columns=['Model ' + str(i) for i in range(n_models)])
    visualize_correlations(df)

    print('Fitting models on full data set...')
    n_test_records = X_test.shape[0]
    y_models_test = np.zeros((n_test_records, n_models))

    transforms = fit_transforms(X, y, transforms)
    X = apply_transforms(X, transforms)
    X_test = apply_transforms(X_test, transforms)

    for k, model in enumerate(models):
        model.fit(X, y)

    print('Generating test data predictions...')
    for k, model in enumerate(models):
        y_models_test[:, k] = model.predict(X_test)

    y_avg_test = y_models_test.sum(axis=1) / n_models

    print('Ensemble complete.')
    return y_avg_test
