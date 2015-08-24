import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from ..utils.utils import *


def visualize_variable_relationships(train_data, viz_type, quantitative_vars, category_vars=None):
    """
    Generates plots showing the relationship between several variables.
    """
    # compare the continuous variable distributions using a violin plot
    sub_data = train_data[quantitative_vars]
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    sb.violinplot(sub_data, ax=ax)
    fig.tight_layout()

    # if categorical variables were provided, visualize the quantitative distributions by category
    if category_vars is not None:
        fig, ax = plt.subplots(len(quantitative_vars), len(category_vars), figsize=(16, 12))
        for i, var in enumerate(quantitative_vars):
            for j, cat in enumerate(category_vars):
                sb.violinplot(train_data[var], train_data[cat], ax=ax[i, j])
        fig.tight_layout()

    # generate plots to directly compare the variables
    if category_vars is None:
        if len(quantitative_vars) == 2:
            sb.jointplot(quantitative_vars[0], quantitative_vars[1], train_data, kind=viz_type, size=16)
        else:
            sb.pairplot(train_data, vars=quantitative_vars, kind='scatter',
                        diag_kind='kde', size=16 / len(quantitative_vars))
    else:
        if len(quantitative_vars) == 1:
            if len(category_vars) == 1:
                sb.factorplot(category_vars[0], quantitative_vars[0], None,
                              train_data, kind='auto', size=16)
            else:
                sb.factorplot(category_vars[0], quantitative_vars[0], category_vars[1],
                              train_data, kind='auto', size=16)
        if len(quantitative_vars) == 2:
            if len(category_vars) == 1:
                sb.lmplot(quantitative_vars[0], quantitative_vars[1], train_data,
                          col=None, row=category_vars[0], size=16)
            else:
                sb.lmplot(quantitative_vars[0], quantitative_vars[1], train_data,
                          col=category_vars[0], row=category_vars[1], size=16)
        else:
            sb.pairplot(train_data, hue=category_vars[0], vars=quantitative_vars, kind='scatter',
                        diag_kind='kde', size=16 / len(quantitative_vars))


def visualize_feature_distributions(train_data, viz_type, plot_size):
    """
    Generates feature distribution plots (histogram or kde) for each feature.
    """
    if viz_type == 'hist':
        hist = True
        kde = False
    else:
        hist = False
        kde = True

    num_features = len(train_data.columns)
    num_plots = num_features / plot_size if num_features % plot_size == 0 else num_features / plot_size + 1

    for i in range(num_plots):
        fig, ax = plt.subplots(4, 4, figsize=(20, 10))
        for j in range(plot_size):
            index = (i * plot_size) + j
            if index < num_features:
                if type(train_data.iloc[0, index]) is str:
                    sb.countplot(x=train_data.columns[index], data=train_data, ax=ax[j / 4, j % 4])
                else:
                    sb.distplot(train_data.iloc[:, index], hist=hist, kde=kde, label=train_data.columns[index],
                                ax=ax[j / 4, j % 4], kde_kws={"shade": True})
        fig.tight_layout()


def visualize_correlations(train_data):
    """
    Generates a correlation matrix heat map.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    colormap = sb.blend_palette(sb.color_palette('coolwarm'), as_cmap=True)
    if len(train_data.columns) < 30:
        sb.corrplot(train_data, annot=True, sig_stars=False, diag_names=True, cmap=colormap, ax=ax)
    else:
        sb.corrplot(train_data, annot=False, sig_stars=False, diag_names=False, cmap=colormap, ax=ax)
    fig.tight_layout()


def visualize_sequential_relationships(train_data, plot_size, smooth=None, window=1):
    """
    Generates line plots to visualize sequential data.  Assumes the data frame index is time series.
    """
    train_data.index.name = None
    num_features = len(train_data.columns)
    num_plots = num_features / plot_size if num_features % plot_size == 0 else num_features / plot_size + 1

    for i in range(num_plots):
        fig, ax = plt.subplots(4, 4, sharex=True, figsize=(20, 10))
        for j in range(plot_size):
            index = (i * plot_size) + j
            if index < num_features:
                if index != 3:  # this column is all 0s in the bike set
                    if smooth == 'mean':
                        train_data.iloc[:, index] = pd.rolling_mean(train_data.iloc[:, index], window)
                    elif smooth == 'var':
                        train_data.iloc[:, index] = pd.rolling_var(train_data.iloc[:, index], window)
                    elif smooth == 'skew':
                        train_data.iloc[:, index] = pd.rolling_skew(train_data.iloc[:, index], window)
                    elif smooth == 'kurt':
                        train_data.iloc[:, index] = pd.rolling_kurt(train_data.iloc[:, index], window)

                    train_data.iloc[:, index].plot(ax=ax[j / 4, j % 4], kind='line', legend=False,
                                                   title=train_data.columns[index])
        fig.tight_layout()


def visualize_transforms(X, y, model_type, n_components, transforms):
    """
    Generates plots to visualize the data transformed by a non-linear manifold algorithm.
    """
    transforms = fit_transforms(X, y, transforms)
    X = apply_transforms(X, transforms)

    if model_type == 'classification':
        class_count = np.count_nonzero(np.unique(y))
        colors = sb.color_palette('hls', class_count)

        for i in range(n_components - 1):
            fig, ax = plt.subplots(figsize=(16, 10))
            for j in range(class_count):
                ax.scatter(X[y == j, i], X[y == j, i + 1], s=50, c=colors[j], label=j)
            ax.set_title('Components ' + str(i + 1) + ' and ' + str(i + 2))
            ax.legend()
            fig.tight_layout()
    else:
        for i in range(n_components - 1):
            fig, ax = plt.subplots(figsize=(16, 10))
            sc = ax.scatter(X[:, i], X[:, i + 1], s=50, c=y, cmap='Reds')
            ax.set_title('Components ' + str(i + 1) + ' and ' + str(i + 2))
            ax.legend()
            fig.colorbar(sc)
            fig.tight_layout()
