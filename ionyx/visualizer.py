import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.preprocessing import LabelEncoder


class Visualizer(object):
    """
    Provides a number of descriptive functions for creating useful visualizations.  Initialize the
    class by passing in a data set and then call the functions individually to create the plots.
    Each method is designed to adapt the character of the visualization based on the inputs provided.
    
    Parameters
    ----------
    data : array-like
        The data set to use to create visualizations.

    fig_size : int, optional, default 16
        Size of the plots.
    """
    def __init__(self, data, fig_size=16):
        self.data = data
        self.fig_size = fig_size

    def feature_distributions(self, viz_type='hist', bins=None, grid_size=4):
        """
        Generates feature distribution plots (histogram or kde) for each feature.

        Parameters
        ----------
        viz_type : {'hist', 'kde', 'both'}, optional, default 'hist'
            Type of plot used for visualization.

        bins : int, optional, default None
            Number of bins to use in histogram plots.

        grid_size : int, optional, default 4
            Number of vertical/horizontal plots to display in a single window.
        """
        if viz_type == 'hist':
            hist = True
            kde = False
        elif viz_type == 'kde':
            hist = False
            kde = True
        elif viz_type == 'both':
            hist = True
            kde = True
        else:
            raise Exception('Visualization type not supported.')

        data = self.data.fillna(0)

        n_features = len(data.columns)
        plot_size = grid_size ** 2
        n_plots = n_features // plot_size if n_features % plot_size == 0 else n_features // plot_size + 1

        for i in range(n_plots):
            fig, ax = plt.subplots(grid_size, grid_size, figsize=(self.fig_size, self.fig_size / 2))
            for j in range(plot_size):
                index = (i * plot_size) + j
                if index < n_features:
                    if type(data.iloc[0, index]) is str:
                        sb.countplot(x=data.columns[index], data=data, ax=ax[j // grid_size, j % grid_size])
                    else:
                        sb.distplot(a=data.iloc[:, index], bins=bins, hist=hist, kde=kde,
                                    label=data.columns[index], ax=ax[j // grid_size, j % grid_size],
                                    kde_kws={"shade": True})
            fig.tight_layout()

    def correlations(self, annotate=False):
        """
        Generates a correlation matrix heat map.

        Parameters
        ----------
        annotate : boolean, optional, default False
            Annotate the heat map with labels.
        """
        corr = self.data.corr()

        if annotate:
            corr = np.round(corr, 2)

        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        fig, ax = plt.subplots(figsize=(self.fig_size, self.fig_size * 3 / 4))
        colormap = sb.blend_palette(sb.color_palette('coolwarm'), as_cmap=True)
        sb.heatmap(corr, mask=mask, cmap=colormap, annot=annotate)
        fig.tight_layout()

    def variable_relationship(self, quantitative_vars, category_vars=None, joint_viz_type='scatter',
                              pair_viz_type='scatter', factor_viz_type='strip', pair_diag_type='kde'):
        """
        Generates plots showing the relationship between several variables.  The combination of plots generated
        depends on the number of quantitative and discrete (categorical or ordinal) variables to be analyzed.
        Plots are rendered using the seaborn statistical visualization package.

        Parameters
        ----------
        quantitative_vars : array-like
            List of variable names to analyze quantitatively.

        category_vars : array-like, optional, default None
            List of variable names to analyze discretely.

        joint_viz_type : {'scatter', 'reg', 'resid', 'kde', 'hex'}, optional, default 'scatter'
            Method to use to display two quantitative variables together.

        pair_viz_type : {'scatter', 'reg'}, optional, default 'scatter'
            Method to use to display more than two quantitative variables together.

        factor_viz_type : {'point', 'bar', 'count', 'box', 'violin', 'strip'}, optional, default 'strip'
            Method to use to display one quantitative variable along with categorical variables.

        pair_diag_type : {'hist', 'kde'}, optional, default 'kde'
            Display type for the diagonal plots in a pair plot.
        """
        if quantitative_vars is None or len(quantitative_vars) == 0:
            raise Exception('Must provide at least one quantitative variable.')

        sub_data = self.data[quantitative_vars]
        fig, ax = plt.subplots(1, 1, figsize=(self.fig_size, self.fig_size * 3 / 4))
        sb.violinplot(data=sub_data, ax=ax)
        fig.tight_layout()

        if category_vars is not None:
            fig, ax = plt.subplots(len(quantitative_vars), len(category_vars),
                                   figsize=(self.fig_size, self.fig_size * 3 / 4))
            if len(quantitative_vars) == 1:
                if len(category_vars) == 1:
                    sb.violinplot(x=quantitative_vars[0], y=category_vars[0], data=self.data, ax=ax)
                else:
                    for i, cat in enumerate(category_vars):
                        sb.violinplot(x=quantitative_vars[0], y=cat, data=self.data, ax=ax[i])
            else:
                for i, var in enumerate(quantitative_vars):
                    if len(category_vars) == 1:
                        sb.violinplot(x=var, y=category_vars[0], data=self.data, ax=ax[i])
                    else:
                        for j, cat in enumerate(category_vars):
                            sb.violinplot(x=var, y=cat, data=self.data, ax=ax[i, j])
            fig.tight_layout()

        if category_vars is None:
            if len(quantitative_vars) == 2:
                sb.jointplot(x=quantitative_vars[0], y=quantitative_vars[1], data=self.data,
                             kind=joint_viz_type, size=self.fig_size)
            else:
                sb.pairplot(data=self.data, vars=quantitative_vars, kind=pair_viz_type,
                            diag_kind=pair_diag_type, size=self.fig_size / len(quantitative_vars))
        else:
            if len(quantitative_vars) == 1:
                if len(category_vars) == 1:
                    sb.factorplot(x=category_vars[0], y=quantitative_vars[0],
                                  data=self.data, kind=factor_viz_type, size=self.fig_size)
                else:
                    sb.factorplot(x=category_vars[0], y=quantitative_vars[0], hue=category_vars[1],
                                  data=self.data, kind=factor_viz_type, size=self.fig_size)
            elif len(quantitative_vars) == 2:
                if len(category_vars) == 1:
                    sb.lmplot(x=quantitative_vars[0], y=quantitative_vars[1],
                              data=self.data, row=category_vars[0], size=self.fig_size)
                else:
                    sb.lmplot(x=quantitative_vars[0], y=quantitative_vars[1], data=self.data,
                              col=category_vars[0], row=category_vars[1], size=self.fig_size)
            else:
                sb.pairplot(data=self.data, hue=category_vars[0], vars=quantitative_vars, kind=pair_viz_type,
                            diag_kind=pair_diag_type, size=self.fig_size / len(quantitative_vars))

    def sequential_relationships(self, time='index', smooth_method=None, window=1, grid_size=4):
        """
        Generates line plots to visualize sequential data.

        Parameters
        ----------
        time : string, optional, default 'index'
            Datetime input column to use for visualization.

        smooth_method : {'mean', 'var', 'skew', 'kurt'}, optional, default None
            Apply a function to the time series to smooth out variations.

        window : int, optional, default 1
            Size of the moving window used to calculate the smoothing function.

        grid_size : int, optional, default 4
            Number of vertical/horizontal plots to display in a single window.
        """
        data = self.data.fillna(0)

        if time is not 'index':
            data = data.reset_index()
            data = data.set_index(time)

        data.index.name = None
        n_features = len(data.columns)
        plot_size = grid_size ** 2
        n_plots = n_features // plot_size if n_features % plot_size == 0 else n_features // plot_size + 1

        for i in range(n_plots):
            fig, ax = plt.subplots(grid_size, grid_size, sharex=True, figsize=(self.fig_size, self.fig_size / 2))
            for j in range(plot_size):
                index = (i * plot_size) + j
                if index < n_features:
                    if type(data.iloc[0, index]) is not str:
                        if smooth_method == 'mean':
                            data.iloc[:, index] = pd.rolling_mean(data.iloc[:, index], window)
                        elif smooth_method == 'var':
                            data.iloc[:, index] = pd.rolling_var(data.iloc[:, index], window)
                        elif smooth_method == 'skew':
                            data.iloc[:, index] = pd.rolling_skew(data.iloc[:, index], window)
                        elif smooth_method == 'kurt':
                            data.iloc[:, index] = pd.rolling_kurt(data.iloc[:, index], window)

                        data.iloc[:, index].plot(ax=ax[j // grid_size, j % grid_size], kind='line',
                                                 legend=False, title=data.columns[index])
            fig.tight_layout()

    def transform(self, transform, X_columns, y_column=None, task=None, n_components=2, scatter_size=50):
        """
        Generates plots to visualize the data transformed by a linear or manifold algorithm.

        Parameters
        ----------
        transform : array-like
            Transform object.  Can be a pipeline with multiple transforms.

        X_columns : list
            List of columns to use to fit the transform.

        y_column : string, optional, default None
            Target column.  Used to color input values for label-based visualizations.

        task : {'classification', 'regression', None}, optional, default None
            Specifies if the data set is being used for classification or regression.  If one of these
            is specified, the plots will color input values using the provided labels.

        n_components : int, optional, default 2
            Number of components of the transformed data set to visualize.

        scatter_size : int, optional, default 50
            Size of the points on the scatter plot.
        """
        X = self.data[X_columns].values
        X = transform.fit_transform(X)

        y = None
        encoder = None
        if y_column:
            y = self.data[y_column].values
            if task == 'classification':
                encoder = LabelEncoder()
                y = encoder.fit_transform(y)

        if y_column and task == 'classification':
            class_count = len(np.unique(y))
            colors = sb.color_palette('hls', class_count)
            for i in range(n_components - 1):
                fig, ax = plt.subplots(figsize=(self.fig_size, self.fig_size * 3 / 4))
                for j in range(class_count):
                    ax.scatter(X[y == j, i], X[y == j, i + 1], s=scatter_size, c=colors[j],
                               label=encoder.classes_[j])
                ax.set_title('Components ' + str(i + 1) + ' and ' + str(i + 2))
                ax.legend()
                fig.tight_layout()
        elif y_column and task == 'regression':
            for i in range(n_components - 1):
                fig, ax = plt.subplots(figsize=(self.fig_size, self.fig_size * 3 / 4))
                sc = ax.scatter(X[:, i], X[:, i + 1], s=scatter_size, c=y, cmap='Blues')
                ax.set_title('Components ' + str(i + 1) + ' and ' + str(i + 2))
                ax.legend()
                fig.colorbar(sc)
                fig.tight_layout()
        else:
            for i in range(n_components - 1):
                fig, ax = plt.subplots(figsize=(self.fig_size, self.fig_size * 3 / 4))
                ax.scatter(X[:, i], X[:, i + 1], s=scatter_size, label='None')
                ax.set_title('Components ' + str(i + 1) + ' and ' + str(i + 2))
                ax.legend()
                fig.tight_layout()

    def feature_importance(self, X_columns, y_column, average=False, task='classification', **kwargs):
        """
        Visualize the predictive importance of each feature in a data set using a trained
        gradient boosting model.

        Parameters
        ----------
        X_columns : list
            List of columns to use to fit the transform.

        y_column : string, optional, default None
            Target column.  Used to color input values for label-based visualizations.

        average : boolean, optional, default False
            Smooth the results by fitting the model multiple times to reduce random variance.

        task : {'classification', 'regression'}, optional, default 'classification'
            Specifies if the target is continuous or categorical.

        **kwargs : dict, optional
            Arguments to pass to the scikit-learn gradient boosting model to improve the quality
            of the fit.  If none are provided then the defaults will be used.
        """
        X = self.data[X_columns]
        y = self.data[y_column]

        if task == 'classification':
            model = GradientBoostingClassifier(**kwargs)
        else:
            model = GradientBoostingRegressor(**kwargs)

        if average:
            feature_importance = np.ones((1, X.shape[1]))
            for i in range(10):
                model.fit(X, y)
                temp = model.feature_importances_.reshape(1, -1)
                feature_importance = np.append(feature_importance, temp, axis=0)
            feature_importance = feature_importance[1:, :].mean(axis=0).reshape(1, -1)
        else:
            model.fit(X, y)
            feature_importance = model.feature_importances_.reshape(1, -1)

        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance.ravel())
        pos = np.arange(sorted_idx.shape[0])

        fig, ax = plt.subplots(figsize=(self.fig_size, self.fig_size * 3 / 4))
        ax.set_title('Variable Importance')
        ax.barh(pos, feature_importance.ravel()[sorted_idx], align='center')
        ax.set_yticks(pos)
        ax.set_yticklabels([X_columns[i] for i in sorted_idx])
        ax.set_xlabel('Relative Importance')
        fig.tight_layout()

    def partial_dependence(self, X_columns, y_column, var_column, average=False,
                           task='classification', grid_resolution=100, **kwargs):
        """
        Visualize the marginal effect of a single variable on a dependent variable, holding
        all other variables constant.  Generated via a trained gradient boosting model.

        Parameters
        ----------
        X_columns : list
            List of columns to use to fit the transform.

        y_column : string, optional, default None
            Target column.  Used to color input values for label-based visualizations.

        var_column : string
            The name of the variable to comparse to the response value.

        average : boolean, optional, default False
            Smooth the results by fitting the model multiple times to reduce random variance.

        task : {'classification', 'regression'}, optional, default 'classification'
            Specifies if the target is continuous or categorical.

        grid_resolution : int, optional, default 100
            Defines the granularity of the segments in the plot.

        **kwargs : dict, optional
            Arguments to pass to the scikit-learn gradient boosting model to improve the quality
            of the fit.  If none are provided then the defaults will be used.
        """
        X = self.data[X_columns]
        y = self.data[y_column]
        index = X_columns.index(var_column)

        distinct = len(np.unique(self.data[var_column]))
        if distinct < grid_resolution:
            grid_resolution = distinct

        if task == 'classification':
            model = GradientBoostingClassifier(**kwargs)
        else:
            model = GradientBoostingRegressor(**kwargs)

        if average:
            response = np.ones((1, grid_resolution))
            axes = np.ones((1, grid_resolution))
            for i in range(10):
                model.fit(X, y)
                a, b = partial_dependence(model, [index], X=X, grid_resolution=grid_resolution)
                response = np.append(response, a, axis=0)
                axes = np.append(axes, b[0].reshape((1, grid_resolution)), axis=0)
            response = response[1:, :].mean(axis=0).reshape((grid_resolution, 1))
            axes = axes[1:, :].mean(axis=0).reshape((grid_resolution, 1))
        else:
            model.fit(X, y)
            response, axes = partial_dependence(model, [index], X=X, grid_resolution=grid_resolution)
            response = response.reshape((grid_resolution, 1))
            axes = axes[0].reshape((grid_resolution, 1))

        df = pd.DataFrame(np.append(axes, response, axis=1), columns=[var_column, y_column])
        df.plot(x=var_column, y=y_column, kind='line', figsize=(self.fig_size, self.fig_size * 3 / 4))
