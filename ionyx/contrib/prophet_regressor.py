import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from fbprophet import Prophet


class ProphetRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn wrapper for the Prophet forecaster.

    Parameters
    ----------
    growth: String 'linear' or 'logistic' to specify a linear or logistic
        trend.
    changepoints: List of dates at which to include potential changepoints. If
        not specified, potential changepoints are selected automatically.
    n_changepoints: Number of potential changepoints to include. Not used
        if input `changepoints` is supplied. If `changepoints` is not supplied,
        then n_changepoints potential changepoints are selected uniformly from
        the first 80 percent of the history.
    yearly_seasonality: Fit yearly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    weekly_seasonality: Fit weekly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    daily_seasonality: Fit daily seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    holidays: pd.DataFrame with columns holiday (string) and ds (date type)
        and optionally columns lower_window and upper_window which specify a
        range of days around the date to be included as holidays.
        lower_window=-2 will include 2 days prior to the date as holidays. Also
        optionally can have a column prior_scale specifying the prior scale for
        that holiday.
    seasonality_prior_scale: Parameter modulating the strength of the
        seasonality model. Larger values allow the model to fit larger seasonal
        fluctuations, smaller values dampen the seasonality. Can be specified
        for individual seasonalities using add_seasonality.
    holidays_prior_scale: Parameter modulating the strength of the holiday
        components model, unless overridden in the holidays input.
    changepoint_prior_scale: Parameter modulating the flexibility of the
        automatic changepoint selection. Large values will allow many
        changepoints, small values will allow few changepoints.
    mcmc_samples: Integer, if greater than 0, will do full Bayesian inference
        with the specified number of MCMC samples. If 0, will do MAP
        estimation.
    interval_width: Float, width of the uncertainty intervals provided
        for the forecast. If mcmc_samples=0, this will be only the uncertainty
        in the trend using the MAP estimate of the extrapolated generative
        model. If mcmc.samples>0, this will be integrated over all model
        parameters, which will include uncertainty in seasonality.
    uncertainty_samples: Number of simulated draws used to estimate
        uncertainty intervals.
    """
    def __init__(
            self,
            growth='linear',
            changepoints=None,
            n_changepoints=25,
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality='auto',
            holidays=None,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            changepoint_prior_scale=0.05,
            mcmc_samples=0,
            interval_width=0.80,
            uncertainty_samples=1000):
        self.growth = growth
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples
        self.model_ = None
        self.preds_ = None

    def fit(self, X, y):
        """
        Fit the Prophet forecast model.

        Parameters
        ----------
        X : array, shape = (n_samples,)
            Dates.
        y : array, shape = (n_samples,)
            Time series values.

        Returns
        -------
        self : Returns an instance of self.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values
        data = pd.DataFrame(X, columns=['ds'])
        data['y'] = y
        self.model_ = Prophet(**self.get_params())
        self.model_.fit(data)

        return self

    def predict(self, X):
        """
        Predict using the Prophet forecast model.

        Parameters
        ----------
        X : array, shape = (n_samples,)
            Dates to generate predictions for.

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        data = pd.DataFrame(X, columns=['ds'])
        self.preds_ = self.model_.predict(data)

        return self.preds_['yhat'].values
