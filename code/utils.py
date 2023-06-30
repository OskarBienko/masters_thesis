import math
import warnings
import numpy as np
import pandas as pd
from typing import List
import plotly.express as px
from scipy.stats import chi2
from scipy.stats import shapiro
import plotly.graph_objects as go
from itertools import combinations
from scipy.stats import chi2_contingency
from scipy.stats.distributions import chi2
from statsmodels.graphics.tsaplots import acf, pacf
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.discrete.discrete_model import BinaryResultsWrapper
warnings.simplefilter(action='ignore', category=InterpolationWarning)
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_log_odds(x: float = None) -> float:
    
    prob = x.sum() / x.count()
    log_odds = np.log(prob/(1-prob))
    return log_odds


def calculate_likelihood_ratio(
    ll0: float = None,
    ll1: float = None,
) -> float:
    return -2 * (ll0-ll1)


def perform_likelihood_ratio_test(
    fitted_model0: BinaryResultsWrapper, 
    fitted_model1: BinaryResultsWrapper,
    alpha: float = 0.05,
    print_inference: bool = False,
) -> float:
    
    L0, L1 = fitted_model0.llf, fitted_model1.llf
    df0, df1 = fitted_model0.df_model, fitted_model1.df_model
    
    chi2_stat = calculate_likelihood_ratio(L0, L1)
    # p_value = 1 - chi2.cdf(chi2_stat, df1-df0).round(3)
    p_value = chi2.sf(chi2_stat, df1-df0).round(3)
    
    if print_inference:
        print('The null hypothesis of the likelihood ratio test is that the full model provides as good fit for the data as the reduced model.')
        print('p-value = P(reject H0|H0 true)')
        print(f'p-value of lrt is equal to: {p_value}')
        if p_value < alpha:
            print(f'p-value < {alpha}, hence there is enough evidence to reject H0 -> there is significant difference in fit!')
        else:
            print(f'p-value > {alpha}, hence there is not enough evidence to reject H0 -> the difference in fit is not significant!')
    return p_value
        

def perform_backward_elimination(
    df: pd.DataFrame = None,
    endog_name: str = None,
    exog: List = None,
    alpha: float = 0.05,
) -> List:
    """Use backward elimination algorithm for feature selection in logistic regression

    :param df: dataframe with endogenous and exogenous variables, defaults to None
    :type df: pd.DataFrame, optional
    :param endog_name: 
    :param exog: list of all exogenous variables to check, defaults to None
    :type exog: List, optional
    :param alpha: significance level, defaults to 0.05
    :type alpha: float, optional
    :return: list with statistically significant exogenous variables
    :rtype: List
    """
    for i in range(len(exog)):
        p_values = list()
        for var in exog:
            exog_temp = exog.copy()
            exog_temp.remove(var)
            # Model 1 is a reduced model
            model1 = Logit(endog=df['target'], exog=df[exog_temp + ['const']]).fit(disp=False)
            # Model 2 is a full model
            model2 = Logit(endog=df['target'], exog=df[exog + ['const']]).fit(disp=False)
            p_value = perform_likelihood_ratio_test(model1, model2)
            p_values.append(p_value)
            max_p_value = max(p_values)
        if max_p_value > alpha:
            boolean_mask = np.array(p_values == max_p_value)
            exog_dropped = np.array(exog)[np.array(boolean_mask)][0]
            exog = np.array(exog)[np.array(~boolean_mask)]
            exog = list(exog)
            print(f'Round no: {i}')
            print(f'LRT p_value: {max_p_value} | Drop: {exog_dropped} variable')
            print('=====================')
            
    return exog
        

def get_interactions(df: pd.DataFrame = None) -> pd.DataFrame:
    """Make interactions by multiplying columns in a dataframe

    :param df: dataframe with candidate exogenous variables, defaults to None
    :type df: pd.DataFrame, optional
    :return: dataframe with interactions
    :rtype: pd.DataFrame
    """

    col_pairs = list(combinations(df.columns, 2))
    dfs = []
    for col_pair in col_pairs:
        divisor = df[col_pair[1]]
        df_divided = df[col_pair[0]].mul(divisor)
        dfs.append(df_divided)

    df_res = pd.concat(dfs, axis=1, join='inner')
    df_res.columns = [f'{tuple[0]}*{tuple[1]}' for tuple in col_pairs]
    return df_res
       
        
def perform_chi_square_independence_test(
    series_1: pd.Series = None,
    series_2: pd.Series = None,
    alpha: float = 0.05,
) -> None:
    
    contigency = pd.crosstab(series_1, series_2)
    p_value = chi2_contingency(contigency)[1].round(3)
    
    print('The null hypothesis of the chi-square test of independence is that the series_1 is independent of series_2.')
    print('p-value = P(reject H0|H0 true)')
    print(f'p-value of the chi-square test of independence is equal to: {p_value}')
    if p_value < alpha:
        print(f'p-value < {alpha}, hence there is enough evidence to reject H0 -> series_2 is not independent of series_1!')
    else:
        print(f'p-value > {alpha}, hence there is not enough evidence to reject H0 -> series_2 is independent of series_1!')
    display(contigency)


def perform_normality_test(
    df: pd.DataFrame = None,
    col_name: str = None,
    alpha: float = 0.05,
) -> None:
    
    p_value_jarque_bera = jarque_bera(df[col_name])[1].round(3)
    p_value_shapiro = np.round(a=shapiro(df[col_name])[1], decimals=3)
    
    print('The null hypothesis of both Jarque-Bera and Shapiro-Wilk tests is that the sample was drawn from a normal distribution.')
    print('p-value = P(reject H0|H0 true)')
    print(f'p-value of Jarque-Bera test is equal to: {p_value_jarque_bera}')
    print(f'p-value of Shapiro-Wilk test is equal to: {p_value_shapiro}')
    if p_value_jarque_bera < alpha and p_value_shapiro < alpha:
        print(f'both Jarque-Bera and Shapiro-Wilk p-values < {alpha}, hence hence there is enough evidence to reject H0 -> the sample does not follow a normal distribution!')
    else:
        print(f'both Jarque-Bera and Shapiro-Wilk p-values > {alpha}, hence there is not enough evidence to reject H0 -> the sample follows a normal distribution!')
        

def perform_stationarity_test(
   df: pd.DataFrame = None,
   col_name: str = None,
   alpha: float = 0.05,
) -> None:
    
   # Stationarity means that the statistical properties of a time series
   # i.e. mean, variance and covariance do not change over time. 
   # Many statistical models require the series to be stationary to make effective and precise predictions.
    
   # Case 1: Both tests conclude that the series is not stationary - The series is not stationary
   # Case 2: Both tests conclude that the series is stationary - The series is stationary
   # Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.
   # Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. Differencing is to be used to make series stationary. The differenced series is checked for stationarity.

   # ADF test
   p_value = np.round(adfuller(x=df[col_name], regression='c', autolag=None)[1], 3)
   print('The null hypothesis of the Augmented Dickey-Fuller is that there is a unit root (series is not stationary).')
   print('The alternative hypothesis is that there is no unit root.')
   print('p-value = P(reject H0|H0 true)')
   print(f'p-value for {col_name} is equal to {str(p_value)}.')
   
   if p_value > alpha:
      print(f'p-value > {alpha}, hence there is not enough evidence to reject H0 -> the process is not stationary!')
   else:
      print(f'p-value < {alpha}, hence there is enough evidence to reject H0 -> thus the process is stationary!')
      
   # KPSS test
   print('\n' + '='*30)
   p_value = np.round(kpss(x=df[col_name], regression='c', nlags='auto')[1], 3)
   print('The null hypothesis of the Kwiatkowski-Phillips-Schmidt-Shin is that the process is trend stationary.')
   print('The alternative hypothesis is that there is a unit root (series is not stationary).')
   print('p-value = P(reject H0|H0 true)')
   print(f'p-value for {col_name} is equal to {str(p_value)}.')
   
   if p_value > alpha:
      print(f'p-value > {alpha}, hence there is not enough evidence to reject H0 -> the process is stationary!')
   else:
      print(f'p-value < {alpha}, hence there is enough evidence to reject H0 -> the process is not stationary!')
      

def perform_autocorrelation_test(
    series: pd.Series = None,
    order: int = None,
    alpha: float = 0.05,
) -> None:
    
    ljungbox_pvalues = acorr_ljungbox(x=series, lags=order)['lb_pvalue'].round(3)
    boolean_mask = ljungbox_pvalues > alpha

    if not ljungbox_pvalues.empty:
            print(f'The null hypothesis of Ljung-Box test is that the autocorrelations of any order up to {order} are all zero.')
            print('p-value = P(reject H0|H0 true)')
            print(f'p-values of Ljung-Box test are: {[pvalue for pvalue in ljungbox_pvalues.values]}')
            print(f'p-values > {alpha}, hence there is not enough evidence to reject H0 -> the series is uncorrelated at lags {[lag for lag in ljungbox_pvalues[boolean_mask].index]}')
            print(f'p-values < {alpha}, hence there is enough evidence to reject H0 -> the series is autocorrelated at lags {[lag for lag in ljungbox_pvalues[~boolean_mask].index]}')
    

def perform_hosmer_lemeshow_test(
    df: pd.DataFrame = None,
    endog_name: str = None,
    pred_name: str = None,
    q: int = None,
) -> float:
    """ Perform the Hosmer-Lemeshow Chi-squared test to judge the goodness of fit for binary data.
    The p-values greater than 0.05 indicate a well-fitted model.
    One rejects the null hypothesis that the observed and predicted probabilities are the same.
    As we know the variable Y is a binary variable which means that it follows Bernoulli(p_i).
    So, the expected value of the variable Y is p_i.
    Also, we should keep in our minds that every value of Y is independent from the other.
    So, the expected number of outcomes of Y = 1 in the g-th decile is p_1 + p_2 + p_3 + ... + p_n where n is the number of p_i which belong in the bin.
    Sources:
        ttps://jbhender.github.io/Stats506/F18/GP/Group13.html
        https://stackoverflow.com/questions/40327399/hosmer-lemeshow-goodness-of-fit-test-in-python
    
    :param df: dataframe with pred_name and endog_name columns, defaults to None
    :type df: pd.DataFrame, optional
    :param endog_name: name of the endogeous variable, defaults to None
    :type endog_name: str, optional  
    :param pred_name: name of the column with prediction, defaults to None
    :type pred_name: str, optional
    :param q: quantiles, defaults to None
    :type q: int, optional
    :return: pvalue
    :rtype: float
    """
    
    df = df.rename(columns={endog_name: 'observed'}).copy()
    df['bins'] = pd.qcut(x=df[pred_name], q=q)
    ys = df['observed'].groupby(by=df['bins']).sum()
    yt = df['observed'].groupby(by=df['bins']).count()
    yn = yt - ys

    yps = df[pred_name].groupby(by=df['bins']).sum()
    ypt = df[pred_name].groupby(by=df['bins']).count()
    ypn = ypt - yps

    hlchi2test = (((ys - yps)**2 / yps) + ((yn - ypn)**2 / ypn)).sum()
    df = q-2
    p_value = 1 - chi2.cdf(hlchi2test, df).round(3)
    return p_value


def print_vif(
    df: pd.DataFrame = None,
    exog: List = None,
) -> None:
    """Calculate the VIF for each variable using statsmodels

    :param df: dataframe, defaults to None
    :type df: pd.DataFrame, optional
    :param exog: list with exogenous variables names, defaults to None
    :type exog: List, optional
    """
    
    vif = dict()
    for i, var in enumerate(exog):
        vif[var] = variance_inflation_factor(exog=df[exog], exog_idx=i).round(2)

    display(pd.DataFrame.from_dict(data=vif, orient='index', columns=['VIF']))
    
    
def print_epv(
    df: pd.DataFrame = None,
    endog_name: str = None,
    exog: List = None,
) -> None:
    """Calculate the number of events per variable (EPV) required in multivariate analysis.
    Source:
        A Simulation Study of the Number of Events per Variable in Logistic Regression Analysis, 1996,
        P. Peduzzi, J. Concato, E. Kemper, T. R. Holford, and A. R. Feinstein
    :param df: dataframe, defaults to None
    :type df: pd.DataFrame, optional
    :param endog_name: name of the endogeous variable, defaults to None
    :type endog_name: str, optional  
    :param exog: list with names of exogenous variables, defaults to None
    :type exog: List, optional
    """
    
    # Get the number of observations
    nobs = len(df)
    
    # Get the number of positive events
    nobs_positive = df[endog_name].sum()
    
    # Get the number of negative events
    nobs_negative = nobs - nobs_positive
    
    # Find the minimum nobs
    nobs_minimum = min(nobs_positive, nobs_negative)

    # Get the number of exogenous variables exluding const
    n_exog = len(exog) - 1

    # Calculate the number of events per variable and round it up
    epv = math.ceil(nobs_minimum / n_exog)

    print(f'Nobs of success: {nobs_positive}')
    print(f'Nobs of failure: {nobs_negative}')
    print(f'Number of predictors: {n_exog}')
    if epv >= 10:
        print(f'The number of events per variable is sufficient: EPV = {epv}.')
    else:
        print(f'The number of events per variable is insufficient: EPV = {epv}.')
        
        
def plot_linearity(
    df: pd.DataFrame = None,
    endog_name: str = None,
    yaxis_name: str = None,
    exog: List = None,
    quantile: int = None
) -> None:
    """Check linear assumption in logistic regression using plotly.
    One way to check if the relationship between the log-odds of the outcome and each continuous independent variable is linear is to: 
    1. Discretize continuous independent variable into quantiles
    2. Calculate probability of expected outcome in each quantile
    3. Calculate log-odds
    4. Plot log-odds versus median independent variable in each quantile
    :param df: dataframe, defaults to None
    :type df: pd.DataFrame, optional
    :param endog_name: name of the endogeous variable, defaults to None
    :type endog_name: str, optional
    :param yaxix_name: label for the y-axis
    :type yaxix_name: str, optional
    :param exog: list with exogenous variables names, defaults to None
    :type exog: List, optional
    :param quantile: number of quantiles. 10 for deciles, 4 for quartiles, etc., defaults to None
    :type quantile: int, optional
    """
    for variable in exog:
        if variable != 'const':
            df_temp = df.groupby(by=pd.qcut(x=df[variable], q=quantile, duplicates='drop')) \
                .agg(func={endog_name: calculate_log_odds, variable: 'median'}) \
                .round(3)
            fig = px.scatter(
                data_frame=df_temp,
                x=variable,
                y=endog_name,
                labels={endog_name: yaxis_name},
                width=500,
                height=500,
                template='ggplot2',
                trendline='ols',
                trendline_color_override='black',
            )
            fig.update_layout(
                legend=dict(
                    orientation='h',
                    entrywidth=70,
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                )
            )
            fig.update_layout(
                margin=dict(
                    l=50,
                    r=30,
                    b=30,
                    t=30,
                    pad=4
                    ),
                paper_bgcolor='rgb(240, 240, 240)'
            )
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', title_standoff=5)
            fig.update_yaxes(showline=True, linewidth=2, linecolor='black', title_standoff=5)
            fig.show(renderer='png', config={'displayModeBar': False})

    
def plot_pacf(
    series: pd.Series = None,
    series_name: str = None,
    nlags: int = None,
    partial: bool = True,
    save: bool = False
) -> None:
    
    """
    This func estimates the (partial) autocorrelation function of a given time series using statsmodels package and then plots it using plotly package
    Note that the default pacf calculation method is Yule-Walker with sample size-adjustment in denominator for acovf
    Note that the default confidence interval is 95%
    
    :param series: observations of time series for which acf/pacf is calculated
    :type series: pd.Series
    :param series_name: at least the name and the frequency of the time series, without special characters, that will be used to create a title
    :type series_name: str
    :param nlags: number of lags to return autocorrelation for
    :type nlags: int
    :param partial: whether to plot pacf
    :type partial: bool (default True)
    :param save: whether to save the plot to the current workding directory
    :type save: bool (default False)
    """
    
    if not isinstance(series, pd.Series):
        raise Exception('Time series is not a pd.Series type!')

    # Define the title depending on the bool argument
    title=f'PACF of {series_name}' if partial else f'ACF of {series_name}'
    
    # Calculate the acf/pacf and the confidence intervals
    corr_array, conf_int_array = pacf(series.dropna(), alpha=0.05, nlags=nlags, method='yw') if partial else acf(series.dropna(), alpha=0.05, nlags=nlags)
    
    # Center the confidence intervals so that it's easy to visually inspect if a given correlation is significantly different from zero
    lower_y = conf_int_array[:,0] - corr_array
    upper_y = conf_int_array[:,1] - corr_array
    
    # Create an empty figure
    fig = go.Figure()

    # Plot the correlations using vertical lines
    [fig.add_scatter(x=(x,x), y=(0,corr_array[x]), mode='lines', line_color='#3f3f3f', hoverinfo='skip') 
        for x in np.arange(len(corr_array))]
    
    # Plot the correlations using markers
    # The <extra></extra> part removes the trace name
    fig.add_scatter(
        x=np.arange(len(corr_array)),
        y=corr_array,
        mode='markers',
        marker_color='#1f77b4',
        marker_size=12,
        hovertemplate=
        'Lag %{x}<br>' +
        'Corr: %{y:.2f}<br>' +
        '<extra></extra>'
    )
    
    # Plot the centered confidence intervals
    fig.add_scatter(x=np.arange(len(corr_array)), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)', hoverinfo='skip')
    fig.add_scatter(x=np.arange(len(corr_array)), y=lower_y, mode='lines', fillcolor='rgba(32, 146, 230,0.3)',
        fill='tonexty', line_color='rgba(255,255,255,0)', hoverinfo='skip')
    
    # Prettify the plot
    fig.update_traces(showlegend=False)
    fig.update_xaxes(tickvals=np.arange(start=0, stop=nlags+1))
    fig.update_yaxes(zerolinecolor='#000000')
    fig.update_layout(title=title, title_x=0.5, width=500, height=300, margin=dict(l=0, r=0, b=0, t=30, pad=1))
    # fig.update_layout(title=title, title_x=0.5, width=500, height=300, hovermode=False, margin=dict(l=0, r=0, b=0, t=30, pad=1))

    # Save the plot to the current working directory
    if save:
        fig.write_image(f'''{title.replace(' ', '_')}.png''')

    # Eventually show the plot
    fig.show(renderer='png')