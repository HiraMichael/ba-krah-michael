import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.othermod.betareg import BetaModel
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor


def perform_canon_specific_analysis(df, outcome, predictors, path):
    # dictionary of descriptive statistics
    descriptive_statistics = compute_descriptive_statistics(df, outcome)
    outcome_table = pd.DataFrame(descriptive_statistics, index=[outcome]).T

    # beta model
    beta_model, exog = perform_beta_regression(df, outcome, predictors)

    # model summary
    model_summary = beta_model.summary()
    with open(path + f'/analysis/tables/{outcome}_model.tex', 'w') as file:
        file.write(model_summary.as_latex())

    # odds ratios
    odds_ratios = convert_coefficients_to_odds_ratios(beta_model, outcome)
    outcome_table = pd.concat([outcome_table, odds_ratios])

    # diagnostics
    plot_residuals(beta_model, path, outcome)
    check_outliers(beta_model, path, outcome)
    check_collinearity(exog, path, outcome)
    check_autocorrelation(beta_model, path, outcome)
    return outcome_table

def compute_descriptive_statistics(df, column_name):
    stats = {
        'Mean': df[column_name].mean(),
        'Std.': df[column_name].std(),
        '25%': df[column_name].quantile(0.25),
        '50% (median)': df[column_name].quantile(0.50),
        '75%': df[column_name].quantile(0.75),
        'Min': df[column_name].min(),
        'Max': df[column_name].max()
    }
    return stats

def perform_beta_regression(df, outcome, predictors):
    df = df.copy()
    # Adjust outcome variable to avoid boundaries (0, 1)
    df['outcome_adjusted'] = df[outcome].clip(0.01, 0.99)

    # Create a lagged version of the outcome variable
    df['lag_outcome'] = df['outcome_adjusted'].shift(1)  # Shift by 1 time step
    predictors.append('lag_outcome')

    # Drop first row
    df = df.dropna()

    # Define predictors
    exog = sm.add_constant(df[predictors])
    endog = df['outcome_adjusted']

    # Fit the Beta regression model
    beta_model = BetaModel(endog, exog).fit(method='bfgs', maxiter=2500)

    return beta_model, exog

def convert_coefficients_to_odds_ratios(beta_model, outcome):
    coefficients = beta_model.params
    odds_ratios = np.exp(coefficients).round(3)
    conf = beta_model.conf_int()
    conf_exp = np.exp(conf).round(3)
    odds_ratios_with_conf = pd.DataFrame({
        outcome: odds_ratios.map(str) +
                               " (" + conf_exp[0].map(str) +
                               ", " + conf_exp[1].map(str) + ")"
    })
    return odds_ratios_with_conf

def plot_residuals(beta_model, path, outcome):
    residuals = beta_model.resid_pearson
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.tight_layout()
    plt.savefig(path + f'/analysis/figures/{outcome}_residuals.pdf', dpi=100, format='pdf')

def check_outliers(beta_model, path, outcome):
    influence = beta_model.get_influence()
    cooks_d = influence.cooks_distance[0]
    plt.stem(cooks_d)
    plt.title('Cook\'s Distance')
    plt.tight_layout()
    plt.savefig(path + f'/analysis/figures/{outcome}_outliers.pdf', dpi=100, format='pdf')

def check_collinearity(exog, path, outcome):
    vif_data = pd.DataFrame({
        "Variable": exog.columns,
        "VIF": [variance_inflation_factor(exog.values, i) for i in range(exog.shape[1])]
    })
    with open(path + f'/analysis/tables/{outcome}_collinearity.tex', 'w') as file:
        file.write(vif_data.to_latex())

def check_autocorrelation(beta_model, path, outcome):
    dw_stat = durbin_watson(beta_model.resid_pearson)
    with open(path + f'/analysis/tables/{outcome}_autocorrelation.txt', 'w') as file:
        file.write(f'Durbin-Watson statistic for autocorrelation: {dw_stat}')

