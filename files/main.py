########################################### IMPORTS #################################################################
# Graphiques
import matplotlib.pyplot as plt
import seaborn as sns
from aquarel import load_theme

# Maths / data
import datetime
import numpy as np
import pandas as pd

# Statsmodels
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.ar_model import AutoReg

# EM
from functools import reduce
from sklearn.decomposition import PCA

# Midas
from midas.mix import mix_freq, mix_freq2
from midas.adl import (
    estimate,
    forecast,
    midas_adl,
    rmse,
    estimate2,
    forecast2,
    midas_adl2,
)

# Data load and save
from os import listdir, mkdir
from os.path import isdir

# Hide warnings in midas
import warnings

warnings.filterwarnings("ignore")

# Graphic theme
theme = load_theme("scientific")
theme.apply()


########################################### FONCTIONS #################################################################


#### Verical realignement functions and factor extraction
def vertical_realignement(df: pd.DataFrame, k: int = 10):
    for index in range(k, df.shape[0]):
        df.iloc[index] = df.iloc[index - k]
    return df.iloc[k:]


def va_dfm(
    data: pd.DataFrame, k_factors: int = 1, factor_order: int = 1, error_order: int = 2
):
    """
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9042056/#:~:text=Abstract,a%20few%20principal%20component%20series.
    """
    va_X = vertical_realignement(data).dropna(axis=1)
    va_X -= va_X.shift(12)
    mod = sm.tsa.DynamicFactor(
        va_X, k_factors=k_factors, factor_order=factor_order, error_order=error_order
    )
    initial_res = mod.fit(method="powell", disp=False)
    res = mod.fit(initial_res.params, disp=False)
    return pd.DataFrame(
        res.factors.filtered.T,
        index=va_X.index,
        columns=[f"factor {factor_nb}" for factor_nb in np.arange(1, k_factors + 1)],
    )


def get_last_k_lags(date, k, dataset):
    date = pd.to_datetime(date)
    idx = dataset.index.get_loc(date)
    last_k_lags = dataset.iloc[max(0, idx - k) : idx]
    return last_k_lags.values.flatten().ravel()


def lagged_dataset_build(y: pd.DataFrame, X: pd.DataFrame, k: int = 1):
    target_idx = y.dropna(axis=0).index
    X_returned = np.zeros(shape=(len(target_idx), k * X.shape[1]))

    for index, date in enumerate(target_idx):
        try:
            X_returned[index, :] = get_last_k_lags(date, k=k, dataset=X)
        except ValueError:  # Si les dimensions ne sont pas bonnes, on tej
            pass

    return y, X


def em_imputing(X, max_iter=3000, eps=1e-08):
    """
    https://joon3216.github.io/research_materials/2019/em_imputation_python.html#imputing-np.nans

    (np.array, int, number) -> {str: np.array or int}

    Precondition: max_iter >= 1 and eps > 0

    Return the dictionary with five keys where:
    - Key 'mu' stores the mean estimate of the imputed data.
    - Key 'Sigma' stores the variance estimate of the imputed data.
    - Key 'X_imputed' stores the imputed data that is mutated from X using
      the EM algorithm.
    - Key 'C' stores the np.array that specifies the original missing entries
      of X.
    - Key 'iteration' stores the number of iteration used to compute
      'X_imputed' based on max_iter and eps specified.
    """
    if max_iter <= 1 or eps <= 0:
        raise ValueError("Les paramètres ne sont pas valides")
    nr, nc = X.shape
    C = np.isnan(X) == False

    # Collect M_i and O_i's
    one_to_nc = np.arange(1, nc + 1, step=1)
    M = one_to_nc * (C == False) - 1
    O = one_to_nc * C - 1

    # Generate Mu_0 and Sigma_0
    Mu = np.nanmean(X, axis=0)
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    S = np.cov(X[observed_rows,].T)
    if np.isnan(S).any():
        S = np.diag(np.nanvar(X, axis=0))

    # Start updating
    Mu_tilde, S_tilde = {}, {}
    X_tilde = X.copy()
    no_conv = True
    iteration = 0
    while no_conv and iteration < max_iter:
        for i in range(nr):
            S_tilde[i] = np.zeros(nc**2).reshape(nc, nc)
            if set(O[i,]) != set(one_to_nc - 1):  # missing component exists
                M_i, O_i = M[i,][M[i,] != -1], O[i,][O[i,] != -1]
                S_MM = S[np.ix_(M_i, M_i)]
                S_MO = S[np.ix_(M_i, O_i)]
                S_OM = S_MO.T
                S_OO = S[np.ix_(O_i, O_i)]
                Mu_tilde[i] = Mu[np.ix_(M_i)] + S_MO @ np.linalg.inv(S_OO) @ (
                    X_tilde[i, O_i] - Mu[np.ix_(O_i)]
                )
                X_tilde[i, M_i] = Mu_tilde[i]
                S_MM_O = S_MM - S_MO @ np.linalg.inv(S_OO) @ S_OM
                S_tilde[i][np.ix_(M_i, M_i)] = S_MM_O
        Mu_new = np.mean(X_tilde, axis=0)
        S_new = np.cov(X_tilde.T, bias=1) + reduce(np.add, S_tilde.values()) / nr
        no_conv = (
            np.linalg.norm(Mu - Mu_new) >= eps
            or np.linalg.norm(S - S_new, ord=2) >= eps
        )
        Mu = Mu_new
        S = S_new
        iteration += 1

    result = {
        "mu": Mu,
        "Sigma": S,
        "X_imputed": X_tilde,
        "C": C,
        "iteration": iteration,
    }

    return result


def impute_em(X: pd.DataFrame, k_factors=1):
    index = X.index
    result = em_imputing(X.values)
    pca = PCA(n_components=k_factors)
    transformed = pca.fit_transform(
        pd.DataFrame(result["X_imputed"], index=X.index, columns=X.columns)
    )
    return pd.DataFrame(
        transformed,
        columns=[f"Facteur {i}" for i in np.arange(1, k_factors + 1)],
        index=index,
    )


########################################### DATA LOADING #################################################################

if __name__ == "__main__":
    # Prends les mêmes dates que l'article
    article_dataframe = pd.DataFrame(
        index=pd.date_range(start="1991-01-01", end="2006-11-01", freq="MS")
    )
    extended_dataframe = pd.DataFrame(
        index=pd.date_range(start="1991-01-01", end="2022-10-01", freq="MS")
    )

    for file_name in listdir("data/"):
        ############### Loading data for the article period ###############
        if "2005" not in file_name:
            temp_df = pd.read_csv(f"data/{file_name}", index_col="DATE")
            temp_df.index = pd.to_datetime(
                temp_df.index
            )  # s'assure d'une compatibilité au join
            temp_df.rename(
                columns={list(temp_df)[0]: file_name[:-4]}, inplace=True
            )  # -4 car ".csv" = 4 caractères
            article_dataframe = article_dataframe.join(temp_df)
        ############### Loading data for the extended period ###############
        temp_df = pd.read_csv(f"data/{file_name}", index_col="DATE")
        temp_df.index = pd.to_datetime(
            temp_df.index
        )  # s'assure d'une compatibilité au join
        temp_df.rename(
            columns={list(temp_df)[0]: file_name[:-4]}, inplace=True
        )  # -4 car ".csv" = 4 caractères
        extended_dataframe = extended_dataframe.join(temp_df)
    print("Data load ok")
    ########################################### DATASETS BUILDING #################################################################

    article_X = article_dataframe.drop(labels="PIB", axis=1)
    article_y = article_dataframe.loc[:, "PIB"]
    article_y = article_y.dropna(axis=0)

    extended_X = extended_dataframe.drop(labels="PIB", axis=1)
    extended_y = extended_dataframe.loc[:, "PIB"]
    extended_y = extended_y.dropna(axis=0)

    article_y = article_y - article_y.shift(4)  # Suppression de l'effet saisonnier
    extended_y = extended_y - extended_y.shift(4)  # Suppression de l'effet saisonnier

    # Dictionnaire pour enregistrer les performances
    rmse_dict = dict()
    rmse_dict_extended = dict()

    # Création des folders de stockage des résultats
    if not isdir("outputs"):
        mkdir(path="outputs")  # Output folder
    if not isdir("outputs/article"):
        mkdir(path="outputs/article/")
    if not isdir("outputs/extended"):
        mkdir(path="outputs/extended/")

    if not isdir("outputs/article/plots"):
        mkdir(path="outputs/article/plots/")
    if not isdir("outputs/article/results"):
        mkdir(path="outputs/article/results/")

    if not isdir("outputs/extended/plots"):
        mkdir(path="outputs/extended/plots/")
    if not isdir("outputs/extended/results"):
        mkdir(path="outputs/extended/results/")
    print("Settings folder ok")
    ########################################### PERIODE DE L'ARTICLE #################################################################
    print("I. Période de l'article")
    print(
        "Note, les parties relatives à l'usage de facteurs déterminées par l'algorithme EM - parties I.4 et II.4 - peuvent être un peu longues (~5 min)."
    )
    plt.figure(figsize=(15, 5))
    plt.title("Evolution du PIB sur les deux périodes", fontsize=13, fontweight="bold")
    plt.scatter(article_y.index, article_y, label="PIB (période article)")
    plt.scatter(
        extended_y["2006-11-01":].index,
        extended_y["2006-11-01":],
        label="PIB (période étendue)",
    )
    plt.legend(loc="upper right")
    plt.savefig("outputs/PIB evolution.png")

    ####### I.1 Benchmark #######
    in_sample_mean = [article_y.iloc[:-9].mean()] * 9

    ar = AutoReg(endog=article_y.iloc[:-9].dropna(axis=0), lags=1)
    fitted_ar = ar.fit()
    fcast_ar = fitted_ar.forecast(9)

    rmse_dict["In sample mean"] = rmse(in_sample_mean, article_y.iloc[-9:])
    rmse_dict["AR"] = rmse(fcast_ar, article_y.iloc[-9:])

    print("I.1 Benchmark ok")

    ####### I.2 Vertical realignment of data and dynamic principal components factors + MIDAS #######

    factor = va_dfm(article_X, k_factors=1)

    plt.figure(figsize=(15, 5))
    plt.title("Evolution PIB / facteur VADPCA", fontsize=13, fontweight="bold")
    plt.scatter(article_y.index, article_y / article_y.mean(), label="PIB normalisé")
    plt.plot(factor.index, factor, label="Facteur", color="orangered")
    plt.legend(loc="upper right")
    plt.savefig("outputs/article/plots/VADPCA factor.png")

    y, yl, x, yf, ylf, xf = mix_freq(
        article_y.dropna(axis=0),
        pd.Series(factor.values.ravel(), index=factor.index),
        0,
        0,
        3,
        start_date=datetime.datetime(1991, 1, 1),
        end_date=datetime.datetime(2004, 10, 1),
    )

    res = estimate(y, yl, x, poly="expalmon")
    res.x

    fc = forecast(xf, ylf, res, poly="expalmon")
    vapdca_midas = fc.join(yf)
    vapdca_midas["error"] = vapdca_midas["yfh"] - vapdca_midas["PIB"]

    rmse_dict["VADPCA_MIDAS_NO_LAGS"] = rmse(vapdca_midas["yfh"], vapdca_midas["PIB"])

    y, yl, x, yf, ylf, xf = mix_freq(
        article_y.dropna(axis=0),
        pd.Series(factor.values.ravel(), index=factor.index),
        3,
        1,
        3,
        start_date=datetime.datetime(1991, 1, 1),
        end_date=datetime.datetime(2004, 10, 1),
    )

    res = estimate(y, yl, x, poly="expalmon")
    res.x

    fc = forecast(xf, ylf, res, poly="expalmon")
    vapdca_midas_lags = fc.join(yf)
    vapdca_midas_lags["error"] = vapdca_midas_lags["yfh"] - vapdca_midas_lags["PIB"]

    rmse_dict["VADPCA_MIDAS_LAGS"] = rmse(
        vapdca_midas_lags["yfh"], vapdca_midas_lags["PIB"]
    )

    plt.figure(figsize=(14, 5))
    plt.title("Nowcasting / forecasting VAPDCA-MIDAS", fontsize=13, fontweight="bold")
    plt.scatter(article_y[-9:].index, article_y[-9:], label="PIB", s=100)

    plt.axvspan(
        xmin=article_y.index[-9],
        xmax=article_y.index[-8],
        color="dodgerblue",
        alpha=0.1,
        label="Nowcasting",
    )
    plt.axvspan(
        xmin=article_y.index[-8],
        xmax=article_y.index[-1],
        color="orangered",
        alpha=0.1,
        label="Forecasting",
    )

    plt.scatter(
        vapdca_midas.index,
        vapdca_midas["yfh"],
        label="Forecast without including lags",
        s=100,
    )
    plt.scatter(
        vapdca_midas_lags.index,
        vapdca_midas_lags["yfh"],
        label="Forecast including lags",
        s=100,
    )
    plt.legend(loc="upper right")
    plt.savefig("outputs/article/results/VADPCA-MIDAS.png")

    print(
        "I.2 Vertical realignment of data and dynamic principal components factors + MIDAS ok"
    )

    ####### I.3 Vertical realignment of data and dynamic principal components factors + Unrestricted MIDAS #######

    y.dropna(axis=0, inplace=True)
    y, X = lagged_dataset_build(article_y, X=factor)

    y_train = y[y.index <= "2004-10-01"]
    X_train = X[X.index <= "2004-10-01"]
    y_train.dropna(axis=0, inplace=True)
    X_train = X_train[X_train.index.isin(y_train.index)]  # # aligne les jeux de train

    y_test = y[y.index > "2004-10-01"]
    X_test = X[X.index > "2004-10-01"]
    y_test.dropna(axis=0, inplace=True)
    X_test = X_test[X_test.index.isin(y_test.index)]  # aligne les jeux de test

    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train)
    results = model.fit()

    y_pred = results.predict(add_constant(X_test))

    plt.figure(figsize=(14, 5))
    plt.title(
        "Nowcasting / forecasting VAPDCA-Unrestricted MIDAS",
        fontsize=13,
        fontweight="bold",
    )
    plt.scatter(article_y[-9:].index, article_y[-9:], label="PIB", s=100)

    plt.axvspan(
        xmin=article_y.index[-9],
        xmax=article_y.index[-8],
        color="dodgerblue",
        alpha=0.1,
        label="Nowcasting",
    )
    plt.axvspan(
        xmin=article_y.index[-8],
        xmax=article_y.index[-1],
        color="orangered",
        alpha=0.1,
        label="Forecasting",
    )

    plt.scatter(y_test.index, y_test, label="PIB", s=100)
    plt.scatter(y_pred.index, y_pred, label="Forecast", s=100)
    plt.legend(loc="upper right")
    plt.savefig("outputs/article/results/VAPDCA-Unrestricted MIDAS.png")

    print(
        "I.3 Vertical realignment of data and dynamic principal components factors + Unrestricted MIDAS ok"
    )

    ####### I.4 Principal component factors and the EM algorithm + MIDAS #######

    # La variable inflation rend la matrice "singular" (i.e det = 0)
    em_factor = impute_em(X=article_X.drop("inflation", axis=1))

    plt.figure(figsize=(15, 5))
    plt.title("Evolution PIB / facteur EMPCA", fontsize=13, fontweight="bold")
    plt.scatter(article_y.index, article_y / article_y.mean(), label="PIB normalisé")
    plt.plot(em_factor.index, em_factor, label="Facteur", color="orangered")
    plt.legend(loc="upper right")
    plt.savefig("outputs/article/plots/EMPCA factor.png")

    y, yl, x, yf, ylf, xf = mix_freq(
        article_y.dropna(axis=0),
        pd.Series(em_factor.values.ravel(), index=em_factor.index),
        0,
        0,
        0,
        start_date=datetime.datetime(1991, 1, 1),
        end_date=datetime.datetime(2004, 10, 1),
    )

    res = estimate(y, yl, x, poly="expalmon")
    res.x

    fc = forecast(xf, ylf, res, poly="expalmon")
    em_pca_midas = fc.join(yf)
    em_pca_midas["error"] = em_pca_midas["yfh"] - em_pca_midas["PIB"]

    rmse_dict["EM_PCA_MIDAS_NO_LAGS"] = rmse(em_pca_midas["yfh"], em_pca_midas["PIB"])

    y, yl, x, yf, ylf, xf = mix_freq(
        article_y.dropna(axis=0),
        pd.Series(em_factor.values.ravel(), index=em_factor.index),
        "3m",
        1,
        3,
        start_date=datetime.datetime(1991, 1, 1),
        end_date=datetime.datetime(2004, 10, 1),
    )

    res = estimate(y, yl, x, poly="expalmon")
    res.x

    fc = forecast(xf, ylf, res, poly="expalmon")
    em_pca_midas_lags = fc.join(yf)
    em_pca_midas_lags["error"] = em_pca_midas_lags["yfh"] - em_pca_midas_lags["PIB"]

    rmse_dict["EM_PCA_MIDAS_LAGS"] = rmse(
        em_pca_midas_lags["yfh"], em_pca_midas_lags["PIB"]
    )

    plt.figure(figsize=(14, 5))
    plt.title("Nowcasting / forecasting EMPCA-MIDAS", fontsize=13, fontweight="bold")

    plt.axvspan(
        xmin=article_y.index[-9],
        xmax=article_y.index[-8],
        color="dodgerblue",
        alpha=0.1,
        label="Nowcasting",
    )
    plt.axvspan(
        xmin=article_y.index[-8],
        xmax=article_y.index[-1],
        color="orangered",
        alpha=0.1,
        label="Forecasting",
    )

    plt.scatter(article_y[-9:].index, article_y[-9:], label="PIB", s=100)
    plt.scatter(
        em_pca_midas.index,
        em_pca_midas["yfh"],
        label="Forecast without lags included",
        s=100,
    )
    plt.scatter(
        em_pca_midas_lags.index,
        em_pca_midas_lags["yfh"],
        label="Forecast with lags included",
        s=100,
    )
    plt.legend(loc="upper right")
    plt.savefig("outputs/article/results/EMPCA MIDAS.png")

    print("I.4 Principal component factors and the EM algorithm + MIDAS ok")

    ####### I.5 Principal component factors and the EM algorithm + Unrestricted MIDAS #######

    y, X = lagged_dataset_build(article_y.dropna(axis=0), X=em_factor)

    y_train = y[y.index <= "2004-10-01"]
    X_train = X[X.index <= "2004-10-01"]

    X_train = X_train[X_train.index.isin(y_train.index)]

    y_test = y[y.index > "2004-10-01"]
    X_test = X[X.index > "2004-10-01"]
    X_test = X_test[X_test.index.isin(y_test.index)]

    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train)
    results = model.fit()

    y_pred = results.predict(add_constant(X_test))

    plt.figure(figsize=(14, 5))
    plt.title(
        "Nowcasting / forecasting EMPCA-Unrestricted MIDAS",
        fontsize=13,
        fontweight="bold",
    )

    plt.axvspan(
        xmin=article_y.index[-9],
        xmax=article_y.index[-8],
        color="dodgerblue",
        alpha=0.1,
        label="Nowcasting",
    )
    plt.axvspan(
        xmin=article_y.index[-8],
        xmax=article_y.index[-1],
        color="orangered",
        alpha=0.1,
        label="Forecasting",
    )

    plt.scatter(y_test.index, y_test, label="PIB", s=100)
    plt.scatter(y_pred.index[0], y_pred.iloc[0], label="Nowcast", s=100)
    plt.scatter(y_pred.index[1:], y_pred.iloc[1:], label="Forecast", s=100)
    plt.legend(loc="upper right")
    plt.savefig("outputs/article/results/EMPCA Unrestricted MIDAS.png")

    print(
        "I.5 Principal component factors and the EM algorithm + Unrestricted MIDAS ok"
    )

    ####### I.6 Analyse des résultats #######

    performances_df = pd.DataFrame.from_dict(
        rmse_dict, orient="index", columns=["RMSE"]
    )
    performances_df["RMSE normalisée"] = performances_df["RMSE"] / article_y.mean()
    performances_df = performances_df.drop("RMSE", axis=1)

    performances_df.style.bar(color="lightgreen")
    performances_df.to_excel("outputs/article/results/results.xlsx")

    print("I.6 Analyse des résultats ok")

    ########################################### PERIODE ETENDUE #################################################################
    print("II. Période étendue")

    ##### II.1 Benchmark #####

    in_sample_mean = [extended_y.iloc[:-9].mean()] * 9

    ar = AutoReg(endog=extended_y.iloc[:-9].dropna(axis=0), lags=1)
    fitted_ar = ar.fit()
    fcast_ar = fitted_ar.forecast(9)

    rmse_dict_extended["In sample mean"] = rmse(in_sample_mean, extended_y.iloc[-9:])
    rmse_dict_extended["AR"] = rmse(fcast_ar, extended_y.iloc[-9:])

    print("II.1 Benchmark ok")

    ##### II.2 Vertical realignment of data and dynamic principal components factors + MIDAS #####

    factor = va_dfm(extended_X, k_factors=1)

    plt.figure(figsize=(15, 5))
    plt.title("Evolution PIB / facteur VADPCA", fontsize=13, fontweight="bold")
    plt.scatter(extended_y.index, extended_y / extended_y.mean(), label="PIB normalisé")
    plt.plot(factor.index, factor, label="Facteur", color="orangered")
    plt.legend(loc="upper right")
    plt.savefig("outputs/article/plots/VADPCA factor.png")

    y, yl, x, yf, ylf, xf = mix_freq(
        extended_y.dropna(axis=0),
        pd.Series(factor.values.ravel(), index=factor.index),
        0,
        0,
        3,
        start_date=datetime.datetime(1991, 1, 1),
        end_date=datetime.datetime(2020, 10, 1),
    )

    res = estimate(y, yl, x, poly="expalmon")
    res.x

    fc = forecast(xf, ylf, res, poly="expalmon")
    vapdca_midas = fc.join(yf)
    vapdca_midas["error"] = vapdca_midas["yfh"] - vapdca_midas["PIB"]

    rmse_dict_extended["VADPCA_MIDAS_NO_LAGS"] = rmse(
        vapdca_midas["yfh"], vapdca_midas["PIB"]
    )

    y, yl, x, yf, ylf, xf = mix_freq(
        extended_y.dropna(axis=0),
        pd.Series(factor.values.ravel(), index=factor.index),
        3,
        1,
        3,
        start_date=datetime.datetime(1991, 1, 1),
        end_date=datetime.datetime(2020, 10, 1),
    )

    res = estimate(y, yl, x, poly="expalmon")
    res.x

    fc = forecast(xf, ylf, res, poly="expalmon")
    vapdca_midas_lags = fc.join(yf)
    vapdca_midas_lags["error"] = vapdca_midas_lags["yfh"] - vapdca_midas_lags["PIB"]

    rmse_dict_extended["VADPCA_MIDAS_LAGS"] = rmse(
        vapdca_midas_lags["yfh"], vapdca_midas_lags["PIB"]
    )

    plt.figure(figsize=(14, 5))
    plt.title("Nowcasting / forecasting VAPDCA-MIDAS", fontsize=13, fontweight="bold")
    plt.scatter(article_y[-9:].index, article_y[-9:], label="PIB", s=100)

    plt.axvspan(
        xmin=article_y.index[-9],
        xmax=article_y.index[-8],
        color="dodgerblue",
        alpha=0.1,
        label="Nowcasting",
    )
    plt.axvspan(
        xmin=article_y.index[-8],
        xmax=article_y.index[-1],
        color="orangered",
        alpha=0.1,
        label="Forecasting",
    )

    plt.scatter(
        vapdca_midas.index,
        vapdca_midas["yfh"],
        label="Forecast without including lags",
        s=100,
    )
    plt.scatter(
        vapdca_midas_lags.index,
        vapdca_midas_lags["yfh"],
        label="Forecast including lags",
        s=100,
    )
    plt.legend(loc="upper right")
    plt.savefig("outputs/extended/results/VADPCA-MIDAS.png")

    print(
        "II.2 Vertical realignment of data and dynamic principal components factors + MIDAS ok"
    )

    ##### II.3 Vertical realignment of data and dynamic principal components factors + Unrestricted MIDAS #####

    plt.figure(figsize=(15, 5))
    plt.title("Evolution PIB / facteur VADPCA", fontsize=13, fontweight="bold")
    plt.scatter(extended_y.index, extended_y / extended_y.mean(), label="PIB normalisé")
    plt.plot(factor.index, factor, label="Facteur", color="orangered")
    plt.legend(loc="upper right")
    plt.savefig("outputs/extended/plots/VADPCA factor.png")

    y.dropna(axis=0, inplace=True)
    y, X = lagged_dataset_build(extended_y, X=factor)

    y_train = y[y.index <= "2020-07-01"]
    X_train = X[X.index <= "2020-07-01"]
    y_train.dropna(axis=0, inplace=True)
    X_train = X_train[X_train.index.isin(y_train.index)]  # # aligne les jeux de train

    y_test = y[y.index > "2020-07-01"]
    X_test = X[X.index > "2020-07-01"]
    y_test.dropna(axis=0, inplace=True)
    X_test = X_test[X_test.index.isin(y_test.index)]  # aligne les jeux de test

    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train)
    results = model.fit()

    y_pred = results.predict(add_constant(X_test))

    plt.figure(figsize=(14, 5))
    plt.title(
        "Nowcasting / forecasting VAPDCA-Unrestricted MIDAS",
        fontsize=13,
        fontweight="bold",
    )
    plt.scatter(extended_y[-9:].index, extended_y[-9:], label="PIB", s=100)
    plt.scatter(y_pred.index, y_pred, label="Forecast", s=100)
    plt.legend(loc="upper right")
    plt.savefig("outputs/extended/results/VAPDCA Unrestricted MIDAS.png")

    rmse_dict_extended["VADPCA_UMIDAS"] = rmse(y_pred, y_test)
    print(
        "II.3 Vertical realignment of data and dynamic principal components factors + Unrestricted MIDAS ok"
    )
    ##### II.4 Principal component factors and the EM algorithm + MIDAS #####

    em_factor = impute_em(X=extended_X.drop("inflation", axis=1))

    plt.figure(figsize=(15, 5))
    plt.title("Evolution PIB / facteur EMPCA", fontsize=13, fontweight="bold")
    plt.scatter(extended_y.index, extended_y / article_y.mean(), label="PIB normalisé")
    plt.plot(em_factor.index, em_factor, label="Facteur", color="orangered")
    plt.legend(loc="upper right")
    plt.savefig("outputs/extended/plots/EMPCA factor.png")

    y, yl, x, yf, ylf, xf = mix_freq(
        extended_y.dropna(axis=0),
        pd.Series(em_factor.values.ravel(), index=em_factor.index),
        0,
        0,
        3,
        start_date=datetime.datetime(1991, 1, 1),
        end_date=datetime.datetime(2020, 7, 1),
    )

    res = estimate(y, yl, x, poly="expalmon")
    res.x

    fc = forecast(xf, ylf, res, poly="expalmon")
    em_pca_midas = fc.join(yf)
    em_pca_midas["error"] = em_pca_midas["yfh"] - em_pca_midas["PIB"]

    rmse_dict_extended["EM_PCA_MIDAS_NO_LAGS"] = rmse(
        em_pca_midas["yfh"], em_pca_midas["PIB"]
    )

    y, yl, x, yf, ylf, xf = mix_freq(
        extended_y.dropna(axis=0),
        pd.Series(em_factor.values.ravel(), index=em_factor.index),
        "3m",
        1,
        3,
        start_date=datetime.datetime(1991, 1, 1),
        end_date=datetime.datetime(2020, 7, 1),
    )

    res = estimate(y, yl, x, poly="expalmon")
    res.x

    fc = forecast(xf, ylf, res, poly="expalmon")
    em_pca_midas_lags = fc.join(yf)
    em_pca_midas_lags["error"] = em_pca_midas_lags["yfh"] - em_pca_midas_lags["PIB"]

    rmse_dict_extended["EM_PCA_MIDAS_LAGS"] = rmse(
        em_pca_midas_lags["yfh"], em_pca_midas_lags["PIB"]
    )

    plt.figure(figsize=(14, 5))
    plt.title("Nowcasting / forecasting EMPCA-MIDAS", fontsize=13, fontweight="bold")

    plt.scatter(extended_y[-9:].index, extended_y[-9:], label="PIB", s=100)
    plt.scatter(
        em_pca_midas.index,
        em_pca_midas["yfh"],
        label="Forecast without lags included",
        s=100,
    )
    plt.scatter(
        em_pca_midas_lags.index,
        em_pca_midas_lags["yfh"],
        label="Forecast with lags included",
        s=100,
    )
    plt.legend(loc="upper right")
    plt.savefig("outputs/extended/results/EMPCA MIDAS.png")

    print("II.4 Principal component factors and the EM algorithm + MIDAS ok")

    ##### II.5 Principal component factors and the EM algorithm + Unrestricted MIDAS #####

    y, X = lagged_dataset_build(extended_y.dropna(axis=0), X=em_factor)

    y_train = y[y.index <= "2020-07-01"]
    X_train = X[X.index <= "2020-07-01"]

    X_train = X_train[X_train.index.isin(y_train.index)]

    y_test = y[y.index > "2020-07-01"]
    X_test = X[X.index > "2020-07-01"]
    X_test = X_test[X_test.index.isin(y_test.index)]

    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train)
    results = model.fit()

    y_pred = results.predict(add_constant(X_test))

    plt.figure(figsize=(14, 5))
    plt.title(
        "Nowcasting / forecasting EMPCA-Unrestricted MIDAS",
        fontsize=13,
        fontweight="bold",
    )

    plt.scatter(y_test.index, y_test, label="PIB", s=100)
    plt.scatter(y_pred.index, y_pred, label="Forecast", s=100)
    plt.legend(loc="upper right")
    plt.savefig("outputs/extended/results/EMPCA Unrestricted MIDAS.png")

    rmse_dict_extended["EM_PCA_UMIDAS"] = rmse(y_pred, y_test)

    print(
        "II.5 Principal component factors and the EM algorithm + Unrestricted MIDAS ok"
    )

    ##### II.6 Analyse des résultats #####

    performances_extended = pd.DataFrame.from_dict(
        rmse_dict_extended, orient="index", columns=["RMSE"]
    )
    performances_extended["RMSE normalisée"] = (
        performances_extended["RMSE"] / article_y.mean()
    )
    performances_extended = performances_extended.drop("RMSE", axis=1)

    performances_extended.style.bar(color="lightgreen")
    performances_extended.to_excel("outputs/extended/results/results.xlsx")
    print("II.6 Analyse des résultats ok")

    ##### II.7 Comparaison périodes article / extension #####

    compare_df = pd.DataFrame(index=performances_extended.index)
    compare_df["Article"] = performances_df["RMSE normalisée"]
    compare_df["Extended"] = performances_extended["RMSE normalisée"]

    # # Create two Seaborn barplots
    # ax1 = sns.barplot(x=compare_df.index, y=compare_df["Article"], ax=axs[0])
    # ax2 = sns.barplot(x=compare_df.index, y=compare_df["Extended"], ax=axs[1])

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Plot the Seaborn barplots on the subplots
    sns.barplot(
        x=compare_df.index,
        y=compare_df["Article"],
        ax=axs[0],
        color="dodgerblue",
        alpha=0.6,
    )
    sns.barplot(
        x=compare_df.index,
        y=compare_df["Extended"],
        ax=axs[1],
        color="orangered",
        alpha=0.6,
    )

    # Set titles for the subplots
    axs[0].set_title("Résultat sur les données de l'article")
    axs[1].set_title("Résultat sur les données étendues")

    axs[0].xaxis.set_tick_params(labelsize=7, rotation=45)
    axs[1].xaxis.set_tick_params(labelsize=7, rotation=45)

    axs[0].set_ylabel("RMSE normalisée")
    axs[1].set_ylabel("")

    # Show the plot
    plt.savefig("outputs/comparaison article_etendue.png")

    compare_df.to_excel("outputs/results_comparaison.xlsx")
    print("II.7 Comparaison périodes article / extension ok")
