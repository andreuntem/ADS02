"""This file contains a set of functions to implement using PCA.
All of them take at least a dataframe df as argument. To test your functions
locally, we recommend using the wine dataset that you can load from sklearn by
importing sklearn.datasets.load_wine"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_cumulated_variance(df, scale):
    """Apply PCA on a DataFrame and return a new DataFrame containing
    the cumulated explained variance from with only the first component,
    up to using all components together. Values should be expressed as
    a percentage of the total variance explained.

    The DataFrame will have one row and each column should correspond to a
    principal component.

    Example:
             PC1        PC2        PC3        PC4    PC5
    0  36.198848  55.406338  66.529969  73.598999  100.0

    If scale is True, you should standardise the data first
    Tip: use the StandardScaler from sklearn

    :param df: pandas DataFrame
    :param scale: boolean, whether to scale or not
    :return: a new DataFrame with cumulated variance in percent
    """
    if scale is True:
        X = StandardScaler().fit_transform(df)
    else:
        X = df.copy()
    pca = PCA()
    pca.fit(X)
    cumsum = [list(np.cumsum(pca.explained_variance_ratio_)*100)]
    col_names = ['PC'+str(i+1) for i in range(X.shape[1])]
    df_out = pd.DataFrame(data=cumsum, columns=col_names)
    return df_out


def get_coordinates_of_first_two(df, scale):
    if scale is True:
        X = StandardScaler().fit_transform(df)
    else:
        X = df.copy()
    pca = PCA()
    pca.fit(X)
    index_names = ['PC1', 'PC2']
    return pd.DataFrame(pca.components_[0:2], columns=df.columns, index=index_names)


def get_most_important_two(df, scale):
    if scale is True:
        X = StandardScaler().fit_transform(df)
    else:
        X = df.copy()

    pca = PCA()
    pca.fit(X)
    pc1 = abs(pca.components_[0])

    list_features = list(df.columns)
    features = []
    for _ in range(2):
        argmax = pc1.argmax()
        feat = list_features[argmax]
        features.append(feat)
        pc1 = np.delete(pc1, argmax)
        list_features.pop(argmax)

    return tuple(features)


def distance_in_n_dimensions(df, point_a, point_b, n, scale):
    points = np.vstack((point_a, point_b))
    if scale is True:
        X = StandardScaler().fit_transform(df)
        df_std = df.std(axis=0, ddof=0).values
        points = (points)/df_std
    else:
        X = df.copy()

    pca = PCA()
    pca.fit(X)
    A = pca.components_[:n]
    points_new = np.dot(points, A.T)
    dist = np.sum((points_new[0] - points_new[1])**2)**.5

    return dist


def find_outliers_pca(df, n, scale):
    if scale is True:
        X = StandardScaler().fit_transform(df)
    else:
        X = df.copy()

    pca = PCA()
    pca.fit(X)
    subspace = pca.components_[0]

    A = np.dot(X, subspace.T)
    a_std = A.std(ddof=0)
    outliers = df[abs(A/a_std) > n]
    return outliers
