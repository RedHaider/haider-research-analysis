import numpy as np
from sklearn.preprocessing import QuantileTransformer


def hidden_representation(model, standardized_features):
    value = standardized_features
    for weight, bias in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        value = np.maximum(value @ weight + bias, 0)
    return value


def moment_match(train, test, train_group, test_group):
    global_mean = train.mean(0)
    global_std = train.std(0) + 1e-6
    transformed_train = train.copy()
    transformed_test = test.copy()
    for group in (0, 1):
        keep_train = train_group == group
        keep_test = test_group == group
        mean = train[keep_train].mean(0)
        std = train[keep_train].std(0) + 1e-6
        transformed_train[keep_train] = (train[keep_train] - mean) / std * global_std + global_mean
        transformed_test[keep_test] = (test[keep_test] - mean) / std * global_std + global_mean
    return transformed_train, transformed_test


def quantile_match(train, test, train_group, test_group):
    transformed_train = np.empty_like(train, dtype=float)
    transformed_test = np.empty_like(test, dtype=float)
    for group in (0, 1):
        keep_train = train_group == group
        keep_test = test_group == group
        transformer = QuantileTransformer(n_quantiles=min(100, keep_train.sum()), output_distribution="normal",
                                          subsample=None, random_state=27)
        transformed_train[keep_train] = transformer.fit_transform(train[keep_train])
        transformed_test[keep_test] = transformer.transform(test[keep_test])
    return transformed_train, transformed_test

