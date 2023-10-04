import pandas as pd
from IPython.core.display_functions import display
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from sklearn import clone
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import seaborn as sns


def backward_elimination(train, valid, features, cat_features, target):
    stats = []

    eliminated_features = []
    best_ginis = []
    # backward elimination
    for iteration in tqdm(range(len(features) - 1)):
        best_gini = None
        worst_feature = None
        stat = {'iteration': iteration}
        for col in features:
            if col in eliminated_features:
                continue

            X_train = train[sorted(set(features) - set(eliminated_features + [col]))]
            X_valid = valid[sorted(set(features) - set(eliminated_features + [col]))]

            model = CatBoostClassifier(verbose=100, eval_metric='AUC', early_stopping_rounds=60, random_state=42,
                                       cat_features=list(set(cat_features) & set(X_train.columns)))
            model.fit(X_train, train[target])
            valid_preds = model.predict_proba(X_valid)[:, 1]
            valid_gini = 2 * roc_auc_score(valid[target], valid_preds) - 1

            stat[col] = valid_gini

            if best_gini is None or best_gini < valid_gini:
                best_gini = valid_gini
                worst_feature = col

        stats.append(stat)

        if best_gini:
            eliminated_features.append(worst_feature)
            best_ginis.append(best_gini)

    stat_2 = pd.DataFrame(stats).T
    stat_2['cnt_nans'] = stat_2.isna().sum(axis=1).values

    stat_2 = stat_2.sort_values('cnt_nans', ascending=False).drop(columns=['cnt_nans'], index=['iteration'])
    display((stat_2 * 100).round(1))

    stat_1 = pd.DataFrame({'names': eliminated_features[::-1], 'ginis': best_ginis[::-1]})

    plt.figure(figsize=(16, 9))
    sns.lineplot(data=stat_1, x='names', y='ginis')
    plt.xticks(rotation=90)
    plt.show()

    num_features = stat_1['ginis'].argmax()
    best_gini = stat_1['ginis'].max()

    features_to_eliminate = stat_1['names'].iloc[num_features:]

    best_features = sorted(set(features) - set(features_to_eliminate))

    print('---' * 5, 'info', '---' * 5, sep='')
    print('Best ginis:', best_gini)
    print('Num features:', len(best_features))
    print('Best features:', ' '.join(best_features))
    print('---' * 12)

    best_model = clone(model)
    best_model.fit(train[best_features], train[target], cat_features=list(set(cat_features) & set(best_features)))

    return stat_1, stat_2, best_model