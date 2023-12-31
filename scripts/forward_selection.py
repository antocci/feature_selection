import pandas as pd
from IPython.core.display_functions import display
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import seaborn as sns


def forward_selection(train, valid, features, cat_features, target):
    stats = []

    picked_features = []
    best_ginis = []
    # forward selection
    for iteration in tqdm(range(len(features))):
        # print('Iteration number', iteration)
        best_gini = None
        best_feature = None
        stat = {'iteration': iteration}
        for col in features:
            if col in picked_features:
                continue

            X_train = train[list(picked_features) + [col]]
            X_valid = valid[list(picked_features) + [col]]

            model = CatBoostClassifier(verbose=0, eval_metric='AUC', early_stopping_rounds=60, random_state=42,
                                       cat_features=list(set(cat_features) & set(X_train.columns)))
            model.fit(X_train, train[target], eval_set=(X_valid, valid[target]))
            valid_preds = model.predict_proba(X_valid)[:, 1]
            valid_gini = 2 * roc_auc_score(valid[target], valid_preds) - 1

            stat[col] = valid_gini

            if best_gini is None or best_gini < valid_gini:
                best_gini = valid_gini
                best_feature = col

        stats.append(stat)

        if best_gini:
            picked_features.append(best_feature)
            best_ginis.append(best_gini)
            print('Picked feature:', best_feature)

    stat_2 = pd.DataFrame(stats).T
    stat_2['cnt_nans'] = stat_2.isna().sum(axis=1).values
    stat_2 = stat_2.sort_values('cnt_nans', ascending=False).drop(columns=['cnt_nans'], index=['iteration'])

    display((stat_2 * 100).round(1))

    stat_1 = pd.DataFrame({'names': picked_features, 'ginis': best_ginis})

    plt.figure(figsize=(16, 9))
    sns.lineplot(data=stat_1, x='names', y='ginis')
    plt.xticks(rotation=90)
    plt.show()

    num_features = stat_1['ginis'].argmax() + 1
    best_gini = stat_1['ginis'].max()
    features_to_eliminate = stat_1['names'].iloc[num_features:]
    best_features = sorted(set(features) - set(features_to_eliminate))

    print('---' * 5, 'info', '---' * 5, sep='')
    print('Best ginis:', best_gini)
    print('Num features:', len(best_features))
    print('Best features:', ' '.join(best_features))
    print('---' * 12)

    return stat_1, stat_2
