import pandas as pd
from IPython.core.display_functions import display
from catboost import CatBoostRegressor
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import seaborn as sns


def forward_selection_regression(train, valid, features, target):
    stats = []
    picked_features = []
    best_rmses = []
    # forward selection
    for iteration in tqdm(range(len(features))):
        best_rmse = None
        best_feature = None
        stat = {'iteration': iteration}
        for col in features:
            if col in picked_features:
                continue

            X_train = train[list(picked_features) + [col]]
            X_valid = valid[list(picked_features) + [col]]

            model = CatBoostRegressor(verbose=0, eval_metric='RMSE', early_stopping_rounds=50, random_state=42,
                                      cat_features=list(set(cat_features) & set(X_train.columns)))
            model.fit(X_train, train[target], eval_set=(X_valid, valid[target]))
            valid_preds = model.predict(X_valid)
            valid_rmse = mean_squared_error(valid[target], valid_preds, squared=False)  # Calculate RMSE

            stat[col] = valid_rmse

            if best_rmse is None or best_rmse > valid_rmse:
                best_rmse = valid_rmse
                best_feature = col

        stats.append(stat)

        if best_rmse:
            picked_features.append(best_feature)
            best_rmses.append(best_rmse)

    stat_2 = pd.DataFrame(stats).T
    stat_2['cnt_nans'] = stat_2.isna().sum(axis=1).values
    stat_2 = stat_2.sort_values('cnt_nans', ascending=False).drop(columns=['cnt_nans'], index=['iteration'])

    display((stat_2).round(8))

    stat_1 = pd.DataFrame({'names': picked_features, 'rmses': best_rmses})

    plt.figure(figsize=(16, 9))
    sns.lineplot(data=stat_1, x='names', y='rmses')
    plt.xticks(rotation=90);
    plt.show()

    num_features = stat_1['rmses'].argmin() + 1
    best_rmse = stat_1['rmses'].min()

    best_features = stat_1['names'].iloc[:num_features]

    print('---' * 5, 'info', '---' * 5, sep='')
    print('Best rmse:', best_rmse)
    print('Num features:', len(best_features))
    print('Best features:', ' '.join(best_features))
    print('---' * 12)
    return stat_1, stat_2
