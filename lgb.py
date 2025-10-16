import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
from lightgbm import LGBMRegressor
import lightgbm as lgb


# =========================
# Configuration
# =========================
class Config:
    LABEL_COLUMN = "ret"
    N_FOLDS = 3
    RANDOM_STATE = 128

    FEATURES = ['open', 'high', 'low', 'close', 'volume', 'num_trades', 'range', 'bid_volume',
                'quote_volume', 'ask_volume', 'buy_volume', 'sell_volume', 'close_1', 'volume_1', 'quote_volume_1',
                'buy_volume_1', 'sell_volume_1', 'bid_volume_1', 'ask_volume_1', 'close_5', 'volume_5',
                'quote_volume_5', 'buy_volume_5', 'sell_volume_5', 'bid_volume_5', 'ask_volume_5', 'close_15',
                'volume_15', 'quote_volume_15', 'buy_volume_15', 'sell_volume_15', 'bid_volume_15', 'ask_volume_15',
                'close_60', 'close_60_std', 'volume_60', 'volume_60_std', 'quote_volume_60', 'quote_volume_60_std',
                'buy_volume_60', 'buy_volume_60_std', 'sell_volume_60', 'sell_volume_60_std', 'bid_volume_60',
                'bid_volume_60_std', 'ask_volume_60', 'ask_volume_60_std', 'close_120', 'close_120_std', 'volume_120',
                'volume_120_std', 'quote_volume_120', 'quote_volume_120_std', 'buy_volume_120', 'buy_volume_120_std',
                'sell_volume_120', 'sell_volume_120_std', 'bid_volume_120', 'bid_volume_120_std', 'ask_volume_120',
                'ask_volume_120_std',
                'VolumeRatio', 'VolRatio', 'ema_12', 'ema_26', 'ema_52', 'rsi_14', 'bb_mid', 'bb_upper', 'bb_lower',
                'macd', 'macd_signal', 'vwap', 'logvolume', 'logquote', 'logbid', 'logask', 'logbuy', 'logsell']


lgb_params = {}
lgb_params['verbosity'] = -1

lgb_params = {

}


LEARNERS = [
    {"name": "lgb", "Estimator": LGBMRegressor, "params": lgb_params}
]


# =========================
# Utility Functions
# =========================
def create_time_decay_weights(n: int, decay: float = 0.01) -> np.ndarray:
    positions = np.arange(n)
    normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()


def load_data(data):
    train_df, test_df = train_test_split(data, test_size=0.15, shuffle=False)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_model_slices(n_samples: int):
    return [
        {"name": "full_data", "cutoff": 0},
        # {"name": "last_95pct", "cutoff": int(0.05 * n_samples)},
        {"name": "last_90pct", "cutoff": int(0.10 * n_samples)},
        # {"name": "last_85pct", "cutoff": int(0.15 * n_samples)},
        {"name": "last_80pct", "cutoff": int(0.20 * n_samples)},
        # {"name": "last_75pct", "cutoff": int(0.25 * n_samples)},
        # {"name": "last_70pct", "cutoff": int(0.30 * n_samples)},
        # {"name": "last_65pct", "cutoff": int(0.35 * n_samples)},
        # {"name": "last_60pct", "cutoff": int(0.40 * n_samples)},
        # {"name": "last_55pct", "cutoff": int(0.45 * n_samples)},
        # {"name": "last_50pct", "cutoff": int(0.50 * n_samples)},
    ]


# =========================
# Training and Evaluation
# =========================
def train_and_evaluate(train_df, test_df):
    FEATURES = Config.FEATURES

    n_samples = len(train_df)
    model_slices = get_model_slices(n_samples)

    oof_preds = {
        learner["name"]: {s["name"]: np.zeros(n_samples) for s in model_slices}
        for learner in LEARNERS
    }
    test_preds = {
        learner["name"]: {s["name"]: np.zeros(len(test_df)) for s in model_slices}
        for learner in LEARNERS
    }

    full_weights = create_time_decay_weights(n_samples)
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
        # print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
        X_valid = train_df.iloc[valid_idx][Config.FEATURES]
        y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN]

        for s in model_slices:
            cutoff = s["cutoff"]
            slice_name = s["name"]
            subset = train_df.iloc[cutoff:].reset_index(drop=True)
            rel_idx = train_idx[train_idx >= cutoff] - cutoff

            X_train = subset.iloc[rel_idx][Config.FEATURES]
            y_train = subset.iloc[rel_idx][Config.LABEL_COLUMN]
            sw = create_time_decay_weights(len(subset))[rel_idx] if cutoff > 0 else full_weights[train_idx]

            print(f"  Training slice: {slice_name}, samples: {len(X_train)}")

            for learner in LEARNERS:
                # model = learner["Estimator"](**learner["params"])
                model = LGBMRegressor(
                    n_estimators=200,  # allow early stopping to find the best round
                    # min_child_samples=20,  # default
                    # subsample=0.9,  # row sampling (bagging)
                    # subsample_freq=1,  # activate bagging
                    # colsample_bytree=0.9,  # feature sampling
                    random_state=Config.RANDOM_STATE,
                    verbosity = -1
                )
                model.fit(X_train, y_train, sample_weight=sw)

                mask = valid_idx >= cutoff
                if mask.any():
                    idxs = valid_idx[mask]
                    oof_preds[learner["name"]][slice_name][idxs] = model.predict(train_df.iloc[idxs][Config.FEATURES])
                if cutoff > 0 and (~mask).any():
                    oof_preds[learner["name"]][slice_name][valid_idx[~mask]] = oof_preds[learner["name"]]["full_data"][valid_idx[~mask]]

                test_preds[learner["name"]][slice_name] += model.predict(test_df[Config.FEATURES])

    # Normalize test predictions
    for learner_name in test_preds:
        for slice_name in test_preds[learner_name]:
            test_preds[learner_name][slice_name] /= Config.N_FOLDS

    return oof_preds, test_preds, model_slices


def ensemble_simple(train_df, test_df, oof_preds, test_preds):
    results_oof = pd.DataFrame()

    results_oof['ret'] = train_df.dropna()['ret']
    results_oof['ols'] = np.mean(list(oof_preds['ols'].values()), axis=0)
    corr_df = pd.DataFrame(np.corrcoef(results_oof[['ret', 'ols']].T.values))
    corr_df.columns = ['ret', 'ols']
    corr_df.index = ['ret', 'ols']
    # best = corr_df.sort_values(by=['ret'], ascending=False).index.tolist()[1]

    corr_recent = pd.DataFrame(np.corrcoef(results_oof[['ret', 'ols']].tail(60).T.values))
    corr_recent.columns = ['ret', 'ols']
    corr_recent.index = ['ret', 'ols']
    # print ("#"*182)
    # print ("#"*182)
    # print ("#"*182)
    # print (datetime.now())
    print(corr_df.iloc[[0]])
    print(corr_recent.iloc[[0]])
    # # print (f'{best} is the best')
    #
    # test_preds_ols = np.mean(list(test_preds['ols'].values()), axis=0)
    #
    # pred_best = test_preds_ols
    #
    # history_std = train_df['ret'].std()
    # close = test_df['close'].tolist()[0]
    # target_price = np.round((100+pred_best)/100*close,4)
    #
    # print ("#"*182)
    # print (f'The prediction is: {pred_best} to {target_price} with t-stat {pred_best/history_std}' )
    # print ("Some tech indicator")
    # cols = ['open', 'high', 'low', 'close', 'volume', 'num_trades', 'range', 'bid_volume', 'quote_volume',
    #             'ask_volume', 'buy_volume', 'sell_volume', 'ret_current', 'return_1m', 'return_5m', 'return_15m',
    #             'volume_15m', 'volume_60m', 'VolumeRatio', 'realvol_15m', 'realvol_60m', 'VolRatio', 'sma_5', 'sma_20',
    #             'ema_12', 'ema_26', 'rsi_14', 'bb_mid', 'bb_upper', 'bb_lower', 'macd', 'macd_signal', 'vwap',
    #             'logvolume', 'logquote', 'logbid', 'logask', 'logbuy', 'logsell']
    # print (test_df[cols])
    #
    # return test_preds


def agg(dataset):
    train_df, test_df = dataset.iloc[0:-1], dataset.iloc[[-1]]
    oof_preds, test_preds, model_slices = train_and_evaluate(train_df.dropna(), test_df)
    # pred = ensemble_simple(train_df, test_df, oof_preds, test_preds)

    results_oof = pd.DataFrame()

    results_oof['ret'] = train_df.dropna()['ret']
    results_oof['lgb'] = np.mean(list(oof_preds['lgb'].values()), axis=0)
    corr_oof1 = np.corrcoef(results_oof[['ret', 'lgb']].tail(120).T.values)[0][1]
    corr_oof2 = np.corrcoef(results_oof[['ret', 'lgb']].tail(180).T.values)[0][1]
    corr_oof3 = np.corrcoef(results_oof[['ret', 'lgb']].tail(60).T.values)[0][1]

    close = test_df['close'].tolist()[0]
    vol_ratio = test_df['VolRatio'].tolist()[0]
    test_preds_lgb = np.mean(list(test_preds['lgb'].values()), axis=0)
    target_price = np.round((100 + test_preds_lgb) / 100 * close, 4)
    # print (corr_oof1, corr_oof2, corr_oof3)
    return target_price[0], vol_ratio, corr_oof3

agg(dataset)
# data = df.copy()
# train_df, test_df = load_data(data.dropna().reset_index())
# oof_preds, test_preds, model_slices = train_and_evaluate(train_df, test_df)
# ensemble_simple(oof_preds, test_preds)

