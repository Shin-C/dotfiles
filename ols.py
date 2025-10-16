import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime

# =========================
# Configuration
# =========================
class Config:
    LABEL_COLUMN = "ret"
    N_FOLDS = 3
    RANDOM_STATE = 42

    FEATURES = ['open', 'high', 'low', 'close', 'volume', 'num_trades', 'range', 'bid_volume', 'quote_volume',
                'ask_volume', 'buy_volume', 'sell_volume', 'ret_current', 'return_1m', 'return_5m', 'return_15m',
                'volume_15m', 'volume_60m', 'VolumeRatio', 'realvol_15m', 'realvol_60m', 'VolRatio', 'sma_5', 'sma_20',
                'ema_12', 'ema_26', 'rsi_14', 'bb_mid', 'bb_upper', 'bb_lower', 'macd', 'macd_signal', 'vwap',
                'logvolume', 'logquote', 'logbid', 'logask', 'logbuy', 'logsell', 'close_mean', 'volume_mean',
                'logvolume_mean', 'logquote_mean', 'logbid_mean', 'logask_mean', 'logbuy_mean', 'logsell_mean',
                'close_std', 'volume_std', 'logvolume_std', 'logquote_std', 'logbid_std', 'logask_std', 'logbuy_std',
                'logsell_std']


lgb_params = {}
lgb_params['verbosity'] = -1

LEARNERS = [
    {"name": "ols", "Estimator": LinearRegression},
]


# =========================
# Utility Functions
# =========================
def create_time_decay_weights(n: int, decay: float = 0.8) -> np.ndarray:
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
        {"name": "last_95pct", "cutoff": int(0.05 * n_samples)},
        {"name": "last_90pct", "cutoff": int(0.10 * n_samples)},
        {"name": "last_85pct", "cutoff": int(0.15 * n_samples)},
        {"name": "last_80pct", "cutoff": int(0.20 * n_samples)},
        {"name": "last_75pct", "cutoff": int(0.25 * n_samples)},
        {"name": "last_70pct", "cutoff": int(0.30 * n_samples)},
        {"name": "last_65pct", "cutoff": int(0.35 * n_samples)},
        {"name": "last_60pct", "cutoff": int(0.40 * n_samples)},
        {"name": "last_55pct", "cutoff": int(0.45 * n_samples)},
        {"name": "last_50pct", "cutoff": int(0.50 * n_samples)},
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
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
        # print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")

        for s in model_slices:
            cutoff = s["cutoff"]
            slice_name = s["name"]
            subset = train_df.iloc[cutoff:].reset_index(drop=True)
            rel_idx = train_idx[train_idx >= cutoff] - cutoff

            X_train = subset.iloc[rel_idx][FEATURES]
            y_train = subset.iloc[rel_idx][Config.LABEL_COLUMN]
            sw = create_time_decay_weights(len(subset))[rel_idx] if cutoff > 0 else full_weights[train_idx]

            for learner in LEARNERS:
                if learner["name"] == "lgb":
                    model = learner["Estimator"](**lgb_params)
                    model.fit(X_train, y_train, sample_weight=sw)
                elif learner['name'] == 'xgb':
                    model = learner["Estimator"]()
                    model.fit(X_train, y_train, sample_weight=sw, verbose=False)
                else:
                    X_train = subset.dropna().iloc[rel_idx][FEATURES]
                    y_train = subset.dropna().iloc[rel_idx][Config.LABEL_COLUMN]
                    sw = create_time_decay_weights(len(subset))[rel_idx] if cutoff > 0 else full_weights[train_idx]
                    model = learner["Estimator"]()
                    model.fit(X_train, y_train, sample_weight=sw)

                mask = valid_idx >= cutoff
                if mask.any():
                    idxs = valid_idx[mask]
                    oof_preds[learner["name"]][slice_name][idxs] = model.predict(train_df.iloc[idxs][FEATURES])
                if cutoff > 0 and (~mask).any():
                    oof_preds[learner["name"]][slice_name][valid_idx[~mask]] = oof_preds[learner["name"]]["full_data"][
                        valid_idx[~mask]]

                test_preds[learner["name"]][slice_name] += model.predict(test_df[FEATURES].values)

    # Normalize test predictions
    for learner_name in test_preds:
        for slice_name in test_preds[learner_name]:
            test_preds[learner_name][slice_name] /= Config.N_FOLDS

    return oof_preds, test_preds, model_slices


def ensemble_simple(train_df, test_df, oof_preds, test_preds):
    results_oof = pd.DataFrame()

    results_oof['ret'] = train_df.dropna()['ret']
    results_oof['ols'] = np.mean(list(oof_preds['ols'].values()), axis=0)
    corr_df = pd.DataFrame(np.corrcoef(results_oof[['ret','ols']].T.values))
    corr_df.columns = ['ret','ols']
    corr_df.index = ['ret', 'ols']
    # best = corr_df.sort_values(by=['ret'], ascending=False).index.tolist()[1]


    print ("#"*182)
    print ("#"*182)
    print ("#"*182)
    print (datetime.now())
    print(corr_df.iloc[[0]])
    # print (f'{best} is the best')

    test_preds_ols = np.mean(list(test_preds['ols'].values()), axis=0)

    pred_best = test_preds_ols

    history_std = train_df['ret'].std()
    close = test_df['close'].tolist()[0]
    target_price = np.round((100+pred_best)/100*close,4)

    print ("#"*182)
    print (f'The prediction is: {pred_best} to {target_price} with t-stat {pred_best/history_std}' )
    print ("Some tech indicator")
    cols = ['open', 'high', 'low', 'close', 'volume', 'num_trades', 'range', 'bid_volume', 'quote_volume',
                'ask_volume', 'buy_volume', 'sell_volume', 'ret_current', 'return_1m', 'return_5m', 'return_15m',
                'volume_15m', 'volume_60m', 'VolumeRatio', 'realvol_15m', 'realvol_60m', 'VolRatio', 'sma_5', 'sma_20',
                'ema_12', 'ema_26', 'rsi_14', 'bb_mid', 'bb_upper', 'bb_lower', 'macd', 'macd_signal', 'vwap',
                'logvolume', 'logquote', 'logbid', 'logask', 'logbuy', 'logsell', 'close_mean', 'volume_mean',
                'logvolume_mean', 'logquote_mean', 'logbid_mean', 'logask_mean', 'logbuy_mean', 'logsell_mean',
                'close_std', 'volume_std', 'logvolume_std', 'logquote_std', 'logbid_std', 'logask_std', 'logbuy_std',
                'logsell_std']
    print (test_df[cols])


    return test_preds


def agg(data):
    train_df,test_df = data.iloc[0:-1], data.iloc[[-1]]
    oof_preds, test_preds, model_slices = train_and_evaluate(train_df.dropna(), test_df)
    ensemble_simple(train_df, test_df, oof_preds, test_preds)

# data = df.copy()
# train_df, test_df = load_data(data.dropna().reset_index())
# oof_preds, test_preds, model_slices = train_and_evaluate(train_df, test_df)
# ensemble_simple(oof_preds, test_preds)
