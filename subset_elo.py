import pandas as pd
import numpy as np
#import plotly.express as px

#import tiktoken
import datetime
import argparse
import os
import math

from glob import glob
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from collections import defaultdict
#from utils import load_questions, load_model_answers
import random


BASELINE_MODEL_NAME = "Retrieved1"


def _logistic(x):
    """Logistic function."""
    return np.exp(-np.logaddexp(0, -x))

def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000):
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["result"] == 0] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["result"] == 2) | (df["result"] == 3)
    tie_idx[len(tie_idx) // 2:] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
    lr.fit(X, Y, sample_weight=1)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    # set anchor as BASELINE = INIT_RATING
    if BASELINE_MODEL_NAME in models.index:
        elo_scores += INIT_RATING - elo_scores[models[BASELINE_MODEL_NAME]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True)))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate ELO ratings for models based on battle outcomes.')
    parser.add_argument('input_file', type=str, help='Input file containing the dataframe (TSV format).')
    #parser.add_argument('output_file', type=str, help='Output file to save the ELO ratings (TXT format).')
    #parser.add_argument('label_column', type=str, help='Column name with the battle outcomes.')

    args = parser.parse_args()

    data = pd.read_csv(args.input_file)
    subsets = list(data['subset'].value_counts().index)

    for subset in subsets:
        df = data[data['subset'] == subset]
        bootstrap_online_elo = compute_mle_elo(df)

        num_rounds = 100
        models_names = bootstrap_online_elo.index

        np.random.seed(42)
        bootstrap_elo_lu = get_bootstrap_result(df, compute_mle_elo, num_rounds)
        bootstrap_elo_lu.to_json("bootstrapping_results.jsonl", lines=True, orient="records")

        stats = pd.DataFrame()
        stats["results"] = None
        stats["results"] = stats['results'].astype('object')
        for i, model in enumerate(bootstrap_online_elo.index):
            assert model in bootstrap_elo_lu.columns

            stats.at[i, "model"] = model
            stats.at[i, "score"] = bootstrap_online_elo[model]
            stats.at[i, "lower"] = np.percentile(bootstrap_elo_lu[model], 2.5)
            stats.at[i, "upper"] = np.percentile(bootstrap_elo_lu[model], 97.5)

            # stats.at[i, "avg_tokens"] = models_answers_lengths.loc[model].mean()
            # stats.at[i, "std_tokens"] = models_answers_lengths.loc[model].std()

            stats.at[i, "results"] = bootstrap_elo_lu[model].tolist()

        decimal = 0
        stats = stats.astype({"score": int, "lower": int, "upper": int})

        print(f'SUBSET {subset}')
        stats.sort_values(by="score", ascending=False, inplace=True)
        for _, row in stats.iterrows():
            interval = str((round(row['lower'] - row['score'], decimal), round(row['upper'] - row['score'], decimal)))
            print(
                f"{row['model'] : <50} | score: {round(row['score'], decimal) : ^5} | 95% CI: {interval : ^12}")

        stats.to_csv(f'{args.input_file}_{subset}_elo.csv')
