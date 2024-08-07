import pandas as pd
import random
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from copy import deepcopy

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10**((rating_b - rating_a) / 400))

def update_elo(rating_a, rating_b, score_a, k=32):
    expected_a = expected_score(rating_a, rating_b)
    new_rating_a = rating_a + k * (score_a - expected_a)
    new_rating_b = rating_b + k * ((1 - score_a) - (1 - expected_a))
    return new_rating_a, new_rating_b

def calculate_elo_ratings(input_file, output_file, label_column, initial_elo=1200):
    df = pd.read_csv(input_file, sep='\t')
    elo_ratings = {model: initial_elo for model in set(df['model_1']).union(df['model_2'])}

    for index, row in df.iterrows():
        model_1 = row['model_1']
        model_2 = row['model_2']
        outcome = row[label_column]

        if outcome == 0:  # model_1 wins
            score_1 = 1
            score_2 = 0
        elif outcome == 1:  # model_2 wins
            score_1 = 0
            score_2 = 1
        elif outcome == 2:  # tie
            score_1 = 0.5
            score_2 = 0.5
        elif outcome == 3:  # both models are bad
            continue  # No change in ratings for this outcome

        rating_1 = elo_ratings[model_1]
        rating_2 = elo_ratings[model_2]

        new_rating_1, new_rating_2 = update_elo(rating_1, rating_2, score_1)

        elo_ratings[model_1] = new_rating_1
        elo_ratings[model_2] = new_rating_2

    with open(output_file, 'w') as f:
        for model, rating in elo_ratings.items():
            f.write(f"{model}: {rating}\n")



DUMMY_PLAYER = 'DUMMY PLAYER'

def extract_game_data(df, label_column):
    df1 = deepcopy(df[df[label_column] < 2])
    df1['Player A'] = df1['model_1']
    df1['Player B'] = df1['model_2']
    df1['Wins A'] = df1[label_column] == 0
    df1['Wins B'] = df1[label_column] == 1

    df1['Wins A'] = df1['Wins A'].astype(int)
    df1['Wins B'] = df1['Wins B'].astype(int)

    df1 = df1[['Player A', 'Player B', 'Wins A', 'Wins B']]

    return df1

def add_dummy_games(game_data, alpha=1):
    players = sorted(list(set(game_data['Player A']) | set(game_data['Player B'])))

    dummy_data = [[p, DUMMY_PLAYER, alpha, alpha] for p in players]
    df = pd.DataFrame(dummy_data, columns=game_data.columns)
    df = pd.concat([game_data, df])

    return df

def compute_rank_scores(game_data, max_iters=1000, error_tol=1e-3):
    winsA = game_data.groupby('Player A').agg(sum)['Wins A'].reset_index()
    winsA = winsA[winsA['Wins A'] > 0]
    winsA.columns = ['Player', 'Wins']
    winsB = game_data.groupby('Player B').agg(sum)['Wins B'].reset_index()
    winsB = winsB[winsB['Wins B'] > 0]
    winsB.columns = ['Player', 'Wins']
    wins = pd.concat([winsA, winsB]).groupby('Player').agg(sum)['Wins']

    num_games = Counter()
    for index, row in game_data.iterrows():
        key = tuple(sorted([row['Player A'], row['Player B']]))
        total = sum([row['Wins A'], row['Wins B']])
        num_games[key] += total

    players = sorted(list(set(game_data['Player A']) | set(game_data['Player B'])))
    ranks = pd.Series(np.ones(len(players)) / len(players), index=players)
    for iters in range(max_iters):
        oldranks = ranks.copy()
        for player in ranks.index:
            denom = np.sum(num_games[tuple(sorted([player, p]))]
                           / (ranks[p] + ranks[player])
                           for p in ranks.index if p != player)
            ranks[player] = 1.0 * wins[player] / denom

        ranks /= sum(ranks)

        if np.sum((ranks - oldranks).abs()) < error_tol:
            break

    if np.sum((ranks - oldranks).abs()) < error_tol:
        print(f" * Converged after {iters} iterations.")
    else:
        print(f" * Max iterations reached ({max_iters} iters).")

    del ranks[DUMMY_PLAYER]

    ranks = ranks.sort_values(ascending=False) \
                 .apply(lambda x: np.log1p(1000 * x) / np.log1p(1000) * 1000) \
                 .astype(int) \
                 .clip(1)

    return ranks


def calculate_bt_model(input_file, output_file, label_column):
    df = pd.read_csv(input_file, sep='\t')

    df1 = extract_game_data(df, label_column)
    games = add_dummy_games(df1)
    ranks = compute_rank_scores(games)

    with open(output_file, 'w') as f:
        for model, rating in ranks.items():
            f.write(f"{model}: {rating}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate ELO ratings for models based on battle outcomes.')
    parser.add_argument('input_file', type=str, help='Input file containing the dataframe (TSV format).')
    parser.add_argument('output_file', type=str, help='Output file to save the ELO ratings (TXT format).')
    parser.add_argument('label_column', type=str, help='Column name with the battle outcomes.')

    args = parser.parse_args()

    calculate_elo_ratings(args.input_file, args.output_file, args.label_column)
    calculate_bt_model((args.input_file, args.output_file, args.label_column))
