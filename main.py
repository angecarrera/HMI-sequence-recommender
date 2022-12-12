import pandas as pd
import numpy as np
from collections import Counter

from Recommender.evaluator.evaluator import *


def read_dataset(name):
    df = pd.read_csv(name)
    df.interactionwu_prep = df.interactionwu_prep.apply(eval)
    return df


def main():
    users_df = pd.read_csv("Recommender/dataset/user_role.csv")
    user_dict = dict(zip(users_df.User, users_df.Role))

    df = read_dataset("Recommender/dataset/sequences_df_prep_simulated_EN.csv")
    cnt = Counter()
    df.interactionwu_prep.map(cnt.update);
    df["user_role"] = df.user.map(user_dict)

    print(df[["user_role", "turn", "interactionwu_prep"]])

    sequence_length = df.interactionwu_prep.map(len).values
    n_sessions_per_user = df.groupby('user').size()
    print('Number of UI elements: {}'.format(len(cnt)))
    print('Number of users: {}'.format(len(df.user.unique())))
    print('Number of interaction sequences: {}'.format(len(df)))

    print('Sequence length:\n\tAverage: {:.2f}\n\tMedian: {}\n\tMin: {}\n\tMax: {}'.format(
        sequence_length.mean(),
        np.quantile(sequence_length, 0.5),
        sequence_length.min(),
        sequence_length.max()))

    print('Sequences per user:\n\tAverage: {:.2f}\n\tMedian: {}\n\tMin: {}\n\tMax: {}'.format(
        n_sessions_per_user.mean(),
        np.quantile(n_sessions_per_user, 0.5),
        n_sessions_per_user.min(),
        n_sessions_per_user.max()))

    print('Most used UI elements: {}'.format(cnt.most_common(5)))

    train, test = last_session_out_split(df)

    print("Train sessions: {} - Test sessions: {}".format(len(train), len(test)))


if __name__ == "__main__":
    main()
