import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
if __name__ == "__main__":
    import os
    os.chdir('/data/Pogona_Pursuit/Arena')
from analysis.strikes.strikes import load_strikes, Loader, StrikeAnalyzer, MissingStrikeData

TRAJ_LEN = 187


def create_dataset(animal_id, max_obs=None, is_normalize=True):
    strikes_ids = []
    for animal_id in ['PV80', 'PV85']:
        strikes_ids += load_strikes(animal_id)
    traj, df = [], []
    cnt = 0
    for sid in strikes_ids:
        try:
            if max_obs and cnt >= max_obs:
                break
            ld = Loader(sid, 'front', is_debug=False, sec_before=3)
            bt = ld.bug_traj_before_strike.drop(columns=['time'])
            if len(bt) < TRAJ_LEN:
                continue
            bt = bt.iloc[-TRAJ_LEN:]
            bt['strike_id'] = sid
            traj.append(bt)
            df.append({'strike_id': sid, 'temperature': ld.avg_temperature,
                       'movement_type': ld.info.get('movement_type'),
                       'hit_x': ld.info.get('x'), 'hit_y': ld.info.get('y')})
            cnt += 1
        except Exception:
            continue
    traj = pd.concat(traj)
    df = pd.DataFrame(df)
    if is_normalize:
        traj_scaler = StandardScaler()
        traj[['x', 'y']] = traj_scaler.fit_transform(traj[['x', 'y']])
        main_scaler = StandardScaler()
        df[['strike_id', 'temperature', 'hit_x', 'hit_y']] = main_scaler.fit_transform(df[['strike_id', 'temperature',
                                                                                           'hit_x', 'hit_y']])
    return traj, df


class StrikeDataset(Dataset):
    def __init__(self, df, traj_df):
        self.cat_features = ['movement_type', 'animal_id']
        self.numerical_features = ['temperature', 'strike_id']
        self.targets = ['hit_x', 'hit_y']
        self.cat_encoders = {}
        self.encode_categoricals(df)
        self.traj = np.vstack(traj_df.groupby('strike_id').apply(pd.DataFrame.to_numpy).apply(
            lambda x: x.reshape(1, *x.shape)).tolist())
        self.x_cat = df.loc[:, self.cat_features].copy().values.astype(np.int64)
        self.numerical = df.loc[:, self.numerical_features].copy().values.astype(np.float32)
        self.y = df.loc[:, self.targets].copy().values.astype(np.float32)

    @property
    def cat_sizes(self):
        cat_embedding_sizes = []
        for e in self.cat_encoders:
            n_categories = len(e.classes_)
            cat_embedding_sizes.append((n_categories, min(50, (n_categories + 1) // 2)))
        return cat_embedding_sizes

    def encode_categoricals(self, df):
        for c in self.cat_features:
            self.cat_encoders[c] = LabelEncoder()
            df[c] = self.cat_encoders[c].fit_transform(df.loc[:, c])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.traj[idx], self.x_cat[idx], self.numerical[idx], self.y[idx]


class StrikeModel(nn.Module):
    def __init__(self, traj_len: int, cat_sizes: list, n_numerical: int, n_targets=2, hidden_size_lstm=64,
                 lin1_size=150, lin2_size=70, lin_dropout=0.3, lstm_dropout=0.3):
        super().__init__()
        self.n_numerical = n_numerical
        hidden_size_lstm = min(hidden_size_lstm, traj_len // 2)
        self.lstm_x = nn.LSTM(traj_len, hidden_size_lstm, num_layers=3, batch_first=True, dropout=lstm_dropout)
        self.lstm_y = nn.LSTM(traj_len, hidden_size_lstm, num_layers=3, batch_first=True, dropout=lstm_dropout)
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(categories, size) for categories, size in cat_sizes])
        self.n_emb = sum(e.embedding_dim for e in self.cat_embeddings)  # length of all embeddings combined
        self.lin1 = nn.Linear(2 * hidden_size_lstm + n_numerical + self.n_emb, lin1_size)
        self.lin2 = nn.Linear(lin1_size, lin2_size)
        self.lin3 = nn.Linear(lin2_size, n_targets)
        self.bn1 = nn.BatchNorm1d(lin1_size)
        self.bn2 = nn.BatchNorm1d(lin2_size)
        self.lin_drop = nn.Dropout(lin_dropout)

    def forward(self, bug_traj, x_cat, *numerical_features):
        assert len(numerical_features) == self.n_numerical
        assert x_cat.shape[1] == len(self.cat_embeddings)
        x1 = [e(x_cat[:, i]) for i, e in enumerate(self.cat_embeddings)]
        x1 = torch.cat(x1, 1)

        bug_x, bug_y = self.lstm_x(bug_traj[:, 0]), self.lstm_y(bug_traj[:, 1])
        bug_x, bug_y = bug_x[:, -1, :], bug_y[:, -1, :]  # last cell, size = (batch_size, hidden_size)
        x = torch.cat([bug_x, bug_y, x1, *numerical_features], 1)
        x = F.relu(self.lin1(x))
        x = self.lin_drop(x)
        x = self.bn1(x)
        x = F.relu(self.lin2(x))
        x = self.lin_drop(x)
        x = self.bn2(x)
        x = self.lin3(x)
        return x


class Trainer:
    def __init__(self, batch_size=4):
        self.cache_dir = Path('/data/Pogona_Pursuit/output/models/strikes_prediction')
        traj, df = self.get_data()
        self.dataset = StrikeDataset(df, traj)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.model = StrikeModel(TRAJ_LEN, self.dataset.cat_sizes, len(self.dataset.numerical_features),
                                 len(self.dataset.targets))
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)

    def get_data(self, is_load=True):
        cache_path = (self.cache_dir / 'test.pkl').as_posix()
        if not is_load:
            traj, df = create_dataset('PV80', max_obs=100)
            with open(cache_path, 'wb') as f:
                pickle.dump({'traj': traj, 'df': df}, f)
        else:
            with open(cache_path, 'rb') as f:
                traj, df = pickle.load(f).values()
        return traj, df

    def train(self, n_epochs=30):
        print('start training of deep model')
        losses = []
        for i in tqdm(range(n_epochs)):
            train_loss = self.train_epoch()
            losses.append(train_loss)

        plt.plot(losses)
        plt.show()

    def train_epoch(self):
        self.model.train()
        total = 0
        sum_loss = 0
        for x1, x2, x3, y in self.loader:
            batch = y.shape[0]
            output = self.model(x1, x2, x3)
            loss = self.loss_fn(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += batch
            sum_loss += batch * (loss.item())
        return sum_loss / total


def main():
    tr = Trainer()
    tr.train(10)


if __name__ == "__main__":
    main()
