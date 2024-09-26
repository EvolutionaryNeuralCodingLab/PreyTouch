import re
import pickle
import time
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import torch
import warnings
from torch import nn
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field
from typing import List
from collections import Counter
from tqdm.auto import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.metrics import (explained_variance_score, roc_auc_score, balanced_accuracy_score, precision_score, \
                             recall_score, confusion_matrix, PrecisionRecallDisplay, RocCurveDisplay, precision_recall_curve,
                             average_precision_score, roc_curve, precision_recall_fscore_support, accuracy_score)
from sklearn.decomposition import PCA
from captum.attr import IntegratedGradients, LayerConductance, NeuronConductance
from pathlib import Path
from analysis.trainer import ClassificationTrainer

TRAJ_DIR = '/media/sil2/Data/regev/datasets/trajs'


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_dim, 1))
        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_length, hidden_dim)
        attention_scores = torch.tanh(torch.matmul(lstm_output, self.attention_weights))
        attention_scores = attention_scores.squeeze(-1)  # shape: (batch_size, seq_length)
        attention_weights = torch.softmax(attention_scores, dim=1)  # shape: (batch_size, seq_length)
        weighted_sum = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(
            1)  # shape: (batch_size, hidden_dim)
        return weighted_sum, attention_weights


class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, is_attn=False, is_weighted=False):
        # x shape: (batch_size, seq_length, input_dim)
        lstm_output, _ = self.lstm(x)
        # lstm_output = lstm_output[:, -1, :]
        # lstm_output shape: (batch_size, seq_length, hidden_dim)
        weighted_sum, attn_weights = self.attention(lstm_output)

        # Apply batch normalization and dropout
        x = self.batch_norm(weighted_sum)
        x = self.dropout(x)
        # Final classification layer
        out = self.fc(x)
        if is_attn:
            return out, attn_weights
        elif is_weighted:
            return out, weighted_sum
        else:
            return out


class LizardTrajDataSet(Dataset):
    def __init__(self, strk_df, trajs, ids, variables, targets_values, is_standardize=True, target_name='block_speed',
                 is_resample=True, sub_section=None, is_shuffled_target=False):
        self.samples = ids
        self.variables = variables
        self.targets = targets_values
        self.target_name = target_name
        self.sub_section = sub_section
        # concatenate the trjaectories into a single dataframe
        self.X = pd.concat([t for t in trajs.values()], axis=0).sort_values(by=['strike_id', 'time']).reset_index(
            drop=True)
        # standarize the data over all trajectory observations
        if is_standardize:
            for col in variables:
                self.X[col] = (self.X[col] - self.X[col].mean()) / self.X[col].std()

        self.X = self.X.query(f'strike_id in {self.samples}')
        self.y = strk_df[target_name].loc[self.samples]
        if is_shuffled_target:
            print(f'Notice! Shuffling randomly the target values')
            self.y = pd.Series(index=self.y.index, data=np.random.choice(self.y.unique(), len(self.y)))

        if is_resample:
            self.resample_trajs()
        if self.X.empty:
            raise ValueError('No samples in the dataset')
        if sub_section:
            self.extract_subsection()

        self.y = self.y.map(lambda x: targets_values.index(x)).to_dict()

    def resample_trajs(self):
        min_count = self.y.value_counts().min()
        self.samples = pd.DataFrame(self.y).groupby(self.target_name).apply(lambda x: x.sample(min_count, random_state=0)).index.get_level_values(1).values.tolist()
        print(f'Resampling trajectories to {min_count} samples from each {self.target_name} class')
        self.X = self.X.query(f'strike_id in {self.samples}')
        self.y = self.y.loc[self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        strike_id = self.samples[idx]
        # select variable of interest
        traj = self.X.query(f'strike_id == {strike_id}').sort_values(by=['time']).reset_index(drop=True)[
            list(self.variables)]
        traj = np.array(traj)
        target_ = np.array([self.y[strike_id]])
        sample = (torch.FloatTensor(traj), torch.FloatTensor(target_).squeeze().type(torch.LongTensor))
        return sample

    def extract_subsection(self):

        def _sample(group):
            # Find the index of the closest value
            closest_idx = (group['total_sec'] - self.sub_section[0]).abs().idxmin()
            # Get the position of the closest index in the group
            closest_position = group.index.get_loc(closest_idx)
            # Get the indexes to sample: from the closest position to closest position + n
            sample_indexes = group.iloc[closest_position:closest_position + self.sub_section[1] + 1].index
            return group.loc[sample_indexes]

        self.X = self.X.groupby('strike_id', group_keys=False).apply(_sample)


@dataclass
class TrajClassifier(ClassificationTrainer):
    batch_size: int = 15
    num_epochs: int = 150
    learning_rate: float = 0.001
    weight_decay: float = 1e-2
    dropout_prob: float = 0.3
    lstm_layers: int = 6
    lstm_hidden_dim: int = 100
    monitored_metric: str = 'val_loss'
    monitored_metric_algo: str = 'min'
    num_loader_workers: int = 0
    threshold: float = 0
    is_resample: bool = True  # resample trajectories to have equal number of samples from each class
    dataset_path: str = TRAJ_DIR
    movement_type: str = 'random_low_horizontal'
    is_hit: bool = False  # keep only hits
    animal_id: str = 'all'
    sub_section: tuple = None  # (start_sec {float}, length of samples after start_sec {int})
    target_name: str = 'block_speed'
    is_shuffled_target: bool = False  # shuffle the target randomly. Used to find baseline for attention.
    targets: List = field(default_factory=lambda: [2, 4, 6, 8])
    feature_names: List = field(default_factory=lambda: ['x', 'y', 'speed'])
    strike_index = None

    # def __post_init__(self):
    #     super().__post_init__()
        # self.device = torch.device('mps')

    def load(self):
        attrs_path = Path(self.model_path) / 'attrs.pkl'
        if not attrs_path.exists():
            self.print(f'No attrs file found. Loading from {self.model_path}')
            s = Path(self.model_path).parts[-2]
            m = re.search(r'traj_classifier_(?P<animal_id>\w+?)_(?P<movement_type>\w+)', s)
            self.movement_type = m.group('movement_type')
            self.animal_id = m.group('animal_id')
        else:
            with attrs_path.open('rb') as f:
                attrs = pickle.load(f)
                for k, v in attrs.items():
                    if k in ['model_path', 'is_debug']:
                        continue
                    setattr(self, k, v)
        super().load()

    def save_model(self):
        super().save_model()
        attrs_path = Path(self.model_path) / 'attrs.pkl'
        with attrs_path.open('wb') as f:
            pickle.dump(vars(self), f)

    def get_dataset(self):
        strk_df, trajs = self.load_data()
        sdf = strk_df.copy()
        sdf = sdf.query(f'{self.target_name} in {list(self.targets)}')
        if self.movement_type:
            sdf = sdf.query('movement_type==@self.movement_type')
        if self.animal_id and self.animal_id != 'all':
            sdf = sdf.query('animal_id==@self.animal_id')
        if self.is_hit:
            sdf = sdf.query('is_hit')
        strikes_ids = sdf.index.values.tolist()

        dataset = LizardTrajDataSet(strk_df, trajs, strikes_ids, self.feature_names, self.targets,
                                    target_name=self.target_name, sub_section=self.sub_section,
                                    is_resample=self.is_resample, is_shuffled_target=self.is_shuffled_target)
        self.print(f'Traj classes count: {pd.Series(dataset.y).value_counts().sort_index().set_axis(self.targets).to_dict()}')

        # set strike index
        example_strike_id = dataset.X.strike_id.unique()[0]
        tf = dataset.X.query(f'strike_id=={example_strike_id}').sort_values(by='time').reset_index()
        row = strk_df.loc[example_strike_id]
        self.strike_index = (tf.time - row.time).dt.total_seconds().abs().idxmin()

        # find the right kfolds size
        for n in range(self.kfolds, 1, -1):
            if (len(dataset) / n) > self.batch_size:
                if self.kfolds != n:
                    print(f'setting kfolds to {n}')
                    self.kfolds = n
                break
            if n == 2:
                raise Exception('Please decrease the batch size')
        return dataset

    def get_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def get_model(self):
        return LSTMWithAttention(input_dim=len(self.feature_names), hidden_dim=self.lstm_hidden_dim,
                                 output_dim=len(self.targets), num_layers=self.lstm_layers,
                                 dropout_prob=self.dropout_prob)

    def evaluate(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
        y_score = F.softmax(y_pred, dim=1)
        y_pred, y_pred_score = self.predict_proba(y_pred)
        y_true, y_pred, y_score = y_true.cpu(), y_pred.cpu(), y_score.cpu()
        try:
            auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        except Exception as e:
            auc = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            accuracy = balanced_accuracy_score(y_true, y_pred)
        return {
            'accuracy': accuracy,
            'auc': auc,
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    def load_data(self):
        filename = 'trajs_10s_after' if self.animal_id not in ['PV80', 'PV42', 'PV85'] else 'trajs_10s_after_msi_regev'
        with open(f'{self.dataset_path}/{filename}.pkl', 'rb') as f:
            d = pickle.load(f)
        strk_df, trajs = d['strk_df'], d['trajs']
        for strike_id, xf in trajs.items():
            xf['speed_x'] = xf.x.diff()
            xf['speed_y'] = xf.y.diff()
            xf['speed'] = np.sqrt(xf.x.diff() ** 2 + xf.y.diff() ** 2)
            trajs[strike_id] = xf.iloc[1:] # remove first row because it had NaN after speed calculation
        return strk_df, trajs

    def get_model_name(self):
        return f'traj_classifier_{self.animal_id}_{self.movement_type}'

    def summary_plots(self, chosen_fold_id):
        fig, axes = plt.subplots(2, 3, figsize=(15, 6))
        self.plot_train_metrics(chosen_fold_id, axes[0, :])
        self.all_data_evaluation(axes=axes[1, :])
        fig.suptitle(f'{self.animal_id} {self.movement_type}', fontsize=20)
        fig.tight_layout()
        if self.cache_dir:
            fig.savefig(self.cache_dir / 'summary_plots.jpg')
        plt.show()

    def get_data_for_evaluation(self, is_test_set=False):
        self.model.eval()
        dataset = self.get_dataset()
        if is_test_set:
            dataset = Subset(dataset, self.test_indices)
        y_true, y_pred, y_score = [], [], []
        attns = {}
        for x, y in (tqdm(dataset) if self.is_debug else dataset):
            outputs, attention_weights = self.model(x.to(self.device).unsqueeze(0), is_attn=True)
            attns.setdefault(self.targets[y.item()], []).append(attention_weights.detach().cpu().numpy())
            label, prob = self.predict_proba(outputs, is_all_probs=True)
            y_true.append(y.item())
            y_pred.append(label.item())
            y_score.append(prob.detach().cpu().numpy())
        y_true, y_pred, y_score = np.vstack(y_true), np.vstack(y_pred), np.vstack(y_score)
        return y_true, y_pred, y_score, attns

    def plot_attention(self, ax, attns):
        mean_att = []
        time_vector = self.sub_section[0] + np.arange(self.sub_section[1] + 1) * (1 / 60)
        for bug_speed in self.targets:
            att = attns[bug_speed]
            att = np.vstack(att).mean(axis=0)
            att[:8] = np.nan
            mean_att.append(att)
            ax.plot(time_vector, att, label=f'{bug_speed}cm/sec', alpha=0.4)

        ax.plot(time_vector, np.vstack(mean_att).mean(axis=0), color='k')
        if self.strike_index:
            ax.axvline(0, color='k', linestyle='--')
        ax.set_xlabel('Time Around Strike [sec]')
        ax.set_ylabel('Attention values')
        ax.legend()

    def all_data_evaluation(self, axes=None, is_test_set=False, is_plot_auc=True, **kwargs):
        if axes is None:
            fig, axes_ = plt.subplots(1, 3, figsize=(18, 4))
        else:
            axes_ = axes
        assert len(axes_) == (3 if is_plot_auc else 2)

        y_true, y_pred, y_score, attns = self.get_data_for_evaluation(is_test_set)
        att_id = 2 if is_plot_auc else 1
        self.plot_attention(axes_[att_id], attns)

        y_true_binary = label_binarize(y_true, classes=np.arange(len(self.targets)))
        self.plot_confusion_matrix(y_true, y_pred, ax=axes_[0])
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        axes_[0].set_title(f'Accuracy: {acc:.2f}')
        # self.plot_precision_curve(y_true_binary, y_score, axes_[1])
        if is_plot_auc:
            self.plot_roc_curve(y_true_binary, y_score, axes_[1])
        # PrecisionRecallDisplay.from_predictions(y_true, y_true, ax=axes_[1])
        # RocCurveDisplay.from_predictions(y_true, y_score, ax=axes_[2])
        if axes is None:
            plt.show()

    def calc_auc(self):
        self.model.eval()
        dataset = self.get_dataset()
        y_true, y_score = [], []
        for x, y in tqdm(dataset):
            outputs, _ = self.model(x.to(self.device).unsqueeze(0), is_attn=True)
            label, prob = self.predict_proba(outputs, is_all_probs=True)
            y_true.append(y.item())
            y_score.append(prob.detach().cpu().numpy())

        y_true, y_score = np.vstack(y_true), np.vstack(y_score)
        y_true_binary = label_binarize(y_true, classes=np.arange(len(self.targets)))
        aucs = {}
        for i, target in enumerate(self.targets):
            aucs[target] = roc_auc_score(y_true_binary[:, i], y_score[:, i])
        return aucs

    def plot_precision_curve(self, y_true_binary, y_score, ax):
        for i, target in enumerate(self.targets):
            precision, recall, _ = precision_recall_curve(y_true_binary[:, i], y_score[:, i])
            ap = average_precision_score(y_true_binary[:, i], y_score[:, i], average="micro")
            display = PrecisionRecallDisplay(precision, recall, average_precision=ap)
            # prevalence_pos_label=Counter(y_true_binary.ravel())[1] / y_true_binary.size
            display.plot(ax=ax, name=str(target))  # plot_chance_level=True
            ax.set_title("Micro-averaged over all classes")

    def plot_roc_curve(self, y_true_binary, y_score, ax):
        for i, target in enumerate(self.targets):
            fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_score[:, i])
            auc = roc_auc_score(y_true_binary[:, i], y_score[:, i])
            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
            display.plot(ax=ax, name=str(target))  # plot_chance_level=True

        micro_auc = roc_auc_score(y_true_binary, y_score, average="micro")
        ax.set_title(f"Micro-averaged over all classes: {micro_auc:.2f}")

    def plot_attention_weights(self, attention_weights, trajectory, ax):
        # attention_weights: Tensor of shape (seq_length,)
        # trajectory: Tensor of shape (seq_length, 2)
        # index: Index of the sample in the batch
        attention = attention_weights.detach().cpu().numpy()
        traj = trajectory.detach().cpu().numpy()
        ax.scatter(traj[:, 0], traj[:, 1], c=attention, cmap='viridis')
        # plt.colorbar(label='Attention Weight')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Attention Weights over Trajectory')

    def check_hidden_states(self, cols=4, axes=None, fig=None, is_legend=True):
        dataset = self.get_dataset()
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes = axes.flatten()

        # self.visualize_features_importance(dataset, axes=axes[:len(self.targets)])
        importance_df = self.visualize_features_importance(dataset, ax=axes[0], is_legend=is_legend)

        # run PCA to visualize targets embedding
        self.model.eval()
        res, targets = [], []
        for x, y in tqdm(dataset):
            x = x.to(self.device).unsqueeze(0)
            outputs, w = self.model(x, is_weighted=True)
            res.append(w.detach().cpu().numpy())
            targets.append(self.targets[y.item()])
        res = np.vstack(res)
        pca = PCA(n_components=2)
        X_embedded = pca.fit_transform(res)
        ax_pca = axes[1]
        sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=targets, palette="deep", ax=ax_pca, legend=is_legend)
        ax_pca.set_title('PCA')
        for ax in axes[len(self.targets)+1:]:
            ax.axis('off')
        if fig is not None:
            fig.tight_layout()
            fig.savefig((Path(self.model_path) / 'hidden_states.png'), dpi=200)
            plt.show()
        return importance_df

    def visualize_features_importance(self, dataset, ax=None, is_legend=True, is_plot=True):
        torch.backends.cudnn.enabled = False
        self.model.eval()
        ig = IntegratedGradients(self.model)

        if ax is None and is_plot:
            fig, ax = plt.subplots(1, len(self.targets), figsize=(18, 3))

        res = {}
        aggregated_attributions = {}
        for x, y in dataset:
            input_ = x.to(self.device).unsqueeze(0)
            baseline = torch.zeros(input_.shape).to(self.device)
            attr, delta = ig.attribute(input_, baseline, target=y.item(), return_convergence_delta=True)
            aggregated_attributions.setdefault(self.targets[y.item()], []).append(attr)

        importance_df = []
        for target, x in aggregated_attributions.items():
            x = torch.cat(x, dim=0)
            importance = np.mean(x.cpu().detach().numpy(), axis=0)
            imp = pd.Series(importance[-1])
            # idx = imp.sort_values().index.tolist()
            # imp = imp.reindex(idx)
            imp.index = [self.feature_names[i] for i in imp.index]
            imp.name = target
            importance_df.append(imp)
            max_x = np.abs(importance[-1]).max()
            res[target] = (imp, max_x)

        importance_df = pd.DataFrame(importance_df)
        res[10] = (importance_df.mean(), 0)
        max_x = max([x[1] for x in res.values()])
        if is_plot:
            x = np.arange(len(self.feature_names))  # the label locations
            width = 0.18  # the width of the bars
            multiplier = 0
            for i, target in enumerate(sorted(list(res.keys()))):
                imp, _ = res[target]
                offset = width * multiplier
                rects = ax.bar(x + offset, imp.values, width, label=f'{target}cm/sec' if target<10 else 'Average')
                # ax.bar_label(rects, padding=3)
                multiplier += 1

            ax.axhline(0, color='k')
            ax.set_xticks(x + width)
            ax.set_xticklabels(self.feature_names)
            if is_legend:
                ax.legend()

        return importance_df


def animals_comparison():
    res_df = []
    for animal_id in ['PV91', 'PV95', 'PV163']:
        for movement_type in ['random_low_horizontal', 'circle']:
            if animal_id == 'PV95' and movement_type == 'circle':
                continue
            try:
                tj = TrajClassifier(save_model_dir=TRAJ_DIR, is_shuffle_dataset=False,
                                    animal_id=animal_id, movement_type=movement_type)
                tj.train(is_plot=True)

                best_i = np.argmin([x['score'] for x in tj.history])
                best_epoch = np.argmin(tj.history[best_i]['metrics']['val_loss'])
                for metric, l in tj.history[best_i]['metrics'].items():
                    res_df.append({'animal_id': animal_id, 'movement_type': movement_type, 'metric': metric,
                                   'value': l[best_epoch]})
            except Exception as e:
                print(e)
    res_df = pd.DataFrame(res_df)
    filename = f'{TRAJ_DIR}/results_{datetime.now().isoformat()}.csv'
    res_df.to_csv(filename)
    plot_comparison(filename)


def hyperparameters_comparison(animal_id='PV91', movement_type='random_low_horizontal', is_resample=False,
                               sub_section=(-2, 180), feature_names=('x', 'y', 'speed')):
    from sklearn.model_selection import ParameterGrid

    res_df = []
    grid = ParameterGrid(dict(dropout_prob=[0.1, 0.3, 0.4, 0.6], lstm_layers=[4, 6, 8], lstm_hidden_dim=[50, 100, 150]))
    for i, params in enumerate(grid):
        print(f'start loop {i+1}/{len(grid)}')
        tj = TrajClassifier(save_model_dir=TRAJ_DIR, is_shuffle_dataset=False, sub_section=sub_section, feature_names=feature_names,
                            is_debug=False, is_resample=is_resample, animal_id=animal_id, movement_type=movement_type, **params)
        tj.train(is_plot=False)
        best_i = np.argmin([x['score'] for x in tj.history])
        best_epoch = np.argmin(tj.history[best_i]['metrics']['val_loss'])
        for metric, l in tj.history[best_i]['metrics'].items():
            res_df.append({'metric': metric, 'value': l[best_epoch], 'animal_id': animal_id,
                           'movement_type': movement_type, **params})

        y_true, y_pred, y_score, attns = tj.get_data_for_evaluation()
        acc = accuracy_score(y_true, y_pred)
        res_df.append({'metric': 'overall_accuracy', 'value': acc, 'animal_id': animal_id,
                       'movement_type': movement_type, 'model_path': tj.model_path, **params})
        torch.cuda.empty_cache()
        time.sleep(1)

    res_df = pd.DataFrame(res_df)
    filename = f'{TRAJ_DIR}/hyperparameters_results_{animal_id}_{movement_type}_{"_".join(feature_names)}_{datetime.now().isoformat()}.csv'
    res_df.to_csv(filename)

    print('\n' + '#' * 50)
    print(f'\nAnimal: {animal_id}, Movement: {movement_type}')
    print(f'sub-section: {sub_section}, Features: {feature_names}')
    for metric in ['overall_accuracy', 'accuracy']:
        print(f'\nBest hyperparameters for {metric}:')
        print(res_df.query(f'metric=="{metric}"').sort_values(by='value', ascending=False).iloc[:3][
                  ['value', 'dropout_prob', 'lstm_hidden_dim', 'lstm_layers']])
    best_model_path = res_df.query(f'metric=="overall_accuracy"').sort_values(by='value', ascending=False).iloc[
        0].model_path
    print(f'\nbest overall model path: {best_model_path}')
    print('\n' + '#' * 50 + '\n')


def find_best_features(movement_type='random_low_horizontal', lstm_layers=4, dropout_prob=0.3, lstm_hidden_dim=50,
                       is_resample=False):
    all_features = ['x', 'y', 'prob', 'speed_x', 'speed_y', 'angle']
    res_df = []
    for L in range(1, len(all_features) + 1):
        for feature_names in itertools.combinations(all_features, L):
            print(f'Start training with features {feature_names}')
            try:
                tj = TrajClassifier(save_model_dir=TRAJ_DIR, is_shuffle_dataset=False, sub_section=(-2, 150),
                                    is_resample=is_resample, is_hit=False,
                                    feature_names=feature_names, animal_id='all', movement_type=movement_type,
                                    lstm_layers=lstm_layers, dropout_prob=dropout_prob, lstm_hidden_dim=lstm_hidden_dim)
                tj.train(is_plot=False)
                best_i = np.argmin([x['score'] for x in tj.history])
                best_epoch = np.argmin(tj.history[best_i]['metrics']['val_loss'])
                for metric, l in tj.history[best_i]['metrics'].items():
                    res_df.append({'features': feature_names, 'metric': metric, 'value': l[best_epoch]})
            except Exception as exc:
                print(f'Error in feature names: {feature_names}; {exc}')

    res_df = pd.DataFrame(res_df)
    filename = f'{TRAJ_DIR}/features_results_{datetime.now().isoformat()}.csv'
    res_df.to_csv(filename)


def run_with_different_seeds(animal_id, movement_type, sub_section, feature_names, lstm_layers=4,
                            dropout_prob=0.1, lstm_hidden_dim=100, n=10, is_run_feature_importance=True, is_plot=True,
                             is_save=True):
    res = {'metrics': [], 'attention': {}, 'feature_importance': []}
    for s in range(n):
        tj = TrajClassifier(save_model_dir=TRAJ_DIR, is_shuffle_dataset=False, sub_section=sub_section,
                            is_resample=False, feature_names=feature_names, seed=s, is_debug=False,
                            animal_id=animal_id, movement_type=movement_type, is_hit=False, lstm_layers=lstm_layers,
                            dropout_prob=dropout_prob, lstm_hidden_dim=lstm_hidden_dim)
        tj.train(is_plot=False)
        best_i = np.argmin([x['score'] for x in tj.history])
        best_epoch = np.argmin(tj.history[best_i]['metrics']['val_loss'])
        for metric, l in tj.history[best_i]['metrics'].items():
            res['metrics'].append({'metric': metric, 'value': l[best_epoch], 'animal_id': animal_id,
                                   'movement_type': movement_type})
        y_true, y_pred, y_score, attns = tj.get_data_for_evaluation()
        for bug_speed, a in attns.items():
            res['attention'].setdefault(bug_speed, []).append(a)
        res['metrics'].append({'metric': 'overall_accuracy', 'value': accuracy_score(y_true, y_pred),
                               'animal_id': animal_id, 'movement_type': movement_type})
        if is_run_feature_importance:
            dataset = tj.get_dataset()
            importance_df = tj.visualize_features_importance(dataset, is_plot=False)
            res['feature_importance'].append(importance_df)

        torch.cuda.empty_cache()
        time.sleep(1)

    res['metrics'] = pd.DataFrame(res['metrics'])
    for bug_speed in res['attention'].keys():
        res['attention'][bug_speed] = np.vstack(res['attention'][bug_speed])
    importance = pd.concat(res['feature_importance']).reset_index().rename(columns={'index': 'bug_speed'})
    imp = importance.groupby('bug_speed')
    imp = imp.mean().merge(imp.std().rename(columns={k: f'{k}_std' for k in feature_names}), left_on='bug_speed', right_index=True)
    res['feature_importance'] = imp

    if is_plot:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))

        sns.barplot(data=res['metrics'], x='metric', y='value', ax=axes[0])
        axes[0].set_xticks(axes[0].get_xticks(), axes[0].get_xticklabels(), rotation=45, ha='right')

        offset = 0.1
        for j, f in enumerate(feature_names):
            for k, bug_speed in enumerate(imp.index):
                axes[1].bar(j + k*offset, imp[f].loc[bug_speed], width=offset)
                axes[1].errorbar(j + k*offset, imp[f].loc[bug_speed], yerr=imp[f'{f}_std'].loc[bug_speed], fmt="o", color="k")
        axes[1].set_xticks(np.arange(len(feature_names)))
        axes[1].set_xticklabels(feature_names)

        mean_att = []
        for bug_speed, a in res['attention'].items():
            a_avg = a.mean(axis=0)
            mean_att.append(a_avg)
            print(a_avg)
            axes[2].plot(a_avg.squeeze(), label=bug_speed, alpha=0.5)
        axes[2].plot(np.vstack(mean_att).mean(axis=0), color='k', linewidth=2)

        fig.tight_layout()
        plt.show()

    if is_save:
        filename = f'{TRAJ_DIR}/different_seeds_{animal_id}_{movement_type}_{datetime.now().isoformat()}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(res, f)
        print(f'Results saved to {filename}')

    return res


def plot_comparison(filename):
    df = pd.read_csv(filename, index_col=0)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    for i, movement_type in enumerate(['random_low_horizontal', 'circle']):
        df_ = df.query(f'movement_type == "{movement_type}" and metric in ["val_loss","accuracy","auc"]')
        sns.barplot(data=df_, x='metric', y='value', hue='animal_id', ax=axes[i])
        axes[i].set_title(movement_type)

    fig.tight_layout()
    plt.show()


def find_optimal_span(animal_id='PV42', movement_type='random_low_horizontal', dt=0.2, span=60, n_seeds=5):
    all_res = []
    for t_start in np.arange(-10, (10-(span/60))+dt, dt):
        print(f'Start training with t_start: {t_start}')
        try:
            r = run_with_different_seeds(animal_id, movement_type, (t_start, span), ['x', 'y', 'speed'],
                                     lstm_layers=4, dropout_prob=0.3, lstm_hidden_dim=100, n=n_seeds,
                                     is_run_feature_importance=True, is_plot=False, is_save=False)
            all_res.append(r)
        except Exception as exc:
            print(f'Error in t_start: {t_start}; {exc}')

    filename = f'{TRAJ_DIR}/optimal_span_{datetime.now().isoformat()}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(all_res, f)
    print(f'\nResults saved to: {filename}')


if __name__ == '__main__':
    # tj = TrajClassifier(save_model_dir=TRAJ_DIR, is_shuffle_dataset=False, sub_section=(-2, 150), is_resample=False,
    #                     animal_id='all', movement_type='random_low_horizontal', is_hit=False, lstm_layers=4,
    #                     dropout_prob=0.3, lstm_hidden_dim=50, is_shuffled_target=True)
    # tj.train(is_plot=True)
    # tj.check_hidden_states()
    find_optimal_span(animal_id='PV91', movement_type='random_low_horizontal')
    # run_with_different_seeds('PV91', 'random_low_horizontal', (-1, 60),
    #                          ['x', 'y', 'speed'], n=10)
    # find_best_features(movement_type='circle', lstm_layers=6, dropout_prob=0.5, is_resample=True)
    # for movement_type in ['random_low_horizontal', 'circle']:
    #     for animal_id in ['PV91', 'PV163', 'PV99', 'PV80', 'PV42', 'PV85', 'all']:
    #         hyperparameters_comparison(animal_id=animal_id, movement_type=movement_type,
    #                                    feature_names=['x', 'y', 'speed'], sub_section=(-2, 120))
    # animals_comparison()
    # plot_comparison('/Users/regev/PhD/msi/Pogona_Pursuit/output/datasets/trajectories/results_2024-06-24T16:43:58.880310.csv')

    # tj = TrajClassifier(
    #     model_path='/media/sil2/Data/regev/datasets/trajs/traj_classifier_all_random_low_horizontal/20240722_105828')
    # tj.check_hidden_states()