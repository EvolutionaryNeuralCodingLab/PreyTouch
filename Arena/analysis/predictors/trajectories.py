import re
import pickle
import time
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
        # A simple linear layer that maps hidden state to a scalar score.
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_length, hidden_dim)
        # Compute unnormalized attention scores for each time step.
        scores = self.attn(lstm_output)  # shape: (batch_size, seq_length, 1)
        scores = torch.tanh(scores)
        scores = scores.squeeze(-1)      # shape: (batch_size, seq_length)
        attn_weights = F.softmax(scores, dim=1)  # shape: (batch_size, seq_length)
        # Compute the context vector as a weighted sum of the LSTM outputs.
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)  # shape: (batch_size, hidden_dim)
        return context, attn_weights


class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, is_attn=False):
        # x shape: (batch_size, seq_length, input_dim)
        lstm_output, (h_n, c_n) = self.lstm(x)
        # lstm_output shape: (batch_size, seq_length, hidden_dim)
        context, attn_weights = self.attention(lstm_output)
        x = self.dropout(context)
        out = self.fc(x)
        if is_attn:
            return out, attn_weights
        else:
            return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        lstm_output, (h_n, c_n) = self.lstm(x)
        x = torch.mean(lstm_output, dim=1)
        out = self.fc(x)
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
        self.X = pd.concat([t for t in trajs.values()], axis=0).sort_values(by=['id', 'time']).reset_index(
            drop=True)
        # standarize the data over all trajectory observations
        if is_standardize:
            self.normalize_trajs()

        self.X = self.X.query(f'id in {self.samples}')
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
        self.X = self.X.query(f'id in {self.samples}')
        self.y = self.y.loc[self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid = self.samples[idx]
        # select variable of interest
        traj = self.X.query(f'id == {sid}').sort_values(by=['time']).reset_index(drop=True)[
            list(self.variables)]
        traj = np.array(traj)
        target_ = np.array([self.y[sid]])
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

        self.X = self.X.groupby('id', group_keys=False).apply(_sample)

    def normalize_trajs(self):
        # we calculate the mean and std over the sequence of -3,3 to match the inputs from previous versions
        X_ = self.X.query('-3<total_sec<3')
        for col in self.variables:
            self.X[col] = (self.X[col] - X_[col].mean()) / X_[col].std()


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
    is_iti: bool = False  # ITI dataset instead of strikes
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
    is_attention: bool = False  # use the LSTMWithAttention model
    strike_index = None

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

    def get_dataset(self, is_print_size=False):
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
        if is_print_size:
            self.print(f'Traj classes count: {pd.Series(dataset.y).value_counts().sort_index().set_axis(self.targets).to_dict()}')

        # set strike index
        example_sid = dataset.X.id.unique()[0]
        tf = dataset.X.query(f'id=={example_sid}').sort_values(by='time').reset_index()
        row = strk_df.loc[example_sid]
        self.strike_index = (tf.time - row.time).dt.total_seconds().abs().idxmin()

        # find the right kfolds size
        for n in range(self.kfolds, 1, -1):
            if (len(dataset) / n) > self.batch_size:
                if self.kfolds != n:
                    self.print(f'setting kfolds to {n}')
                    self.kfolds = n
                break
            if n == 2:
                raise Exception('Please decrease the batch size')
        return dataset

    def get_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    # def get_scheduler(self, optimizer):
    #     return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=self.monitored_metric_algo, patience=10)

    def get_model(self):
        if not self.is_attention:
            return LSTMModel(input_dim=len(self.feature_names), hidden_dim=self.lstm_hidden_dim,
                             output_dim=len(self.targets), num_layers=self.lstm_layers,
                             dropout_prob=self.dropout_prob)
        else:
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
        if not self.is_iti:
            # filename = 'trajs_before_after_10s_strike'
            filename = 'trajs_before_120_after_60s_strike'
        else:
            filename = 'trajs_10s_iti'
        with open(f'{self.dataset_path}/{filename}.pkl', 'rb') as f:
            d = pickle.load(f)
        strk_df, trajs = d['strk_df' if not self.is_iti else 'trial_df'], d['trajs']
        for sid, xf in trajs.items():
            xf['speed_x'] = xf.x.diff()
            xf['speed_y'] = xf.y.diff()
            xf['speed'] = np.sqrt(xf.x.diff() ** 2 + xf.y.diff() ** 2)
            trajs[sid] = xf.iloc[1:] # remove first row because it had NaN after speed calculation
        return strk_df, trajs

    def get_model_name(self):
        model_name = f'traj_classifier_{self.animal_id}_{self.movement_type}'
        if self.is_iti:
            model_name += '_iti'
        return model_name

    def summary_plots(self, chosen_fold_id):
        # fig, axes = plt.subplots(2, 3, figsize=(15, 6))
        fig = plt.figure(figsize=(16, 6))
        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
        gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0], wspace=0.1)
        self.plot_train_metrics(chosen_fold_id, [fig.add_subplot(g) for g in gs_top])
        gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer_gs[1], wspace=0.3)
        self.all_data_evaluation(axes=[fig.add_subplot(g) for g in gs_bottom])
        fig.suptitle(f'{self.animal_id} {self.movement_type}', fontsize=20)
        # fig.tight_layout()
        if self.cache_dir:
            fig.savefig(self.cache_dir / 'summary_plots.jpg')
        plt.show()

    def get_data_for_evaluation(self, is_test_set=False):
        self.model.eval()
        dataset = self.get_dataset()
        if is_test_set:
            dataset = Subset(dataset, self.test_indices)
        y_true, y_pred, y_score = [], [], []
        for x, y in dataset:
            outputs = self.model(x.to(self.device).unsqueeze(0))
            label, prob = self.predict_proba(outputs, is_all_probs=True)
            y_true.append(y.item())
            y_pred.append(label.item())
            y_score.append(prob.detach().cpu().numpy())
        y_true, y_pred, y_score = np.vstack(y_true), np.vstack(y_pred), np.vstack(y_score)
        return y_true, y_pred, y_score

    def all_data_evaluation(self, axes=None, is_test_set=False, is_plot_auc=True, ablate_all_except=True,
                            is_overall_ablation=False, **kwargs):
        if axes is None:
            fig, axes_ = plt.subplots(1, 4, figsize=(18, 4))
        else:
            axes_ = axes
        assert len(axes_) == (4 if is_plot_auc else 3)

        y_true, y_pred, y_score = self.get_data_for_evaluation(is_test_set)
        att_id = 2 if is_plot_auc else 1
        # self.plot_attention(axes_[att_id], attns)
        self.plot_segment_importance(axes_[att_id])
        self.plot_ablation(axes_[att_id + 1], ablate_all_except=ablate_all_except, is_overall_ablation=is_overall_ablation)

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

    def calc_ablation(self, segment=None, ablate_all_except=False):
        self.model.eval()
        dataset = self.get_dataset()
        ablations = {}
        if segment is not None:
            assert len(segment) == 2, 'Segment should be a tuple of start and end times'
            mask = (self.time_vector >= segment[0]) & (self.time_vector <= segment[1])

        for i, feature_name in enumerate(self.feature_names + ['control']):
            y_true, y_pred = [], []
            for x, y in dataset:
                if feature_name != 'control':
                    if ablate_all_except:
                        x = self._ablate_all_except(x, segment, i)
                    else:
                        if segment is not None:
                            x[mask, i] = 0.0
                        else:
                            x[:, i] = 0.0
                outputs = self.model(x.to(self.device).unsqueeze(0))
                label, prob = self.predict_proba(outputs, is_all_probs=True)
                y_true.append(y.item())
                y_pred.append(label.item())
            acc = accuracy_score(y_true, y_pred)
            ablations[feature_name] = acc

        ablations = {k: v - ablations['control'] if not ablate_all_except else v for k, v in ablations.items() if k != 'control'}
        return ablations

    def _ablate_all_except(self, x, segment, feature_id):
        for i in range(len(self.feature_names)):
            if i == feature_id:
                if segment is not None:
                    mask = (self.time_vector >= segment[0]) & (self.time_vector <= segment[1])
                    x[~mask, i] = 0.0
            else:
                x[:, i] = 0.0
        return x

    def plot_ablation(self, ax, ablate_all_except=False, is_overall_ablation=False):
        if is_overall_ablation:
            ablations_dict = self.calc_ablation(segment=(self.time_vector[0], self.time_vector[-1]),
                                                ablate_all_except=ablate_all_except)
            ax.bar(ablations_dict.keys(), ablations_dict.values())
        else:
            af = []
            for start_t in np.linspace(self.time_vector[0], self.time_vector[-1]-0.2, 3):
                seg = (start_t, start_t + 0.2)
                ablations_dict = self.calc_ablation(segment=seg, ablate_all_except=ablate_all_except)
                ablations_dict['segment'] = np.round(np.mean(seg), 1)
                af.append(ablations_dict)
            af = pd.DataFrame(af)
            af = af.set_index('segment').stack().reset_index().rename(columns={'level_1': 'feature', 0: 'ablation'})
            sns.barplot(data=af, x='segment', y='ablation', hue='feature', ax=ax)
            ax.set_xlabel('segment mid [sec]')

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
        ax.set_title(f"Micro-averaged over all classes: {micro_auc:.2f}", fontsize=12)

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

    def calc_segment_importance(self, max_allocation=27840):
        torch.backends.cudnn.enabled = False
        self.model.eval()
        ig = IntegratedGradients(self.model)
        dataset = self.get_dataset()

        max_examples = None
        input_tensors = {}
        for x, y in dataset:
            if not max_examples:
                max_examples = int(np.floor(max_allocation / torch.mul(*x.size()).item()))
            y = y.item()
            input_tensors.setdefault(y, []).append(x.to(self.device))
        segments = {}
        for i, input_tensor in input_tensors.items():
            input_tensor = torch.stack(input_tensor)[:max_examples, :, :]
            attributions, delta = ig.attribute(input_tensor, target=i, return_convergence_delta=True)
            seg = attributions.mean(dim=0).sum(dim=-1).detach().cpu().numpy()
            segments[self.targets[i]] = np.abs(seg)
        return segments

    def plot_segment_importance(self, ax):
        segments = self.calc_segment_importance()
        for bug_speed, seg in segments.items():
            ax.plot(self.time_vector, seg, label=str(bug_speed), alpha=0.5)
        ax.plot(self.time_vector, np.mean(list(segments.values()), axis=0), label='Average', color='k', linewidth=3)

    def plot_attention(self, ax, attns):
        mean_att = []
        for bug_speed in self.targets:
            att = attns[bug_speed]
            att = np.vstack(att).mean(axis=0)
            att[:8] = np.nan
            mean_att.append(att)
            ax.plot(self.time_vector, att, label=f'{bug_speed}cm/sec', alpha=0.4)

        ax.plot(self.time_vector, np.vstack(mean_att).mean(axis=0), color='k')
        if self.strike_index:
            ax.axvline(0, color='k', linestyle='--')
        ax.set_xlabel('Time Around Strike [sec]')
        ax.set_ylabel('Attention values')
        ax.set_title('Attention')
        ax.legend()

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

    @property
    def time_vector(self):
        return self.sub_section[0] + np.arange(self.sub_section[1] + 1) * (1 / 60)


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
                               sub_section=(-1, 60), feature_names=('x', 'y', 'speed'), **kwargs):
    from sklearn.model_selection import ParameterGrid

    res_df = []
    grid = ParameterGrid(dict(dropout_prob=[0.2, 0.4, 0.6], lstm_layers=[1, 2, 3, 4], lstm_hidden_dim=[64, 128]))
    for i, params in enumerate(grid):
        if params['lstm_layers'] == 1:
            if params['dropout_prob'] != 0.4:
                continue
            else:
                params['dropout_prob'] = 0
        print(f'start loop {i+1}/{len(grid)}')
        tj = TrajClassifier(save_model_dir=TRAJ_DIR, is_shuffle_dataset=False, sub_section=sub_section, feature_names=feature_names,
                            is_debug=False, is_resample=is_resample, animal_id=animal_id, movement_type=movement_type, **kwargs, **params)
        tj.train(is_plot=False)
        best_i = np.argmin([x['score'] for x in tj.history])
        best_epoch = np.argmin(tj.history[best_i]['metrics']['val_loss'])
        for metric, l in tj.history[best_i]['metrics'].items():
            res_df.append({'metric': metric, 'value': l[best_epoch], 'animal_id': animal_id,
                           'movement_type': movement_type, **params})

        y_true, y_pred, y_score = tj.get_data_for_evaluation()
        acc = accuracy_score(y_true, y_pred)
        res_df.append({'metric': 'overall_accuracy', 'value': acc, 'animal_id': animal_id,
                       'movement_type': movement_type, 'model_path': tj.model_path, **params})

        print(f'{",".join([f"{k}:{v}" for k, v in params.items()])} - Overall Accuracy: {acc:.2f}')
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


def run_with_different_seeds(animal_id, movement_type, feature_names, sub_section=(-1, 60),
                             n=10, is_run_feature_importance=True, is_plot=True, is_save=True, **kwargs):
    res = {'metrics': [], 'ig': {}, 'ablation': []}
    for s in range(n):
        print(f'\n>>>>> Start iteration {s+1}/{n} for seed {s}...')
        tj = TrajClassifier(save_model_dir=TRAJ_DIR, feature_names=feature_names, seed=s, sub_section=sub_section,
                            animal_id=animal_id, movement_type=movement_type, **kwargs)
        tj.train(is_plot=False)
        tj.is_debug = False
        best_fold = tj.get_best_model()
        best_epoch = tj.history[best_fold]['chosen_epoch'] - 1
        for metric, l in tj.history[best_fold]['metrics'].items():
            res['metrics'].append({'metric': metric, 'value': l[best_epoch], 'animal_id': animal_id,
                                   'movement_type': movement_type, 'model_path': tj.model_path})

        y_true, y_pred, y_score = tj.get_data_for_evaluation()
        ig_segments = tj.calc_segment_importance()
        for bug_speed, seg in ig_segments.items():
            res['ig'].setdefault(bug_speed, []).append(seg)
        res['metrics'].append({'metric': 'overall_accuracy', 'value': accuracy_score(y_true, y_pred),
                               'animal_id': animal_id, 'movement_type': movement_type, 'model_path': tj.model_path})
        if is_run_feature_importance:
            af = []
            for start_t in np.linspace(tj.time_vector[0], tj.time_vector[-1] - 0.2, 3):
                seg = (start_t, start_t + 0.2)
                ablations_dict = tj.calc_ablation(segment=seg, ablate_all_except=True)
                ablations_dict['segment'] = np.mean(seg)
                af.append(ablations_dict)
            af = pd.DataFrame(af)
            af = af.set_index('segment').stack().reset_index().rename(columns={'level_1': 'feature', 0: 'ablation'})
            res['ablation'].append(af)

        torch.cuda.empty_cache()
        time.sleep(1)

    res['metrics'] = pd.DataFrame(res['metrics'])
    # stack the ig vectors to one matrix for each bug speed
    for bug_speed in res['ig'].keys():
        res['ig'][bug_speed] = np.vstack(res['ig'][bug_speed])
    # convert the ablations dicts into a DataFrame with columns for feature-name and ablation
    res['ablation'] = pd.concat(res['ablation'])

    if is_plot:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
        # plot all metrics as bars with errors
        sns.barplot(data=res['metrics'], x='metric', y='value', ax=axes[0])
        axes[0].set_xticks(axes[0].get_xticks(), axes[0].get_xticklabels(), rotation=45, ha='right')
        # plot ablation results as bars with errors
        sns.barplot(data=res['ablation'], x='segment', y='ablation', hue='feature', ax=axes[1])
        axes[1].set_xlabel('segment mid [sec]')
        # print ig curves for each bug speed and average ig curve
        mean_ig = []
        time_vector = sub_section[0] + np.arange(sub_section[1] + 1) * (1 / 60)
        for bug_speed, a in res['ig'].items():
            a_avg = a.mean(axis=0)
            mean_ig.append(a_avg)
            axes[2].plot(time_vector, a_avg.squeeze(), label=bug_speed, alpha=0.5)
        axes[2].plot(time_vector, np.vstack(mean_ig).mean(axis=0), color='k', linewidth=2)
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
    all_res = {}
    for t_start in np.arange(-10, (10-(span/60))+dt, dt):
        print(f'Start training with t_start: {t_start}')
        try:
            r = run_with_different_seeds(animal_id, movement_type,  ['x', 'y', 'speed'], sub_section=(t_start, span),
                                     lstm_layers=4, dropout_prob=0.3, lstm_hidden_dim=100, n=n_seeds,
                                     is_run_feature_importance=True, is_plot=False, is_save=False)
            all_res[t_start] = r
        except Exception as exc:
            print(f'Error in t_start: {t_start}; {exc}')

    filename = f'{TRAJ_DIR}/optimal_span_{datetime.now().isoformat()}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(all_res, f)
    print(f'\nResults saved to: {filename}')


if __name__ == '__main__':
    # tj = TrajClassifier(save_model_dir=TRAJ_DIR, is_shuffle_dataset=False, sub_section=(0, 60), is_resample=False,
    #                     animal_id='PV91', movement_type='random_low_horizontal', is_hit=False, lstm_layers=4,
    #                     dropout_prob=0.3, lstm_hidden_dim=50, is_shuffled_target=True, is_iti=True)
    # tj.train(is_plot=True)
    # tj.check_hidden_states()

    hyperparameters_comparison(animal_id='PV42', movement_type='random_low_horizontal', monitored_metric='val_loss', monitored_metric_algo='min',
                               feature_names=['x', 'y', 'speed_x', 'speed_y'], sub_section=(-1, 60))

    # find_optimal_span(animal_id='PV163', movement_type='random_low_horizontal')

    # for animal_id in ['PV95', 'PV99', 'PV80']:  # ['PV42', 'PV91', 'PV95', 'PV99', 'PV80']
    #     # for sub_section in [(-2, 60), (-1.5, 60), (-1, 60), (0, 59)]:
    #     for sub_section in [(-1, 60), (-1, 119)]:
    #         try:
    #             run_with_different_seeds(animal_id, 'random_low_horizontal',
    #                                      ['x', 'y', 'speed_x', 'speed_y'], n=30, is_shuffled_target=False,
    #                                      is_resample=False, lstm_layers=2, dropout_prob=0.4, lstm_hidden_dim=64, num_epochs=150,
    #                                      sub_section=sub_section, monitored_metric='val_loss', monitored_metric_algo='min')
    #         except Exception as exc:
    #             print(f'Error in animal_id: {animal_id}; {exc}')


    # find_best_hyperparameters(movement_type='random_low_horizontal', lstm_layers=4, dropout_prob=0.3, is_resample=False)

    # find_best_features(movement_type='circle', lstm_layers=6, dropout_prob=0.5, is_resample=True)
    # for movement_type in ['random_low_horizontal', 'circle']:
    #     for animal_id in ['PV91', 'PV163', 'PV99', 'PV80', 'PV85', 'all']:  # 'PV42',
    #         hyperparameters_comparison(animal_id=animal_id, movement_type=movement_type,
    #                                    feature_names=['x', 'y', 'speed'], sub_section=(-1, 60))
    # animals_comparison()
    # plot_comparison('/Users/regev/PhD/msi/Pogona_Pursuit/output/datasets/trajectories/results_2024-06-24T16:43:58.880310.csv')

    # tj = TrajClassifier(
    #     model_path='/media/sil2/Data/regev/datasets/trajs/traj_classifier_all_random_low_horizontal/20240722_105828')
    # tj.check_hidden_states()