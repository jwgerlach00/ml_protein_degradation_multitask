import torch
from torch import nn
import joblib
import numpy as np
import copy
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from typing import List
from plotly import graph_objects as go
from linkerology_multitask import LABEL_MAPPER


class SavedTrainerPlotter:
    def __init__(self, best_model:nn.Module, X_train:np.ndarray, X_val:np.ndarray, Y_train:np.ndarray, Y_val:np.ndarray,
                 targets:List[str], component:bool) -> None:
        self.__device = torch.device('cpu') # Use CPU for ease of moving tensors to numpy
        self.model = best_model.to(self.__device)
        self.__targets = targets

        self.X_train = X_train.copy()
        self.X_val = X_val.copy()
        if component:
            self.X_train = self._X_component(self.X_train)
            self.X_val = self._X_component(self.X_val)

        self.Y_train = Y_train.copy()
        self.Y_val = Y_val.copy()

        self.Y_train_p = self._predict(self.X_train)
        self.Y_val_p = self._predict(self.X_val)

    @staticmethod
    def _X_component(X:np.ndarray) -> np.ndarray:
        """
        Flattens X if PROTAC component multitask dataset is being used. Use prior to feeding component NN model.

        :param X: ECFP matrix
        :type X: np.ndarray
        :return: Flattened matrix
        :rtype: np.ndarray
        """
        return X.reshape((X.shape[0], -1))

    def _predict(self, X:np.ndarray) -> np.ndarray:
        """
        Predicts on X using self.model and casts to Numpy.

        :param X: Shaped ECFP array
        :type X: np.ndarray
        :return: Predictions array
        :rtype: np.ndarray
        """
        return self.model(torch.tensor(X).float()).detach().numpy()
    
    def plot_roc(self) -> go.Figure:
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        flat_axes = axes.flat
        for i in range(self.Y_val.shape[1]):

            for j in ['train', 'val']:
                Y = getattr(self, f'Y_{j}')
                Y_p = getattr(self, f'Y_{j}_p')

                filter = Y[:, i] != -100

                fpr, tpr, thresholds = roc_curve(Y[:, i][filter], Y_p[:, i][filter])
                auc_score = auc(fpr, tpr)

                flat_axes[i].plot(fpr, tpr, label=f'{j} AUC = {auc_score:.2f}', color=('blue' if j == 'train' else 'red'))

            flat_axes[i].grid()
            flat_axes[i].set_title(f'ROC {LABEL_MAPPER[self.__targets[i]]}', fontsize=12)
            
            flat_axes[i].plot([0, 1], [0, 1], 'k--')  # Diagonal line for random classifier
            flat_axes[i].set_xlabel('False Positive Rate')
            flat_axes[i].set_ylabel('True Positive Rate')
            flat_axes[i].legend(loc='lower right')

        plt.delaxes(flat_axes[-1])
        plt.tight_layout()
        return fig


if __name__ == '__main__':
    import json


    directory = 'DATA/processed/40nm_parent_ECFP12_2048bit'
    # directory = 'DATA/component_binary_processed_data_ecfp12_2048'

    X_train = np.load(f'{directory}/linkerology_multitask_X_train.npy')
    Y_train = np.load(f'{directory}/linkerology_multitask_Y_train.npy')
    X_test = np.load(f'{directory}/linkerology_multitask_X_test.npy')
    Y_test = np.load(f'{directory}/linkerology_multitask_Y_test.npy')
    with open(f'{directory}/hyperparams.json') as f:
        targets = json.load(f)['targets']

    trainer = joblib.load('output/40nm_parent_ECFP12_2048bit_64_100_0001/trainer.joblib')
    # trainer = joblib.load('component_binary_processed_out_64/trainer_test.joblib')
    model = copy.deepcopy(trainer.model)
    model.load_state_dict(copy.deepcopy(trainer.best_model_state_dict))

    mm = SavedTrainerPlotter(model, X_train, X_test, Y_train, Y_test, targets, component=True)
    mm.plot_roc()
