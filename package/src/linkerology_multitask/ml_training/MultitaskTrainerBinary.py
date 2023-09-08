import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from typing import Tuple, List
from linkerology_multitask.ml_training.MultitaskNNDatasets import MultitaskNNDataset, ComponentMultitaskNNDataset
from linkerology_multitask.ml_training._Trainer import _Trainer
from sklearn import metrics
import joblib
from random import randint


class MultitaskTrainerBinary(_Trainer):
    def __init__(self, multitask_model, num_targets:int, dataset_metadata:dict) -> None:
        super(MultitaskTrainerBinary, self).__init__(multitask_model, num_targets, dataset_metadata)
        self.component = dataset_metadata['component']

        # Hyperparams
        self.learning_rate = 0.0001
        # self.learning_rate = 0.00001
        self.weight_decay = 1e-4
        self.num_epochs = 200
        # self.num_epochs = 200
        self.batch_size = 64

        self.__base_criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                          weight_decay=self.weight_decay)

        self.per_target_train_accuracy_history = None
        self.per_target_train_f1_history = None
        self.per_target_val_accuracy_history = None
        self.per_target_val_f1_history = None

        self.best_model_state_dict = None

    def loss_fn(self, Y_p:torch.Tensor, Y:torch.Tensor, fill_random:bool):
        if fill_random:
            Y_filled = self.fill_random_blanks(Y)

        losses = []
        for i in range(self.num_targets):
            masked_Y_p_i, masked_Y_i = self._mask(Y_p[:, i], Y[:, i], -100)
            if fill_random:
                fill_masked_Y_p_i, fill_masked_Y_i = self._mask(Y_p[:, i], Y_filled[:, i], -100)
                loss = self._calc_loss_for_one_target(fill_masked_Y_p_i, fill_masked_Y_i, self.__base_criterion)
            else:
                loss = self._calc_loss_for_one_target(masked_Y_p_i, masked_Y_i, self.__base_criterion)

            if loss is not None:
                losses.append(loss)
                
        agg_loss = torch.sum(torch.stack(losses)) if losses else None
        return agg_loss
    
    def calc_metrics(self, X_full:np.ndarray, Y_full:np.ndarray) -> Tuple[float, List[float], float,
                                                                                        List[float]]:
        if self.component:
            X_full = X_full.reshape((X_full.shape[0], -1))
        Y_p = self.model(torch.tensor(X_full, device=self.device).float()).detach().cpu()
        masked_Y_p, masked_Y = self._mask(Y_p, torch.tensor(Y_full).detach(), -100)
        masked_Y_p = masked_Y_p.numpy()
        masked_Y_p = (masked_Y_p >= 0.5).astype(int) # threshold
        masked_Y = masked_Y.numpy()
        overall_accuracy = metrics.accuracy_score(masked_Y, masked_Y_p) # might want to just take mean of all the per targets to compare to the single target model
        overall_f1 = metrics.f1_score(masked_Y, masked_Y_p)

        per_target_accuaracy = []
        per_target_f1 = []
        for target_i in range(Y_full.shape[1]):
            _Y = Y_full[:, target_i]
            _Y_p = Y_p[:, target_i]
            _Y_p, _Y = self._mask(_Y_p, torch.tensor(_Y).detach(), -100)
            _Y_p = _Y_p.numpy()
            _Y_p = (_Y_p >= 0.5).astype(int) # threshold
            _Y = _Y.numpy()
            per_target_accuaracy.append(metrics.accuracy_score(_Y, _Y_p))
            per_target_f1.append(metrics.f1_score(_Y, _Y_p))
        
        return overall_accuracy, per_target_accuaracy, overall_f1, per_target_f1
        
    def fill_random_blanks(self, Y:torch.Tensor, random_num:int=1) -> torch.Tensor:
        Y = torch.clone(Y)
        for r_i in range(Y.shape[0]):
            blank_indexes = torch.where(Y[r_i, :] == -100)[0].tolist()
            random_blank_indexes = [blank_indexes.pop(randint(0, len(blank_indexes) - 1)) for _ in range(random_num)]
            Y[r_i, random_blank_indexes] = 0
        return Y

    def train(self, X_train:np.ndarray, X_val:np.ndarray, Y_train:np.ndarray, Y_val:np.ndarray) -> None:
        Dataset = ComponentMultitaskNNDataset if self.component else MultitaskNNDataset
        train_loader = torch.utils.data.DataLoader(Dataset(X_train, Y_train, self.device), batch_size=self.batch_size)
        val_loader = torch.utils.data.DataLoader(Dataset(X_val, Y_val, self.device), batch_size=self.batch_size)

        for epoch in range(self.num_epochs):
            print(f'Epoch: {epoch}')
            
            print('training...')
            self.model.train()
            _train_loss_history = []
            per_target_train_accuracy = None
            per_target_train_f1 = None
            for _X_train, _Y_train in tqdm(train_loader):

                # Make predictions
                self.optimizer.zero_grad()
                _Y_train_p = self.model(_X_train)

                loss = self.loss_fn(_Y_train_p, _Y_train, fill_random=False)

                if loss is not None:
                    loss.backward()
                    _train_loss_history.append(loss.item())

                # Optimizer step
                self.optimizer.step()

            train_accuracy, per_target_train_accuracy, train_f1, per_target_train_f1 = self.calc_metrics(X_train, Y_train)
            # Apppend detached loss to history
            self.train_loss_history.append(sum(_train_loss_history) / len(_train_loss_history))
            # Append overall metrics to history
            self.train_accuracy_history.append(train_accuracy)
            self.train_f1_history.append(train_f1)
            # Append per target metrics to history
            self.per_target_train_accuracy_history = self._stack_per_target_history(self.per_target_train_accuracy_history, per_target_train_accuracy, take_nanmean=False)
            self.per_target_train_f1_history = self._stack_per_target_history(self.per_target_train_f1_history, per_target_train_f1, take_nanmean=False)

            print('validating...')
            self.model.eval()
            _val_loss_history = []
            per_target_val_accuracy = None
            per_target_val_f1 = None
            with torch.no_grad():
                for _X_val, _Y_val in tqdm(val_loader):
                    _Y_val_p = self.model(_X_val)
                    loss = self.loss_fn(_Y_val_p, _Y_val, fill_random=False)

                    if loss is not None:
                        _val_loss_history.append(loss.item())

            val_accuracy, per_target_val_accuracy, val_f1, per_target_val_f1 = self.calc_metrics(X_val, Y_val)
            # Append detached loss to history
            self.val_loss_history.append(sum(_val_loss_history) / len(_val_loss_history))
            # Append overall metrics to history
            self.val_accuracy_history.append(val_accuracy)
            self.val_f1_history.append(val_f1)
            # Append per target metrics to history
            self.per_target_val_accuracy_history = self._stack_per_target_history(self.per_target_val_accuracy_history, per_target_val_accuracy, take_nanmean=False)
            self.per_target_val_f1_history = self._stack_per_target_history(self.per_target_val_f1_history, per_target_val_f1, take_nanmean=False)

            if min(self.val_loss_history) == self.val_loss_history[-1]:
                self.best_model_state_dict = self.model.state_dict()

        self.plot()

    def plot(self) -> None:
        self.plot_loss_history('Summed Classification Multitask BCE Loss', 'BCE Loss', 'binary_loss')
        self.plot_accuracy_history('Binary Classification Multitask Accuracy', 'Accuracy', 'binary_accuracy')
        self.plot_f1_history('Binary Classification Multitask F1', 'F1 Score', 'binary_f1')
        self.plot_per_target_history(self.per_target_train_accuracy_history, 'Per Target Train Accuracy', 'Accuracy',
                                     'binary_per_target_train_accuracy')
        self.plot_per_target_history(self.per_target_val_accuracy_history, 'Per Target Validation Accuracy', 'Accuracy',
                                     'binary_per_target_val_accuracy')
        self.plot_per_target_history(self.per_target_train_f1_history, 'Per Target Train F1', 'F1 Score',
                                     'binary_per_target_train_f1')
        self.plot_per_target_history(self.per_target_val_f1_history, 'Per Target Validation F1', 'F1 Score',
                                     'binary_per_target_val_f1')


if __name__ == '__main__':
    from package.src.linkerology_multitask.ml_models.BinaryMultitask import MultitaskNNBinary
    from linkerology_multitask.ml_models import BinaryMultitask
    from linkerology_multitask import LABEL_MAPPER
    import os
    import json
    import random


    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    directory = 'DATA/processed/40nm_parent_ECFP12_2048bit'

    X_train = np.load(f'{directory}/linkerology_multitask_X_train.npy')
    X_test = np.load(f'{directory}/linkerology_multitask_X_test.npy')
    Y_train = np.load(f'{directory}/linkerology_multitask_Y_train.npy')
    Y_test = np.load(f'{directory}/linkerology_multitask_Y_test.npy')

    with open(os.path.join(directory, 'hyperparams.json'), 'r') as f:
        dataset_metadata=json.load(f)

    num_targets = Y_test.shape[1]

    bit_vector_length = X_train.shape[1]*X_train.shape[2] if dataset_metadata['component'] else X_train.shape[1]

    model = BinaryMultitask(bit_vector_length=bit_vector_length, num_targets=num_targets, device=torch.device('cuda'))
    trainer = MultitaskTrainerBinary(model, num_targets=num_targets, dataset_metadata=dataset_metadata)

    out_dir = f'output/{directory.split("/")[-1]}_{trainer.batch_size}_{trainer.num_epochs}_'\
        f'{str(trainer.learning_rate).split(".")[-1]}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    os.chdir(out_dir)

    trainer.train(X_train, X_test, Y_train, Y_test)
    min_loss_index = np.array(trainer.val_loss_history).argmin()
    print(min_loss_index)
    print(max(trainer.val_accuracy_history))
    # print(trainer.per_target_val_accuracy_history)
    print(trainer.per_target_val_accuracy_history[min_loss_index])
    print([LABEL_MAPPER[target] for target in trainer.dataset_metadata['targets']])

    print(trainer.best_model_state_dict)

    joblib.dump(trainer, 'trainer.joblib')