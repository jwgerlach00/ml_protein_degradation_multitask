import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from typing import Tuple
from linkerology_multitask.ml_training.MultitaskNNDatasets import MultitaskNNDataset
from linkerology_multitask.ml_training._Trainer import _Trainer
from sklearn.metrics import r2_score


class MultitaskTrainerReg(_Trainer):
    def __init__(self, multitask_model, num_targets:int, dataset_metadata:dict) -> None:
        super(MultitaskTrainerReg, self).__init__(multitask_model, num_targets, dataset_metadata)

        # Hyperparams
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.num_epochs = 200 # 1000
        self.batch_size = 64

        self.__base_criterion = nn.MSELoss()
        # self.__base_criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.per_target_train_detached_loss_history = None
        self.per_target_val_detached_loss_history = None

        self.best_model_state_dict = None


    def loss_fn(self, Y_p:torch.Tensor, Y:torch.Tensor) -> Tuple[torch.Tensor, np.ndarray, float, np.ndarray]:
        per_target_detached_loss = np.array([])
        per_target_r2_score = np.array([])
        losses = []
        r2_scores = []
        for i in range(self.num_targets):
            masked_Y_p_i, masked_Y_i = self._mask(Y_p[:, i], Y[:, i], np.nan)
            loss = self._calc_loss_for_one_target(masked_Y_p_i, masked_Y_i, self.__base_criterion)
            metrics = self._calc_metrics_for_one_target(masked_Y_p_i, masked_Y_i, ['r2'])

            if loss is not None:
                losses.append(loss)

            r2 = metrics[0] if metrics else None
            if r2 is not None:
                r2_scores.append(r2)

            per_target_detached_loss = np.append(per_target_detached_loss, (loss.item() if loss is not None else np.nan))
            per_target_r2_score = np.append(per_target_r2_score, (r2 if r2 is not None else np.nan))

        agg_loss = torch.mean(torch.stack(losses)) if losses else None
        agg_r2 = float(np.nanmean(r2_scores)) if r2_scores else None # nan because not well defined w/ < 2 samples
        return agg_loss, per_target_detached_loss, agg_r2, per_target_r2_score

    def train(self, X_train:np.ndarray, X_val:np.ndarray, Y_train:np.ndarray, Y_val:np.ndarray) -> None:
        train_loader = torch.utils.data.DataLoader(MultitaskNNDataset(X_train, Y_train, self.device), batch_size=self.batch_size)
        val_loader = torch.utils.data.DataLoader(MultitaskNNDataset(X_val, Y_val, self.device), batch_size=self.batch_size)

        for epoch in range(self.num_epochs):
            print(f'Epoch: {epoch}')
            
            print('training...')
            self.model.train()
            _train_loss_history = []
            _r2_loss_history = []
            _per_target_train_detached_loss_history = None
            for _X_train, _Y_train in tqdm(train_loader):

                # Make predictions
                self.optimizer.zero_grad()
                _Y_train_p = self.model(_X_train)

                loss, per_target_detached_loss, _, per_target_r2_score = self.loss_fn(_Y_train_p, _Y_train)
                # loss, accuracy, per_target_accuracy = self.calc_loss_and_accuracy(_Y_train_p, _Y_train, return_per_target_accuracies=True)

                if loss is not None:
                    loss.backward()
                    _train_loss_history.append(loss.item())

                # if r2_score is not None:
                #     _r2_loss_history.append(r2_score)

                # Optimizer step
                self.optimizer.step()

                _per_target_train_detached_loss_history = self._stack_per_target_history(_per_target_train_detached_loss_history, per_target_detached_loss, take_nanmean=False)

            self.full_r2_score(X_train, Y_train)
                # print(sum(_r2_loss_history) / len(_r2_loss_history))

            #     if accuracy is not None:
            #         _train_accuracy_history.append(accuracy)

            print('validating...')
            self.model.eval()
            _val_loss_history = []
            _per_target_val_detached_loss_history = None
            with torch.no_grad():
                for _X_val, _Y_val in tqdm(val_loader):
                    _Y_val_p = self.model(_X_val)
                    loss, per_target_detached_loss, _, per_target_r2_score = self.loss_fn(_Y_val_p, _Y_val)
                    # loss, accuracy, per_target_accuracy = self.calc_loss_and_accuracy(_Y_val_p, _Y_val, return_per_target_accuracies=True)

                    if loss is not None:
                        _val_loss_history.append(loss.item())

                    _per_target_val_detached_loss_history = self._stack_per_target_history(_per_target_val_detached_loss_history, per_target_detached_loss, take_nanmean=False)

            #         if accuracy is not None:
            #             _val_accuracy_history.append(accuracy) # for normal plotting against training
            self.full_r2_score(X_val, Y_val)
                    
            self.train_loss_history.append(sum(_train_loss_history) / len(_train_loss_history))
            self.val_loss_history.append(sum(_val_loss_history) / len(_val_loss_history))

            self.per_target_train_detached_loss_history = self._stack_per_target_history(self.per_target_train_detached_loss_history, _per_target_train_detached_loss_history, take_nanmean=True)
            self.per_target_val_detached_loss_history = self._stack_per_target_history(self.per_target_val_detached_loss_history, _per_target_val_detached_loss_history, take_nanmean=True)

            if min(self.val_loss_history) == self.val_loss_history[-1]:
                self.best_model_state_dict = self.model.state_dict()
        
        self.plot()
    
    def plot(self) -> None:
        self.plot_loss_history('Summed Regression Multitask MSE Loss', 'MSE Loss', 'regression_loss')
        self.plot_per_target_history(self.per_target_train_detached_loss_history, 'Per Target Train MSE', 'MSE', 'per_target_train_mse')
        self.plot_per_target_history(self.per_target_val_detached_loss_history, 'Per Target Validation MSE', 'MSE', 'per_target_val_mse')

    def full_r2_score(self, full_X:np.ndarray, full_Y:np.ndarray):
        Y_p = self.model(torch.tensor(full_X, device=self.device).float()).detach().cpu()
        masked_Y_p, masked_Y = self._mask(Y_p, torch.tensor(full_Y).detach(), np.nan)
        masked_Y_p = masked_Y_p.numpy()
        masked_Y = masked_Y.numpy()
        print(r2_score(masked_Y, masked_Y_p))



if __name__ == '__main__':
    from linkerology_multitask.ml_models import MultitaskNNReg
    import os
    import json
    import joblib
    import random

    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    directory = 'DATA/processed/reg_parent_ECFP12_2048bit'

    X_train = np.load(f'{directory}/linkerology_multitask_X_train.npy')
    X_test = np.load(f'{directory}/linkerology_multitask_X_test.npy')
    Y_train = np.load(f'{directory}/linkerology_multitask_Y_train.npy')
    Y_test = np.load(f'{directory}/linkerology_multitask_Y_test.npy')

    # scaler = StandardScaler()
    # Y_train = scaler.fit_transform(Y_train)
    # Y_test = scaler.transform(Y_test)

    with open(os.path.join(directory, 'hyperparams.json'), 'r') as f:
        dataset_metadata=json.load(f)

    num_targets = Y_test.shape[1]

    model = MultitaskNNReg(bit_vector_length=X_train.shape[1], num_targets=num_targets, device=torch.device('cuda'))
    trainer = MultitaskTrainerReg(model, num_targets=num_targets, dataset_metadata=dataset_metadata)

    out_dir = f'output/{directory.split("/")[-1]}_{trainer.batch_size}_{trainer.num_epochs}_{str(trainer.learning_rate).split(".")[-1]}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    os.chdir(out_dir)

    trainer.train(X_train, X_test, Y_train, Y_test)
    joblib.dump(trainer, 'trainer_test.joblib')
    # print(max(trainer.val_accuracy_history))
    # print(trainer.per_target_val_accuracy_history)