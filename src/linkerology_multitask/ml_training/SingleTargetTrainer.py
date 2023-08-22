import torch
import pandas as pd
from collections import OrderedDict
import numpy as np
from typing import Optional, Tuple
import copy
import joblib
import os
from linkerology_multitask.ml_training import MultitaskTrainerBinary
from linkerology_multitask.ml_models import BinaryMultitask
from linkerology_multitask import LABEL_MAPPER


class SingleTargetTrainer:
    def __init__(self) -> None:
        self.per_target_accuracy_df = None

    def train_all_single(self, X_train:np.ndarray, X_val:np.ndarray, Y_train:np.ndarray, Y_val:np.ndarray,
                         dataset_metadata:dict, device:torch.device, load_pretrained_model:bool=False,
                         model_state_dict_for_pretrain:Optional[OrderedDict]=None, freeze_layers:bool=False) -> None:
        # """
        # Iterates through each protein target using the target dimension of Y_train and Y_val and trains the multitask \
        #     model with each individually. Although the multitask model is used for consistency, this is not a multitask \
        #         model since there is only one target for each independent model.

        # :param X_train: Train ECFP feature set
        # :type X_train: np.ndarray
        # :param X_val: Validation ECFP feature set
        # :type X_val: np.ndarray
        # :param Y_train: Train binary labels entries (dim0) over both protein target (dim1) and organism (dim2)
        # :type Y_train: np.ndarray
        # :param Y_val: Validation binary labels entries (dim0) over both protein target (dim1) and organism (dim2)
        # :type Y_val: np.ndarray
        # :param targets: List of targets cooresponding to order of dim1 in Y_train and Y_val
        # :type targets: List[str]
        # :param device: Torch device to train on
        # :type device: torch.device
        # :param freeze_model: Whether to freeze the first 2 layers of model, defaults to False
        # :type freeze_model: bool
        # :param model_state_dict_for_freeze: Parameters to load on model if freeze_model is True, defaults to None
        # :type model_state_dict_for_freeze: Optional[OrderedDict]
        # :raises ValueError: Target dimension (dim1) doesn't match between train and val
        # :raises ValueError: freeze_model is True but model_state_dict is None
        # """
        if Y_train.shape[1] != Y_val.shape[1]:
            raise ValueError('Y_train and Y_test must have the same size target dimension but Y_train has '
                            f'{Y_train.shape[1]} and Y_test has {Y_val.shape[1]}')
        if freeze_layers and not load_pretrained_model:
            raise ValueError('freeze_layers cannot be True while load_pretrained_model is False.')
        if load_pretrained_model and (model_state_dict_for_pretrain is None):
            raise ValueError('If load_pretrained_model is True then model_state_dict_for_pretrain must not be None.')
        
        self.per_target_accuracy_df = pd.DataFrame({'metric': ['accuracy', 'f1']})
        for target_i in range(Y_train.shape[1]):
            target = LABEL_MAPPER[dataset_metadata['targets'][target_i]]
            print(f'target #{target_i+1}: {target}')

            Y_train_i = Y_train[:, target_i]
            Y_train_i = Y_train_i[:, np.newaxis]
            Y_val_i = Y_val[:, target_i]
            Y_val_i = Y_val_i[:, np.newaxis]
            
            if load_pretrained_model:
                model = self._pretrained_single_target_model(X_train.shape[1], device, model_state_dict_for_pretrain,
                                                             target_i, freeze_layers=freeze_layers)
            else:
                model = self._get_single_target_model(X_train.shape[1], device, target_i)
            # Instantiate Trainer object
            trainer = MultitaskTrainerBinary(model, num_targets=1, dataset_metadata=dataset_metadata)
            trainer.num_epochs = 150
            # Directory to store outputs from trainer
            out_dir = f'{target}'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            os.chdir(out_dir)
            # Train
            trainer.train(X_train, X_val, Y_train_i, Y_val_i)

            # per_target_accuracy_df
            val_accuracy, val_f1 = self.get_per_target_val_metrics(trainer)
            self.per_target_accuracy_df[target] = [val_accuracy, val_f1]

            joblib.dump(trainer, 'trainer.joblib')

            os.chdir('..')

    @staticmethod
    def get_per_target_val_metrics(trainer:MultitaskTrainerBinary) -> Tuple[np.ndarray, np.ndarray]:
        min_loss_index = np.array(trainer.val_loss_history).argmin()
        return (
            trainer.per_target_val_accuracy_history[min_loss_index][0],
            trainer.per_target_val_f1_history[min_loss_index][0]
        )

    @staticmethod
    def _pretrained_single_target_model(bit_vector_length:int, device:torch.device,
                                        model_state_dict:Optional[OrderedDict], target_index:int,
                                        freeze_layers:bool) -> BinaryMultitask:
        model_state_dict = SingleTargetTrainer._convert_multi_state_dict_to_single(model_state_dict, target_index)
        model = SingleTargetTrainer._get_single_target_model(bit_vector_length, device, target_index)
        model.load_state_dict(model_state_dict)
        if freeze_layers:
            SingleTargetTrainer._freeze_first_two_layers(model)
        return model
    
    @staticmethod
    def _get_single_target_model(bit_vector_length:int, device:torch.device, target_index:int) -> BinaryMultitask:
        return BinaryMultitask(bit_vector_length=bit_vector_length, num_targets=1, device=device,
                               out_linear_layer_offset=target_index)
    
    @staticmethod
    def _convert_multi_state_dict_to_single(model_state_dict:OrderedDict, target_index:int) -> OrderedDict:
        target_state_dict = copy.deepcopy(model_state_dict)
        # Remove all but the layer equal to the target index
        remove_from_target_state_dict = list(range(0, 8))
        remove_from_target_state_dict.pop(target_index)
        for i in remove_from_target_state_dict:
            layer_name = f'out_linear_layer_1_{i}'
            del target_state_dict[f'{layer_name}.weight']
            del target_state_dict[f'{layer_name}.bias']
        return target_state_dict

    @staticmethod
    def _freeze_first_two_layers(model:BinaryMultitask) -> None:
        """
        Freezes linear1 and linear2 of model inplace.

        :param model: Model with loaded parameters
        :type model: nn.Module
        """
        # Freeze first two layers
        for param in model.linear1.parameters():
            param.requires_grad = False
        for param in model.linear2.parameters():
            param.requires_grad = False
