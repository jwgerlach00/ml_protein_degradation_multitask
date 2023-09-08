import numpy as np
import torch
from typing import Tuple, Optional, Iterable, Union, List
from abc import ABC, abstractmethod
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, f1_score, r2_score
from linkerology_multitask import LABEL_MAPPER


class _Trainer(ABC):
    POSSIBLE_METRICS = ['accuracy', 'f1', 'r2']

    def __init__(self, multitask_model, num_targets:int, dataset_metadata:dict) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = multitask_model
        self.model.to(self.device)
    
        self.num_targets = num_targets
        self.dataset_metadata = dataset_metadata
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []
        self.train_f1_history = []
        self.val_f1_history = []

    @abstractmethod
    def train() -> None:
        pass
    
    @abstractmethod
    def loss_fn(): # what is returned varies
        pass

    @staticmethod
    def _mask(Y_p_i:torch.Tensor, Y_i:torch.Tensor, mask_label:Union[float, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        if np.isnan(mask_label):
            mask = ~torch.isnan(Y_i)
        else:
            mask = (Y_i != mask_label)
        masked_Y_p_i = Y_p_i[mask]
        masked_Y_i = Y_i[mask]
        return masked_Y_p_i, masked_Y_i
    
    @staticmethod
    def _calc_loss_for_one_target(masked_Y_p_i, masked_Y_i, criterion) -> Optional[np.ndarray]:
        loss = criterion(masked_Y_p_i, masked_Y_i)
        if not torch.isnan(loss):
            return loss
        return None

    @staticmethod
    def _calc_accuracy_for_one_target(masked_Y_p_i, masked_Y_i) -> Optional[np.ndarray]:
        masked_Y_p_i_class_labels = (masked_Y_p_i >= 0.5).int()
        num_valid = masked_Y_i.size(0)
        if num_valid > 0:
            return (masked_Y_p_i_class_labels == masked_Y_i).sum().item() / num_valid
        return None
    
    @staticmethod
    def _calc_metrics_for_one_target(masked_Y_p_i, masked_Y_i, metrics:List[str]) -> Optional[List[np.ndarray]]:
        for m in metrics:
            if m not in _Trainer.POSSIBLE_METRICS:
                raise ValueError(f'{m} is not a valid metric. Use only metrics in {_Trainer.POSSIBLE_METRICS}')

        num_valid = masked_Y_i.size(0)

        if num_valid > 0:
            out = []
            if 'accuracy' in metrics:
                out.append(accuracy_score(masked_Y_i.detach().cpu(), masked_Y_p_i.detach().cpu()))
            if 'f1' in metrics:
                out.append(f1_score(masked_Y_i.detach().cpu(), masked_Y_p_i.detach().cpu()))
            if 'r2' in metrics:
                # print(masked_Y_i.detach().cpu(), masked_Y_p_i.detach().cpu())
                # print(masked_Y_i.detach().cpu())
                out.append(r2_score(masked_Y_i.detach().cpu(), masked_Y_p_i.detach().cpu()))
            return out
        return None
    
    @staticmethod
    def _stack_per_target_history(history:np.ndarray, entry:np.ndarray, take_nanmean:bool) -> np.ndarray:
        entry = np.nanmean(entry, axis=0) if take_nanmean else entry
        return entry if history is None else np.vstack((history, entry))
    
    def plot_loss_history(self, title:str, y_label:str, save_name:str):
        return self._plot_train_vs_val(self.train_loss_history, self.val_loss_history, title, y_label, save_name)
    
    def plot_accuracy_history(self, title:str, y_label:str, save_name:str):
        return self._plot_train_vs_val(self.train_accuracy_history, self.val_accuracy_history, title, y_label, save_name)
    
    def plot_f1_history(self, title:str, y_label:str, save_name:str):
        return self._plot_train_vs_val(self.train_f1_history, self.val_f1_history, title, y_label, save_name)
    
    @staticmethod
    def _plot_train_vs_val(train_data:Iterable[float], val_data:Iterable[float], title:str, y_label:str, save_name:str):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(train_data))), y=train_data, name='train', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=list(range(len(val_data))), y=val_data, name='validation', line=dict(color='red')))

        fig.update_layout(title=title, xaxis_title='Epoch', yaxis_title=y_label, title_x=0.5, title_y=0.9)
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))

        fig.write_image(f'{save_name}.png')
        return fig

    def plot_per_target_history(self, per_target_history:np.ndarray, title:str, y_label:str, save_name=Optional[str]):
        target_labels = [LABEL_MAPPER[target] for target in self.dataset_metadata['targets']]
        fig = go.Figure()
        
        # Plot for each target
        for i in range(self.num_targets):
            target_val_loss = per_target_history[:, i]
            fig.add_trace(go.Scatter(x=list(range(len(target_val_loss))), y=target_val_loss, name=target_labels[i]))
        
        fig.update_layout(
            title=title,
            title_x=0.5,
            legend=dict(x=1, y=0.5),
            plot_bgcolor='white',
            xaxis_title='Epoch',
            yaxis_title=y_label
        )

        fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

        if save_name:
            fig.write_image(f'{save_name}.png')
        
        return fig
