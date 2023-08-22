import numpy as np
from typing import List, Optional
from plotly import graph_objects as go
from linkerology_multitask import LABEL_MAPPER


class ProcessedDataPlotter:
    def __init__(self, train_arr:np.ndarray, val_arr:np.ndarray, target_columns:List[str],
                 dataset_label:Optional[str]=None) -> None:
        self.train_arr = np.copy(train_arr)
        self.val_arr = np.copy(val_arr)
        self.dataset_label = dataset_label
        self.train_counts = (train_arr != -100).sum(axis=0)
        self.val_counts = (val_arr != -100).sum(axis=0)
        self.target_labels = [LABEL_MAPPER[target] for target in target_columns]

    @property
    def train_distribution_pie(self) -> go.Figure:
        fig = go.Figure(data=[go.Pie(values=self.train_counts, labels=self.target_labels, sort=False)])
        fig.update_traces(hoverinfo='label+percent', textinfo='value')
        fig.update_layout(
            title=f'Train Distribution: {sum(self.train_counts)} samples',
            title_x=0.5,
            height=600,
            width=800
        )
        return fig
    
    @property
    def train_activity_bar(self) -> go.Figure:
        out_train = [(np.sum(x == 0), np.sum(x == 1)) for x in [self.train_arr[:, i] for i in 
                                                                range(self.train_arr.shape[1])]]

        fig = go.Figure(data=[
            go.Bar(name='0s', x=self.target_labels, y=[i[0] for i in out_train]),
            go.Bar(name='1s', x=self.target_labels, y=[i[1] for i in out_train])
        ])

        fig.update_layout(
            barmode='stack',
            xaxis_title='Arrays',
            yaxis_title='Count',
            height=600,
            width=800,
            title=(f'{self.dataset_label} ' if self.dataset_label else '') + 'Train Activity Distribution',
            title_x=0.5
        )
        return fig
    
    @property
    def val_distribution_pie(self) -> go.Figure:
        fig = go.Figure(data=[go.Pie(values=self.val_counts, labels=self.target_labels, sort=False)])

        fig.update_traces(
            hoverinfo='label+percent',
            textinfo='value'
        )

        fig.update_layout(
            title=(f'{self.dataset_label} ' if self.dataset_label else '') + \
                f'Validation Distribution: {sum(self.val_counts)} samples',
            title_x=0.5,
            height=600,
            width=800
        )
        return fig
    
    @property
    def val_activity_bar(self) -> go.Figure:
        out_val = [(np.sum(x == 0), np.sum(x == 1)) for x in [self.val_arr[:, i] for i in range(self.val_arr.shape[1])]]

        fig = go.Figure(data=[
            go.Bar(name='0s', x=self.target_labels, y=[i[0] for i in out_val]),
            go.Bar(name='1s', x=self.target_labels, y=[i[1] for i in out_val])
        ])

        fig.update_layout(
            barmode='stack',
            xaxis_title='Arrays',
            yaxis_title='Count',
            height=600,
            width=800,
            title=(f'{self.dataset_label} ' if self.dataset_label else '') + 'Validation Activity Distribution',
            title_x=0.5
        )
        return fig
    

if __name__ == '__main__':
    import json


    # # 80/20 temporal split agnostic to target protein
    # agnostic_train_arr = np.load('DATA/test_no_indep/linkerology_multitask_Y_train.npy')
    # agnostic_val_arr = np.load('DATA/test_no_indep/linkerology_multitask_Y_test.npy')
    # with open('DATA/test_no_indep/hyperparams.json') as f:
    #     targets = json.load(f)['targets']

    # plotter = ProcessedDataPlotter(agnostic_train_arr, agnostic_val_arr, targets, 'Agnostic')
    # figs = []
    # figs.append(plotter.train_distribution_pie)
    # figs.append(plotter.train_activity_bar)
    # figs.append(plotter.val_distribution_pie)
    # figs.append(plotter.val_activity_bar)


    # # 80/20 temporal split for each target protein (independent splits)
    # indep_val_arr = np.load('DATA/test_indep/linkerology_multitask_Y_test.npy')
    # indep_train_arr = np.load('DATA/test_indep/linkerology_multitask_Y_train.npy')
    # with open('DATA/test_indep/hyperparams.json') as f:
    #     targets = json.load(f)['targets']

    # plotter = ProcessedDataPlotter(indep_train_arr, indep_val_arr, targets, 'Independent')
    # figs.append(plotter.train_distribution_pie)
    # figs.append(plotter.train_activity_bar)
    # figs.append(plotter.val_distribution_pie)
    # figs.append(plotter.val_activity_bar)


    # [fig.show() for fig in figs]

    # directory = 'brad_binarization_indep_temporal_split'
    directory = '40nm_final'
    train_arr = np.load(f'DATA/{directory}/linkerology_multitask_Y_train.npy')
    val_arr = np.load(f'DATA/{directory}/linkerology_multitask_Y_test.npy')
    with open(f'DATA/{directory}/hyperparams.json') as f:
        targets = json.load(f)['targets']

    plotter = ProcessedDataPlotter(train_arr, val_arr, targets, 'Brad')
    figs = []
    figs.append(plotter.train_distribution_pie)
    figs.append(plotter.train_activity_bar)
    figs.append(plotter.val_distribution_pie)
    figs.append(plotter.val_activity_bar)
    [fig.show() for fig in figs]
