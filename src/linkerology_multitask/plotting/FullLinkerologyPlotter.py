from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import PandasTools
from linkerology_multitask import LABEL_MAPPER


class FullLinkerologyPlotter:
    def __init__(self, full_df:pd.DataFrame) -> None:
        # self.full_df = full_df.copy()
        dc50_cols = [col for col in full_df.columns if col.startswith('Degradation') and 'IC50' in col and 'Neomorphic' not in col and '(Mod)' not in col]
        self.dc50_df = full_df[dc50_cols].dropna(how='all')
        self.dc50_melted = self._dc50_df_entry_per_assay()

    def _dc50_df_entry_per_assay(self) -> pd.DataFrame:
        # Melt
        dc50_melted = pd.melt(self.dc50_df, var_name='assay', value_name='dc50_nm').dropna(how='any')
        # Assign target column (AR, ER, LRRK2, etc.)
        dc50_melted['Target'] = [LABEL_MAPPER[assay] for assay in dc50_melted['assay']]
        # Sort by new target column
        dc50_melted = dc50_melted.sort_values(by='Target')
        return dc50_melted

    def plot_activity_distribution(self, qualifier:Optional[str]=None, nm_threshold:Optional[float]=None,
                                   nbins:int=10) -> go.Figure:
        qualifier_factory = {
            '>': lambda : self.dc50_melted['dc50_nm'] > nm_threshold,
            '>=': lambda : self.dc50_melted['dc50_nm'] >= nm_threshold,
            '<': lambda : self.dc50_melted['dc50_nm'] < nm_threshold,
            '<=': lambda : self.dc50_melted['dc50_nm'] <= nm_threshold
        }

        thresholded_dc50_melted = self.dc50_melted.where(qualifier_factory[qualifier]()) if nm_threshold else \
            self.dc50_melted.copy()

        fig = px.histogram(thresholded_dc50_melted, x="dc50_nm", color="Target", nbins=nbins)

        threshold_text = f'{qualifier} {nm_threshold} nM' if nm_threshold else 'all'
        fig.update_layout(
            height=500,
            width=900,  # Adjust the width to your desired value
            title=f'Linkerology Activity Distribution ({threshold_text}): {len(thresholded_dc50_melted)} samples',
        )
        fig.update_layout(
            title_x=0.5, # separate to coorespond to new figure size,
            bargap=0.1
        )
        fig.update_xaxes(
            title='DC50 (nM)'
        )
        fig.update_yaxes(
            title='Count'
        )
        return fig
    
    def plot_target_distribution_pie(self) -> go.Figure:
        value_counts = self.dc50_melted['Target'].value_counts().sort_index()
        fig = go.Figure(data=[go.Pie(values=value_counts.values, labels=value_counts.index, sort=False)])
        fig.update_traces(hoverinfo='label+percent', textinfo='value')
        fig.update_layout(
            title='Target Distribution',
            title_x=0.5,
            height=600,
            width=800
        )
        return fig
    
    def plot_num_targets_pie(self) -> go.Figure:
        not_nan = ~np.isnan(self.dc50_df.to_numpy())

        fig = px.pie(values=Counter(not_nan.sum(axis=1)), names=[1, 2])
        fig.update_layout(
            title='Number of Targets per Datapoint',
            title_x=0.5,
            height=600,
            width=800
        )
        return fig
    
    def plot_kde_grid(self):
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        flat = axes.flat

        for i in range(self.dc50_df.shape[1]):
            values = self.dc50_df.iloc[:, i].dropna().values
            sns.kdeplot(values.astype(float), color='blue', bw_adjust=0.4, ax=flat[i])
            plt.xlabel('DC50 (nM)')
            flat[i].set_title(f'KDE {LABEL_MAPPER[self.dc50_cols[i]]}', fontsize=12)

        plt.delaxes(flat[-1])
        plt.tight_layout()

        return fig
        

if __name__ == '__main__':
    binary_nm_threshold = 40

    # df = pd.read_excel('DATA/raw/full_df.xlsx', index_col='Unnamed: 0')
    df = PandasTools.LoadSDF('../../data/linkerology_with_qualifiers.sdf')

    plotter = FullLinkerologyPlotter(df)
    figs = []
    figs.extend([
        plotter.plot_activity_distribution(), # all
        plotter.plot_activity_distribution('<', binary_nm_threshold),
        plotter.plot_activity_distribution('>=', binary_nm_threshold)
    ])
    figs.append(plotter.plot_target_distribution_pie())
    figs.append(plotter.plot_num_targets_pie())
    [fig.show() for fig in figs]