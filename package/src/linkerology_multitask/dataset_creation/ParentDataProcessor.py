from typing import List, Dict, Tuple
import pandas as pd
from collections import Counter
import numpy as np
import copy
import os
from math import floor
import json
from linkerology_multitask.dataset_creation.ProtacsFeaturizer import ProtacsFeaturizer


class ParentDataProcessor:
    DEFAULT_HYPERPARAMS = {
        'fingerprint': {
            'radius': 3,
            'bit_vector_length': 1024
        },
        'drop_ar': False
    }
    CLASSIFICATION_METHODS = {
        'active_frac_per_target',
        'nm_threshold',
        'normalize_to_lrrk2'
    }

    def __init__(self, full_df:pd.DataFrame, smiles_column_name:str='Parent Structure',
                 hyperparams:dict=DEFAULT_HYPERPARAMS) -> None:
        self.project_column_name = 'Project Name'
        self.date_column_name = 'Registration Date'

        self.full_df = full_df.copy()
        self.smiles_column_name = smiles_column_name
        self.hyperparams = copy.deepcopy(hyperparams)
        self.hyperparams['component'] = False # because this is for PARENT only
        # self.__nan_class_label = nan_class_label
        self.__nan_class_label = None

        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        self.non_dc50_cols_to_keep = [self.smiles_column_name]
        self.dc50_column_names = self._get_dc50_column_names()
        
        self.targets = None

        self.full_df = self.full_df.dropna(subset=self.dc50_column_names, how='all').reset_index(drop=True)

    
    @property
    def projects(self) -> List[str]:
        """
        List of all projects in self.full_df.

        :return: All projects
        :rtype: List[str]
        """
        return self.full_df[self.project_column_name].unique().tolist()
    
    @property
    def count(self) -> Dict[str, int]:
        """
        Counts of records corresponding to each Project Name. Sorted decending.

        :return: Project-count map
        :rtype: Dict[str, int]
        """
        counter = Counter(self.full_df[self.project_column_name])
        return dict(counter.most_common())
    
    def build_classification_dataset(self, classification_method:str, test_ratio:float=0.2, nan_class_label:int=-100,
                                     active_frac_per_target:float=0.2, classification_nm_threshold:float=40.0,
                                     project_indep_temporal_split:bool=True) -> None:
        """
        Processes self.full_df and sets self.X and self.X according to self.hyperparams.
        """
        if classification_method not in ParentDataProcessor.CLASSIFICATION_METHODS:
            raise ValueError(f'"classification_method" must be one of {ParentDataProcessor.CLASSIFICATION_METHODS}. Received \
                             "{classification_method}"')

        self.__nan_class_label = nan_class_label

        if project_indep_temporal_split:
            train_df, test_df = self.project_indep_temporal_split(test_ratio)
        else:
            train_df, test_df = self.temporal_split(test_ratio)

        train_df = self._get_only_smiles_and_dc50_data(train_df)
        test_df = self._get_only_smiles_and_dc50_data(test_df)

        if classification_method == 'active_frac_per_target':
            train_df = self._active_frac_per_target_convert_to_class_labels(train_df, active_frac_per_target)
            test_df = self._active_frac_per_target_convert_to_class_labels(test_df, active_frac_per_target)
        elif classification_method == 'nm_threshold':
            train_df = self._nm_threshold_convert_to_class_labels(train_df, classification_nm_threshold)
            test_df = self._nm_threshold_convert_to_class_labels(test_df, classification_nm_threshold)
        elif classification_method == 'normalize_to_lrrk2':
            train_df = self._normalize_to_lrrk2_and_convert_to_class_labels(train_df)
            test_df = self._normalize_to_lrrk2_and_convert_to_class_labels(test_df)

        self._set_X(train_df, test_df)
        self._set_Y(train_df, test_df)
    
    def build_regression_dataset(self, test_ratio:float=0.2, project_indep_temporal_split:bool=True) -> None:
        """
        Processes self.full_df and sets self.X and self.X according to self.hyperparams.
        """
        if project_indep_temporal_split:
            train_df, test_df = self.project_indep_temporal_split(test_ratio)
        else:
            train_df, test_df = self.temporal_split(test_ratio)

        train_df = self._get_only_smiles_and_dc50_data(train_df)
        test_df = self._get_only_smiles_and_dc50_data(test_df)

        self._set_X(train_df, test_df)
        self._set_Y(train_df, test_df)
        self.Y_train = self.Y_train.astype(float)
        self.Y_test = self.Y_test.astype(float)

    def temporal_split(self, test_ratio:float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Sort for temporal
        self.full_df = self._sort_df_by_date(self.full_df, self.date_column_name)

        train_test_split_index = floor(len(self.full_df)*(1 - test_ratio))

        train_df = self.full_df.iloc[:train_test_split_index].reset_index(drop=True)
        test_df = self.full_df.iloc[train_test_split_index:].reset_index(drop=True)
        return train_df, test_df

    def project_indep_temporal_split(self, test_ratio:float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        grouped = self.full_df.groupby(self.full_df[self.project_column_name])
        train_df = None
        test_df = None
        for _, group_df in grouped:
            group_df = self._sort_df_by_date(group_df.reset_index(drop=True), self.date_column_name)
    
            train_test_split_index = floor(len(group_df)*(1 - test_ratio))
            group_train_df = group_df.iloc[:train_test_split_index]
            group_test_df = group_df.iloc[train_test_split_index:]

            if train_df is None:
                train_df = group_train_df
            else:
                train_df = pd.concat((train_df, group_train_df), axis=0)
            if test_df is None:
                test_df = group_test_df
            else:
                test_df = pd.concat((test_df, group_test_df), axis=0)

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    @staticmethod
    def _sort_df_by_date(df, date_column_name:str) -> pd.DataFrame:
        return df.iloc[pd.to_datetime(df[date_column_name], format='%d-%b-%Y').sort_values(ascending=True).index].reset_index(drop=True)

    def write_data(self, out_path:str='.') -> None:
        """
        Saves self.X and self.Y to out_path. Files will be named "out_path"/linkerology_multitask_(X/Y).npy

        :param out_path: Absolute path to save arrays to, defaults to '.'
        :type out_path: str
        :raises TypeError: self.X or self.Y are not np.ndarray
        """
        # if not isinstance(self.X, np.ndarray and isinstance(self.Y, np.ndarray)):
        #     raise TypeError(f'Both X and Y of DataProcessor must be type np.ndarray. X is {type(self.X)} and Y is {type(self.Y)}.')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self.hyperparams['targets'] = self.targets
        with open(os.path.join(out_path, 'hyperparams.json'), 'w') as f:
            json.dump(self.hyperparams, f)
        np.save(os.path.join(out_path, 'linkerology_multitask_X_train.npy'), self.X_train)
        np.save(os.path.join(out_path, 'linkerology_multitask_X_test.npy'), self.X_test)
        np.save(os.path.join(out_path, 'linkerology_multitask_Y_train.npy'), self.Y_train)
        np.save(os.path.join(out_path, 'linkerology_multitask_Y_test.npy'), self.Y_test)

    def _get_only_smiles_and_dc50_data(self, df) -> pd.DataFrame:
        """
        Returns only the columns of self.full_df that correspond to DC50 data, and self.smiles_col_name.

        :return: Subset of self.full_df
        :rtype: pd.DataFrame
        """
        return df[self.non_dc50_cols_to_keep + self.dc50_column_names]
    
    def _get_dc50_column_names(self) -> List[str]:
        """
        Gets list of column name strings in self.full_df that start with "Degradation" and have "IC50" in them.

        :return: List of degradation columns in self.full_df given syntax assumptions
        :rtype: List[str]
        """
        return [col for col in self.full_df.columns if col.startswith('Degradation') and 'IC50' in col and '(Mod)' not in col]
    
    def _nm_threshold_convert_to_class_labels(self, df:pd.DataFrame, nm_threshold:float) -> pd.DataFrame:
        only_dc50_df = df.loc[:, ~df.columns.isin(self.non_dc50_cols_to_keep)].astype(float)
        binary_df = only_dc50_df.apply(lambda x: np.where(x >= nm_threshold, 0, np.where(x < nm_threshold, 1, np.nan))) # first check if greater than or equl to threshold, then (if false) check if less than threshold, it neither set to nan
        binary_df = binary_df.fillna(self.__nan_class_label).astype(int)
        binary_df[self.non_dc50_cols_to_keep] = df[self.non_dc50_cols_to_keep]
        return binary_df
    
    def _active_frac_per_target_convert_to_class_labels(self, df:pd.DataFrame, active_frac_per_target:float) -> pd.DataFrame:
        """
        Converts every column in df besides columns in self.non_dc50_cols_to_keep to 0, 1 class labels. NaN values are converted to self.__nan_class_label.

        :param df: DataFrame only containing self.non_dc50_cols_to_keep and data columns to be conveted
        :type df: pd.DataFrame
        :param active_frac_per_target: Fraction of desired active labels per target
        :type active_frac_per_target: float
        :return: DataFrame with self.non_dc50_cols_to_keep and class labeling
        :rtype: pd.DataFrame
        """
        df = df.copy()
        for col in df.columns:
            if col in self.non_dc50_cols_to_keep:
                continue
            # Convert string numbers to float
            col_data = df[col].astype(float)
            # Get the threshold at which (active_frac_per_target)*100% of the non-NaN data is labeled as active NOTE: assumes lower value is active (1)
            threshold = col_data.quantile(active_frac_per_target)
            # Perform labeling using threshold, retain NaN values
            binary_series = col_data.where(col_data.isna(), col_data <= threshold)
            # Replace NaN values w/ self.nan_class_label
            df[col] = binary_series.fillna(self.__nan_class_label).astype(int)
        return df
    
    def _normalize_to_lrrk2_and_convert_to_class_labels(self, df:pd.DataFrame, lrrk2_lower_nm_threshold:float=3, lrrk2_upper_nm_threshold:float=12) -> pd.DataFrame:
        lrrk2_col_name = 'Degradation_Endogenous_WT_LRRK2_C-Term-HiBit_HEK293_24h_DR_384 (v1);GMean;IC50 (nM)'
        lrrk2_median = df[lrrk2_col_name].median()
        print(df)

        for col in df.columns:
            if col in self.non_dc50_cols_to_keep:
                continue

            if col == lrrk2_col_name: # col is LRRK2, don't normalize
                normalized_series = df[col].astype(float)
            else: # normalize
                factor = df[col].median() / lrrk2_median
                normalized_series = df[col].astype(float)*factor

            binary_series = pd.Series(np.where(normalized_series > lrrk2_upper_nm_threshold, 0, np.where(normalized_series < lrrk2_lower_nm_threshold, 1, np.nan)))
            df[col] = binary_series
        df = df.dropna(subset=self.dc50_column_names, how='all').reset_index(drop=True) # before fillna
        df[self.dc50_column_names] = df[self.dc50_column_names].fillna(self.__nan_class_label).astype(int) # replacing nan w/ nan_class_label down here so we can dropna first
        return df
        
    
    def _set_X(self, train_df:pd.DataFrame, test_df:pd.DataFrame) -> None:
        """
        Sets self.X np.ndarray by converting df[self.smiles_column_name] to fingerprints according to self.hyperparams.

        :param df: DataFrame containing self.smiles_column_name
        :type df: pd.DataFrame
        """
        self.X_train, self.X_test = [ProtacsFeaturizer.smiles_to_fingerprints(
            df[self.smiles_column_name], radius=self.hyperparams['fingerprint']['radius'],
            bit_vector_length=self.hyperparams['fingerprint']['bit_vector_length'],
            return_array=True) for df in [train_df, test_df]]

    def _set_Y(self, train_df:pd.DataFrame, test_df:pd.DataFrame) -> None:
        """
        Sets self.Y np.ndarray to every column in df besides df[self.non_dc50_cols_to_keep] to a 2D array.

        :param df: DataFrame optionally containing self.non_dc50_cols_to_keep (will be excluded) and column data to be set to Y
        :type df: pd.DataFrame
        """
        ar_col_name = 'Degradation_AR_High Content_VCaP (v1);GMean;IC50 (nM)'
        only_dc50_cols = set(train_df.columns) - set(self.non_dc50_cols_to_keep) # filter out non-dc50 columns
        only_dc50_cols = [col for col in train_df.columns if col in only_dc50_cols] # put back in original order
        if self.hyperparams['drop_ar']:
            only_dc50_cols = [col for col in only_dc50_cols if col != ar_col_name]
        self.targets = only_dc50_cols # targets are shared between train and test since they came from the same dataset
        self.Y_train = train_df.loc[:, only_dc50_cols].to_numpy()
        self.Y_test = test_df.loc[:, only_dc50_cols].to_numpy()


if __name__ == '__main__':
    from rdkit.Chem import PandasTools


    # LOAD DATA
    full_df = PandasTools.LoadSDF('../../data/DataView_CopyofProcessed_linkerology_060223.sdf_1__export.sdf')
    # full_df = PandasTools.LoadSDF('../../data/DataView_Linkerology_Data_Dump_Concat_070623.sdf') # includes additional compounds


    # 40NM PARENT
    hyperparams = {
        'fingerprint': {
            'radius': 6,
            'bit_vector_length': 2048
        },
        'component': False,
        'drop_ar': False
    }

    processor = ParentDataProcessor(full_df, hyperparams=hyperparams)
    # processor.project_column_name = 'Concat;Project Name'
    # processor.date_column_name = 'Max;Registration Date'
    # processor.build_classification_dataset(classification_method='normalize_to_lrrk2', project_indep_temporal_split=True)

    processor.build_classification_dataset(classification_method='nm_threshold', project_indep_temporal_split=True)
    name = '40nm_parent'

    # processor.build_regression_dataset(project_indep_temporal_split=True)
    # name = 'reg_parent'

    out_path = f'DATA/processed/{name}_ECFP{hyperparams["fingerprint"]["radius"]*2}_{hyperparams["fingerprint"]["bit_vector_length"]}bit'
    if hyperparams['drop_ar']:
        out_path += '_drop_ar'
    processor.write_data(out_path)
