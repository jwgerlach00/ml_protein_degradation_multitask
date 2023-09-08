from typing import Union, List, Dict
import pandas as pd
from collections import Counter
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from math import floor
import copy


class LinkerologySampler:
    DEFAULT_HYPERPARAMS = {
        'fingerprint': {
            'radius': 6,
            'bit_vector_length': 2048
        },
        'class_labeling': {
            'drop_ambiguous': True,
            'lower_threshold': 0.5,
            'upper_threshold': 0.8
        }
    }

    def __init__(self, full_df:pd.DataFrame) -> None:
        self.full_df = full_df
    
    @property
    def projects(self):
        return self.full_df['Project Name'].unique().tolist()
    
    @property
    def count(self) -> Dict[str, int]:
        """Retrieves counts of records corresponding to each Project Name. Sorts decending.

        :return: Project-count map
        :rtype: Dict[str, int]
        """
        counter = Counter(self.full_df['Project Name'])
        return dict(counter.most_common())

    def sample_project(self, project_name:str, target_col_name:str, hyperparams:dict=DEFAULT_HYPERPARAMS) -> pd.DataFrame:
        """Retrieves only data for a set of project names and strips irrelivant fields for those projects.

        :param project_names: List of names of projects in 'Project Name' field of full_df
        :type project_name: str
        :return: Only data for the projects
        :rtype: pd.DataFrame
        """
        out = self.full_df.where(self.full_df['Project Name'] == project_name).dropna(axis=0, how='all')
        # out = self.full_df[self.full_df['Project Name'].isin(project_names)].dropna(axis=0, how='all')
        out = out.dropna(axis=1, how='all') # drop unused fields for the project
        out = out.reset_index(drop=True)
        return Sample(out, target_col_name, hyperparams)


class Sample:
    def __init__(self, df:pd.DataFrame, target_col_name:str, hyperparams:dict):
        self.df = df.copy()
        self.target_col_name = target_col_name
        self.hyperparams = copy.deepcopy(hyperparams)
        if hyperparams['task_type'] == 'classification':
            self.add_classes_to_df(self.hyperparams['class_labeling']['drop_ambiguous'], self.hyperparams['class_labeling']['lower_threshold'], self.hyperparams['class_labeling']['upper_threshold'])

    @staticmethod
    def smiles_to_fingerprints(smiles_list:List[str], radius:int=3, bit_vector_length:int=1024):
        return Sample.mols_to_fingerprints([Chem.MolFromSmiles(smiles) for smiles in smiles_list], radius, bit_vector_length)

    @staticmethod
    def mols_to_fingerprints(mols:List[Chem.rdchem.Mol], radius:int=3, bit_vector_length:int=1024, return_array:bool=True) -> Union[np.ndarray, list]:
        """Converts RDKit Mol objects to Morgan Fingerprints. Either creates an array or a list of RDKit Fingerprint objects.

        :param mols: RDKit Mols
        :type mols: List[Chem.rdchem.Mol]
        :param return_array: Whether to return array or Fingerprint objects, defaults to True
        :type return_array: bool, optional
        :return: Fingerprints
        :rtype: Union[np.ndarray, list]
        """
        if return_array: # convert to numpy with each row having length (bit_vector_length, specified by hyperparams)
            fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bit_vector_length) for mol in mols]
            X = np.empty((0, bit_vector_length))
            for fp in fingerprints:
                arr = np.array([])
                Chem.DataStructs.ConvertToNumpyArray(fp, arr)
                X = np.vstack((X, arr)) # Add row
            return X
        else: # no bit-vector
            return [AllChem.GetMorganFingerprint(mol, radius=radius) for mol in mols]

    def add_classes_to_df(self, drop_ambiguous:bool, lower_threshold:float, upper_threshold:float):
        self.df['class'] = self.dc50_to_multiclass(self.df[self.target_col_name].astype(float).tolist(), lower_threshold, upper_threshold)
        if drop_ambiguous:
            self.df = self.df.where(self.df['class'] != 'ambiguous')

    @staticmethod
    def dc50_to_multiclass(dc50_values:List[float], lower_threshold:float=0.5, upper_threshold:float=0.8):
        out = []
        for value in dc50_values:
            if value <= lower_threshold:
                out.append('active')
            elif value >= upper_threshold:
                out.append('inactive')
            else:
                out.append('ambiguous') 
        return out
    
    def temporal_split(self, test_ratio:float):
        if self.hyperparams['task_type'] == 'classification':
            y_col_name = 'class'
        elif self.hyperparams['task_type'] == 'regression':
            y_col_name = self.target_col_name
        else:
            raise ValueError('(hyperparams.task_type) must be (1) of: "classification", "regression".')

        # Sort for temporal
        sorted_df = self._sort_by_date()
        # Drop entries with missing target data
        sorted_df = sorted_df.dropna(how='any', subset=[self.target_col_name]).reset_index(drop=True)

        train_test_split_index = floor(len(sorted_df)*(1 - test_ratio))
        X_train = sorted_df['Parent Structure'].tolist()[:train_test_split_index]
        X_test = sorted_df['Parent Structure'].tolist()[train_test_split_index:]

        y_train = sorted_df[y_col_name].tolist()[:train_test_split_index]
        y_test = sorted_df[y_col_name].tolist()[train_test_split_index:]

        X_train = self.smiles_to_fingerprints(X_train, self.hyperparams['fingerprint']['radius'], self.hyperparams['fingerprint']['bit_vector_length'])
        X_test = self.smiles_to_fingerprints(X_test, self.hyperparams['fingerprint']['radius'], self.hyperparams['fingerprint']['bit_vector_length'])

        return X_train, X_test, y_train, y_test
    
    def random_split(self, test_ratio:float):
        if self.hyperparams['task_type'] == 'classification':
            y_col_name = 'class'
        elif self.hyperparams['task_type'] == 'regression':
            y_col_name = self.target_col_name
        else:
            raise ValueError('(hyperparams.task_type) must be (1) of: "classification", "regression".')

        # Drop entries with missing target data
        df = self.train_df.dropna(how='any', subset=[self.target_col_name]).reset_index(drop=True)

        # Shuffle the dataframe
        shuffled_df = df.sample(frac=1, random_state=self.random_state)

        train_test_split_index = int(len(shuffled_df) * (1 - test_ratio))

        X_train = shuffled_df['Parent Structure'].tolist()[:train_test_split_index]
        X_test = shuffled_df['Parent Structure'].tolist()[train_test_split_index:]

        y_train = shuffled_df[y_col_name].tolist()[:train_test_split_index]
        y_test = shuffled_df[y_col_name].tolist()[train_test_split_index:]

        X_train = self.smiles_to_fingerprints(X_train, self.hyperparams['fingerprint']['radius'], self.hyperparams['fingerprint']['bit_vector_length'])
        X_test = self.smiles_to_fingerprints(X_test, self.hyperparams['fingerprint']['radius'], self.hyperparams['fingerprint']['bit_vector_length'])

        return X_train, X_test, y_train, y_test
    
    def _sort_by_date(self) -> pd.DataFrame:
        # Convert to recognized datetime format, sort, take index, use index on existing df
        return self.df.iloc[pd.to_datetime(self.df['Registration Date'], format='%d-%b-%Y').sort_values(ascending=True).index]
    