import pandas as pd
from linkerology_multitask.dataset_creation.ParentDataProcessor import ParentDataProcessor
from linkerology_multitask.dataset_creation.ProtacsFeaturizer import ProtacsFeaturizer


class ComponentDataProcessor(ParentDataProcessor):
    def __init__(self, full_df:pd.DataFrame, target_warhead_column_name:str, linker_column_name:str,
                 e3_warhead_column_name:str, hyperparams:dict=ParentDataProcessor.DEFAULT_HYPERPARAMS):
        super().__init__(full_df=full_df, smiles_column_name=None, hyperparams=hyperparams) # None SMILES because we \
            # have each component instead
        self.non_dc50_cols_to_keep = [target_warhead_column_name, linker_column_name, e3_warhead_column_name]

        self.target_warhead_column_name = target_warhead_column_name
        self.linker_column_name = linker_column_name
        self.e3_warhead_column_name = e3_warhead_column_name

        self.full_df = self.full_df.dropna(subset=[self.target_warhead_column_name, self.linker_column_name,
                                                   self.e3_warhead_column_name], how='any').reset_index(drop=True)

    def _set_X(self, train_df:pd.DataFrame, test_df:pd.DataFrame) -> None:
        self.X_train, self.X_test = [
            ProtacsFeaturizer(
                target_warhead_smiles_list=df[self.target_warhead_column_name].tolist(),
                linker_smiles_list=df[self.linker_column_name].tolist(),
                e3_warhead_smiles_list=df[self.e3_warhead_column_name].tolist()
            ).protac_component_ecfp_matrix(self.hyperparams['fingerprint']['radius'],
                                           self.hyperparams['fingerprint']['bit_vector_length'])
        for df in [train_df, test_df]]


if __name__ == '__main__':
    from rdkit.Chem import PandasTools


    full_df = PandasTools.LoadSDF('../../data/DataView_CopyofProcessed_linkerology_060223.sdf_1__export.sdf')

    hyperparams = {
        'fingerprint': {
            'radius': 6,
            'bit_vector_length': 2048
        },
        'component': True,
        'drop_ar': False
    }
    processor = ComponentDataProcessor(full_df, target_warhead_column_name='Target Warhead Structure',
                                       linker_column_name='Linker Smiles', e3_warhead_column_name='CRBN Warhead Smiles',
                                       hyperparams=hyperparams)
    
    # processor.build_classification_dataset(classification_method='normalize_to_lrrk2', project_indep_temporal_split=True)
    processor.build_classification_dataset(classification_method='nm_threshold', project_indep_temporal_split=True)
    name = '40nm_component'

    # processor.build_regression_dataset(project_indep_temporal_split=True)
    # name = 'reg_component'

    out_path = f'DATA/processed/{name}_ECFP{hyperparams["fingerprint"]["radius"]*2}_' + \
        f'{hyperparams["fingerprint"]["bit_vector_length"]}bit'
    if hyperparams['drop_ar']:
        out_path += '_drop_ar'
    processor.write_data(out_path)
