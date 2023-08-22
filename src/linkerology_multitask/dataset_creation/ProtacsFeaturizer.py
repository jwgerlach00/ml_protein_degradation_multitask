import numpy as np
from typing import List, Optional, Union
from rdkit import Chem
from rdkit.Chem import AllChem


class ProtacsFeaturizer:
    def __init__(self, target_warhead_smiles_list:List[str], linker_smiles_list:List[str],
                 e3_warhead_smiles_list:List[str]) -> None:
        if len(target_warhead_smiles_list) != len(linker_smiles_list) != len(e3_warhead_smiles_list):
            raise ValueError('target_warhead_smiles_list, linker_smiles_list, and e3_warhead_smiles_list must all be \
                             the same length.')
    
        self.target_warhead_smiles_list = target_warhead_smiles_list
        self.linker_smiles_list = linker_smiles_list
        self.e3_warhead_smiles_list = e3_warhead_smiles_list
        

    def protac_component_ecfp_matrix(self, radius:int, bit_vector_length:int) -> np.ndarray:
        out = []
        for smiles_list in [self.target_warhead_smiles_list, self.linker_smiles_list, self.e3_warhead_smiles_list]:
            out.append(self.smiles_to_fingerprints(smiles_list, radius=radius, bit_vector_length=bit_vector_length,
                                                   return_array=True))
        return np.stack(out, axis=1)
        
    @staticmethod
    def smiles_to_fingerprints(smiles_list:List[str], radius:int=3, bit_vector_length:Optional[int]=1024,
                               return_array:bool=True) -> Union[np.ndarray, list]:
        """
        Wrapper for Protac.mols_to_fingerprints. Converts SMILES to Mols before converting to fingerprints.

        :param smiles_list: List of SMILES strings
        :type smiles_list: List[str]
        :param radius: Atom radius for AllChem.GetMorganFingerprint, defaults to 3
        :type radius: int, optional
        :param bit_vector_length: Length of folded-down vector output by AllChem.GetMorganFingerprint, defaults to 1024
        :type bit_vector_length: Optional[int], optional
        :param return_array: Whether to return an array (True) or a list of RDKit Fingerprint objects (False), defaults
            to True
        :type return_array: bool, optional
        :return: Array of bit vectors or RDKit finger object
        :rtype: Union[np.ndarray, list]
        """
        return ProtacsFeaturizer.mols_to_fingerprints([Chem.MolFromSmiles(smiles) for smiles in smiles_list],
                                                      radius=radius, bit_vector_length=bit_vector_length,
                                                      return_array=return_array)

    @staticmethod
    def mols_to_fingerprints(mols:List[Chem.rdchem.Mol], radius:int=3, bit_vector_length:Optional[int]=1024,
                             return_array:bool=True) -> Union[np.ndarray, list]:
        """
        Converts RDKit Mol objects to Morgan Fingerprints. Either creates an array or a list of RDKit Fingerprint
        objects.

        :param mols: RDKit Mols
        :type mols: List[Chem.rdchem.Mol]
        :param radius: Atom radius for AllChem.GetMorganFingerprint, defaults to 3
        :type radius: int, optional
        :param bit_vector_length: Length of folded-down vector output by AllChem.GetMorganFingerprint, defaults to 1024
        :type bit_vector_length: Optional[int], optional
        :param return_array: Whether to return an array (True) or a list of RDKit fingerprint objects (False), defaults
            to True
        :type return_array: bool, optional
        :return: Array of bit vectors or RDKit finger object
        :rtype: Union[np.ndarray, list]
        """
        if return_array: # convert to numpy with each row having length (bit_vector_length, specified by hyperparams)
            fingerprints = []
            for mol in mols:
                try:
                    fingerprints.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bit_vector_length))
                except Exception:
                    print(mol)
                    exit()
            # fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bit_vector_length)
            #                 for mol in mols]
            X = np.empty((0, bit_vector_length))
            for fp in fingerprints:
                arr = np.array([])
                Chem.DataStructs.ConvertToNumpyArray(fp, arr)
                X = np.vstack((X, arr)) # Add row
            return X
        else: # no bit-vector
            return [AllChem.GetMorganFingerprint(mol, radius=radius) for mol in mols]