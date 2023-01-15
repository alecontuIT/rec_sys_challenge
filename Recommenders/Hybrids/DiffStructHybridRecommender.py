from numpy import linalg as LA
import numpy as np
from Recommenders.DataIO import DataIO
from Recommenders.Hybrids.BaseHybridRecommender import BaseHybridRecommender
import utils

class DiffStructHybridRecommender(BaseHybridRecommender):
    """ DiffStructHybridRecommender: You cannot combine 2 or more DiffStructHybridRecommender here, only one is allowed

    """

    RECOMMENDER_NAME = "DiffStructHybridRecommender"

    def __init__(self, URM_train, recs_on_urm_splitted=None, dataset_version="interactions-all-ones", not_trained_recs_classes=[], trained_recs=[]):
        super(DiffStructHybridRecommender, self).__init__(URM_train, recs_on_urm_splitted, dataset_version, not_trained_recs_classes, trained_recs)
        
        
        
    def fit(self, normalize=None, alphas_sum_to_one=False, **alphas):
        self.alphas = []
        self.normalize = normalize
        self.alphas_sum_to_one = alphas_sum_to_one
        self.normalization_const = 0
        
        if len(alphas.values()) < 1 or self.num_rec != len(alphas.values()):
            print("The number of weights does not match with the number of recommenders to combine!")
        
        for alpha in alphas.values():
            self.normalization_const = self.normalization_const + alpha
            self.alphas.append(alpha)

        

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        items_weights = 0
        for rec_idx in range(self.num_rec):
            items_weights_single_rec = self.trained_recs_list[rec_idx]._compute_item_score(user_id_array, items_to_compute)
            items_weights_single_rec = self.normalizing(items_weights_single_rec)
            weight = self.alphas[rec_idx]
            if self.alphas_sum_to_one:
                weight /= self.normalization_const
            items_weights += weight * items_weights_single_rec

        return items_weights
    
    
    
    def normalizing(self, items_weights_single_rec):
        if self.normalize == None:
            return items_weights_single_rec
        elif self.normalize == "L1":
            return items_weights_single_rec / LA.norm(items_weights_single_rec, 1)
        elif self.normalize == "L2":
            return items_weights_single_rec / LA.norm(items_weights_single_rec, 2)
        elif self.normalize == "inf":
            return items_weights_single_rec / LA.norm(items_weights_single_rec, np.inf)
        elif self.normalize == "-inf":
            return items_weights_single_rec / LA.norm(items_weights_single_rec, -np.inf)
        elif self.normalize == "fro":
            return items_weights_single_rec / LA.norm(items_weights_single_rec, "fro")
        
        

    def save_model(self, folder_path, file_name = None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
            
        self._print("Saving model in file '{}'".format(folder_path + file_name))
        
        data_dict_to_save = {"num_rec": self.num_rec,                       #
                             "recs_classes_names": self.recs_classes_names, # Always to be saved in an HybridRecommender
                             "dataset_version": self.dataset_version,       #
                             "hybrids_versions": self.hybrids_versions,     #
                             
                             "alphas": self.alphas, 
                             "normalization_const" : self.normalization_const,
                             "normalize": self.normalize,
                             "alphas_sum_to_one": self.alphas_sum_to_one}


        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")