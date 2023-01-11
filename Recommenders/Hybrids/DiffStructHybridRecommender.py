from Recommenders.BaseRecommender import BaseRecommender
import scipy.sparse as sps
from numpy import linalg as LA
import numpy as np
from Recommenders.DataIO import DataIO
from Recommenders.Hybrids.FastLoadWrapperRecommender import FastLoadWrapperRecommender
import utils

class DiffStructHybridRecommender(BaseRecommender):
    """ DiffStructHybridRecommender: You cannot combine 2 or more DiffStructHybridRecommender here, only one is allowed

    """

    RECOMMENDER_NAME = "DiffStructHybridRecommender"
    RECOMMENDER_VERSION = "best_version"

    def __init__(self, URM_train, dataset_version="interactions-all-ones", recs_classes_list=[], load_scores_from_saved=True, recs_on_urm_splitted=None, user_id_array_val=None, new_item_scores_file_name_root = None, alphas_sum_to_one=False):
        super(DiffStructHybridRecommender, self).__init__(URM_train)
        self.recs_classes_list = recs_classes_list
        self.dataset_version = dataset_version
        self.num_rec = len(recs_classes_list)
        self.load_scores_from_saved = load_scores_from_saved
        self.new_item_scores_file_name_root = new_item_scores_file_name_root
        self.alphas_sum_to_one = alphas_sum_to_one
        
        self.user_id_array_val = user_id_array_val
        if self.load_scores_from_saved: # if recs_classes_list == [] we are loading from file
            assert (self.user_id_array_val is not None)
        else:
            assert (self.user_id_array_val is None)
            
        self.recs_on_urm_splitted = recs_on_urm_splitted
        if self.recs_on_urm_splitted:
            assert (self.new_item_scores_file_name_root is None)
        else:
            assert (self.new_item_scores_file_name_root is not None)
            
        self.trained_recs_list = []
        self.recs_classes_names = []
        self.there_is_an_hybrid = False
        
        for rec_class in self.recs_classes_list:
            if rec_class is DiffStructHybridRecommender:
                    self.there_is_an_hybrid = True
                    
            if not self.load_scores_from_saved:
                rec = utils.load_best_model(self.URM_train, 
                                            rec_class, 
                                            dataset_version=self.dataset_version, 
                                            optimization=self.recs_on_urm_splitted)
            else:
                rec = FastLoadWrapperRecommender(self.URM_train, 
                                                 rec_class, 
                                                 dataset_version=self.dataset_version,
                                                 user_id_array_val=self.user_id_array_val,
                                                 fast=True,
                                                 recs_on_urm_splitted = self.recs_on_urm_splitted,
                                                 new_item_scores_file_name_root = self.new_item_scores_file_name_root)
            self.trained_recs_list.append(rec)
            self.recs_classes_names.append(rec.RECOMMENDER_NAME)
        
        
        
    def fit(self, normalize=None, **alphas):
        self.alphas = []
        self.normalize = normalize
        self.normalization_const = 0
        idx = 0
        self.RECOMMENDER_VERSION = ""
        
        if len(alphas.values()) < 1 or self.num_rec != len(alphas.values()):
            print("The number of weights does not match with the number of recommenders to combine!")
        
        for alpha in alphas.values():
            self.normalization_const = self.normalization_const + alpha
            self.alphas.append(alpha)
            #self.RECOMMENDER_VERSION += "alpha" + str(idx) + "-" + str(alpha) + "_"
            idx += 1
        #self.RECOMMENDER_VERSION += "norm-" + str(self.normalize)

        

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

            
        if self.there_is_an_hybrid:
            kwargs = {"load_scores_from_saved": self.load_scores_from_saved, 
                      "recs_on_urm_splitted": self.recs_on_urm_splitted, 
                      "user_id_array_val": self.user_id_array_val, 
                      "new_item_scores_file_name_root": self.new_item_scores_file_name_root}
            
            old_hybrid = utils.load_best_model(self.URM_train,
                                               DiffStructHybridRecommender, 
                                               dataset_version=self.dataset_version, 
                                               optimization=True,
                                               **kwargs) # get hyperparams from 'optimization' folder
            idx_hybrid = 0
            
            '''
            aaa = {"alphas": self.alphas, 
                             "num_rec": self.num_rec,
                             "normalization_const" : self.normalization_const,
                             "normalize": self.normalize,
                             "recs_classes_names": self.recs_classes_names,
                             "load_scores_from_saved": self.load_scores_from_saved,
                             "dataset_version": self.dataset_version}
            bbb = {"alphas": old_hybrid.alphas, 
                             "num_rec": old_hybrid.num_rec,
                             "normalization_const" : old_hybrid.normalization_const,
                             "normalize": old_hybrid.normalize,
                             "recs_classes_names": old_hybrid.recs_classes_names,
                             "load_scores_from_saved": old_hybrid.load_scores_from_saved,
                             "dataset_version": old_hybrid.dataset_version}
            print("#######################################\n" + "BIGGER HYBRID BLACK BOX ELEMENTS: " + str(aaa) + "\n#######################################")
            print("#######################################\n" + "SMALL OLD HYBRID: " + str(bbb) + "\n#######################################")
            '''

            for idx in range(len(self.recs_classes_list)):
                if self.recs_classes_list[idx] == DiffStructHybridRecommender:
                    idx_hybrid = idx
                    
            new_normalization_const = old_hybrid.normalization_const * self.normalization_const
            new_num_rec = old_hybrid.num_rec + self.num_rec - 1
            
            old_alphas = []
            for alpha in old_hybrid.alphas:
                new_alpha = alpha * self.alphas[idx_hybrid]
                old_alphas.append(new_alpha)
                
            new_alphas = []    
            for alpha in self.alphas:
                new_alpha = alpha * old_hybrid.normalization_const
                new_alphas.append(new_alpha)
            
            i = 0
            for alpha in old_alphas:
                new_alphas.insert(idx_hybrid + i, alpha)
                i += 1
            new_alphas.pop(idx_hybrid + i)
            
            old_recs_classes_names = old_hybrid.recs_classes_names
            new_recs_classes_names = self.recs_classes_names
            
            i = 0
            for rec_name in old_recs_classes_names:
                new_recs_classes_names.insert(idx_hybrid + i, rec_name)
                i += 1
            new_recs_classes_names.pop(idx_hybrid + i)
            
            data_dict_to_save = {"alphas": new_alphas, 
                             "num_rec": new_num_rec,
                             "normalization_const" : new_normalization_const,
                             "normalize": self.normalize,
                             "recs_classes_names": new_recs_classes_names,
                             "dataset_version": self.dataset_version}
                    
        else:
            data_dict_to_save = {"alphas": self.alphas, 
                             "num_rec": self.num_rec,
                             "normalization_const" : self.normalization_const,
                             "normalize": self.normalize,
                             "recs_classes_names": self.recs_classes_names,
                             "dataset_version": self.dataset_version}


        dataIO = DataIO(folder_path=folder_path)
        '''
        print("#######################################\n" + "FINAL HYBRID: " + str(data_dict_to_save) + "\n#######################################")
        '''
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")
        
        
        
    def load_model(self, folder_path, file_name = None):
        super(DiffStructHybridRecommender, self).load_model(folder_path, file_name)

        self.trained_recs_list = []
        for rec_class_name in self.recs_classes_names:
            rec_class = utils.get_rec_class_by_name(rec_class_name)
            self.recs_classes_list.append(rec_class)
            if not self.load_scores_from_saved:
                rec = utils.load_best_model(self.URM_train, 
                                            rec_class, 
                                            dataset_version=self.dataset_version,
                                            recs_on_urm_splitted = self.recs_on_urm_splitted,
                                            optimization=self.recs_on_urm_splitted)
            else:
                rec = FastLoadWrapperRecommender(self.URM_train, 
                                                 rec_class, 
                                                 dataset_version=self.dataset_version,
                                                 user_id_array_val=self.user_id_array_val,
                                                 fast=True,
                                                 recs_on_urm_splitted = self.recs_on_urm_splitted,
                                                 new_item_scores_file_name_root = self.new_item_scores_file_name_root)
            self.trained_recs_list.append(rec)