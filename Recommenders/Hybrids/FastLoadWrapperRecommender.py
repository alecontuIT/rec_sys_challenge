from Recommenders.BaseRecommender import BaseRecommender
import numpy as np
import utils

class FastLoadWrapperRecommender(BaseRecommender):
    ''' This class can be used only for recommenders already optimized! So the rec_class must have 
    a folder 'optimization'
    a folder 'best'
    a file 'recommendation.csv
    '''
    RECOMMENDER_NAME = ""
    def __init__(self, URM_train, rec_class, dataset_version="interactions-all-ones", use_block = False, fast=True, user_id_array_val=None):
        self.rec_class = rec_class
        self.RECOMMENDER_NAME += self.rec_class.RECOMMENDER_NAME
        self.dataset_version = dataset_version
        self.fast = fast
        self.use_block = use_block
        self.user_id_array_val = user_id_array_val
        utils.save_item_scores(self.rec_class, URM_train, self.user_id_array_val, self.dataset_version, fast=fast)
        
    def fit(self):
        return
    
    def _compute_item_score(self, user_id_array, items_to_compute = None):
        scores = utils.load_item_scores(self.rec_class, self.dataset_version, self.fast)
        if self.use_block and self.user_id_array_val != None:
            mask = np.isin(self.user_id_array_val, user_id_array)
            return scores[mask]
        return scores
    
    def save_model(self, folder_path, file_name = None):
        return
    
    def load_model(self, folder_path, file_name = None):
        return
            

