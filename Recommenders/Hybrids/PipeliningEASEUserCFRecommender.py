from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.DataIO import DataIO
import scipy.sparse as sp
import numpy as np


class PipeliningEASEUserCFRecommender(BaseRecommender):
    
    RECOMMENDER_NAME = "PipelineHybrid_EASE_UserKNN_CF_Recommender"
    
        
    def __init__(self, rec_input, rec_output_class, URM_train: sp.csr_matrix):
        super().__init__(URM_train)
        self.r1 = rec_input
        self.r2_class = rec_output_class
    
    def fit(self,
            topK=50,
            shrink=100,
            similarity='cosine',
            normalize=True,
            feature_weighting = "none",
            URM_bias = False,
            k=5,
            **similarity_args):
        self.k = k
        extra_rows = []
        extra_cols = []
        extra_data = []
        for user_id in range(self.URM_train.shape[0]):
            extra_cols.extend(self.r1.recommend(user_id, cutoff=self.k))
            extra_rows.extend([user_id for _ in range(self.k)])
            extra_data.extend([1 for _ in range(self.k)])

        URM_coo = self.URM_train.tocoo()
        self.URM_new_csr = sp.csr_matrix((np.append(URM_coo.data, extra_data),
                                 (np.append(URM_coo.row, extra_rows), np.append(URM_coo.col, extra_cols))),
                                shape=URM_coo.shape)
        
        self.r2 = self.r2_class(self.URM_new_csr)
        self.r2.fit(topK, shrink, similarity, normalize, feature_weighting, URM_bias, **similarity_args)
    
    def save_model(self, folder_path, file_name = None):
        self.r2.save_model(folder_path, file_name)

        
    def _compute_item_score(self, user_id_array, items_to_compute = None):
        _, scores_batch = self.r2.recommend(user_id_array, cutoff = 10, remove_seen_flag=False, return_scores = True)
        return scores_batch
    
