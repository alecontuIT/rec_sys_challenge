from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.DataIO import DataIO
from lightfm import LightFM
import numpy as np
import scipy.sparse as sps
from copy import deepcopy

class LightFMRecommender(BaseRecommender, Incremental_Training_Early_Stopping):
    """LightFMCBFRecommender"""

    RECOMMENDER_NAME = "LightFMRecommender"

    def __init__(self, URM_train, ICM_train, UCM_train):
        super(LightFMRecommender, self).__init__(URM_train)
        self.lightFM_model = None
        
        if not ICM_train == None:
            self.ICM_train = ICM_train.copy()
            # Need to hstack item_features to ensure each ItemIDs are present in the model
            eye = sps.eye(self.n_items, self.n_items).tocsr()
            self.ICM_train = sps.hstack((eye, self.ICM_train)).tocsr()
        else:
            self.ICM_train = None
            
        if not UCM_train == None:
            self.UCM_train = UCM_Train.copy()
            # Need to hstack user_features to ensure each UserIDs are present in the model
            eye = sps.eye(self.n_users, self.n_users).tocsr()
            self.UCM_train = sps.hstack((eye, self.UCM_train)).tocsr()
        else:
            self.UCM_train = None


    def fit(self, 
            # fit
            sample_weight=None, 
            epochs=1, 
            num_threads=8, 
            verbose=False,
        
            # Constructor Params
            no_components=20, 
            k=5, 
            n=10, 
            learning_schedule='adagrad', 
            loss='warp',
            learning_rate=0.05,
            rho=0.95,
            epsilon=1e-06,
            item_alpha=0.0,
            user_alpha=0.0,
            max_sampled=10,
            random_state=None,
           
            **earlystopping_kwargs):
        
        self.sample_weight = sample_weight
        self.num_threads = num_threads
        self.verbose = verbose
        
        # Let's fit a WARP model
        if self.lightFM_model == None:
            self.lightFM_model = LightFM(no_components=no_components, 
                                         k=k, 
                                         n=n, 
                                         learning_schedule=learning_schedule, 
                                         loss=loss,
                                         learning_rate=learning_rate,
                                         rho=rho,
                                         epsilon=epsilon,
                                         item_alpha=item_alpha,
                                         user_alpha=user_alpha,
                                         max_sampled=max_sampled,
                                         random_state=1)

        if earlystopping_kwargs:
            self._train_with_early_stopping(epochs,
                                            algorithm_name = self.RECOMMENDER_NAME,
                                            **earlystopping_kwargs)
        else:
            self._run_epoch(epochs)
        
        self.lightFM_model = self.lightFM_model_best

        

    def _compute_item_score(self, user_id_array, items_to_compute = None):
        # Create a single (n_items, ) array with the item score, then copy it for every user
        if items_to_compute is None:
            items_to_compute = np.arange(self.n_items)
        else:
            items_to_compute = np.array(items_to_compute)

        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        for user_index, user_id in enumerate(user_id_array):
            # try:
            item_scores[user_index,items_to_compute] = self.lightFM_model.predict(int(user_id),
                                                                                 items_to_compute,
                                                                                 item_features = self.ICM_train,
                                                                                 user_features = self.UCM_train,
                                                                                 num_threads = self.num_threads)

        return item_scores
    
    
    
    def _prepare_model_for_validation(self):
        pass


    
    def _update_best_model(self):
        self.lightFM_model_best = deepcopy(self.lightFM_model)

        

    def _run_epoch(self, num_epoch):
        self.lightFM_model.fit_partial(self.URM_train, 
                                       user_features=self.UCM_train, 
                                       item_features=self.ICM_train,
                                       sample_weight=self.sample_weight,
                                       epochs=num_epoch,
                                       num_threads=self.num_threads,
                                       verbose=self.verbose)
        

    
    def save_model(self, folder_path, file_name = None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = self.lightFM_model.get_params()
        
        # Not serializable object RandomState
        del data_dict_to_save["random_state"]

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")
        
        
        
    def load_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)
        
        self.lightFM_model.set_params(data_dict)

        self._print("Loading complete")