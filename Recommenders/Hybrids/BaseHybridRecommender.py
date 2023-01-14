from Recommenders.BaseRecommender import BaseRecommender
import os
from Recommenders.DataIO import DataIO
import utils

class BaseHybridRecommender(BaseRecommender):

    RECOMMENDER_NAME = "BaseHybridRecommender"
    
    def __init__(self, URM_train, recs_on_urm_splitted=None, dataset_version="interactions-all-ones", not_trained_recs_classes=[], trained_recs=[]):
        super(BaseHybridRecommender, self).__init__(URM_train)

        self.RECOMMENDER_VERSION = ""
        
        ##################################################
        self.recs_on_urm_splitted = recs_on_urm_splitted # This attribute should not be saved: it's provided
        ################################################## in the constructor.

        self.dataset_version = dataset_version      
        self.num_rec = 2 #len(not_trained_recs_classes) + len(trained_recs)
            
        self.trained_recs_list = []
        self.recs_classes_list = not_trained_recs_classes.copy()
        self.recs_classes_names = []

        for rec_class in self.recs_classes_list:
            assert not issubclass(rec_class, BaseHybridRecommender), "Error: hybrid recommenders must be provided as objects, already fitted.\n"
            rec = utils.load_best_model(self.URM_train, 
                                            rec_class, 
                                            dataset_version=self.dataset_version, 
                                            optimization=self.recs_on_urm_splitted)

            self.trained_recs_list.append(rec)
            self.recs_classes_names.append(rec.RECOMMENDER_NAME)
            self.RECOMMENDER_VERSION += rec.RECOMMENDER_NAME[:(len(rec.RECOMMENDER_NAME)-len("Recommender"))]
            
        self.hybrids_versions = []
        for rec in trained_recs:
            self.trained_recs_list.append(rec)
            if issubclass(rec.__class__, BaseHybridRecommender):
                self.hybrids_versions.append(rec.RECOMMENDER_VERSION)
            self.recs_classes_list.append(rec.__class__)
            self.recs_classes_names.append(rec.RECOMMENDER_NAME)
            self.RECOMMENDER_VERSION += rec.RECOMMENDER_NAME[:(len(rec.RECOMMENDER_NAME)-len("Recommender"))]
        
        
        
    def load_model(self, folder_path, file_name = None):
        super(BaseHybridRecommender, self).load_model(folder_path, file_name)

        self.trained_recs_list = []
        hybrid_idx = 0
        for rec_class_name in self.recs_classes_names:
            rec_class = utils.get_rec_class_by_name(rec_class_name)
            self.recs_classes_list.append(rec_class)
            
            if issubclass(rec_class, BaseHybridRecommender):
                rec = rec_class(self.URM_train, 
                                recs_on_urm_splitted=self.recs_on_urm_splitted, 
                                dataset_version = self.dataset_version)
                rec.load_model_by_version(self.hybrids_versions[hybrid_idx])
                hybrid_idx += 1
                
            else:
                rec = utils.load_best_model(self.URM_train, 
                                            rec_class, 
                                            dataset_version=self.dataset_version,
                                            optimization=self.recs_on_urm_splitted)
            self.trained_recs_list.append(rec)
            self.RECOMMENDER_VERSION += rec.RECOMMENDER_NAME[:(len(rec.RECOMMENDER_NAME)-len("Recommender"))]
            
            
            
    def load_model_by_version(self, version):
        folder = "recommendations"
        folder = os.path.join(folder, self.dataset_version)
        folder = os.path.join(folder, self.RECOMMENDER_NAME)
        folder = os.path.join(folder, version)
        if self.recs_on_urm_splitted:
            folder = os.path.join(folder, "optimization")
        else:
            folder = os.path.join(folder, "best")
        self.load_model(folder, self.RECOMMENDER_NAME + "_best_model.zip")
        
        
        
    def get_best_res_on_validation(self, version, metric="MAP"):
        folder = "recommendations"
        folder = os.path.join(folder, self.dataset_version)
        folder = os.path.join(folder, self.RECOMMENDER_NAME)
        folder = os.path.join(folder, version)
        folder = os.path.join(folder, "optimization")
        data_loader = DataIO(folder_path = folder)
        hyperparams_file = self.RECOMMENDER_NAME + "_metadata.zip"
        if os.path.exists(os.path.join(folder, hyperparams_file)):
            search_metadata = data_loader.load_data(hyperparams_file)
            return search_metadata["result_on_validation_best"][metric]
        else:
            return {}
        
        
        
    def get_best_res_on_validation_during_search(self, metric="MAP"):
        folder = "recommendations"
        folder = os.path.join(folder, self.dataset_version)
        folder = os.path.join(folder, self.RECOMMENDER_NAME)
        folder = os.path.join(folder, "hyperparams_search")
        data_loader = DataIO(folder_path = folder)
        hyperparams_file = self.RECOMMENDER_NAME + "_metadata.zip"
        if os.path.exists(os.path.join(folder, hyperparams_file)):
            search_metadata = data_loader.load_data(hyperparams_file)
            return search_metadata["result_on_validation_best"][metric]
        else:
            return {}