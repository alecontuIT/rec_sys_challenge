import os
import numpy as np
import utils
from Recommenders.Hybrids.DiffStructHybridRecommender import DiffStructHybridRecommender
from recmodels import ItemKNNSimilarityHybridRec
from Evaluation.Evaluator import EvaluatorHoldout
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Categorical, Real, Integer
        
class IterativeHybridOptimizer():
    def __init__(self, URM_all, URM_train, URM_val, rec_classes_list, dataset_version, load_scores_from_saved=True):
        self.URM_train = URM_train
        self.URM_all = URM_all
        self.rec_classes_list = rec_classes_list
        self.dataset_version = dataset_version
        self.URM_val = URM_val
        self.load_scores_from_saved = load_scores_from_saved
        
        # load models trained on train part of the dataset
        self.recs = []
        self.validation_results = []
        for rec_class in self.rec_classes_list:
            val_res = utils.get_best_res_on_validation(rec_class, dataset_version=self.dataset_version)
            self.validation_results.append(val_res)
            print(val_res)
        
        # sort by validation results
        sorted_idx = np.argsort(self.validation_results)[::-1]
        self.validation_results = np.take_along_axis(np.array(self.validation_results), sorted_idx, None)
        self.rec_classes_list = np.take_along_axis(np.array(self.rec_classes_list), sorted_idx, None)
        print(self.validation_results, self.rec_classes_list)
    
    def incremental_bayesian_search(self, n_cases, perc_random_starts, block_size=None, cutoff=10):
        pass
    
    def on_end_search(**kwargs):
        rec = utils.load_best_model(self.URM_all, 
                      self.recommender_hybrid, 
                      self.dataset_version, 
                      True,
                      **kwargs)
        utils.submission(rec, dataset_version, override=True)
        
        folder = utils.get_folder_best_model(self.recommender_hybrid, self.dataset_version)
        for file in folder:
            if "hyperparams_search" in file or self.recommender_hybrid.RECOMMENDER_VERSION in file:
                utils.copy_all_files(os.path.join(folder, file), 
                                     os.path.join(os.path.join(folder, self.final_folder), file), 
                                     remove_source=True)
            
            
class DiffStructHybridOptimizer(IterativeHybridOptimizer):
    def incremental_bayesian_search(self, n_cases, perc_random_starts, block_size=None, cutoff=10):
        self.alphas = []
        rec1_class = self.rec_classes_list[0]
        self.final_folder = rec1_class.RECOMMENDER_NAME
        old_val_res = self.validation_results[0]
        self.recommender_hybrid = DiffStructHybridRecommender

        hyperparameters_range_dictionary = utils.get_hybrid_weights_range_dict(2, low=0., high=10., prior="uniform")
        hyperparameters_range_dictionary["normalize"] = Categorical(["L1", None, "fro", "inf", "-inf"])
        idx = 0
        evaluator_validation = EvaluatorHoldout(self.URM_val, cutoff_list=[cutoff])
        if block_size == None:
            block_size = len(evaluator_validation.users_to_evaluate)
            
        for rec2_class in self.rec_classes_list[1:]:
            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [self.URM_train, 
                                               self.dataset_version, 
                                               [rec1_class, rec2_class], 
                                               self.load_scores_from_saved, 
                                               True, 
                                               evaluator_validation.users_to_evaluate,
                                               None] ,   
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {},
                EARLYSTOPPING_KEYWORD_ARGS = {}, 
            )
            
#            output_folder_path = str(idx + 1) + "_"
            utils.bayesian_search(
                self.recommender_hybrid, 
                recommender_input_args, 
                hyperparameters_range_dictionary, 
                evaluator_validation,
                dataset_version=self.dataset_version,
                n_cases=n_cases,
                perc_random_starts=perc_random_starts,
 #               cust_output_folder_path = output_folder_path,
                block_size = block_size
            )
            
            val_res = utils.get_best_res_on_validation(self.recommender_hybrid, 
                                                 dataset_version=self.dataset_version, 
                                                 optimization=True)#,
                                                 #custom_folder_name = output_folder_path)
            if val_res > old_val_res:
                print("******* Validation Metric improved! Model {} kept. *******".format(rec2_class.RECOMMENDER_NAME))
                old_val_res = val_res
                
                optimized_hybrid = utils.load_model_from_hyperparams_search_folder(self.URM_train,
                                                             self.recommender_hybrid, 
                                                             dataset_version=self.dataset_version, 
                                                             load_scores_from_saved=self.load_scores_from_saved, 
                                                             recs_on_urm_splitted=True, 
                                                             user_id_array_val=evaluator_validation.users_to_evaluate)
                path = utils.get_hyperparams_search_output_folder(self.recommender_hybrid, self.dataset_version)
                utils.copy_all_files(path, 
                                     path[:(len(path)-1)] + str(idx) + "/", 
                                     remove_source=False)
                utils.optimization_terminated(optimized_hybrid, self.dataset_version, override = True)
                rec1_class = self.recommender_hybrid
                
                # Fix the normalization after first hybrid found
                hyperparameters_range_dictionary["normalize"] = Categorical([optimized_hybrid.normalize])
                self.final_folder += "-" + rec2_class.RECOMMENDER_NAME
                
            else:
                print("******* Validation Metric not improved! Model {} discarded. *******".format(rec2_class.RECOMMENDER_NAME))
            
            idx += 1
        
        kwargs = {"load_scores_from_saved": True, 
          "recs_on_urm_splitted": False, 
          "user_id_array_val": utils.get_users_for_submission(), 
          "new_item_scores_file_name_root": "on_all_urm_"}
        on_end_search(**kwargs)
        
            
            
class ItemSimilarityKNNHybridOptimizer(IterativeHybridOptimizer):
    def incremental_bayesian_search(self, n_cases, perc_random_starts, block_size=None, cutoff=10):
        self.alphas = []
        rec1_class = self.rec_classes_list[0]
        old_val_res = self.validation_results[0]
        self.recommender_hybrid = ItemKNNSimilarityHybridRec
        self.final_folder = rec1_class.RECOMMENDER_NAME

        hyperparameters_range_dictionary = {}
        hyperparameters_range_dictionary["topK"] = Integer(5, 1000)
        hyperparameters_range_dictionary["alpha"] =  Real(low=0, high=1, prior="uniform")
        
        idx = 0
        evaluator_validation = EvaluatorHoldout(self.URM_val, cutoff_list=[cutoff])
        if block_size == None:
            block_size = len(evaluator_validation.users_to_evaluate)
            
        for rec2_class in self.rec_classes_list[1:]:
            rec1 = utils.load_best_model(self.URM_train, 
                                   rec1_class, 
                                   self.dataset_version, 
                                   optimization=True)
            rec2 = utils.load_best_model(self.URM_train, 
                                   rec2_class, 
                                   self.dataset_version, 
                                   optimization=True)
            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [self.URM_train] ,   
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [rec1.W_sparse, rec2.W_sparse],
                FIT_KEYWORD_ARGS = {"similarities_string": rec1.RECOMMENDER_NAME + "_" + rec2.RECOMMENDER_NAME + "_" },
                EARLYSTOPPING_KEYWORD_ARGS = {}, 
            )
            
#            output_folder_path = str(idx + 1) + "_"
            utils.bayesian_search(
                self.recommender_hybrid, 
                recommender_input_args, 
                hyperparameters_range_dictionary, 
                evaluator_validation,
                dataset_version=self.dataset_version,
                n_cases=n_cases,
                perc_random_starts=perc_random_starts,
 #               cust_output_folder_path = output_folder_path,
                block_size = block_size
            )
            
            val_res = utils.get_best_res_on_validation(self.recommender_hybrid, 
                                                 dataset_version=self.dataset_version, 
                                                 optimization=True)#,
                                                 #custom_folder_name = output_folder_path)
            if val_res > old_val_res:
                print("******* Validation Metric improved! Model {} kept. *******".format(rec2_class.RECOMMENDER_NAME))
                old_val_res = val_res
                
                optimized_hybrid = utils.load_model_from_hyperparams_search_folder(self.URM_train,
                                                             self.recommender_hybrid, 
                                                             dataset_version=self.dataset_version)
                path = utils.get_hyperparams_search_output_folder(self.recommender_hybrid, self.dataset_version)
                utils.copy_all_files(path, 
                                     path[:(len(path)-1)] + str(idx) + "/", 
                                     remove_source=False)
                utils.optimization_terminated(optimized_hybrid, self.dataset_version, override = True)
                rec1_class = self.recommender_hybrid
                self.final_folder += "-" + rec2_class.RECOMMENDER_NAME
                
            else:
                print("******* Validation Metric not improved! Model {} discarded. *******".format(rec2_class.RECOMMENDER_NAME))
            
            idx += 1
            
        on_end_search()