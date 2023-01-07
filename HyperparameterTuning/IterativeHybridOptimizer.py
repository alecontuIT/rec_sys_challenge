import os
import numpy as np
import utils
from Recommenders.Hybrids.DiffStructHybridRecommender import DiffStructHybridRecommender
from Evaluation.Evaluator import EvaluatorHoldout
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Categorical
        
class OptimizerDiffStructHybridRecommender():
    def __init__(self, URM_train, URM_val, rec_classes_list, dataset_version, load_scores_from_saved=True, user_id_array_val=None):
        self.URM_train = URM_train
        self.rec_classes_list = rec_classes_list
        self.dataset_version = dataset_version
        self.URM_val = URM_val
        self.load_scores_from_saved = load_scores_from_saved
        self.user_id_array_val = user_id_array_val
        
        # load models trained on train part of the dataset
        self.recs = []
        self.validation_results = []
        for rec_class in self.rec_classes_list:
            val_res = utils.get_best_res_on_validation(rec_class, dataset_version=self.dataset_version)
            self.validation_results.append(val_res)
        
        # sort by validation results
        sorted_idx = np.argsort(self.validation_results)[::-1]
        self.validation_results = np.take_along_axis(np.array(self.validation_results), sorted_idx, None)
        self.rec_classes_list = np.take_along_axis(np.array(self.rec_classes_list), sorted_idx, None)
        print(self.validation_results, self.rec_classes_list)
            
            
            
    def incremental_bayesian_search(self, n_cases, perc_random_starts, block_size=None, cutoff=10):
        self.alphas = []
        rec1_class = self.rec_classes_list[0]
        old_val_res = self.validation_results[0]
        hyb_rec_class = DiffStructHybridRecommender

        hyperparameters_range_dictionary = utils.get_hybrid_weights_range_dict(2, low=0., high=1., prior="uniform")
        hyperparameters_range_dictionary["normalize"] = Categorical(["L1", None, "fro", "inf", "-inf"])
            
        for rec2_class in self.rec_classes_list[1:]:
            evaluator_validation = EvaluatorHoldout(self.URM_val, cutoff_list=[cutoff])
            if block_size == None:
                block_size = len(evaluator_validation.users_to_evaluate)
            
            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [self.URM_train, 
                                               self.dataset_version, 
                                               [rec1_class, rec2_class], 
                                               self.load_scores_from_saved, 
                                               True, 
                                               evaluator_validation.users_to_evaluate] ,   
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {},
                EARLYSTOPPING_KEYWORD_ARGS = {}, 
            )
            
#            output_folder_path = str(idx + 1) + "_"
            utils.bayesian_search(
                hyb_rec_class, 
                recommender_input_args, 
                hyperparameters_range_dictionary, 
                evaluator_validation,
                dataset_version=self.dataset_version,
                n_cases=n_cases,
                perc_random_starts=perc_random_starts,
 #               cust_output_folder_path = output_folder_path,
                block_size = block_size
            )
            
            val_res = utils.get_best_res_on_validation(hyb_rec_class, 
                                                 dataset_version=self.dataset_version, 
                                                 optimization=True)#,
                                                 #custom_folder_name = output_folder_path)
            if val_res > old_val_res:
                print("******* Validation Metric improved! Model {} kept. *******".format(rec2_class.RECOMMENDER_NAME))
                old_val_res = val_res
                
                optimized_hybrid = utils.load_model_from_hyperparams_search_folder(self.URM_train,
                                                             hyb_rec_class, 
                                                             dataset_version=self.dataset_version, 
                                                             optimization=True,  # get hyperparams from 'optimization' folder
                                                             load_scores_from_saved=self.load_scores_from_saved, 
                                                             user_id_array_val=evaluator_validation.users_to_evaluate)
                utils.optimization_terminated(optimized_hybrid, self.dataset_version, override = True)
                rec1_class = hyb_rec_class
                
                # Fix the normalization after first hybrid found
                hyperparameters_range_dictionary["normalize"] = Categorical([optimized_hybrid.normalize])
                
            else:
                print("******* Validation Metric not improved! Model {} discarded. *******".format(rec2_class.RECOMMENDER_NAME))
            
            idx += 1