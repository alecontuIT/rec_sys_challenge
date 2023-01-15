import os
import numpy as np
import utils
from Recommenders.Hybrids.BaseHybridRecommender import BaseHybridRecommender
from Recommenders.Hybrids.DiffStructHybridRecommender import DiffStructHybridRecommender
from recmodels import ItemKNNSimilarityHybridRec
from Evaluation.Evaluator import EvaluatorHoldout
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Categorical, Real, Integer
        
class IterativeHybridOptimizer():
    def __init__(self, URM_all, URM_train, URM_val, dataset_version, not_trained_recs_classes=[], trained_recs=[]):
        
        self.URM_train = URM_train
        self.URM_val = URM_val
        self.URM_all = URM_all
        
        self.dataset_version = dataset_version      
            
        self.trained_recs = trained_recs
        self.rec_classes_list = not_trained_recs_classes
        
        self.validation_results = []
        for rec in self.trained_recs:
            if issubclass(rec.__class__, BaseHybridRecommender):
                val_res = rec.get_best_res_on_validation(rec.RECOMMENDER_VERSION, metric="MAP")
            else:
                val_res = utils.get_best_res_on_validation(rec_class, dataset_version=self.dataset_version)
            
            self.validation_results.append(val_res)
            
            print(val_res)
        
        
        for rec_class in self.rec_classes_list:
            assert not issubclass(rec_class, BaseHybridRecommender), "Error: hybrid recommenders must be provided as objects, already fitted.\n"

            rec = utils.load_best_model(self.URM_train, 
                                            rec_class, 
                                            dataset_version=self.dataset_version, 
                                            optimization=True)
            
            val_res = utils.get_best_res_on_validation(rec_class, dataset_version=self.dataset_version)
            self.validation_results.append(val_res)
            self.trained_recs.append(rec)
            print(val_res)
        
        sorted_idx = np.argsort(self.validation_results)[::-1]
        self.validation_results = np.take_along_axis(np.array(self.validation_results), sorted_idx, None)
        self.trained_recs = np.take_along_axis(np.array(self.trained_recs), sorted_idx, None)
        
        self.is_fitted_mask = [True] * len(self.validation_results)
        
        print(self.validation_results)     
        
        
    
    def incremental_bayesian_search(self, n_cases, perc_random_starts, block_size=None, cutoff=10):
        pass
    
    
    
    def on_end_search(**kwargs):
        rec = utils.load_best_model(self.URM_all, 
                      self.recommender_hybrid, 
                      self.dataset_version, 
                      True,
                      **kwargs)
        utils.submission(rec, dataset_version, override=True)
        
        folder = "recommendations"
        folder = os.path.join(folder, dataset_version)
        folder = os.path.join(folder, DiffStructHybridRecommender.RECOMMENDER_NAME) 
        for file in os.listdir(folder):
            if "hyperparams_search" in file or DiffStructHybridRecommender.RECOMMENDER_VERSION in file:
                utils.copy_all_files(os.path.join(folder, file), 
                             os.path.join(os.path.join(folder, self.final_folder), file), 
                             remove_source_folder=True)
                
            
            
class DiffStructHybridOptimizer(IterativeHybridOptimizer):
    def get_hybrid_weights_range_dict(self, number_of_recommender, low=0., high=1., prior="uniform"):
        hyperpar_dict = {}
        for i in range(number_of_recommender):
            weight = "w"+str(i)
            hyperpar_dict[weight] = Real(low=low, high=high, prior=prior)
        return hyperpar_dict
    
    
    
    def incremental_bayesian_search(self, n_cases, perc_random_starts, block_size=None, cutoff=10, allow_normalization=True, allow_alphas_sum_to_one=False):
        self.alphas = []
        old_val_res = self.validation_results[0]
        self.recommender_hybrid = DiffStructHybridRecommender
        optimized_hybrid = None

        hyperparameters_range_dictionary = self.get_hybrid_weights_range_dict(2, low=0., high=10., prior="uniform")
        if allow_normalization:
            hyperparameters_range_dictionary["normalize"] = Categorical(["L1", None, "fro", "inf"])
        else:
            hyperparameters_range_dictionary["normalize"] = Categorical([None])
            
        hyperparameters_range_dictionary["alphas_sum_to_one"] = Categorical([allow_alphas_sum_to_one])

        evaluator_validation = EvaluatorHoldout(self.URM_val, cutoff_list=[cutoff])
        if block_size == None:
            block_size = len(evaluator_validation.users_to_evaluate)
            
        not_trained_recs_classes_arg = []
        not_trained_recs_classes_idx = 0
        trained_recs_arg = []
        trained_recs_idx = 0
        
        if self.is_fitted_mask[0]:
            trained_recs_arg.append(self.trained_recs[trained_recs_idx])
            if issubclass(self.trained_recs[trained_recs_idx].__class__, BaseHybridRecommender):
                string_name = self.trained_recs[trained_recs_idx].RECOMMENDER_VERSION
            else:
                string_name = self.trained_recs[trained_recs_idx].RECOMMENDER_NAME
            trained_recs_idx += 1
        else:
            not_trained_recs_classes_arg.append(self.rec_classes_list[not_trained_recs_classes_idx])
            string_name = self.rec_classes_list[not_trained_recs_classes_idx].RECOMMENDER_NAME
            not_trained_recs_classes_idx += 1

            
        print("******* Best model in the list provided: {}, with MAP = {} *******".format(string_name, old_val_res))
        for idx in range(1, len(self.validation_results)):
            if self.is_fitted_mask[idx]:
                trained_recs_arg.append(self.trained_recs[trained_recs_idx])
                if issubclass(self.trained_recs[trained_recs_idx].__class__, BaseHybridRecommender):
                    string_name = self.trained_recs[trained_recs_idx].RECOMMENDER_VERSION
                else:
                    string_name = self.trained_recs[trained_recs_idx].RECOMMENDER_NAME
                trained_recs_idx += 1
            else:
                not_trained_recs_classes_arg.append(self.rec_classes_list[not_trained_recs_classes_idx])
                string_name = self.rec_classes_list[not_trained_recs_classes_idx].RECOMMENDER_NAME
                not_trained_recs_classes_idx += 1
                
            print("******* Optimize with model {} *******".format(string_name))
                
            dict_args = {
                "recs_on_urm_splitted": True, 
                "dataset_version": self.dataset_version, 
                "not_trained_recs_classes": not_trained_recs_classes_arg, 
                "trained_recs": trained_recs_arg
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [self.URM_train] ,   
                CONSTRUCTOR_KEYWORD_ARGS = dict_args,
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {},
                EARLYSTOPPING_KEYWORD_ARGS = {}, 
            )
            
            utils.bayesian_search(
                self.recommender_hybrid, 
                recommender_input_args, 
                hyperparameters_range_dictionary, 
                evaluator_validation,
                dataset_version=self.dataset_version,
                n_cases=n_cases,
                perc_random_starts=perc_random_starts,
                block_size = block_size
            )
            
            tmp = self.recommender_hybrid(self.URM_train, True, self.dataset_version)
            val_res = tmp.get_best_res_on_validation_during_search()
            
            if val_res > old_val_res:
                print("******* Validation Metric improved! Validation result of Model {} equal to {}. Model kept. *******".format(string_name, val_res))
                old_val_res = val_res
                
                optimized_hybrid = utils.load_model_from_hyperparams_search_folder(self.URM_train,
                                                             self.recommender_hybrid, 
                                                             dataset_version=self.dataset_version)

                utils.optimization_terminated(optimized_hybrid, self.dataset_version, override = True)
                trained_recs_arg = []
                not_trained_recs_classes_arg = []
                trained_recs_arg.append(optimized_hybrid)
                
                # Fix the normalization after first hybrid found
                if allow_normalization:
                    hyperparameters_range_dictionary["normalize"] = Categorical([optimized_hybrid.normalize])
                
            else:
                print("******* Validation Metric not improved! Validation result of Model {} equal to {}. Model discarded. *******".format(string_name, val_res))
                not_trained_recs_classes_arg = []
            
        return optimized_hybrid
    
    
    
    def inverse_incremental_bayesian_search(self, n_cases, perc_random_starts, block_size=None, cutoff=10, allow_normalization=False, allow_alphas_sum_to_one=True):
        self.alphas = []
        idx_res = -1
        old_val_res = self.validation_results[idx_res]
        self.recommender_hybrid = DiffStructHybridRecommender
        optimized_hybrid = None

        hyperparameters_range_dictionary = self.get_hybrid_weights_range_dict(2, low=0., high=10., prior="uniform")
        if allow_normalization:
            hyperparameters_range_dictionary["normalize"] = Categorical(["L1", None, "fro", "inf", "-inf"])
        else:
            hyperparameters_range_dictionary["normalize"] = Categorical([None])
            
        hyperparameters_range_dictionary["alphas_sum_to_one"] = Categorical([allow_alphas_sum_to_one])

        evaluator_validation = EvaluatorHoldout(self.URM_val, cutoff_list=[cutoff])
        if block_size == None:
            block_size = len(evaluator_validation.users_to_evaluate)
            
        not_trained_recs_classes_arg = []
        not_trained_recs_classes_idx = -1
        trained_recs_arg = []
        trained_recs_idx = -1
        
        if self.is_fitted_mask[-1]:
            trained_recs_arg.append(self.trained_recs[trained_recs_idx])
            if issubclass(self.trained_recs[trained_recs_idx].__class__, BaseHybridRecommender):
                string_name = self.trained_recs[trained_recs_idx].RECOMMENDER_VERSION
            else:
                string_name = self.trained_recs[trained_recs_idx].RECOMMENDER_NAME
            trained_recs_idx -= 1
        else:
            not_trained_recs_classes_arg.append(self.rec_classes_list[not_trained_recs_classes_idx])
            string_name = self.rec_classes_list[not_trained_recs_classes_idx].RECOMMENDER_NAME
            not_trained_recs_classes_idx -= 1

            
        print("******* Worst model in the list provided: {}, with MAP = {} *******".format(string_name, old_val_res))
        for idx in range(len(self.validation_results) - 1, -1, -1):
            if self.is_fitted_mask[idx]:
                trained_recs_arg.append(self.trained_recs[trained_recs_idx])
                if issubclass(self.trained_recs[trained_recs_idx].__class__, BaseHybridRecommender):
                    string_name = self.trained_recs[trained_recs_idx].RECOMMENDER_VERSION
                else:
                    string_name = self.trained_recs[trained_recs_idx].RECOMMENDER_NAME
                
                trained_recs_idx -= -1
                
            else:
                not_trained_recs_classes_arg.append(self.rec_classes_list[not_trained_recs_classes_idx])
                string_name = self.rec_classes_list[not_trained_recs_classes_idx].RECOMMENDER_NAME
                not_trained_recs_classes_idx -= -1
            
            print("******* Optimize with model {} *******".format(string_name))
                
            dict_args = {
                "recs_on_urm_splitted": True, 
                "dataset_version": self.dataset_version, 
                "not_trained_recs_classes": not_trained_recs_classes_arg, 
                "trained_recs": trained_recs_arg
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [self.URM_train] ,   
                CONSTRUCTOR_KEYWORD_ARGS = dict_args,
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {},
                EARLYSTOPPING_KEYWORD_ARGS = {}, 
            )
            
            utils.bayesian_search(
                self.recommender_hybrid, 
                recommender_input_args, 
                hyperparameters_range_dictionary, 
                evaluator_validation,
                dataset_version=self.dataset_version,
                n_cases=n_cases,
                perc_random_starts=perc_random_starts,
                block_size = block_size
            )
            
            tmp = self.recommender_hybrid(self.URM_train, True, self.dataset_version)
            val_res = tmp.get_best_res_on_validation_during_search()
            
            if val_res > old_val_res:
                print("******* Validation Metric improved! Validation result of Model {} equal to {}. Model kept. *******".format(string_name, val_res))
                old_val_res = val_res
                
                optimized_hybrid = utils.load_model_from_hyperparams_search_folder(self.URM_train,
                                                             self.recommender_hybrid, 
                                                             dataset_version=self.dataset_version)

                utils.optimization_terminated(optimized_hybrid, self.dataset_version, override = True)
                trained_recs_arg = []
                not_trained_recs_classes_arg = []
                trained_recs_arg.append(optimized_hybrid)
                
                # Fix the normalization after first hybrid found
                if allow_normalization:
                    hyperparameters_range_dictionary["normalize"] = Categorical([optimized_hybrid.normalize])
                
            else:
                print("******* Validation Metric not improved! Validation result of Model {} equal to {}. Model discarded. *******".format(string_name, val_res))
                not_trained_recs_classes_arg = []
            
        return optimized_hybrid
        
            
            
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