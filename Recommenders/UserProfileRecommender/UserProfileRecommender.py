import utils
import numpy as np 
import scipy.sparse as sps
import matplotlib.pyplot as plt
from Evaluation.Evaluator import EvaluatorHoldout
import Data_manager.split_functions.split_train_validation_random_holdout as split
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Hybrids.BaseHybridRecommender import BaseHybridRecommender
from Recommenders.DataIO import DataIO
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Categorical, Real, Integer

class UserProfileRec(BaseRecommender):
    RECOMMENDER_NAME = "UserProfileRecommender"
    def __init__(self, URM_train, ICM_train, cf_rec_classes = [], cb_rec_classes = [], cf_rec_versions = [], dataset_version='interactions-all-ones', best_model_for_user_profile_perc = 0.7, seed=None):
        super(UserProfileRec, self).__init__(URM_train)
        self.cf_rec_classes = cf_rec_classes.copy()
        self.cb_rec_classes = cb_rec_classes.copy()
        self.dataset_version = dataset_version
        self.ICM_train = ICM_train
        self.best_model_for_user_profile_perc = best_model_for_user_profile_perc
        self.cf_fitted_recs_dict_to_select = {}
        self.cb_fitted_recs_dict_to_select = {}
        self.URM_train_to_select = None 
        self.URM_val_to_select = None
        self.seed = seed
        self.cf_rec_versions = cf_rec_versions.copy()
        
        
    def results_of_all_recs_per_group(self, plot=True):
        if (self.URM_train_to_select == None and self.URM_val_to_select == None):
            self.URM_train_to_select, self.URM_val_to_select = split.split_train_in_two_percentage_global_sample(self.URM_train, train_percentage = self.best_model_for_user_profile_perc, seed=self.seed)
        
        users_to_ignore, _ = self.get_users_not_and_in_group(self.URM_train_to_select)
        if (len(self.cf_fitted_recs_dict_to_select) == 0 and len(self.cb_fitted_recs_dict_to_select) == 0):
            self.cf_fitted_recs_dict_to_select, self.cb_fitted_recs_dict_to_select = self.fit_all_recs(
                self.cf_rec_classes,
                self.cf_rec_versions,
                self.cb_rec_classes, 
                self.URM_train_to_select, 
                True,                               
                self.ICM_train)
            
        results_cf_rec_per_group = {}
        results_cb_rec_per_group = {}
    
        group_id = 0
        for users_to_ignore_per_group in users_to_ignore:
            print("Evaluating Group {}:".format(group_id))
            evaluator = EvaluatorHoldout(self.URM_val_to_select, cutoff_list=[self.cutoff], ignore_users=users_to_ignore_per_group)
    
            for (rec_class, rec_version), recommender in self.cf_fitted_recs_dict_to_select.items():
                result_df, results_run_string = evaluator.evaluateRecommender(recommender)
                print(results_run_string)
                if (rec_class, rec_version) in results_cf_rec_per_group:
                    results_cf_rec_per_group[(rec_class, rec_version)].append(result_df.loc[self.cutoff][self.metric])
                else:
                    results_cf_rec_per_group[(rec_class, rec_version)] = [result_df.loc[self.cutoff][self.metric]]
                
            for (rec_class, rec_version), recommender in self.cb_fitted_recs_dict_to_select.items():
                result_df, results_run_string = evaluator.evaluateRecommender(recommender)
                print(results_run_string)
                if (rec_class, rec_version) in results_cb_rec_per_group:
                    results_cb_rec_per_group[(rec_class, rec_version)].append(result_df.loc[self.cutoff][self.metric])
                else:
                    results_cb_rec_per_group[(rec_class, rec_version)] = [result_df.loc[self.cutoff][self.metric]]
                
            group_id += 1
            print("\n")
        
        if plot == True:
            self.plot_metric_per_user_group(results_cf_rec_per_group, results_cb_rec_per_group)
            
        return results_cf_rec_per_group, results_cb_rec_per_group # dict {RecClass: [metric[group0], metric[group1], ...]}

            
            
            
    def plot_metric_per_user_group(self, results_cf_rec_per_group, results_cb_rec_per_group):
        _ = plt.figure(figsize=(14, 9))
        s = 10
        for (rec_class, rec_version), recommender in self.cf_fitted_recs_dict_to_select.items():
            results = results_cf_rec_per_group[(rec_class, rec_version)]
            plt.scatter(x=np.arange(0,len(results)), y=results, label=rec_class.RECOMMENDER_NAME, s=s, marker='_')
            
        for (rec_class, rec_version), recommender in self.cb_fitted_recs_dict_to_select.items():
            results = results_cb_rec_per_group[(rec_class, rec_version)]
            plt.scatter(x=np.arange(0,len(results)), y=results, label=rec_class.RECOMMENDER_NAME, s=s, marker='_')
            
        plt.ylabel(self.metric)
        plt.xlabel('User Group')
        plt.legend()
        plt.show()
        
        
        
    def get_users_not_and_in_group(self, URM_train):
        users_not_in_group_list = []
        users_in_group_list = []
        
        if self.clustering_strategy == "equal-parts":
            profile_length = np.ediff1d(sps.csr_matrix(URM_train).indptr)
            block_size = int(len(profile_length) * (1/self.num_groups)) + 1
            sorted_users = np.argsort(profile_length)
            print("Users group size: " + str(block_size))
            print("\n")
    
            for group_id in range(0, self.num_groups):
                start_pos = group_id * block_size
                end_pos = min((group_id + 1) * block_size, len(profile_length))
    
                users_in_group = sorted_users[start_pos:end_pos]
                users_in_group_list.append(users_in_group)
                users_in_group_p_len = profile_length[users_in_group]
                print("Group {}, #users in group {}, average p.len {:.2f}, median {}, min {}, max {}".format(
                    group_id, 
                    users_in_group.shape[0],
                    users_in_group_p_len.mean(),
                    np.median(users_in_group_p_len),
                    users_in_group_p_len.min(),
                    users_in_group_p_len.max()))
        
                users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
                users_not_in_group = sorted_users[users_not_in_group_flag]
                users_not_in_group_list.append(users_not_in_group)
        
        elif self.clustering_strategy == "kmeans_ucm":
            ucm = utils.get_ucm() #crea metodo in utils
            X = ucm[["ProfileSeen","SeenInteractionCount","ProfileInfo","InfoInteractionCount"]].values
            k = 8
            ucm["cluster"] = utils.clusterize(X, k) #aggiungi metodo in utils
            ucm["user_id"] = ucm.index
            user_ids = ucm["user_id"]
            users_not_in_group_list = []
            users_in_group_list = []
            for i in range(k):
                users_in_group = ucm[ucm['cluster'] == i]
                users_in_group = users_in_group["user_id"]
                users_in_group_list.append(np.array(users_in_group))
                users_not_in_group_flag = np.isin(user_ids, users_in_group, invert=True)
                users_not_in_group = user_ids[users_not_in_group_flag]
                users_not_in_group_list.append(np.array(users_not_in_group))
        elif self.clustering_strategy == "kmeans_profile_length":
            print("\n")
    
        return users_not_in_group_list, users_in_group_list



    def fit_all_recs(self, cf_rec_classes, cf_rec_versions, cb_rec_classes, URM, with_validation, ICM):
        cf_recs_dict = {}
        cb_recs_dict = {}
                
        if self.seed is None:
            for recommender_class in cf_rec_classes:
                recommender_object = recommender_class(URM)
                recommender_object.fit(**(utils.get_best_model_hyperparameters(recommender_class, dataset_version=self.dataset_version)))
                cf_recs_dict[(recommender_class, recommender_object.RECOMMENDER_VERSION)] = recommender_object

            for recommender_class in cb_rec_classes:
                recommender_object = recommender_class(URM, ICM_train=ICM)
                recommender_object.fit(**(utils.get_best_model_hyperparameters(recommender_class, dataset_version=self.dataset_version)))
                cb_recs_dict[(recommender_class, recommender_object.RECOMMENDER_VERSION)] = recommender_object
        
        else:
            for recommender_class, recommender_version in zip(cf_rec_classes, cf_rec_versions):
                if issubclass(recommender_class, BaseHybridRecommender):
                    recommender_object = recommender_class(URM, 
                                                           recs_on_urm_splitted=with_validation, 
                                                           dataset_version=self.dataset_version)
                    recommender_object.load_model_by_version(recommender_version)

                else:
                    recommender_object = utils.load_best_model(URM, 
                                            recommender_class, 
                                            dataset_version=self.dataset_version, 
                                            optimization=with_validation)

                cf_recs_dict[(recommender_class, recommender_version)] = recommender_object
                
            for recommender_class in cb_rec_classes:
                recommender_object = utils.load_best_model(URM, 
                                            recommender_class, 
                                            dataset_version=self.dataset_version, 
                                            optimization=with_validation)
                try:
                    version = recommender_object.RECOMMENDER_VERSION
                except Exception as e:
                    version = ""
                cb_recs_dict[(recommender_class, version)] = recommender_object

    
        print("\n")
        return cf_recs_dict, cb_recs_dict
    
    
    
    # Assumption: the higher the metric, the better the result
    def fit(self, num_groups, cutoff=10, metric="MAP", plot=True, clustering_strategy="equal-parts"):
        '''
        select best recommender for each group
        '''
        self.clustering_strategy = clustering_strategy
        self.num_groups = num_groups
        self.cutoff = cutoff
        self.metric = metric
        results_cf_rec_per_group, results_cb_rec_per_group = self.results_of_all_recs_per_group(plot=plot)
        self.best_rec_per_group_list = []
        best_recs_cf = []
        best_recs_cb = []
        self.recs_names_per_group = []
    
        for group_idx in range(self.num_groups):
            curr_best = 0
            curr_best_res = 0
            curr_best_cf = 0
            for (rec_class, rec_version), results in results_cf_rec_per_group.items():
                if results[group_idx] > curr_best_res:
                    curr_best_res = results[group_idx]
                    curr_best = (rec_class, rec_version)
                    curr_best_cf = (rec_class, rec_version)
            for (rec_class, rec_version), results in results_cb_rec_per_group.items():
                if results[group_idx] > curr_best_res:
                    curr_best_res = results[group_idx]
                    curr_best = (rec_class, rec_version)
            if curr_best_cf == curr_best:
                best_recs_cf.append(curr_best_cf)
            else:
                best_recs_cb.append(curr_best)
            self.best_rec_per_group_list.append(curr_best)
    
        self.best_cf_recs_class_no_duplicates = []
        self.best_cf_recs_version_no_duplicates = []
        for (rec_class, rec_version) in best_recs_cf:
            if rec_class not in self.best_cf_recs_class_no_duplicates:
                self.best_cf_recs_class_no_duplicates.append(rec_class)
                self.best_cf_recs_version_no_duplicates.append(rec_version)
            
        self.best_cb_recs_class_no_duplicates = []
        for (rec_class, rec_version) in best_recs_cb:
            if rec_class not in self.best_cb_recs_class_no_duplicates:
                self.best_cb_recs_class_no_duplicates.append(rec_class)
                
        self.best_cf_fitted_rec_dict, self.best_cb_fitted_rec_dict = self.fit_all_recs(
            self.best_cf_recs_class_no_duplicates,
            self.best_cf_recs_version_no_duplicates,
            self.best_cb_recs_class_no_duplicates, 
            self.URM_train, 
            False,
            self.ICM_train)
        
        for (rec_class, rec_version) in self.best_rec_per_group_list:
            self.recs_names_per_group.append(rec_class.RECOMMENDER_NAME)
        
        version = "n_groups-" + str(self.num_groups)
        for (rec_class, rec_version) in self.best_cf_fitted_rec_dict.keys():
            version += "_" + rec_class.RECOMMENDER_NAME + "-" + rec_version
            
        for (rec_class, rec_version) in self.best_cb_fitted_rec_dict.keys():
            version += "_" + rec_class.RECOMMENDER_NAME + "-" + rec_version

        self.RECOMMENDER_VERSION = version
        
        
        
    def fit_best_recs(self, URM, with_validation):
        self.best_cf_fitted_rec_dict, self.best_cb_fitted_rec_dict = self.fit_all_recs(
            self.best_cf_recs_class_no_duplicates,
            self.best_cf_recs_version_no_duplicates,
            best_cb_recs_class_no_duplicates, 
            URM, 
            with_validation,
            self.ICM_train)
        
        for (rec_class, rec_version) in self.best_rec_per_group_list:
            self.recs_names_per_group.append(rec_class.RECOMMENDER_NAME)
        
        version = "n_groups-" + str(self.num_groups)
        for (rec_class, rec_version) in self.best_cf_fitted_rec_dict.keys():
            version += "_" + rec_class.RECOMMENDER_NAME + "-" + rec_version
            
        for (rec_class, rec_version) in self.best_cb_fitted_rec_dict.keys():
            version += "_" + rec_class.RECOMMENDER_NAME + "-" + rec_version

        self.RECOMMENDER_VERSION = version
                        
    

    def _compute_item_score(self, user_id_array, items_to_compute = None):
        '''
        rec_class_per_group: list of n recommender classes, with n that is the number of users groups
        '''
        all_recs = {}
        all_recs.update(self.best_cf_fitted_rec_dict)
        all_recs.update(self.best_cb_fitted_rec_dict)
        
        print("num all recs best: #" +  str(len(all_recs)))
    
        _, users_in_group = self.get_users_not_and_in_group(self.URM_train)
        
        print("num users groups (masks): #" +  str(len(users_in_group)))
    
        scores = np.ndarray((0, self.n_items))
        users = np.ndarray((0,))
        group_id = 0
        for (rec_class, rec_version) in self.best_rec_per_group_list:
            rec_for_group = all_recs.get((rec_class, rec_version))
            user_to_recommend = users_in_group[group_id] 
            mask = np.isin(user_to_recommend, user_id_array)
            user_to_recommend = user_to_recommend[mask]
            recommendation_for_group = rec_for_group._compute_item_score(user_id_array=user_to_recommend,
                                                                         items_to_compute=items_to_compute)
            print("scores.shape:" + str(scores.shape))
            scores = np.concatenate([scores, recommendation_for_group])
            print("scores.shape:" + str(scores.shape) +"\n")
            print("users.shape:" + str(users.shape))
            users = np.concatenate([users, user_to_recommend])
            print("users.shape:" + str(users.shape) + "\n")
            group_id += 1

        indices_to_sort = np.argsort(users)
        users = users[indices_to_sort]
        print("final users.shape:" + str(users.shape) + "\n")
        scores = scores[indices_to_sort, :]
        print("final scores.shape:" + str(scores.shape) +"\n")

        return scores 
    
    
    
    def optimize_2hybrids_on_group(self, num_groups, hybrid_class, trained_recs_arg, n_cases=50, perc_random_starts=0.3, cutoff=10, metric="MAP", clustering_strategy="equal-parts", block_size=None, allow_normalization=True, allow_alphas_sum_to_one=False, root="", previous_hyb=[]):
        self.clustering_strategy = clustering_strategy
        self.num_groups = num_groups
        self.cutoff = cutoff
        self.metric = metric
        
        import os
        
        hyperparameters_range_dictionary = self.get_hybrid_weights_range_dict(2, low=0., high=10., prior="uniform")
        if allow_normalization:
            hyperparameters_range_dictionary["normalize"] = Categorical(["L1", None, "fro", "inf"])
        else:
            hyperparameters_range_dictionary["normalize"] = Categorical([None])
        hyperparameters_range_dictionary["alphas_sum_to_one"] = Categorical([allow_alphas_sum_to_one])
        
        if (self.URM_train_to_select == None and self.URM_val_to_select == None):
            self.URM_train_to_select, self.URM_val_to_select = split.split_train_in_two_percentage_global_sample(
                self.URM_train, 
                train_percentage = self.best_model_for_user_profile_perc, 
                seed=self.seed)
        
        users_to_ignore, _ = self.get_users_not_and_in_group(self.URM_train_to_select)
        
        groupidx = 1
        for users_not_in_group in users_to_ignore:
            evaluator_validation = EvaluatorHoldout(self.URM_val_to_select, 
                                                    cutoff_list=[10], 
                                                    ignore_users=users_not_in_group)
            
            if len(trained_recs_arg) == 1:
                trained_recs_arg.append(previous_hyb[groupidx-1])
                
            dict_args = {
                "recs_on_urm_splitted": True, 
                "dataset_version": self.dataset_version, 
                "not_trained_recs_classes": [], 
                "trained_recs": trained_recs_arg
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [self.URM_train_to_select] ,   
                CONSTRUCTOR_KEYWORD_ARGS = dict_args,
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {},
                EARLYSTOPPING_KEYWORD_ARGS = {}, 
            )
            
            utils.bayesian_search(
                hybrid_class, 
                recommender_input_args, 
                hyperparameters_range_dictionary, 
                evaluator_validation,
                dataset_version=self.dataset_version,
                n_cases=n_cases,
                perc_random_starts=perc_random_starts,
                block_size = block_size,
                cust_output_folder = os.path.join(root,
                                                  os.path.join( 
                                                  os.path.join("per_group_hyperparams_search", "group" + str(groupidx)), "optimization"))
            )
            
            groupidx += 1
    
    
    
    
    def save_model(self, folder_path, file_name = None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))
        
        self.recs_names_per_group = []
        for rec_class in self.best_rec_per_group_list:
            self.recs_names_per_group.append(rec_class.RECOMMENDER_NAME)
            
        self.cf_rec_names = []
        for rec_class in self.cf_rec_classes:
            self.cf_rec_names.append(rec_class.RECOMMENDER_NAME)
        
        self.cb_rec_names = []
        for rec_class in self.cb_rec_classes:
            self.cb_rec_names.append(rec_class.RECOMMENDER_NAME)
            

        data_dict_to_save = {"recs_names_per_group": self.recs_names_per_group,
                            "cf_rec_names": self.cf_rec_names,
                            "cb_rec_names": self.cb_rec_names,
                            "num_group": self.num_groups,
                            "clustering_strategy": self.clustering_strategy}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")
        
        
        
        
    def load_model(self, folder_path, file_name = None):
        ''' TO BE IMPLEMENTED
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])
                
        for rec_name in self.res_names_per_group

        self._print("Loading complete")
        '''
        
        
        
    def get_hybrid_weights_range_dict(self, number_of_recommender, low=0., high=1., prior="uniform"):
        hyperpar_dict = {}
        for i in range(number_of_recommender):
            weight = "w"+str(i)
            hyperpar_dict[weight] = Real(low=low, high=high, prior=prior)
        return hyperpar_dict