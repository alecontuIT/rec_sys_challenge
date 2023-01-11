import utils
import numpy as np 
import scipy.sparse as sps
import matplotlib.pyplot as plt
from Evaluation.Evaluator import EvaluatorHoldout
import Data_manager.split_functions.split_train_validation_random_holdout as split
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Hybrids.FastLoadWrapperRecommender import FastLoadWrapperRecommender
from Recommenders.DataIO import DataIO

class UserProfileRec(BaseRecommender):
    RECOMMENDER_NAME = "UserProfileRecommender"
    def __init__(self, URM_train, ICM_train, cf_rec_classes = [], cb_rec_classes = [], dataset_version='interactions-all-ones', best_model_for_user_profile_perc = 0.7, seed=None):
        super(UserProfileRec, self).__init__(URM_train)
        self.cf_rec_classes = cf_rec_classes
        self.cb_rec_classes = cb_rec_classes
        self.dataset_version = dataset_version
        self.ICM_train = ICM_train
        self.best_model_for_user_profile_perc = best_model_for_user_profile_perc
        self.cf_fitted_recs_dict_to_select = {}
        self.cb_fitted_recs_dict_to_select = {}
        self.URM_train_to_select = None 
        self.URM_val_to_select = None
        self.seed = seed
        
        
        
    def results_of_all_recs_per_group(self, plot=True):
        if (self.URM_train_to_select == None and self.URM_val_to_select == None):
            self.URM_train_to_select, self.URM_val_to_select = split.split_train_in_two_percentage_global_sample(self.URM_train, train_percentage = self.best_model_for_user_profile_perc, seed=self.seed)
        
        users_to_ignore, _ = self.get_users_not_and_in_group(self.URM_train_to_select)
        if (len(self.cf_fitted_recs_dict_to_select) == 0 and len(self.cb_fitted_recs_dict_to_select) == 0):
            self.cf_fitted_recs_dict_to_select, self.cb_fitted_recs_dict_to_select = self.fit_all_recs(self.cf_rec_classes, 
                                                                self.cb_rec_classes, 
                                                                self.URM_train_to_select, 
                                                                self.URM_val_to_select,                               
                                                                self.ICM_train)
        results_cf_rec_per_group = {}
        results_cb_rec_per_group = {}
    
        group_id = 0
        for users_to_ignore_per_group in users_to_ignore:
            print("Evaluating Group {}:".format(group_id))
            evaluator = EvaluatorHoldout(self.URM_val_to_select, cutoff_list=[self.cutoff], ignore_users=users_to_ignore_per_group)
    
            for rec_class, recommender in self.cf_fitted_recs_dict_to_select.items():
                result_df, _ = evaluator.evaluateRecommender(recommender)
                if rec_class in results_cf_rec_per_group:
                    results_cf_rec_per_group[rec_class].append(result_df.loc[self.cutoff][self.metric])
                else:
                    results_cf_rec_per_group[rec_class] = [result_df.loc[self.cutoff][self.metric]]
                
            for rec_class, recommender in self.cb_fitted_recs_dict_to_select.items():
                result_df, _ = evaluator.evaluateRecommender(recommender)
                if rec_class in results_cb_rec_per_group:
                    results_cb_rec_per_group[rec_class].append(result_df.loc[self.cutoff][self.metric])
                else:
                    results_cb_rec_per_group[rec_class] = [result_df.loc[self.cutoff][self.metric]]
                
            group_id += 1
            print("\n")
        
        if plot == True:
            self.plot_metric_per_user_group(results_cf_rec_per_group, results_cb_rec_per_group)
            
        return results_cf_rec_per_group, results_cb_rec_per_group # dict {RecClass: [metric[group0], metric[group1], ...]}

            
            
            
    def plot_metric_per_user_group(self, results_cf_rec_per_group, results_cb_rec_per_group):
        _ = plt.figure()
        for rec_class, recommender in self.cf_fitted_recs_dict_to_select.items():
            results = results_cf_rec_per_group[rec_class]
            plt.scatter(x=np.arange(0,len(results)), y=results, label=rec_class.RECOMMENDER_NAME, marker='_')
            
        for rec_class, recommender in self.cb_fitted_recs_dict_to_select.items():
            results = results_cb_rec_per_group[rec_class]
            plt.scatter(x=np.arange(0,len(results)), y=results, label=rec_class.RECOMMENDER_NAME, marker='_')
            
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
        
        elif self.clustering_strategy == "kmeans":
            
            print("\n")
    
        return users_not_in_group_list, users_in_group_list



    def fit_all_recs(self, cf_rec_classes, cb_rec_classes, URM, URM_val, ICM):
        cf_recs_dict = {}
        cb_recs_dict = {}

        if self.seed is not None or URM_val is None:
            for recommender_class in cf_rec_classes:
                recommender_object = recommender_class(URM)
                recommender_object.fit(**(utils.get_best_model_hyperparameters(recommender_class, dataset_version=self.dataset_version)))
                cf_recs_dict[recommender_class] = recommender_object

            for recommender_class in cb_rec_classes:
                recommender_object = recommender_class(URM, ICM_train=ICM)
                recommender_object.fit(**(utils.get_best_model_hyperparameters(recommender_class, dataset_version=self.dataset_version)))
                cb_recs_dict[recommender_class] = recommender_object
        
        else:
            evaluator = EvaluatorHoldout(self.URM_val_to_select, cutoff_list=[self.cutoff])
            for recommender_class in cf_rec_classes:
                recommender_object = FastLoadWrapperRecommender(URM, 
                                                                recommender_class, 
                                                                dataset_version=self.dataset_version, 
                                                                user_id_array_val=evaluator.users_to_evaluate, 
                                                                recs_on_urm_splitted=True)
                
                cf_recs_dict[recommender_class] = recommender_object

            for recommender_class in cb_rec_classes:
                recommender_object = FastLoadWrapperRecommender(URM, 
                                                                recommender_class, 
                                                                dataset_version=self.dataset_version, 
                                                                user_id_array_val=evaluator.users_to_evaluate, 
                                                                recs_on_urm_splitted=True)
                cb_recs_dict[recommender_class] = recommender_object
    
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
        best_rec_class_per_group_list = []
        best_recs_cf = []
        best_recs_cb = []
    
        for group_idx in range(self.num_groups):
            curr_best = 0
            curr_best_res = 0
            curr_best_cf = 0
            for rec_class, results in results_cf_rec_per_group.items():
                if results[group_idx] > curr_best_res:
                    curr_best_res = results[group_idx]
                    curr_best = rec_class
                    curr_best_cf = rec_class
            for rec_class, results in results_cb_rec_per_group.items():
                if results[group_idx] > curr_best_res:
                    curr_best_res = results[group_idx]
                    curr_best = rec_class
            if curr_best_cf == curr_best:
                best_recs_cf.append(curr_best_cf)
            else:
                best_recs_cb.append(curr_best)
            best_rec_class_per_group_list.append(curr_best)
    
        best_cf_recs_class_no_duplicates = []
        for i in best_recs_cf:
            if i not in best_cf_recs_class_no_duplicates:
                best_cf_recs_class_no_duplicates.append(i)
            
        best_cb_recs_class_no_duplicates = []
        for i in best_recs_cb:
            if i not in best_cb_recs_class_no_duplicates:
                best_cb_recs_class_no_duplicates.append(i)
                
        self.best_cf_fitted_rec_dict, self.best_cb_fitted_rec_dict = self.fit_all_recs(best_cf_recs_class_no_duplicates,
                                                                    best_cb_recs_class_no_duplicates, 
                                                                    self.URM_train, 
                                                                    self.URM_train,
                                                                    self.ICM_train)
        self.best_rec_class_per_group_list = best_rec_class_per_group_list
        
        version = "n_groups-" + str(self.num_groups)
        for elem in self.best_cf_fitted_rec_dict.keys():
            version += "_" + elem.RECOMMENDER_NAME
            
        for elem in self.best_cb_fitted_rec_dict.keys():
            version += "_" + elem.RECOMMENDER_NAME

        self.RECOMMENDER_VERSION = version
                        
    

    def _compute_item_score(self, user_id_array, items_to_compute = None):
        '''
        rec_class_per_group: list of n recommender classes, with n that is the number of users groups
        '''
        all_recs = {}
        all_recs.update(self.best_cf_fitted_rec_dict)
        all_recs.update(self.best_cb_fitted_rec_dict)
    
        _, users_in_group = self.get_users_not_and_in_group(self.URM_train)
    
        scores = np.ndarray((0, self.n_items))
        users = np.ndarray((0,))
        group_id = 0
        for rec_class in self.best_rec_class_per_group_list:
            rec_for_group = all_recs.get(rec_class)
            user_to_recommend = users_in_group[group_id] 
            mask = np.isin(user_to_recommend, user_id_array)
            user_to_recommend = user_to_recommend[mask]
            recommendation_for_group = rec_for_group._compute_item_score(user_id_array=user_to_recommend,
                                                                         items_to_compute=items_to_compute)
            scores = np.concatenate([scores, recommendation_for_group])
            users = np.concatenate([users, user_to_recommend])
            group_id += 1

        indices_to_sort = np.argsort(users)
        users = users[indices_to_sort]
        scores = scores[indices_to_sort, :]

        return scores 
    
    
    def save_model(self, folder_path, file_name = None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))
        
        self.recs_names_per_group = []
        for rec_class in self.best_rec_class_per_group_list:
            self.recs_names_per_group.append(rec_class.RECOMMENDER_NAME)
            
        self.cf_rec_names = []
        for rec_class in self.cf_rec_classes:
            self.cf_rec_names.append(rec_class.RECOMMENDER_NAME)
        
        self.cb_rec_names = []
        for rec_class in self.cb_rec_classes:
            self.cb_rec_names.append(rec_class.RECOMMENDER_NAME)
            

        data_dict_to_save = {"best_rec_class_per_group_list": self.recs_names_per_group,
                            "cf_rec_names": self.cf_rec_names,
                            "cb_rec_names": self.cb_rec_names}

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