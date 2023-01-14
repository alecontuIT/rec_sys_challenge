import scipy.sparse as sps
import numpy as np
import pandas as pd
import os
import shutil
import warnings
import math
import Data_manager.split_functions.split_train_validation_random_holdout as split
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchSingleCase import SearchSingleCase
from Recommenders.DataIO import DataIO
from recmodels import *
from skopt.space import Real
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative
from sklearn.cluster import KMeans
    
    
    
def urm_all_ones_visualized():
    URM_summed = urm_visualization_all_ones_summed()
    nnz_inds = URM_summed.nonzero()
    URM_all_ones = sps.csr_matrix((np.ones(len(nnz_inds[0])), (nnz_inds[0], nnz_inds[1])), shape=URM_summed.shape)
    return URM_all_ones



def urm_all_ones():
    URM_summed = urm_all_ones_summed()
    nnz_inds = URM_summed.nonzero()
    URM_all_ones = sps.csr_matrix((np.ones(len(nnz_inds[0])), (nnz_inds[0], nnz_inds[1])), shape=URM_summed.shape)
    return URM_all_ones



def urm_all_ones_summed():
    # Import dataframe for URM matrix
    root_path = "data"
    interactions = pd.read_csv(os.path.join(root_path, "interactions_and_impressions.csv"), low_memory=False)
    data_ICM_type = pd.read_csv(os.path.join(root_path,"data_ICM_type.csv"), low_memory=False)
    interactions["Data"] = 1
    interactions = interactions.drop(columns=["Impressions"])
    all_items = pd.concat([interactions["ItemID"], data_ICM_type["item_id"]], ignore_index=True).unique()
    n_users = len(interactions["UserID"].unique())
    n_items = len(all_items) # should be considered also the items present in the ICM but not here?
    
    URM_summed = sps.csr_matrix((interactions["Data"].values,
                          (interactions["UserID"].values, interactions["ItemID"].values)),
                             shape = (n_users, n_items))
    return URM_summed



def urm_seen_or_info(value_seen=1, value_info=0.5): # so 1 if seen, 0.5 if only info page opened
    assert value_info < value_seen and value_info != value_seen and value_info != 0, "value_seen = {} and value_info = {} are incompatible".format(value_seen, value_info)
    root_path = "data"
    interactions = pd.read_csv(os.path.join(root_path, "interactions_and_impressions.csv"), low_memory=False)
    data_ICM_type = pd.read_csv(os.path.join(root_path,"data_ICM_type.csv"), low_memory=False)
    interactions.loc[interactions["Data"] == 1, "Data"] = value_info
    interactions.loc[interactions["Data"] == 0, "Data"] = value_seen
    interactions = interactions.drop(columns=["Impressions"])
    all_items = pd.concat([interactions["ItemID"], data_ICM_type["item_id"]], ignore_index=True).unique()
    n_users = len(interactions["UserID"].unique())
    n_items = len(all_items) # should be considered also the items present in the ICM but not here?
    
    interactions.drop_duplicates(inplace=True)
    interactions.sort_values(by=['Data', 'UserID', 'ItemID'], inplace=True)
    interactions.drop_duplicates(subset = ['UserID', 'ItemID'], inplace=True)
    
    interactions.to_csv(os.path.join(root_path, "seen-or-info.csv"))
    URM_seen_or_info = sps.csr_matrix((interactions["Data"].values,
                          (interactions["UserID"].values, interactions["ItemID"].values)),
                             shape = (n_users, n_items))
    return URM_seen_or_info



def urm_visualization_all_ones_summed():
    # Import dataframe for URM matrix
    root_path = "data"
    interactions = pd.read_csv(os.path.join(root_path, "interactions_and_impressions.csv"), low_memory=False)
    data_ICM_type = pd.read_csv(os.path.join(root_path,"data_ICM_type.csv"), low_memory=False)
    interactions.drop(interactions[interactions["Data"] == 1].index, inplace = True)
    interactions["Data"] = 1
    interactions.drop(columns=["Impressions"], inplace = True)
    all_items = pd.concat([interactions["ItemID"], data_ICM_type["item_id"]], ignore_index=True).unique()
    n_users = len(interactions["UserID"].unique())
    n_items = len(all_items) # should be considered also the items present in the ICM but not here?
    
    URM_summed = sps.csr_matrix((interactions["Data"].values,
                          (interactions["UserID"].values, interactions["ItemID"].values)),
                             shape = (n_users, n_items))
    return URM_summed



def urm_info_all_ones_summed():
    # Import dataframe for URM matrix
    root_path = "data"
    interactions = pd.read_csv(os.path.join(root_path, "interactions_and_impressions.csv"), low_memory=False)
    n_users = len(interactions["UserID"].unique())
    data_ICM_type = pd.read_csv(os.path.join(root_path,"data_ICM_type.csv"), low_memory=False)
    interactions.drop(interactions[interactions["Data"] == 0].index, inplace = True)
    interactions["Data"] = 1
    interactions.drop(columns=["Impressions"], inplace = True)
    all_items = pd.concat([interactions["ItemID"], data_ICM_type["item_id"]], ignore_index=True).unique()
    n_items = len(all_items) # should be considered also the items present in the ICM but not here?
    URM_summed = sps.csr_matrix((interactions["Data"].values,
                          (interactions["UserID"].values, interactions["ItemID"].values)),
                             shape = (n_users, n_items))
    return URM_summed



def icm_types():
    # Import dataframe for ICM matrix
    root_path = "data"
    interactions = pd.read_csv(os.path.join(root_path, "interactions_and_impressions.csv"), low_memory=False)
    data_ICM_type = pd.read_csv(os.path.join(root_path,"data_ICM_type.csv"), low_memory=False)
    all_items = pd.concat([interactions["ItemID"], data_ICM_type["item_id"]], ignore_index=True).unique() # should be considered the items present in the ICM but not in interactions? should be considered the items present in interactions but not in ICM?
    all_types = data_ICM_type["feature_id"].unique()
    
    # to have features with consecutive id
    mapped_id, original_id = pd.factorize(data_ICM_type["feature_id"].unique())
    feature_original_ID_to_index = pd.Series(mapped_id, index=original_id)
    data_ICM_type["feature_id"] = data_ICM_type["feature_id"].map(feature_original_ID_to_index)
    
    n_items = len(all_items)
    n_types = len(all_types)
    ICM_csr = sps.csr_matrix((data_ICM_type["data"].values,
                          (data_ICM_type["item_id"].values, data_ICM_type["feature_id"].values)),
                            shape = (n_items, n_types))
    return ICM_csr



def icm_length():
    # Import dataframe for ICM matrix
    root_path = "data"
    interactions = pd.read_csv(os.path.join(root_path, "interactions_and_impressions.csv"), low_memory=False)
    icm_length = pd.read_csv(os.path.join(root_path,"data_ICM_length.csv"), low_memory=False)
    all_items = pd.concat([interactions["ItemID"], icm_length["item_id"]], ignore_index=True).unique() 
    icm_length.drop(columns=["feature_id"], inplace=True)
    missing_items = pd.DataFrame()
    missing_items["item_id"] = all_items[np.isin(all_items, icm_length["item_id"], invert=True)] 
    missing_items["data"] = 1
    icm_length = pd.concat([icm_length, missing_items], ignore_index=True)
    icm_length.sort_values(by=["item_id"])
    return icm_length
    
    
    
def icm():
    # Import dataframe for URM matrix
    root_path = "data"
    interactions = pd.read_csv(os.path.join(root_path, "interactions_and_impressions.csv"), low_memory=False)
    ICM_one_hot = pd.read_csv(os.path.join(root_path,"ICM_one_hot.csv"), low_memory=False)
    index= pd.DataFrame()
    index["index"] = ICM_one_hot.index
    all_items_id = pd.concat([interactions["ItemID"], index["index"]], ignore_index=True).unique()
    n_items = len(all_items_id)
    n_features = len(ICM_one_hot.columns)
    ICM_one_hot = ICM_one_hot.reindex(list(range(0, n_items))).reset_index(drop=True).fillna(0)
    ICM_one_hot = sps.csr_matrix(ICM_one_hot, shape = (n_items, n_features))
    return ICM_one_hot



def get_info_norm_urm(train_percentage = 0.7, seed=1234):
    urm_info = urm_info_all_ones_summed()
    urm_visualizations = urm_visualization_all_ones_summed()
    urm_train_vis, _ = split.split_train_in_two_percentage_global_sample(urm_visualizations, 
                                                                                      train_percentage = train_percentage,
                                                                                     seed=seed)
    urm_train_info, urm_validation_info = split.split_train_in_two_percentage_global_sample(urm_info, 
                                                                                      train_percentage = train_percentage,
                                                                                     seed=seed)
    del urm_info
    del urm_visualizations
    users_stats = statistics_per_user(urm_train_vis, urm_train_info)
    users_stats[users_stats["ProfileInfo"] == 0] = 1

    urm_train_info = urm_train_info / np.array(users_stats["ProfileInfo"])[:,None]
    urm_train_info = sps.csr_matrix(urm_train_info.astype(np.float))
    sps.save_npz(os.path.join("data","info_norm.npz"), urm_train_info)

    return urm_train_info, urm_validation_info    
    
    
    
def get_vis_norm_urm(train_percentage = 0.7, seed=1234):
    urm_visualizations = urm_visualization_all_ones_summed()
    urm_info = urm_info_all_ones_summed()
    urm_train_vis, urm_validation_vis = split.split_train_in_two_percentage_global_sample(urm_visualizations, 
                                                                                      train_percentage = train_percentage,
                                                                                     seed=seed)
    urm_train_info, _ = split.split_train_in_two_percentage_global_sample(urm_info, 
                                                                                      train_percentage = train_percentage,
                                                                                     seed=seed)
    del urm_info
    del urm_visualizations
    icm_len = icm_length()
    users_stats = statistics_per_user(urm_train_vis, urm_train_info)
    users_stats[users_stats["ProfileSeen"] == 0] = 1
    urm_train_vis = urm_train_vis / np.transpose(np.array(icm_len["data"]))
    urm_train_vis = urm_train_vis / np.array(users_stats["ProfileSeen"])[:,None]
    urm_train_vis = sps.csr_matrix(urm_train_vis.astype(np.float))

    sps.save_npz(os.path.join("data","vis_norm.npz"), urm_train_vis)

    return urm_train_vis, urm_validation_vis
                 
                 

def statistics_per_user(urm_seen_train, urm_info_train):
    ucm = pd.DataFrame();

    profile_length_seen = np.ediff1d(sps.csr_matrix(urm_seen_train).indptr)
    seen_count = np.sum(urm_seen_train, axis=1)
    ucm["ProfileSeen"] = profile_length_seen
    ucm["SeenInteractionCount"] = seen_count

    info_count = np.sum(urm_info_train, axis=1)
    profile_length_info = np.ediff1d(sps.csr_matrix(urm_info_train).indptr)
    ucm["ProfileInfo"] = profile_length_info
    ucm["InfoInteractionCount"] = info_count
    ucm = ucm.convert_dtypes()
    return ucm



def statistics_per_item(urm_seen_train, urm_info_train):
    item_stats = pd.DataFrame();

    profile_length_seen = np.ediff1d(sps.csr_matrix(urm_seen_train.T).indptr)
    seen_count = np.sum(urm_seen_train, axis=0)
    seen_count = seen_count.transpose()
    item_stats["ProfileSeen"] = profile_length_seen.T
    item_stats["SeenInteractionCount"] = seen_count

    info_count = np.sum(urm_info_train, axis=0)
    info_count = info_count.transpose()
    profile_length_info = np.ediff1d(sps.csr_matrix(urm_info_train.T).indptr)
    item_stats["ProfileInfo"] = profile_length_info.T
    item_stats["InfoInteractionCount"] = info_count
    item_stats = item_stats.convert_dtypes()
    return item_stats



def get_URM_stacked(URM_csr):
    URM_csr = sps.vstack([URM_csr, icm().T])
    URM_csr = sps.csr_matrix(URM_csr)
    return URM_csr



def get_ICM_stacked(URM_csr):
    ICM_csr = sps.csr_matrix(get_URM_stacked(URM_csr).T)
    return ICM_csr



def get_urm_visualization_summed_transformed(train_percentage, k=None, seed=None, transformation="minmax"):
    URM = urm_visualization_all_ones_summed()
    URM_all_ones = urm_all_ones()
    URM_train, URM_validation = split.split_train_in_two_percentage_global_sample(URM, 
                                                                                  train_percentage = train_percentage,
                                                                                  seed=seed)
    _, URM_validation_all_ones = split.split_train_in_two_percentage_global_sample(URM_all_ones, 
                                                                                  train_percentage = train_percentage,
                                                                                  seed=seed)
    URM_val_coo = URM_validation.tocoo()
    URM_val_all_ones_coo = URM_validation_all_ones.tocoo()
    #assert (URM_val_coo.row == URM_val_all_ones_coo.row).all() and (URM_val_coo.col == URM_val_all_ones_coo.col).all() and (URM_val_all_ones_coo.data == np.ones(len(URM_val_all_ones_coo.data))).all(), "The validation and training set overlap!"
    
    if k is not None and k > 0: 
        URM_train = scale_URM_per_clusters(URM_train, k, transformation)
        URM = scale_URM_per_clusters(URM, k, transformation)
    else:
        URM_train = transform_sparse_matrix(URM_train, transformation)
        URM = transform_sparse_matrix(URM, transformation)
        
    return URM, URM_train, URM_validation_all_ones



def get_urm_info_summed_transformed():
    return



def get_URM_all_ones_summed_transformed(train_percentage, k=None, seed=None, transformation="minmax"):
    URM = urm_all_ones_summed()
    URM_all_ones = urm_all_ones()
    URM_train, URM_validation = split.split_train_in_two_percentage_global_sample(URM, 
                                                                                  train_percentage = train_percentage,
                                                                                  seed=seed)
    _, URM_validation_all_ones = split.split_train_in_two_percentage_global_sample(URM_all_ones, 
                                                                                  train_percentage = train_percentage,
                                                                                  seed=seed)
    URM_val_coo = URM_validation.tocoo()
    URM_val_all_ones_coo = URM_validation_all_ones.tocoo()
    assert (URM_val_coo.row == URM_val_all_ones_coo.row).all() and (URM_val_coo.col == URM_val_all_ones_coo.col).all() and (URM_val_all_ones_coo.data == np.ones(len(URM_val_all_ones_coo.data))).all(), "The validation and training set overlap!"
    
    if k is not None and k > 0: 
        URM_train = scale_URM_per_clusters(URM_train, k, transformation)
        URM = scale_URM_per_clusters(URM, k, transformation)
    else:
        URM_train = transform_sparse_matrix(URM_train, transformation)
        URM = transform_sparse_matrix(URM, transformation)
        
    return URM, URM_train, URM_validation_all_ones



def transform_sparse_matrix(sp_matrix, transformation):
    sp_matrix_coo = sp_matrix.tocoo()
    
    if (transformation == "logistic"):
        cluster["Data"] = cluster["Data"].apply(logistic_scale_implicit_rating)
    elif (transformation == "tanh"):
        cluster["Data"] = cluster["Data"].apply(tanh_scale_implicit_rating)
    elif (transformation == "minmax"):
        transformed_data = range_scaling(sp_matrix_coo.data)
    elif (transformation == "std"):
        transformed_data = std_norm(sp_matrix_coo.data)
    else:
        print("Wrong transformation type!")
        
    transformed_matrix = sps.csr_matrix((transformed_data,
                          (sp_matrix_coo.row, sp_matrix_coo.col)),
                             shape = sp_matrix_coo.shape) 
    return transformed_matrix

    
    
def scale_URM_per_clusters(URM, k, transformation):
    df = get_df_from_urm(URM)
    clusters_list = get_clusters_dfs_from_df(df, k)
    
    clusters = []
    for cluster in clusters_list:
        global max_implicit_rating
        global min_implicit_rating
        max_implicit_rating = cluster["Data"].max()
        min_implicit_rating = cluster["Data"].min()
        if (transformation == "logistic"):
            cluster["Data"] = cluster["Data"].apply(logistic_scale_implicit_rating)
        elif (transformation == "tanh"):
            cluster["Data"] = cluster["Data"].apply(tanh_scale_implicit_rating)
        elif (transformation == "minmax"):
            cluster["Data"] = cluster["Data"].apply(ranged_min_max_scale_implicit_rating)
        else:
            print("Wrong scale type!")
        clusters.append(cluster)
    URM = pd.concat(clusters)
    URM = URM.sort_values(by=['UserID',"ItemID"])
    URM.drop(columns=["cluster"], inplace = True)
    data_ICM_type = pd.read_csv(os.path.join("data","data_ICM_type.csv"), low_memory=False)
    all_items = pd.concat([URM["ItemID"], data_ICM_type["item_id"]], ignore_index=True).unique()
    n_users = len(URM["UserID"].unique())
    n_items = len(all_items)
    URM_scaled = sps.csr_matrix((URM["Data"].values,
                          (URM["UserID"].values, URM["ItemID"].values)),
                             shape = (n_users, n_items))    
    
    return URM_scaled
    
    
    
def get_df_from_urm(URM):
    coo = URM.tocoo(copy=False)
    df = pd.DataFrame({'UserID': coo.row, 'ItemID': coo.col, 'Data': coo.data}
                 )[['UserID', 'ItemID', 'Data']].sort_values(['UserID', 'ItemID']
                 ).reset_index(drop=True)
    return df



def get_clusters_dfs_from_df(df, k):
    df["Data"] = df.groupby(["UserID", "ItemID"])["Data"].transform("sum")
    df.drop_duplicates(inplace=True)
    
    df_copy = df.copy()
    df_copy.drop(columns=["ItemID"], inplace = True)
    df_copy["Data"] = df_copy.groupby(["UserID"])["Data"].transform("sum")
    df_copy.drop_duplicates(inplace=True)
    df_copy.reset_index(inplace=True)
    # Extract the data values
    X = df_copy['Data'].values.reshape(-1, 1)
    # Create a KMeans model with 2 clusters
    kmeans = KMeans(n_clusters=k)
    # Fit the model to the data
    kmeans.fit(X)
    # Add a new column to the DataFrame with the cluster labels
    df_copy['cluster'] = kmeans.labels_
    df_merged = pd.merge(df[["UserID","ItemID","Data"]], df_copy[["UserID","cluster"]], on='UserID')
    
    # Create an empty list to store the separate datasets
    clustered_data = []

    # Iterate through each cluster
    for i in range(k):
        # Create a new dataframe for the current cluster
        cluster_df = df_merged[df_merged['cluster'] == i]
        # Save the dataframe to the list
        clustered_data.append(cluster_df)
        
    return clustered_data



def logistic(x):
    return 1 / (1 + math.exp(-x))



def logistic_scale_implicit_rating(implicit_rating, alpha=1, min_rating=0, max_rating=5):
    normalized_rating = implicit_rating/max_implicit_rating
    return min_rating + (max_rating - min_rating) * logistic(1 * normalized_rating)



def tanh_scale_implicit_rating(implicit_rating, alpha=1, min_rating=0, max_rating=5):
    normalized_rating = implicit_rating/max_implicit_rating
    return min_rating + (max_rating - min_rating) * math.tanh(alpha * implicit_rating)



def range_scaling(x, a=1, b=5):
    max_x = np.max(x)
    min_x = np.min(x)
    return ((b - a)/(max_x - min_x)) * (x - min_x) + a



def ranged_min_max_scale_implicit_rating(implicit_rating):
    return (4/(max_implicit_rating - min_implicit_rating))*(implicit_rating-min_implicit_rating)+1


def std_norm(x):
    mean_x = np.mean(x)
    stddev_x = np.std(x)
    return (x - mean_x) / stddev_x

        
def clusterize(X, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    return kmeans.labels_

def get_ucm():
    urm_visualizations = urm_visualization_all_ones_summed()
    urm_info = urm_info_all_ones_summed()
    urm_train_vis, _ = split.split_train_in_two_percentage_global_sample(urm_visualizations, 
                                                                                      train_percentage = 0.7,
                                                                                     seed=1234)
    urm_train_info, _ = split.split_train_in_two_percentage_global_sample(urm_info, 
                                                                                      train_percentage = 0.7,
                                                                                     seed=1234)
    ucm = statistics_per_user(urm_train_vis, urm_train_info)
    return ucm

def get_data_global_sample(dataset_version, train_percentage = 0.70, setSeed=False, k=None, transformation=None, value_seen=None, value_info=None):
    if setSeed == True:
        seed = 1234
    else:
        seed = None
        
    #warnings.filterwarnings('ignore')
    URM_csr = urm_all_ones()
    ICM = icm()
    URM_train, URM_validation = split.split_train_in_two_percentage_global_sample(URM_csr, 
                                                                                  train_percentage = train_percentage,
                                                                                  seed=seed)
    
    if (dataset_version == "interactions-all-ones"):
        return URM_csr, URM_train, URM_validation, ICM
    
    elif (dataset_version == "stacked"):
        URM_stacked = get_URM_stacked(URM_csr)
        URM_stacked_train = get_URM_stacked(URM_train)
        ICM_stacked = get_ICM_stacked(URM_train)
        ICM_stacked_train = get_ICM_stacked(URM_train)
        return URM_csr, URM_train, URM_validation, URM_stacked, URM_stacked_train, ICM_stacked, ICM_stacked_train
    
    elif (dataset_version == "interactions-summed"):
        return urm_all_ones_summed(), icm_types()
    
    elif (dataset_version == "custom"):
        assert value_seen is not None and value_info is not None, "value_seen or value_info is None!"
        URM = urm_seen_or_info(value_seen, value_info)
        URM_train, URM_validation = split.split_train_in_two_percentage_global_sample(URM, 
                                                                                  train_percentage = train_percentage,
                                                                                  seed=seed)
        return URM, URM_train, URM_validation, ICM
    
    elif (dataset_version == "interactions-summed-transformed"):
        assert transformation is not None, "transformation is None!"
        URM_csr, URM_train, URM_validation = get_URM_all_ones_summed_transformed(train_percentage, k, seed, transformation)
        return URM_csr, URM_train, URM_validation, ICM
    
    elif (dataset_version == "visualizations-summed-transformed"):
        assert transformation is not None, "transformation is None!"
        URM_csr, URM_train, URM_validation = get_urm_visualization_summed_transformed(train_percentage, k, seed, transformation)
        return URM_csr, URM_train, URM_validation, ICM
    
    else:
        print("Wrong dataset name. Try: \n - interactions-all-ones \n - stacked \n - interactions-summed \n - interactions-summed-transformed")

        
        
def get_data_user_wise(dataset_version, train_percentage = 0.70):
    #warnings.filterwarnings('ignore')
    URM_csr = urm_all_ones()
    ICM = icm()
    URM_train, URM_validation = split.split_train_in_two_percentage_user_wise(URM_csr, train_percentage = train_percentage)
    
    if (dataset_version == "interactions-all-ones"):
        return URM_csr, URM_train, URM_validation, ICM
    
    elif (dataset_version == "stacked"):
        URM_stacked = get_URM_stacked(URM_csr)
        URM_train, URM_validation = split.split_train_in_two_percentage_user_wise(URM_stacked, train_percentage = train_percentage)
        ICM_stacked = get_ICM_stacked(URM_csr)
        return URM_stacked, URM_train, URM_validation, ICM_stacked
    
    elif (dataset_version == "interactions-summed"):
        return urm_all_ones_summed(), icm_types()
    else:
        print("Wrong dataset name. Try: \n - interactions-all-ones \n - stacked \n - interactions-summed")
        

    
def get_users_for_submission():
    root_path = "data"
    users = pd.read_csv(os.path.join(root_path, "data_target_users_test.csv"))
    return users["user_id"]



def global_effects(URM_biased, shrink_user=5000, shrink_item=3000):
    # 1) global average
    mu = URM_biased.data.sum(dtype=np.float32) / URM_biased.nnz
    URM_unbiased = URM_biased.copy()
    URM_unbiased.data = URM_unbiased.data - mu

    # 2) item average bias
    # compute the number of non-zero elements for each column
    col_nnz = np.ediff1d(sps.csc_matrix(URM_biased).indptr)
    item_bias = URM_unbiased.sum(axis=0) / (col_nnz + shrink_item)
    item_bias = np.asarray(item_bias).ravel()  # converts 2-d matrix to 1-d array without anycopy
    item_bias[col_nnz==0] = -np.inf
        
    nz = URM_unbiased.nonzero()
    URM_unbiased[nz] -= item_bias[nz[1]]

    # 3) user average bias
    # This computes the mean of the row excluding the missing values
    row_nnz = np.ediff1d(sps.csr_matrix(URM_unbiased).indptr)
    user_bias = URM_unbiased.sum(axis=1).ravel() / (row_nnz + shrink_user)
    user_bias = np.asarray(user_bias).ravel()
    user_bias[row_nnz==0] = -np.inf
        
    # 4) summ all the contributes
    URM_unbiased[nz] = user_bias[nz[0]] + item_bias[nz[1]] + mu
    return URM_unbiased 
    


def bayesian_search(recommender_class, recommender_input_args, hyperparameters_range_dictionary, evaluator_validation, dataset_version="interactions-all-ones", n_cases = 60, perc_random_starts = 0.3, metric_to_optimize = "MAP", cutoff_to_optimize = 10, cust_output_folder_path=None, block_size=None, resume_from_saved=False, ICM=None, ICM_name=None):

    n_random_starts = int(n_cases * perc_random_starts)
    output_folder_path = get_hyperparams_search_output_folder(recommender_class, dataset_version=dataset_version, custom_folder_name=cust_output_folder_path)
    
    if recommender_class is TopPopRec:
        recommender_input_args_local = recommender_input_args
        urm = recommender_input_args_local.CONSTRUCTOR_POSITIONAL_ARGS[0]

        hyperparameterSearch = SearchSingleCase(recommender_class, evaluator_validation=evaluator_validation)
        hyperparameterSearch.search(recommender_input_args,
                                   fit_hyperparameters_values={},
                                   metric_to_optimize = metric_to_optimize,
                                   cutoff_to_optimize = cutoff_to_optimize,
                                   output_folder_path = output_folder_path,
                                   output_file_name_root = recommender_class.RECOMMENDER_NAME,
                                   resume_from_saved = resume_from_saved,
                                   save_model = "best",
                                   )
    
    elif recommender_class is ItemKNNCBFRec or recommender_class is ItemKNNCFRec or recommender_class is UserKNNCFRec:
        if recommender_class is ItemKNNCFRec:
            recommender_class = ItemKNNCFRecommender
            knn_cf = True
        elif recommender_class is UserKNNCFRec:
            recommender_class = UserKNNCFRecommender
            knn_cf = True
        elif recommender_class is ItemKNNCBFRec:
            recommender_class = ItemKNNCBFRecommender
            knn_cbf = True
        else:
            knn_cf = False
        
        if knn_cf: #if knn_cf:
            recommender_input_args_local = recommender_input_args
            urm = recommender_input_args_local.CONSTRUCTOR_POSITIONAL_ARGS[0] # the URM 
            runHyperparameterSearch_Collaborative(recommender_class, 
                                                  urm, 
                                                  n_cases = n_cases, 
                                                  n_random_starts = n_random_starts,
                                                  resume_from_saved = resume_from_saved,
                                                  evaluator_validation = evaluator_validation,
                                                  metric_to_optimize = metric_to_optimize, 
                                                  cutoff_to_optimize = cutoff_to_optimize,
                                                  output_folder_path = output_folder_path)
        elif knn_cbf:
            recommender_input_args_local = recommender_input_args
            urm = recommender_input_args_local.CONSTRUCTOR_POSITIONAL_ARGS[0] # the URM 
            runHyperparameterSearch_Content(recommender_class, 
                                            urm, 
                                            ICM_object = ICM, 
                                            ICM_name = ICM_name, 
                                            n_cases = n_cases, 
                                            n_random_starts = n_random_starts, 
                                            resume_from_saved = resume_from_saved,
                                            evaluator_validation = evaluator_validation,  
                                            metric_to_optimize = metric_to_optimize, 
                                            cutoff_to_optimize = cutoff_to_optimize,
                                            output_folder_path = output_folder_path, 
                                            parallelizeKNN = True, 
                                            allow_weighting = True, 
                                            allow_bias_ICM = True)
        
    else:
        hyperparameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, block_size = block_size)
        hyperparameterSearch._set_skopt_params(n_jobs=-1)
        hyperparameterSearch.search(recommender_input_args,
                       hyperparameter_search_space = hyperparameters_range_dictionary,
                       n_cases = n_cases,
                       n_random_starts = n_random_starts,
                       save_model = "best",
                       resume_from_saved = resume_from_saved,
                       output_folder_path = output_folder_path, # Where to save the results
                       output_file_name_root = recommender_class.RECOMMENDER_NAME, # How to call the files
                       metric_to_optimize = metric_to_optimize,
                       cutoff_to_optimize = cutoff_to_optimize)

    
    
def optimization_terminated(recommender, dataset_version, override = False):
    recommendations_folder_root = "recommendations"
    recommendations_folder_root = os.path.join(recommendations_folder_root, dataset_version)
    recommendations_folder_root = os.path.join(recommendations_folder_root, recommender.RECOMMENDER_NAME) 
    recommendations_folder = os.path.join(recommendations_folder_root, recommender.RECOMMENDER_VERSION)
    if ((not os.path.exists(recommendations_folder))):
        try:
            os.makedirs(recommendations_folder)
        except OSError:
            recommendations_folder = os.path.join(recommendations_folder_root, recommender.RECOMMENDER_VERSION[:255])
            os.makedirs(recommendations_folder)
        
    if (len(os.listdir(recommendations_folder)) == 0) or override:
        hyperparam_search_folder = os.path.join(recommendations_folder_root, "hyperparams_search")
        if os.path.exists(hyperparam_search_folder):
            train_folder = os.path.join(recommendations_folder, "optimization")
            if not os.path.exists(train_folder):
                    os.makedirs(train_folder)
            print(hyperparam_search_folder, train_folder)
            copy_all_files(hyperparam_search_folder, train_folder, remove_source=False)
        
    else:
        print("Error! It already exists the folder " + recommendations_folder)
    
    
    
def submission(recommender, dataset_version, override = False):
    optimization_terminated(recommender, dataset_version, override = override)
    recommendations_folder_root = "recommendations"
    recommendations_folder_root = os.path.join(recommendations_folder_root, dataset_version)
    recommendations_folder_root = os.path.join(recommendations_folder_root, recommender.RECOMMENDER_NAME)
    recommendations_folder_version = get_folder_best_model(recommender.__class__, dataset_version)
        
    rec_file_path = os.path.join(recommendations_folder_version, "recommendations.csv")
    if (os.path.exists(rec_file_path) and override) or (not os.path.exists(rec_file_path)):
        users = get_users_for_submission()
        
        tmp = recommender.recommend(user_id_array=users, cutoff=10)
        well_formatted = []
        for i in tmp:
            well_formatted.append( " ".join([str(x) for x in i]))
            
        submission = pd.DataFrame()
        submission["user_id"] = users
        submission["item_list"] = well_formatted
        submission.to_csv(rec_file_path, index=False)
        
        hyperparam_search_folder = os.path.join(recommendations_folder_root, "hyperparams_search")
        if os.path.exists(hyperparam_search_folder):
            shutil.rmtree(hyperparam_search_folder)
            
        submission_folder = os.path.join(recommendations_folder_version, "best")
        if not os.path.exists(submission_folder):
            os.makedirs(submission_folder)
        recommender.save_model(folder_path=submission_folder)
        
    else:
        print("Error! It already exists the file ", rec_file_path)

        
        
def get_hyperparams_search_output_folder(recommender_class, dataset_version="interactions-all-ones", custom_folder_name=None):
    folder = "recommendations"
    output_folder_path = os.path.join(folder, dataset_version)
    output_folder_path = os.path.join(output_folder_path, recommender_class.RECOMMENDER_NAME)
    if custom_folder_name != None:
        hyper_search_folder = custom_folder_name
    else:
        hyper_search_folder = "hyperparams_search"
    output_folder_path = os.path.join(output_folder_path, hyper_search_folder)
                                      
    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        
    return output_folder_path + "/"

                

def choose_num_train_epochs(recommender, validation_evaluator, max_epochs=10, valid_every_n = 1, metric = "MAP", cutoff=10, **fit_args):
    validation_results = pd.DataFrame(columns=["epoch", metric])
    for i in range(int(max_epochs/valid_every_n)):
        recommender.fit(epochs=valid_every_n, **fit_args)
        results_df, result_string = validation_evaluator.evaluateRecommender(recommender)
        print(result_string)
        validation_results.loc[i] = [int((i+1) * valid_every_n), results_df.iloc[0][metric]]
    maxidx = validation_results.idxmax()
    return validation_results.iat[maxidx[metric], 0], validation_results

    
    
def get_best_model_hyperparameters(recommender_class, dataset_version="interactions-all-ones", optimization=True, metric="MAP", custom_folder_name = None):
    folder = "recommendations"
    
    ## During Bayesian Search
    if optimization:
        folder = os.path.join(folder, dataset_version)
        folder = os.path.join(folder, recommender_class.RECOMMENDER_NAME)
        if custom_folder_name == None:
            hyp_search_folder = "hyperparams_search"
        else:
            hyp_search_folder = custom_folder_name
        folder = os.path.join(folder, hyp_search_folder)
        
    ## After Bayesian Search    
    else:
        folder = get_folder_best_model(recommender_class, dataset_version)
        folder = os.path.join(folder, "optimization")

    data_loader = DataIO(folder_path = folder)
    hyperparams_file = recommender_class.RECOMMENDER_NAME + "_metadata.zip"
    if os.path.exists(os.path.join(folder, hyperparams_file)):
        search_metadata = data_loader.load_data(hyperparams_file)
        return search_metadata["hyperparameters_best"]
    else:
        return {}

    
    
def get_best_res_on_validation(recommender_class, dataset_version="interactions-all-ones", optimization=False, metric="MAP", custom_folder_name = None):
    folder = "recommendations"
    
    ## During Bayesian Search
    if optimization:
        folder = os.path.join("recommendations", dataset_version)
        folder = os.path.join(folder, recommender_class.RECOMMENDER_NAME)
        if custom_folder_name == None:
            hyp_search_folder = "hyperparams_search"
        else:
            hyp_search_folder = custom_folder_name
        folder = os.path.join(folder, hyp_search_folder)
        
    ## After Bayesian Search
    else:
        folder = get_folder_best_model(recommender_class, dataset_version)
        folder = os.path.join(folder, "optimization")

    data_loader = DataIO(folder_path = folder)
    hyperparams_file = recommender_class.RECOMMENDER_NAME + "_metadata.zip"
    if os.path.exists(os.path.join(folder, hyperparams_file)):
        search_metadata = data_loader.load_data(hyperparams_file)
        return search_metadata["result_on_validation_best"][metric]
    else:
        return {}

        
        
####################### AFTER HYPERPARAMS SEARCH OF "recommender_class"       
def get_folder_best_model(recommender_class, dataset_version="interactions-all-ones"):
    folder = "recommendations"
    folder = os.path.join(folder, dataset_version)
    folder = os.path.join(folder, recommender_class.RECOMMENDER_NAME)
    list_dir_no_hid = listdir_nohidden(folder)
    if (len(list_dir_no_hid) == 1 and list_dir_no_hid[0] != "hyperparams_search"):
        return os.path.join(folder, list_dir_no_hid[0])
    if (len(list_dir_no_hid) >= 2):
        i = 0
        while i < len(list_dir_no_hid):
            if list_dir_no_hid[i][:18] != "hyperparams_search":
                return os.path.join(folder, list_dir_no_hid[i])
            i += 1
            
    print("Error: not present best model folder")



def save_item_scores(recommender_class, URM, user_id_array, dataset_version, fast=True, on_validation=True, new_item_scores_file_name_root=None):
    '''on_validation must be true if URM is the URM_train (so the URM after splitting)
    '''
    folder = get_folder_best_model(recommender_class, dataset_version)
    if new_item_scores_file_name_root is None:
        file_name = ""
    else:
        file_name = new_item_scores_file_name_root
    file_name += "item_scores"
    scores_file = os.path.join(folder, file_name + ".npy")
    if not os.path.exists(scores_file) or recommender_class is DiffStructHybridRecommender:
        assert (URM is not None) and (user_id_array is not None)
        kwargs = {}
        if recommender_class is DiffStructHybridRecommender:
            kwargs = {"load_scores_from_saved": True, 
                      "recs_on_urm_splitted": on_validation, 
                      "user_id_array_val": user_id_array, 
                      "new_item_scores_file_name_root": new_item_scores_file_name_root}
        recommender = load_best_model(URM, 
                                      recommender_class, 
                                      dataset_version=dataset_version, 
                                      optimization=on_validation, 
                                      **kwargs)
        item_scores = recommender._compute_item_score(user_id_array)
        
        recommender._print("Saving item_score in file '{}'".format(scores_file)) 
        if fast:
            np.save(scores_file, item_scores, allow_pickle=False)
        else:
            data_dict_to_save = {"item_scores": item_scores} 
            dataIO = DataIO(folder_path=folder)
            dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)
        recommender._print("Saving complete")
    
    
    
def load_item_scores(recommender_class,  dataset_version, fast = True, new_item_scores_file_name_root=None):
    folder = get_folder_best_model(recommender_class, dataset_version)
    if new_item_scores_file_name_root == None:
        new_item_scores_file_name_root = ""
    file_name = new_item_scores_file_name_root + "item_scores"
    if not fast:
        dataIO = DataIO(folder_path=folder)
        return dataIO.load_data(file_name=file_name)
    else:
        return np.load(os.path.join(folder, file_name + ".npy"),  allow_pickle=False)
    
    

def fit_best_recommender(recommender_class, URM, dataset_version, **kwargs):
    best_hyperparameters = get_best_model_hyperparameters(recommender_class, dataset_version)
    recommender = recommender_class(*get_kwargs_constructor(recommender_class, URM, dataset_version), **kwargs)
    recommender.fit(**best_hyperparameters)
    return recommender
    
    
    
def load_best_model(URM, rec_class, dataset_version="interactions-all-ones", optimization=False, **kwargs):
    rec = rec_class(*get_kwargs_constructor(rec_class, URM, dataset_version), **kwargs)
    if optimization:
        folder = os.path.join(get_folder_best_model(rec_class, dataset_version), "optimization")
        rec.load_model(folder, file_name = rec.RECOMMENDER_NAME + "_best_model.zip" )
    else:
        folder = os.path.join(get_folder_best_model(rec_class, dataset_version), "best")
        rec.load_model(folder, file_name = rec.RECOMMENDER_NAME + ".zip" )
    return rec



def load_model_from_hyperparams_search_folder(URM, rec_class, dataset_version="interactions-all-ones", **kwargs):
    rec = rec_class(*get_kwargs_constructor(rec_class, URM, dataset_version), **kwargs)
    folder = get_hyperparams_search_output_folder(rec_class, dataset_version=dataset_version, custom_folder_name=None)
    rec.load_model(folder, file_name = rec.RECOMMENDER_NAME + "_best_model.zip" )
    return rec

    
    
################### ALWAYS
def get_hybrid_weights_range_dict(number_of_recommender, low=0., high=1., prior="uniform"):
    hyperpar_dict = {}
    for i in range(number_of_recommender):
        weight = "w"+str(i)
        hyperpar_dict[weight] = Real(low=low, high=high, prior=prior)
    return hyperpar_dict
    
    
    
def listdir_nohidden(path):
    list_dir = os.listdir(path)
    for f in list_dir:
        if f.startswith('.'):
            list_dir.remove(f)
    return list_dir



def copy_all_files(source_folder, destination_folder, remove_source=False):
    if os.path.exists(source_folder):
        for file_name in os.listdir(source_folder):
        
            # construct full file path
            source = os.path.join(source_folder, file_name)
            destination = os.path.join(destination_folder, file_name)
        
            # copy only files
            if os.path.isfile(source):
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                shutil.copy(source, destination)
                if remove_source:
                    os.remove(source)



def get_kwargs_constructor(rec_class, URM, dataset_version="interactions-all-ones"):
    if rec_class is ItemKNNCBFRec:
        return [URM, icm()]
    if rec_class is EASE_R_Rec:
        sparse_threshold = 0.1
        return [URM, sparse_threshold]
    if rec_class is SLIM_BPRRec:
        memory_threshold = 0.9
        return [URM, memory_threshold]
    if rec_class is LightFMRecommender:
        return [URM, icm(), ucm()]
    if rec_class is DiffStructHybridRecommender:
        return [URM, dataset_version]
    else:
        return [URM]

    
    
def get_all_rec_classes():
    return [TopPopRec, 
            ItemKNNCBFRec, 
            ItemKNNCFRec, 
            UserKNNCFRec, 
            IALSRec, 
            SLIM_BPRRec, 
            P3AlphaRec, 
            RP3BetaRec, 
            EASE_R_Rec, 
            MatrixFactorizationBPRRec, 
            FunkSVDRec, 
            AsySVDRec, 
            PureSVDRec, 
            PureSVDItemRec, 
            ScaledPureSVDRec, 
            SVDFeatureRec]



def get_rec_class_by_name(rec_class_name):
    if TopPopRec.RECOMMENDER_NAME == rec_class_name:
        return TopPopRec 
    elif ItemKNNCBFRec.RECOMMENDER_NAME == rec_class_name:
        return ItemKNNCBFRec
    elif ItemKNNCFRec.RECOMMENDER_NAME == rec_class_name:
        return ItemKNNCFRec 
    elif UserKNNCFRec.RECOMMENDER_NAME == rec_class_name:
        return UserKNNCFRec 
    elif IALSRec.RECOMMENDER_NAME == rec_class_name:
        return IALSRec 
    elif SLIM_BPRRec.RECOMMENDER_NAME == rec_class_name:
        return SLIM_BPRRec 
    elif P3AlphaRec.RECOMMENDER_NAME == rec_class_name:
        return P3AlphaRec 
    elif RP3BetaRec.RECOMMENDER_NAME == rec_class_name:
        return RP3BetaRec 
    elif EASE_R_Rec.RECOMMENDER_NAME == rec_class_name:
        return EASE_R_Rec 
    elif MatrixFactorizationBPRRec.RECOMMENDER_NAME == rec_class_name:
        return MatrixFactorizationBPRRec
    elif FunkSVDRec.RECOMMENDER_NAME == rec_class_name:
        return FunkSVDRec
    elif AsySVDRec.RECOMMENDER_NAME == rec_class_name:
        return AsySVDRec 
    elif PureSVDRec.RECOMMENDER_NAME == rec_class_name:
        return PureSVDRec 
    elif PureSVDItemRec.RECOMMENDER_NAME == rec_class_name:
        return PureSVDItemRec 
    elif ScaledPureSVDRec.RECOMMENDER_NAME == rec_class_name:
        return ScaledPureSVDRec 
    elif SVDFeatureRec.RECOMMENDER_NAME == rec_class_name:
        return SVDFeatureRec
    elif LightFMRecommender.RECOMMENDER_NAME == rec_class_name:
        return LightFMRecommender
    elif DiffStructHybridRecommender.RECOMMENDER_NAME == rec_class_name:
        return DiffStructHybridRecommender
    