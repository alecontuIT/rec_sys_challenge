import numpy as np
import scipy.sparse as sps
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
from Data_manager.split_functions.split_train_validation_cold_items import _zero_out_values
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

def split_users_in_n_groups_by_masks(URM_all, masks, train_percentage = 0.8):
    """
    The function splits an URM in two matrices selecting the number of interactions one user at a time
    :param URM_train:
    :param train_percentage:
    :param verbose:
    :return:
    """

    assert train_percentage >= 0.0 and train_percentage<=1.0, "train_percentage must be a value between 0.0 and 1.0, provided was '{}'".format(train_percentage)
    
    # check valid masks:
    n_users, n_items = URM_all.shape
    all_ones = np.ones(n_users)
    result_vec = 0
    for m in masks: 
        result_vec = np.add(res, m)
        
    for res in result_vec:
        assert (res == 1)

    # Use CSR for user-wise split
    URM_all = sps.csr_matrix(URM_all)
    n_users, n_items = URM_all.shape
    URMs = []
    URMs_train = []
    URMs_valid = []

    for users_mask in masks:
        users = np.arange(0, n_users, dtype = np.int)[users_mask]
        URM = _zero_out_values(URM_all.copy(), rows_to_zero = users)
        URMs.append(URM)
        URM_train, URM_valid = split_train_in_two_percentage_global_sample(URM, train_percentage = train_percentage)
        URMs_train.append(URM_train)
        URMs_valid.append(URM_valid)

    return URMs, URMs_train, URMs_valid