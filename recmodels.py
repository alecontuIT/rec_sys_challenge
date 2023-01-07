from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.Hybrids.DiffStructHybridRecommender import DiffStructHybridRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender, PureSVDItemRecommender, ScaledPureSVDRecommender
from Recommenders.MatrixFactorization.SVDFeatureRecommender import SVDFeature
from Recommenders.FactorizationMachines.FMRecommender import LightFMRecommender



class TopPopRec(TopPop):
    def fit(self):
        super(TopPopRec, self).fit()
        self.RECOMMENDER_VERSION = "classic"

        

class ItemKNNCBFRec(ItemKNNCBFRecommender):  
    def fit(self, 
            topK=50, 
            shrink=100, 
            normalize = True, 
            similarity = "cosine", 
            feature_weighting = "none", 
            ICM_bias = None, 
            **similarity_args):
        super(ItemKNNCBFRec, self).fit( 
            topK, 
            shrink,similarity, 
            normalize, 
            feature_weighting, 
            ICM_bias, 
            **similarity_args)
        self.RECOMMENDER_VERSION = "topK-" + str(topK) + "_shrink-" + str(shrink) +"_feature_weighting-"+feature_weighting+ "_sim-" + similarity
            
            

class ItemKNNCFRec(ItemKNNCFRecommender):
    def fit(self, 
            topK=50, 
            shrink=100, 
            similarity='cosine', 
            normalize=True, 
            feature_weighting = "none", 
            URM_bias = False, 
            **similarity_args):
        super(ItemKNNCFRec, self).fit(
            topK, 
            shrink, 
            similarity, 
            normalize, 
            feature_weighting,
            URM_bias, 
            **similarity_args)
        self.RECOMMENDER_VERSION = "topK-" + str(topK) + "_shrink-" + str(shrink) +"_feature_weighting-"+feature_weighting+ "_sim-" + similarity
                    
            

class UserKNNCFRec(UserKNNCFRecommender):
    def fit(self, 
            topK=50, 
            shrink=100, 
            similarity='cosine', 
            normalize=True, 
            feature_weighting = "none", 
            URM_bias = False, 
            **similarity_args):
        super(UserKNNCFRec, self).fit(
            topK, 
            shrink, 
            similarity, 
            normalize, 
            feature_weighting,
            URM_bias, 
            **similarity_args)
        self.RECOMMENDER_VERSION = "topK-" + str(topK) + "_shrink-" + str(shrink) +"_feature_weighting-"+feature_weighting+ "_sim-" + similarity
            
    
    
class IALSRec(IALSRecommender):
    def fit(self, 
            epochs = 300,
            num_factors = 20,
            confidence_scaling = "linear",
            alpha = 1.0,
            epsilon = 1.0,
            reg = 1e-3,
            init_mean=0.0,
            init_std=0.1,
            **earlystopping_kwargs):
        super(IALSRec, self).fit(
            epochs,
            num_factors,
            confidence_scaling,
            alpha,
            epsilon,
            reg, 
            init_mean,
            init_std,
            **earlystopping_kwargs)
        self.RECOMMENDER_VERSION =  "numfact-" + str(num_factors) + "_alpha-" + str(alpha) + "_reg-" + str(reg)



class SLIM_BPRRec(SLIM_BPR_Cython):        
    def fit(self, 
            epochs=300,
            positive_threshold_BPR = None,
            train_with_sparse_weights = None,
            allow_train_with_sparse_weights = True,
            symmetric = True,
            random_seed = None,
            lambda_i = 0.0, 
            lambda_j = 0.0, 
            learning_rate = 1e-4, 
            topK = 200,
            sgd_mode='adagrad', 
            gamma=0.995, 
            beta_1=0.9, 
            beta_2=0.999,
            **earlystopping_kwargs):
        super(SLIM_BPRRec, self).fit(
            epochs = epochs,
            positive_threshold_BPR = positive_threshold_BPR,
            train_with_sparse_weights = train_with_sparse_weights,
            allow_train_with_sparse_weights = allow_train_with_sparse_weights,
            symmetric = symmetric,
            random_seed = random_seed,
            lambda_i = lambda_i, 
            lambda_j = lambda_j, 
            learning_rate = learning_rate, 
            topK = topK,
            sgd_mode = sgd_mode, 
            gamma = gamma, 
            beta_1 = beta_1, 
            beta_2 = beta_2,
            **earlystopping_kwargs)
        self.RECOMMENDER_VERSION =  "epochs-" + str(epochs) + "_topK-" + str(topK) + "_lambda_i-" + str(lambda_i) + "_lambda_j-" + str(lambda_j) + "_learning_rate-" + str(learning_rate) + "_sym-" + str(symmetric) + "_sgd-" + str(sgd_mode) 
        
        
        
class P3AlphaRec(P3alphaRecommender):        
    def fit(self, 
            topK = 100,
            alpha = 1.,
            min_rating = 0,
            implicit = False,
            normalize_similarity = False):
        super(P3AlphaRec, self).fit(
            topK,
            alpha,
            min_rating,
            implicit,
            normalize_similarity)
        self.RECOMMENDER_VERSION =  "topK-" + str(topK) + "_alpha-" + str(alpha) + "_min_rating-" + str(min_rating) + "_implicit-" + str(implicit) + "_normalize_similarity-" + str(normalize_similarity)
        
        
        
class RP3BetaRec(RP3betaRecommender):        
    def fit(self, 
            alpha = 1.,
            beta = 0.6,
            min_rating = 0,
            topK = 100,
            implicit = False,
            normalize_similarity = True):
        super(RP3BetaRec, self).fit(
            alpha,
            beta,
            min_rating,
            topK,
            implicit,
            normalize_similarity)
        self.RECOMMENDER_VERSION =  "topK-" + str(topK) + "_beta-" + str(beta) + "_alpha-" + str(alpha) + "_min_rating-" + str(min_rating) + "_implicit-" + str(implicit) + "_normalize_similarity-" + str(normalize_similarity)

        
        
class EASE_R_Rec(EASE_R_Recommender):        
    def fit(self, 
            topK = None,
            l2_norm = 1e3,
            normalize_metrics = False):
        super(EASE_R_Rec, self).fit(
            topK,
            l2_norm,
            normalize_metrics)
        self.RECOMMENDER_VERSION =  "topK-" + str(topK) + "_l2_norm-" + str(l2_norm) + "_normalize_metrics-" + str(normalize_metrics)
                
        
        
class MatrixFactorizationBPRRec(MatrixFactorization_BPR_Cython):
    def fit(self, 
            epochs=300, 
            batch_size = 1000,
            num_factors=10, 
            positive_threshold_BPR = None,
            learning_rate = 0.001,
            use_bias = True,
            use_embeddings = True,
            sgd_mode='sgd',
            negative_interactions_quota = 0.0,
            dropout_quota = None,
            init_mean = 0.0, 
            init_std_dev = 0.1,
            user_reg = 0.0, 
            item_reg = 0.0, 
            bias_reg = 0.0, 
            positive_reg = 0.0, 
            negative_reg = 0.0,
            random_seed = None,
            **earlystopping_kwargs):
        super(MatrixFactorizationBPRRec, self).fit(epochs = epochs, 
            batch_size = batch_size,
            num_factors=num_factors, 
            positive_threshold_BPR = positive_threshold_BPR,
            learning_rate = learning_rate,
            use_bias = use_bias,
            use_embeddings = use_embeddings,
            sgd_mode = sgd_mode,
            negative_interactions_quota = negative_interactions_quota,
            dropout_quota = dropout_quota,
            init_mean = init_mean, 
            init_std_dev = init_std_dev,
            user_reg = user_reg, 
            item_reg = item_reg, 
            bias_reg = bias_reg, 
            positive_reg = positive_reg, 
            negative_reg = negative_reg,
            random_seed = 1,
            **earlystopping_kwargs)
        self.RECOMMENDER_VERSION = "epochs-" + str(epochs) + "_batch-" + str(batch_size) + "_nfactors-" + str(num_factors) + "_learnrate-" + str(learning_rate) + "_usebias-" + str(use_bias) + "_useembed-" + str(use_embeddings) + "_sgdmode-" + str(sgd_mode) + "_neginter-" + str(negative_interactions_quota) + "_dropout-" + str(dropout_quota) + "_userreg-" + str(user_reg) +  "_itemreg-" + str(item_reg) + "_biasreg-" + str(bias_reg) + "_posreg-" + str(positive_reg) + "_negreg-" + str(negative_reg)

        
        
class FunkSVDRec(MatrixFactorizationBPRRec):
    RECOMMENDER_NAME = "MatrixFactorization_FunkSVD_Cython_Recommender"

    
    
class AsySVDRec(MatrixFactorizationBPRRec):
    RECOMMENDER_NAME = "MatrixFactorization_AsySVD_Cython_Recommender"

    
        
class PureSVDRec(PureSVDRecommender):
    def fit(self, num_factors=100, random_seed = None):
        super(PureSVDRec, self).fit(num_factors=num_factors, random_seed=random_seed)
        self.RECOMMENDER_VERSION = "numfactors-" + str(num_factors)
        
        
        
class PureSVDItemRec(PureSVDItemRecommender):
    def fit(self, num_factors=100, topK = None, random_seed = None):
        super(PureSVDItemRec, self).fit(num_factors=num_factors, topK=topK, random_seed=random_seed)
        self.RECOMMENDER_VERSION = "numfactors-" + str(num_factors) + "_topK-" + str(topK)
    
    
    
class ScaledPureSVDRec(ScaledPureSVDRecommender):
    def fit(self, num_factors = 100, random_seed = None, scaling_items = 1.0, scaling_users = 1.0):
        super(ScaledPureSVDRec, self).fit(num_factors=num_factors, random_seed=random_seed, scaling_items=scaling_items, scaling_users=scaling_users)
        self.RECOMMENDER_VERSION = "numfactors-" + str(num_factors) + "_scalingitems-" + str(scaling_items) + "_scalingusers-" + str(scaling_users)

        
        
class SVDFeatureRec(SVDFeature):
    def fit(self, epochs=30, num_factors=32, learning_rate=0.01,
            user_reg=0.0, item_reg=0.0, user_bias_reg=0.0, item_bias_reg=0.0,
            temp_file_folder = None):
        super(SVDFeatureRec, self).fit(epochs=epochs, num_factors=num_factors, learning_rate=learning_rate,
            user_reg=user_reg, item_reg=item_reg, user_bias_reg=user_bias_reg, item_bias_reg=item_bias_reg,
            temp_file_folder = temp_file_folder)
        self.RECOMMENDER_VERSION = "epochs-" + str(epochs) + "_nfactors-" + str(num_factors) + "_learnrate-" + str(learning_rate) + "_userreg-" + str(user_reg) +  "_itemreg-" + str(item_reg) + "_userbiasreg-" + str(user_bias_reg) + "__itembiasreg-" + str(item_bias_reg)

            