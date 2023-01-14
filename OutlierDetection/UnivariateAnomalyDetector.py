from scipy import stats
import numpy as np

def get_anomaly_limits(X):
    mad = stats.median_abs_deviation(X, scale='normal')
    for c in range(X.shape[1]):
        mean = X[:,c].mean()
        std = X[:,c].std()
        median = np.median(X[:,c])


        lower = mean - 3*std
        upper = mean + 3*std
        mad_lower = median - 3*mad[c]
        mad_upper = median + 3*mad[c]

        print("Variable X%d (%.3f,%.3f)"%(c,mean,std))
        print("\tStandard Deviation Limits (%.3f,%.3f)"%(lower,upper))
        print("\tMAD Limits (%.3f,%.3f)"%(mad_lower,mad_upper))
        return (lower,upper), (mad_lower,mad_upper)
        
        
        
def get_stddev_outliers(X):
    for c in range(X.shape[1]):
        mean = X[:,c].mean()
        std = X[:,c].std()
        lower = mean - 3*std
        upper = mean + 3*std
        outliers = ((X[:,c]>upper) | (X[:,c]<lower))

        if (outliers.sum()>0):
            print("Variable X%d (%.3f,%.3f) has %d outliers"%(c,mean,std,outliers.sum()))
            print("\t===============================")
            for i,v in enumerate(X[outliers==True,c]):
                print("\t%d\t%.3f"%(i,v))
            print("\t===============================")
            return outliers

        else:
            print("Variable X%d has no outliers")
            

            
def get_mad_outliers(X, tolerance=3):
    mad = stats.median_abs_deviation(X, scale='normal')

    # the coefficient is usually twice the mad multiplied by a tolerance factor
    for c in range(X.shape[1]):
        mean = np.mean(X[:,c])
        std = np.std(X[:,c])
        median = np.median(X[:,c])
        lower = median - tolerance*mad[c]
        upper = median + tolerance*mad[c]
        outliers = ((X[:,c]>upper) | (X[:,c]<lower))

        if (outliers.sum()>0):
            print("Variable X%d (%.3f,%.3f) has %d outliers"%(c,mean,std,outliers.sum()))
            print("\t===============================")
            for i,v in enumerate(X[outliers==True,c]):
                print("\t%d\t%.3f"%(i,v))
            print("\t===============================")
            return outliers

        else:
            print("Variable X%d has no outliers")