from imblearn.combine import SMOTETomek
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalize(X_train, X_val):
    print("\nNormalization...")
    scaler =  MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)
    print(X_train.shape)
    print(X_val.shape)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)  # only scaling no balancing -> as should be similar to test set

    return X_train, X_val, scaler


def balance(X_train, y_train):
    # BALANCING (as given set is highly imbalanced)
    # oversampling: ADASYN, RandomOverSampler, SMOTE
    # undersampling: AllKNN, ClusterCentroids, CondensedNearestNeighbour, EditedNearestNeighbours, InstanceHardnessThreshold,NearMiss,NeighbourhoodCleaningRule,OneSidedSelection,RandomUnderSampler,RepeatedEditedNearestNeighbours,TomekLinks
    # over and undersampling: SMOTEENN SMOTETomek
    print("\nBalancing...")
    sm = SMOTETomek()
    X_train = np.nan_to_num(X_train)
    X_train, y_train =  sm.fit_sample(X_train, y_train)

    # New statistics
    nr_labels_pos = np.count_nonzero(y_train)
    nr_labels_neg = len(y_train) - nr_labels_pos
    print("New Training labels: %d : %d" % (nr_labels_pos, nr_labels_neg))

    return X_train, y_train
