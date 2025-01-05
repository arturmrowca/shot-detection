"""
Feature based approach
"""
import numpy as np
from sklearn.metrics.classification import precision_recall_fscore_support, classification_report
import copy
from matplotlib import pyplot as plt
import warnings
import pandas as pd
from processing.preprocess.segmentation import SEGMENTATION
from processing.util.visualization import visualize_windows

warnings.filterwarnings("ignore")


def score_set(X, Y, Xv, ytrue, clf):
    # train
    clf.fit(X, Y)

    # predict
    y_estimated = clf.predict(Xv)

    # prep evaluate
    ypred = y_estimated

    # correct with window around
    ypred = correct_y_val(ytrue, ypred)

    # do evaluate
    p, r, f1, s = precision_recall_fscore_support(ytrue, ypred, labels=None, average=None, sample_weight=None)

    score = f1[1] # choose score to optimize here -> score of underrepresented class
    return  score


def fw_bw_selection(X_train, y_train, X_val, y_val, clf):
    """ My own implementation of a FW BW search to find optimal features"""

    score = 0.0
    print("Feature Selction FW BW - classifier %s" % str(clf.__class__.__name__))
    X_train = pd.DataFrame(X_train, columns = ["c%d" % i for i in range(X_train.shape[1])])
    X_val = pd.DataFrame(X_val, columns = ["c%d" % i for i in range(X_val.shape[1])])

    # run SFFS
    all_feats = copy.deepcopy(list(X_train.columns))#copy.deepcopy(sel)
    features = []#all_feats
    prev_score = 0

    while list(all_feats):
        # inclusion
        # get best feature
        best = None
        cur_max = 0
        for f in all_feats:
            subs = features + [f]
            score = score_set(X_train[subs], y_train, X_val[subs], y_val, clf)
            if score > cur_max:
                cur_max = score
                best = f
        if not best is None:
            features += [best]
            all_feats.remove(best)

        score = score_set(X_train[features], y_train, X_val[features], y_val, clf)
        print("Current: %s - %s" % (str(score), str(features)))

        # exclusion - if no improvement
        best_features = []
        prev_score = score
        if len(features)>1:
            best = None
            #print("Exclusion")
            for f in features:
                subs = copy.deepcopy(features)
                subs.remove(f)
                score = score_set(X_train[subs], y_train, X_val[subs], y_val, clf)
                if score > prev_score and score > cur_max:
                    cur_max = score
                    best = f
                    best_features = features
            if not best == None: # if none: then no improvement
                features.remove(best)
                score = score_set(X_train[features], y_train, X_val[features], y_val, clf)
                print("Current: %s - %s" % (str(score), str(features)))
    best_features = np.array([int(p.replace("c","")) for p in best_features])
    print("Best-features %s" % str([best_features[i] for i in range(len(best_features))]))
    return best_features


def feature_selection(X_train, y_train, X_val, y_val, clf, best_features = [], apply_selection = True):
    """
    Performs a feature selection using FW BW search in a wrapper evaluation
    maximizing the F1 score
    :return:
    """
    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)

    # Do FW BW search
    if apply_selection:
        print("\nFeatures from FW BW search...")
        if not best_features:
            best_features = fw_bw_selection(X_train, y_train, X_val, y_val, clf)
        try:
            X_train = X_train.as_matrix()[:, best_features]
            X_val = X_val.as_matrix()[:, best_features]
        except:
            X_train = X_train[:, best_features]
            X_val = X_val[:, best_features]

    return X_train, X_val


def correct_y_val(y_val, y_est_val):
    """
    If true and estimated values are within a proximity of eps
    (e.g. window 5 was classified shot but the ground truth set shot is in window 7)
    Then, set the estimate to having 1 in window 7
    This is fair within eps as no two shots can be fired within 100 ms e.g.
    """
    tmp = np.array(np.where(y_val==1)[0])
    eps = 2 # 2 == +-one window / but window might overlap! shot detection within this is considered true
    for pos in list(np.where(y_est_val==1)[0]):
        rem = tmp[(tmp<=pos+eps) & (tmp>=pos-eps)]
        # shot detected within eps window
        if len(rem) > 0:
            y_est_val[pos] = 0
            y_est_val[rem[0]] = 1
    return y_est_val

def correct_y_val_slide(y_val, y_est_val):
    '''
    If windows are overlapping, set the non overlapping part of the estimate to
    to be equal to the true value, y_val = ground truth, y_est_val = estimate
    e.g. y_est_val   00011111000
         y_val       00000111100
        As can be seen the estimate detected the shot correctly so it is fair if
        in the statistics it is also considered as correct
        thus, change it for this overlap to
        y_est_val    00000111100

        also connect shots within range e.g. 001001 -> will be 001111
    '''

    # in the estimate connect predictions within eps
    idx_list = sorted(list(set(np.where(y_est_val==1)[0])))
    eps = 6 # 300 ms if another shot was found within 300 ms connect those
    for i in range(len(idx_list)-1):
        idx = idx_list[i]
        next_idx = idx_list[i+1]
        if next_idx - idx <= eps:
            y_est_val[idx:next_idx] = 1

    # find overlap between ground truth and prediction
    # if overlap given assume shot is found
    idx_list = sorted(list(set(np.where((y_est_val == y_val) & y_val==1)[0])))
    prev_idx = 0
    for i in range(len(idx_list)-1):# overlaps that were found
        idx = idx_list[i]

        if idx-1 == prev_idx:
            prev_idx = idx
            continue
        if y_est_val[idx] != y_val[idx]:
            continue

        # for both set estimate to 0 where wrong and to 1 where right
        # idx back until 0
        if idx == 0: continue
        for k in [i for i in range(0, idx)][::-1]:
            if y_val[k] == 0:
                break
            y_est_val[k] = 1

        prev_val =[1, 1]
        for k in range(idx, len(y_est_val)):
            if y_val[k] == 1:
                y_est_val[k] = 1

            prev_val = [prev_val[-1], y_est_val[k]]
            if y_val[k] == 0:
                if y_est_val[k] == 0:
                    break
                else:
                    y_est_val[k] = 0

                if prev_val[0] == 0 and prev_val[1] == 0:
                    break

        prev_idx = idx

    return y_est_val


def do_pca(X_train, X_val, n_pca, decision):
    """Do the PCA on the given data, if decision = True, a plot is shown which helps to
    decide what number of PCA components to use"""
    from sklearn.decomposition import PCA

    if decision:
        # Plotting the Cumulative Summation of the Explained Variance to decide how many components to use
        pca = PCA()
        pca_fit = pca.fit(X_train)
        plt.figure()
        plt.plot(np.cumsum(pca_fit.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Variance (%)')  # for each component
        plt.title('Explained Variance in Data set')
        plt.show()

    else:
        print("Apply PCA with components: %s" % n_pca)
        pca = PCA(n_components=min([n_pca, X_train.shape[1]]))
        pca_fit = pca.fit(X_train)
        X_train = pca_fit.transform(X_train)
        X_val  = pca_fit.transform(X_val)

    return X_train, X_val


def fb_testing(X_test, y_test, clf, segmentation):

    # Evaluate classifier on given test data
    y_est_test = clf.predict(X_test)

    # do label fairness adjustment
    if segmentation in [SEGMENTATION.MAX_SPEED_1_SEC]:
        y_est_val = correct_y_val(y_test, y_est_test)
    elif segmentation in [SEGMENTATION.SLIDING_WINDOW]:
        # post process - if single classification assume outlier
        idx_list = list(np.where(y_est_test==1))
        eps = 6 # 300 ms if nothing else detected shot within this range this is an outlier
        for i in range(1, len(idx_list)-1):
            prev_idx = idx_list[i-1]
            idx = idx_list[i]
            next_idx = idx_list[i+1]
            if next_idx - idx <= eps and idx - prev_idx <= eps:
                y_est_test[idx] = 0

        # clean prediction result for fair comparison
        y_est_val = correct_y_val_slide(y_test, y_est_test)

    print("\nPerformance Test set")
    class_report = classification_report(y_test, y_est_test, target_names=["No Shot", "Shot"])
    print(class_report)

    plt.scatter(range(len(y_test)), y_test/2, c="r")
    plt.scatter(range(len(y_est_val)), y_est_val, c="g")
    plt.legend(["ground truth", "prediction"], loc="lower right")
    plt.ylim([0.3, 1.1])
    plt.show()


def fb_train(X_train, y_train, X_val, y_val, clf, train_plot_bad, Ws_train, Ws_val, segmentation, fb_testing):

    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)

    # Train classifier
    clf.fit(X_train, y_train)
    if fb_testing:return clf

    # Evaluate classifier on training and on validation set
    y_est_train = clf.predict(X_train)
    y_est_val = clf.predict(X_val)

    # make fairness adjustment
    if segmentation in [SEGMENTATION.MAX_SPEED_1_SEC]:
        y_est_val = correct_y_val(y_val, y_est_val)
    elif segmentation in [SEGMENTATION.SLIDING_WINDOW]:
        # post process - if single classification assume outlier
        idx_list = list(np.where(y_est_val==1))
        eps = 6 # 300 ms if nothing else detected shot within this range this is an outlier
        for i in range(1, len(idx_list)-1):
            prev_idx = idx_list[i-1]
            idx = idx_list[i]
            next_idx = idx_list[i+1]
            if next_idx - idx <= eps and idx - prev_idx <= eps:
                y_est_val[idx] = 0

        # clean prediction result for fair comparison
        y_est_val = correct_y_val_slide(y_val, y_est_val)

    print("\nPerformance Training set")
    class_report = classification_report(y_train, y_est_train, target_names=["No Shot", "Shot"])
    print(class_report)

    print("\nPerformance Validation set")
    class_report = classification_report(y_val, y_est_val, target_names=["No Shot", "Shot"])
    print(class_report)

    if train_plot_bad:
        # samples where shot was detected
        sub_indices = np.where((y_est_val == y_val) & (y_val == 1))[0]
        print("Plotting samples that were identified")
        print(sub_indices)
        if len(sub_indices) > 0 and Ws_val:
            visualize_windows([Ws_val[index] for index in sub_indices], y_val[sub_indices], only_shot=False,
                              compare=[], exclude=["absxpos", "absypos"])

        # samples where shot was not detected
        sub_indices = np.where((y_est_val != y_val) & (y_val == 1))[0]
        print("Plotting samples that could not be identified")
        print(sub_indices)
        if len(sub_indices) > 0 and Ws_val:
            visualize_windows([Ws_val[index] for index in sub_indices], y_val[sub_indices], only_shot=False, compare=[],
                              exclude=["absxpos", "absypos"])

    plt.scatter(range(len(y_val)), y_val/2, c="r")
    plt.scatter(range(len(y_est_val)), y_est_val, c="g")
    plt.legend(["ground truth", "prediction"], loc="lower right")
    plt.ylim([0.3, 1.1])
    plt.show()

    return clf

