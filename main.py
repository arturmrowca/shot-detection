from enum import Enum
from sklearn.ensemble.forest import RandomForestClassifier
from processing.a_feature_base import feature_selection, do_pca, fb_train, fb_testing
from processing.b_lstm import lstm_train, lstm_test
from processing.preprocess.ingest import load_kinexon_sets, train_test_validation
from processing.preprocess.norm import normalize, balance
from processing.preprocess.segmentation import SEGMENTATION, prepare_set
from processing.util.visualization import visualize_pairwise, visualize_windows
from numpy import random

class APPROACH(Enum):
    FEATURE_BASED = 1 # Extraction, Selection Transformation, Random Forest
    LSTM = 2 # LSTM on raw data


if __name__ == "__main__":

    # Define approach to use
    approach = APPROACH.FEATURE_BASED # Different ML methods
    segmentation = SEGMENTATION.SLIDING_WINDOW # Different segmentation techniques
    resegment = False # if true perform segmentation - else run Machine Learning
    #random.seed(123)# enable if verifying results

    # Params for FB Approach
    fb_normalize = True # use normalization
    fb_balance = True # use balancing
    fb_feature_selection = True # run feature selection FW BW on Validation
    fb_apply_selection = True # apply fw bw feature selection
    #->FOR MAXIMA SEG: fb_best_features = [33, 25, 17, 35, 6, 5, 24, 8, 9, 23, 0, 30, 14, 41, 42, 43, 16, 18, 37, 1, 29, 27, 15, 7, 12]#24, 35, 4, 3, 19, 34, 27, 22, 5, 32, 41, 14, 38, 15, 23, 10, 8, 2, 6, 13, 28, 11, 36, 18, 37]# Found from feature selection
    #->FOR SLIDING SEG: fb_best_features = [38, 33, 25, 36, 35, 12, 22, 6, 29, 43, 9, 13, 5, 27, 24, 0, 23, 30, 15, 42, 37, 1, 40, 2, 11]
    fb_best_features = [38, 33, 25, 36, 35, 12, 22, 6, 29, 43, 9, 13, 5, 27, 24, 0, 23, 30, 15, 42, 37, 1, 40, 2, 11]
    fb_clf = RandomForestClassifier()
    fb_pca = False # run pca -> seems to not make it better leave it false
    fb_n_pca = 20 # number of pca components
    fb_decision_pca = False # show plot of relevant components
    fb_train_plot_bad = False # plot examples that were misclassified
    fb_test = True # run test on testset

    # Params for LSTM approach
    lstm_training = False
    lstm_testing = True
    lstm_lstm1_dim=16
    lstm_lstm2_dim=8
    lstm_dropout_perc=0.1
    lstm_l1_penalty=.01
    lstm_learning_rate=.01
    lstm_loss='mean_squared_error'
    lstm_batch_size=1000
    lstm_epochs=800 # 200?
    lstm_validation_split=0.1
    fair_correct = True # If True treats a shot detection within eps 300 ms as correct
    dec_threshold = 0.8 # higher means only high confidence shots are considered such
    balance_ratio = 0.8

    # -----------------------------------
    #    SEGMENT DATA
    # -----------------------------------
    if resegment:
        prepare_set(segmentation_type = segmentation)

    else:
        # -----------------------------------
        #    PREPARE DATA
        # -----------------------------------
        # Load data
        Xs, ys, Ws = load_kinexon_sets(5, segmentation)

        # Visualize
        if False: visualize_pairwise(Xs[0], ys[0])
        if False: visualize_windows(Ws[0], ys[0], only_shot=True, compare=["absxpos"])

        # Train, validation, test split
        X_train, X_val, X_test, y_train, y_val, y_test, Ws_train, Ws_val, Ws_test = train_test_validation(Xs, ys, Ws, train=[0,3], val=[1], test=[2, 4])# test=4 und 2
        print(X_train.shape)

        # -----------------------------------
        #    Apply classical ML Approaches
        # -----------------------------------
        if approach == APPROACH.FEATURE_BASED:

            # ------>   PREPROCESSING    <------
            # Normalization
            if fb_normalize:X_train, X_val, scaler = normalize(X_train, X_val)

            # Balancing
            if fb_balance:X_train, y_train = balance(X_train, y_train)

            # Run Feature Selection
            if fb_feature_selection:X_train, X_val = feature_selection(X_train, y_train, X_val, y_val, fb_clf, fb_best_features, fb_apply_selection)

            # Run PCA
            if fb_pca:X_train, X_val = do_pca(X_train, X_val, fb_n_pca, fb_decision_pca)

            # ------>   TRAINING    <------
            clf = fb_train(X_train, y_train, X_val, y_val, fb_clf, fb_train_plot_bad, Ws_train, Ws_val, segmentation, fb_testing)

            # ------>   TESTING    <------
            # Same preprocessing
            if fb_normalize:X_test = scaler.transform(X_test)
            if fb_feature_selection and fb_apply_selection:X_test = X_test[:, fb_best_features]
            if fb_test:fb_testing(X_test, y_test, clf, segmentation)

        # -----------------------------------
        #    Apply classical DL Approaches
        # -----------------------------------
        if approach == APPROACH.LSTM:

            # ------>   TRAINING    <------
            if lstm_training:lstm_train(X_train, y_train, X_val, y_val, lstm1_dim=lstm_lstm1_dim
                                        ,lstm2_dim=lstm_lstm2_dim,dropout_perc=lstm_dropout_perc
                                        ,l1_penalty=lstm_l1_penalty,learning_rate=lstm_learning_rate
                                        ,loss=lstm_loss,batch_size=lstm_batch_size,epochs=lstm_epochs
                                        ,validation_split=lstm_validation_split, balance = balance_ratio)

            # ------>   TESTING    <------
            if lstm_testing:lstm_test(X_test, y_test, X_train, y_train, segmentation, dec_threshold, fair_correct=fair_correct)
