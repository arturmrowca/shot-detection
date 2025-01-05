import os
import dill
import numpy as np
from processing.preprocess.segmentation import SEGMENTATION


def load_kinexon_sets(set_number, type):

    Xs = [] # contains X part of data
    ys = [] # contains y part of data
    Ws = [] # contains windows as pandas dataframe of data (not for LSTM)

    for i in range(set_number):
        # --------> try to read from file <--------
        if type == SEGMENTATION.MAX_SPEED_1_SEC:
            storage_file_path = r"storage/windowed_set_%d.dill" % (i)
        elif type == SEGMENTATION.SLIDING_WINDOW:
            storage_file_path = r"storage/windowed_set_%d_3_30.dill" % (i)
        elif type == SEGMENTATION.LSTM_RAW:
            storage_file_path = r"storage/raw_set_%d.dill" % (i)
        else:
            raise FileNotFoundError("No files defined for type %s" % str(type))

        if os.path.exists(storage_file_path):
            print("Reading from file '%s'" % str(storage_file_path))
            with open(storage_file_path, 'rb') as in_strm:
                data = dill.load(in_strm)
                try:
                    [X, y, w] = data
                except:
                    [X, y] = data
                    w = None
            Xs += [X]
            ys += [y]
            Ws += [w]

    if Ws[0] is not None:
        print("Columns %s" % str(Ws[0][0].columns))

    return Xs, ys, Ws


def train_test_validation(Xs, ys, Ws, train=[0, 3], val = [2], test = [1, 4]):

    # Data sets with features and labels
    X_train, y_train = np.concatenate([Xs[i] for i in train]), np.concatenate([ys[i] for i in train])
    X_val, y_val = np.concatenate([Xs[i] for i in val]), np.concatenate([ys[i] for i in val])
    X_test, y_test = np.concatenate([Xs[i] for i in test]), np.concatenate([ys[i] for i in test])

    # If windows are available store those as well (optional)
    try:
        Ws_train = []
        for i in train:
            Ws_train += Ws[i]
    except:
        Ws_train = []
    try:
        Ws_val = []
        for i in val:
            Ws_val += Ws[i]
    except:
        Ws_val = []
    try:
        Ws_test = []
        for i in test:
            Ws_test += Ws[i]
    except:
        Ws_test = []

    # Print stats
    print("\nData splitting\nSplit with \t\t\t%d : %d : %d" % (len(y_train), len(y_val), len(y_test)))
    nr_labels_pos = np.count_nonzero(y_train)
    nr_labels_neg = len(y_train) - nr_labels_pos
    print("Training labels: \t%d : %d" % (nr_labels_pos, nr_labels_neg))
    nr_labels_pos = np.count_nonzero(y_val)
    nr_labels_neg = len(y_val) - nr_labels_pos
    print("Validation labels: \t%d : %d" % (nr_labels_pos, nr_labels_neg))
    nr_labels_pos = np.count_nonzero(y_test)
    nr_labels_neg = len(y_test) - nr_labels_pos
    print("Test labels: \t\t%d : %d" % (nr_labels_pos, nr_labels_neg))

    return X_train, X_val, X_test, y_train, y_val, y_test, Ws_train, Ws_val, Ws_test
