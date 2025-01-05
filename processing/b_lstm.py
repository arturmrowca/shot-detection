"""
Feature based approach
"""
import copy
"""
import matplotlib.pyplot as plt
from keras.engine.saving import load_model
from keras.optimizers import Adam
from keras.regularizers import l1
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import classification_report
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np
from processing.a_feature_base import correct_y_val_slide
from processing.preprocess.segmentation import SEGMENTATION


def lstm_train(X_train, y_train, X_val, y_val,lstm1_dim=32,lstm2_dim=16,dropout_perc=0,l1_penalty=.01,learning_rate=.01
    ,loss='mean_squared_error',batch_size=200,epochs=10,validation_split=0.1, balance = 1.0):

    # Restrict GPU utilisation
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.2
    #config.gpu_options.visible_device_list = "0"  # restrict to one GPU
    #set_session(tf.Session(config=config))
    config = None

    # Check input and record meta data X == (batches, seq_len, features)
    (n_samples, n_timesteps, n_features) = X_train.shape
    input_x = np.concatenate([X_train, X_val])
    input_y = np.concatenate([y_train, y_val])

    # Architecture
    net_model = Sequential()

    # Use two LSTMs with a dense layer
    net_model.add(LSTM(
                lstm1_dim,
                return_sequences=True,
                dropout=dropout_perc,
                recurrent_dropout=dropout_perc,
            )
        )
    net_model.add(
            LSTM(
                lstm2_dim,
                return_sequences=True,
                dropout=dropout_perc,
                recurrent_dropout=dropout_perc,
            )
        )
    # Use a Dense layer as result
    net_model.add(Flatten())
    net_model.add(
            Dense(
                1,
                activation='relu',
                kernel_regularizer=l1(l1_penalty)
            )
    )


    # Compile model
    opt_rms = Adam(lr=learning_rate) # Optimizer
    net_model.compile(
        optimizer=opt_rms,
        loss=loss
    )

    # Balance
    print("Balancing...")
    rel = input_x[input_y==1]
    rel_y = input_y[input_y==1]
    ratio = int(balance* (float(len(input_x[input_y==0]))/float(len(input_x[input_y==1])))) - 1
    print("Old ratio %s" % str(ratio))
    new_xs = [rel]
    new_ys = [rel_y]
    for i in range(ratio):
        new_xs += [rel]
        new_ys += [rel_y]
    input_x = np.concatenate([input_x] + new_xs)
    input_y = np.concatenate([input_y] + new_ys)


    ratio = float(len(input_x[input_y==0]))/float(len(input_x[input_y==1]))
    print("New ratio %s" % str(ratio))

    # Fit the model
    fit_history = net_model.fit(
        x=input_x,
        y=input_y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        shuffle=True,
        verbose=1
    )
    net_model.save(r"storage/lstm_trained.h5")


def lstm_test(X_test, y_test, X_train, y_train, segmentation, dec_threshold, predict_train = False, fair_correct = False):

    # prediction is continuous
    # this value decides what to condier a shot
    decision_threshold = dec_threshold # bigger than this is considered shot
    y_pred_org = None
    model = load_model(r"storage/lstm_trained.h5")
    #print(model.summary()) # plots the model if this is of interest

    print("\n\nDoing prediction...")
    y_pred = model.predict(X_test)
    y_pred[y_pred <= decision_threshold] = int(0)
    y_pred[y_pred > decision_threshold] = int(1)
    if predict_train:
        y_train_pred = model.predict(X_test)
        y_train_pred[y_train_pred <=decision_threshold] = int(0)
        y_train_pred[y_train_pred >decision_threshold] = int(1)


    # fairness adjustment
    if segmentation in [SEGMENTATION.LSTM_RAW]:
        if fair_correct:
            y_pred_org = copy.deepcopy(y_pred)

            # clean prediction result for fair comparison
            y_pred = correct_y_val_slide(y_test, y_pred)

    if predict_train:
        print("\nPerformance Training set")
        class_report = classification_report(y_train, y_train_pred, target_names=["No Shot", "Shot"])
        print(class_report)

    print("\nPerformance Test set")
    class_report = classification_report(y_test, y_pred, target_names=["No Shot", "Shot"])
    print(class_report)

    plt.scatter(range(len(y_test)), y_test/2, c="r")
    plt.scatter(range(len(y_pred)), y_pred, c="g")
    if y_pred_org is None:
        y_pred_org = y_pred
    plt.scatter(range(len(y_pred_org)), y_pred_org * 3/4, c="b")
    plt.legend(["ground truth", "prediction used for eval", "prediction original"], loc="lower right")
    plt.ylim([0.3, 1.1])
    plt.show()


"""

