from enum import Enum
import pandas as pd
import os
import numpy as np
from scipy.stats import skew, kurtosis
import dill
import warnings
from processing.util.tools import parallelize_stuff
warnings.filterwarnings("ignore")


class SEGMENTATION(Enum):
    MAX_SPEED_1_SEC = 0 # cut at points of max speed, with 1 Sec beofe
    SLIDING_WINDOW = 1 # sliding window with fixed size
    LSTM_RAW = 2 # raw lstm with labels


def had_shot(ts_min, ts_max, ts_list):
    # If a shot label is within this time window return true
    if np.any((ts_list < ts_max) & (ts_list > ts_min)):
        return True
    return False


def segment_raw_shot_area(ev_dfs, pos_dfs, fixed_i = -1):
    """ LSTM Segmentation """

    step = 3  # step size
    window_size = 20  # windows of 20 == 1 sec
    eps_before = 5 # shot area +- eps values will be considered a shot
    eps_after = 5
    relevant_cols = ["speed", "xpos", "ypos", "xsmooth", "ysmooth", "absxpos", "absypos"]

    for i in range(len(ev_dfs)):
        if fixed_i != -1:
            i = fixed_i
            pos_df = pos_dfs[0]
            ev_df = ev_dfs[0]
        else:
            pos_df = pos_dfs[i]
            ev_df = ev_dfs[i]

        pos_df["label"] = 0
        for k in range(len(ev_df)):
            print("Processing %d / %d" % (k, len(ev_df)))
            row = ev_df.iloc[k]
            event_time = row["ts"]
            upper = event_time + eps_after * 50 # ms
            lower = event_time - eps_before * 50 # ms

            # set label
            pos_df.loc[(pos_df["ts"] <= upper) & (pos_df["ts"] >= lower),"label"] = 1

        pos_df["absxpos"] = pos_df["xpos"]
        pos_df["absypos"] = pos_df["ypos"]
        pos_df["xpos"] -= pos_df["xpos"].iloc[0]
        pos_df["ypos"] -= pos_df["ypos"].iloc[0]
        pos_df["xsmooth"] -= pos_df["xsmooth"].iloc[0]
        pos_df["ysmooth"] -= pos_df["ysmooth"].iloc[0]
        pos_df["xypos"] -= pos_df["xypos"].iloc[0]

        # Normalization relative position
        # shot to right -> increase in x
        # shot to left -> decrease in x
        # same for y -> shot from top or bottom is still shot
        # ---> normalize such that decrease is inverted
        if pos_df["xpos"].iloc[0] > pos_df["xpos"].iloc[-1]:  # invert it
            pos_df["xpos"] -= pos_df["xpos"].max()
            pos_df["xpos"] *= -1
        if pos_df["ypos"].iloc[0] > pos_df["ypos"].iloc[-1]:  # invert it
            pos_df["ypos"] -= pos_df["ypos"].max()
            pos_df["ypos"] *= -1
        if pos_df["xsmooth"].iloc[0] > pos_df["xsmooth"].iloc[-1]:  # invert it
            pos_df["xsmooth"] -= pos_df["xsmooth"].max()
            pos_df["xsmooth"] *= -1
        if pos_df["ysmooth"].iloc[0] > pos_df["ysmooth"].iloc[-1]:  # invert it
            pos_df["ysmooth"] -= pos_df["ysmooth"].max()
            pos_df["ysmooth"] *= -1

        # Normalize values to be between 0 and 1
        pos_df["absxpos"] -= pos_df["absxpos"].min()
        pos_df["absxpos"] /= pos_df["absxpos"].max()
        pos_df["absypos"] -= pos_df["absypos"].min()
        pos_df["ypos"] /= pos_df["absypos"].max()
        pos_df["speed"] -= pos_df["speed"].min()
        pos_df["speed"] /= pos_df["speed"].max()
        pos_df["xpos"] -= pos_df["xpos"].min()
        pos_df["xpos"] /= pos_df["xpos"].max()
        pos_df["ypos"] -= pos_df["ypos"].min()
        pos_df["ypos"] /= pos_df["ypos"].max()
        pos_df["xsmooth"] -= pos_df["xsmooth"].min()
        pos_df["xsmooth"] /= pos_df["xsmooth"].max()
        pos_df["ysmooth"] -= pos_df["ysmooth"].min()
        pos_df["ysmooth"] /= pos_df["ysmooth"].max()

        #  Store window of fixed size
        wins = []
        labels = []
        for k in range(0, len(pos_df), step):

            # Process window
            if k % 99 == 0: print("Processing %d of %d" % (k, len(pos_df)))
            try:
                window = pos_df.iloc[k: k + window_size]
            except:
                print("Skip")
                continue

            win_mat = window[relevant_cols].as_matrix()
            if win_mat.shape[0] ==window_size:
                wins += [win_mat]
                labels += [np.any(window["label"] == 1)]

        X = np.stack(wins)
        y = np.array(labels)

        storage_file_path = r"storage/raw_set_%d.dill" % (i)
        with open(storage_file_path, "wb") as dill_file:
            dill.dump([X, y], dill_file, recurse=True)
        print("Stored for future use to %s" % str(storage_file_path))


def segment_turning_point(ev_dfs, pos_dfs, fixed_i = -1):
    """
    Performs a segmentation based on turning points and stores results to file.
    Used in feature-based approach
    :return:
    """

    # ----------------------------------------
    #   perform segmentation at turning points
    # ----------------------------------------
    min_length = 0.5  # segments smaller than this are dropped in seconds
    window_prev = True  # if true use from each maximum a fixed size window before it
    window_prev_size = 20  # 20 == 1 sec
    recompute = True
    prepared_set = []

    for i in range(len(ev_dfs)):
        if fixed_i != -1:
            i = fixed_i
            pos_df = pos_dfs[0]
            ev_df = ev_dfs[0]
        else:
            pos_df = pos_dfs[i]
            ev_df = ev_dfs[i]
        print("-----------" + "\nDataset %d\n" % i + "-------------")

        # --------> try to read from file <--------
        storage_file_path = r"storage/windowed_set_%d.dill" % (i)
        if os.path.exists(storage_file_path) and not recompute:
            print("Reading from file")
            with open(storage_file_path, 'rb') as in_strm:
                data = dill.load(in_strm)
                [X, y] = data
            prepared_set += [[X, y]]


        # --------> else extract features per window <--------
        else:

            # Find segmentation indices
            mins = turning_points_min(np.array(pos_df["speed"]))
            maxs = turning_points_max(np.array(pos_df["speed"]))
            segmentation_indices = mins | maxs
            segmentation_indices = np.where(segmentation_indices == 1)[0]

            # Initialize result sets
            window_features = []
            labels = []
            start_times = []
            windows_collect = []

            # Segment at those points
            for r in range(segmentation_indices.shape[0] - 1):
                if r % 20 == 0: print("Processing %d of %d" % (r, segmentation_indices.shape[0]))

                # Process window
                win_from = segmentation_indices[r]
                win_to = segmentation_indices[r + 1]
                if window_prev:
                    win_from = max([0, win_to - window_prev_size])  # use window before
                window = pos_df.iloc[win_from: win_to]

                # also drop windows smaller than threshold
                seconds = (window["ts"].max() - window["ts"].min()) * 0.001
                if seconds < min_length: continue

                # Reset x and y pos to absolute zero, where it started
                start_time = window["ts"].min()
                window["absxpos"] = window["xpos"]
                window["absypos"] = window["ypos"]
                window["xpos"] -= window["xpos"].iloc[0]
                window["ypos"] -= window["ypos"].iloc[0]
                window["xsmooth"] -= window["xsmooth"].iloc[0]
                window["ysmooth"] -= window["ysmooth"].iloc[0]
                window["xypos"] -= window["xypos"].iloc[0]

                # Normalization relative position
                # shot to right -> increase in x
                # shot to left -> decrease in x
                # same for y -> shot from top or bottom is still shot
                # ---> normalize such that decrease is inverted
                if window["xpos"].iloc[0] > window["xpos"].iloc[-1]:  # invert it
                    window["xpos"] -= window["xpos"].max()
                    window["xpos"] *= -1
                if window["ypos"].iloc[0] > window["ypos"].iloc[-1]:  # invert it
                    window["ypos"] -= window["ypos"].max()
                    window["ypos"] *= -1
                if window["xsmooth"].iloc[0] > window["xsmooth"].iloc[-1]:  # invert it
                    window["xsmooth"] -= window["xsmooth"].max()
                    window["xsmooth"] *= -1
                if window["ysmooth"].iloc[0] > window["ysmooth"].iloc[-1]:  # invert it
                    window["ysmooth"] -= window["ysmooth"].max()
                    window["ysmooth"] *= -1

                # Normalize values to be between 0 and 1
                window["speed"] -= window["speed"].min()
                window["speed"] /= window["speed"].max()
                window["xpos"] -= window["xpos"].min()
                window["xpos"] /= window["xpos"].max()
                window["ypos"] -= window["ypos"].min()
                window["ypos"] /= window["ypos"].max()
                window["xsmooth"] -= window["xsmooth"].min()
                window["xsmooth"] /= window["xsmooth"].max()
                window["ysmooth"] -= window["ysmooth"].min()
                window["ysmooth"] /= window["ysmooth"].max()

                # Set label on Shot
                if had_shot(window["ts"].min(), window["ts"].max(), ev_df["ts"]):
                    # If shot in this window
                    label = 1
                else:
                    label = 0

                # Extract features
                features = []
                valid_features = True

                for col in ["speed", "xpos", "ypos", "xsmooth", "ysmooth", "absxpos",
                            "absypos"]:  # "xpos", "ypos", xypos should be a better kernel here as direction should not matter

                    # Min
                    features += [window[col].min()]

                    # Max
                    features += [window[col].max()]

                    # Mean
                    features += [window[col].mean()]

                    # Variance
                    var = np.var(window[col])
                    features += [var]

                    # Skew
                    sk = skew(window[col])
                    features += [sk]

                    # Kurtosis
                    kurt = kurtosis(window[col])
                    features += [kurt]

                    if col == "speed":
                        # Include Acceleration
                        try:
                            grad = np.gradient(window[col])
                        except:
                            # no gradient calculation possible
                            valid_features = False
                            break

                        # Variance of acceleration
                        features += [np.var(grad)]

                        # Mean acceleration
                        features += [np.abs(np.mean(grad))]

                if valid_features:
                    window_features += [features]
                    start_times += [start_time]
                    labels += [label]
                    windows_collect += [window]

            X = np.array(window_features)  # feature matrix per window
            y = np.array(labels)  # label matrix per window

            # write to file
            with open(storage_file_path, "wb") as dill_file:
                dill.dump([X, y, windows_collect], dill_file, recurse=True)
            print("Stored for future use to %s" % str(storage_file_path))

            prepared_set += [[X, y, windows_collect]]
    print("---> Done all data sets segmented and prepared")


def segment_window(ev_dfs, pos_dfs, fixed_i = -1):
    """ SLIDING window segmentation. Used in feature-based approach"""

    from scipy.stats import skew, kurtosis
    import dill

    step = 3  # step size
    window_size = 30  # windows of 20 == 1 sec
    prepared_set = []

    # Do windowing for all data sets, assign label per window
    for i in range(len(ev_dfs)):
        if fixed_i != -1:
            i = fixed_i
            pos_df = pos_dfs[0]
            ev_df = ev_dfs[0]
        else:
            pos_df = pos_dfs[i]
            ev_df = ev_dfs[i]
        print("------------" + "\nDataset %d\n" % i + "------------")

        # try to read from file
        storage_file_path = r"storage/windowed_set_%d_%d_%d.dill" % (i, step, window_size)

        # Initialize result sets
        window_features = []
        labels = []
        start_times = []
        windows_collect = []


        for k in range(0, len(pos_df), step):

            # Process window
            if k % 99 == 0: print("Processing %d of %d" % (k, len(pos_df)))
            try:
                window = pos_df.iloc[k: k + window_size]
            except:
                print("Skip")
                continue

            # Reset x and y pos to absolute zero, where it started
            start_time = window["ts"].min()
            window["absxpos"] = window["xpos"]
            window["absypos"] = window["ypos"]
            window["xpos"] -= window["xpos"].iloc[0]
            window["ypos"] -= window["ypos"].iloc[0]
            window["xsmooth"] -= window["xsmooth"].iloc[0]
            window["ysmooth"] -= window["ysmooth"].iloc[0]
            window["xypos"] -= window["xypos"].iloc[0]

            # Normalization relative position
            # shot to right -> increase in x
            # shot to left -> decrease in x
            # same for y -> shot from top or bottom is still shot
            # ---> normalize such that decrease is inverted
            if window["xpos"].iloc[0] > window["xpos"].iloc[-1]:  # invert it
                window["xpos"] -= window["xpos"].max()
                window["xpos"] *= -1
            if window["ypos"].iloc[0] > window["ypos"].iloc[-1]:  # invert it
                window["ypos"] -= window["ypos"].max()
                window["ypos"] *= -1
            if window["xsmooth"].iloc[0] > window["xsmooth"].iloc[-1]:  # invert it
                window["xsmooth"] -= window["xsmooth"].max()
                window["xsmooth"] *= -1
            if window["ysmooth"].iloc[0] > window["ysmooth"].iloc[-1]:  # invert it
                window["ysmooth"] -= window["ysmooth"].max()
                window["ysmooth"] *= -1

            # Normalize values to be between 0 and 1
            window["speed"] -= window["speed"].min()
            window["speed"] /= window["speed"].max()
            window["xpos"] -= window["xpos"].min()
            window["xpos"] /= window["xpos"].max()
            window["ypos"] -= window["ypos"].min()
            window["ypos"] /= window["ypos"].max()
            window["xsmooth"] -= window["xsmooth"].min()
            window["xsmooth"] /= window["xsmooth"].max()
            window["ysmooth"] -= window["ysmooth"].min()
            window["ysmooth"] /= window["ysmooth"].max()

            # Set label on Shot
            if had_shot(window["ts"].min(), window["ts"].max(), ev_df["ts"]):

                # If shot in this window -> but speed is falling -> use window before
                if window["speed"].iloc[0] > window["speed"].iloc[-1]:
                    labels[-1] = 1
                    label = 0

                else:
                    label = 1
            else:
                label = 0

            # Extract features
            features = []
            valid_features = True

            for col in ["speed", "xpos", "ypos", "xsmooth", "ysmooth", "absxpos",
                        "absypos"]:  # "xpos", "ypos", xypos should be a better kernel here as direction should not matter

                # Min
                features += [window[col].min()]

                # Max
                features += [window[col].max()]

                # Mean
                features += [window[col].mean()]

                # Variance
                var = np.var(window[col])
                features += [var]

                # Skew
                sk = skew(window[col])
                features += [sk]

                # Kurtosis
                kurt = kurtosis(window[col])
                features += [kurt]

                if col == "speed":
                    # Include Acceleration
                    try:
                        grad = np.gradient(window[col])
                    except:
                        # no gradient calculation possible
                        valid_features = False
                        break

                    # Variance of acceleration
                    features += [np.var(grad)]

                    # Mean acceleration
                    features += [np.abs(np.mean(grad))]

            if valid_features:
                window_features += [features]
                start_times += [start_time]
                labels += [label]
                windows_collect += [window]

        X = np.array(window_features)  # feature matrix per window
        y = np.array(labels)  # label matrix per window

        # write to file
        with open(storage_file_path, "wb") as dill_file:
            dill.dump([X, y], dill_file, recurse=True)
        print("Stored for future use to %s" % str(storage_file_path))

        prepared_set += [[X, y]]



def turning_points_max(a):
    """ Returns all maxima """
    return np.r_[1, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], 1]

def turning_points_min(a):
    """ Returns all minima """
    return np.r_[1, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], 1]


def prepare_set(data_folder = "data", segmentation_type = SEGMENTATION.MAX_SPEED_1_SEC):
    """
    Ingest, preprocess and segment the data
    :return:
    """

    # INGEST
    positions_files = ["180325_match1sthalf_positions.csv", "180325_match2ndhalf_positions.csv",
                       "180428_match_positions.csv"]
    events_files = ["180325_match1sthalf_events.csv", "180325_match2ndhalf_events.csv", "180428_match_events.csv"]

    pos_dfs = [pd.read_csv(os.path.join(data_folder, f), sep=";") for f in positions_files]
    ev_dfs = [pd.read_csv(os.path.join(data_folder, f), sep=";") for f in events_files]

    print("Columns pos: \t%s\nColumns ev \t%s" % (
    str(", ".join(list(pos_dfs[0].columns))), str(", ".join(list(ev_dfs[0].columns)))))


    # MERGE
    sep_str = "".join(["-"] * 20)
    for i in range(len(pos_dfs)):
        print("\n" + sep_str + "\nDataset %d\n" % i + sep_str)
        for col in ["sensor id", "mapped id", "full name"]:
            print("%s - %s" % (str(col), str(pos_dfs[i][col].unique())))
        for col in ["Player ID", "Name", "Event type"]:
            print("%s - %s" % (str(col), str(ev_dfs[i][col].unique())))

        # seems that mapped id corresponds to player id - make it consistent to match
        # -> make it more consistent by removing not needed columns
        # -> also rename some columns for better readability
        ev_dfs[i] = ev_dfs[i][["Timestamp", "Distance (m)", "Ball speed (km/h)", "Player ID"]].rename(
            columns={"Timestamp": "ts", "Distance (m)": "distance", "Ball speed (km/h)": "speed",
                     "Player ID": "idball"})
        pos_dfs[i] = pos_dfs[i][["ts in ms", "x in m", "y in m", "mapped id"]].rename(
            columns={"ts in ms": "ts", "x in m": "xpos", "y in m": "ypos", "mapped id": "idball"})

        print("New Sets: \n\nEvents:")
        print(ev_dfs[i].head(3))
        print("\nPositions:")
        print(pos_dfs[i].head(3))

        # check for nan columns
        print("\nEvent -> Columns with NaN values:\n%s" % str(ev_dfs[i].isnull().any()))
        print("\nPosition -> Columns with NaN values:\n%s" % str(pos_dfs[i].isnull().any()))

    print("\nNew columns")
    print("Columns pos: \t%s\nColumns ev \t%s" % (
    str(", ".join(list(pos_dfs[0].columns))), str(", ".join(list(ev_dfs[0].columns)))))



    # Split data set according to ball id -> giving one data set per ball
    # also crop each data frame 1 minute before and 1 minute after first and last shot
    prev_pos_dfs = pos_dfs
    prev_ev_dfs = ev_dfs
    ev_dfs = []
    pos_dfs = []
    for i in range(len(prev_pos_dfs)):
        for ball_id in list(prev_pos_dfs[i]["idball"].unique()):
            pos_dfs += [prev_pos_dfs[i][prev_pos_dfs[i]["idball"] == ball_id]]
            ev_dfs += [prev_ev_dfs[i][prev_ev_dfs[i]["idball"] == ball_id]]

            # latest
            gap = 20 * 60  # 60 seconds
            pos_dfs[-1] = pos_dfs[-1][pos_dfs[-1]["ts"] < ev_dfs[-1]["ts"].max() + gap]
            pos_dfs[-1] = pos_dfs[-1][pos_dfs[-1]["ts"] > ev_dfs[-1]["ts"].min() - gap]

    print("New number of datasets -> %s" % str(len(pos_dfs)))



    # Extract speed as additional information from this
    # (not normalized but rather an approximate -> but trend is sufficient here)
    import numpy as np

    def smooth(x, window_len=11, window='hanning'):
        if window_len < 3:
            return x
        # ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')
        y = np.convolve(w / w.sum(), s, mode='valid')
        return y

    for i in range(len(ev_dfs)):
        # smooth it for this
        timeline = pd.to_datetime(pos_dfs[i]["ts"], unit="ms")

        x_pos = pd.Series(pos_dfs[i]["xpos"].values, timeline)
        y_pos = pd.Series(pos_dfs[i]["ypos"].values, timeline)

        win_size = 41
        half_size = int(win_size / 2)
        x_pos_smoothed = smooth(x_pos, win_size)[half_size:-half_size]  # shift by half of window size
        y_pos_smoothed = smooth(y_pos, win_size)[half_size:-half_size]

        pos_dfs[i]["xspeed"] = pd.Series(x_pos_smoothed,
                                         timeline).diff().values * 3600.0  # speed in m/ms -> *3600 km/h  speed in x direction
        pos_dfs[i]["yspeed"] = pd.Series(y_pos_smoothed, timeline).diff().values * 3600.0
        pos_dfs[i]["xspeed"] = pos_dfs[i]["xspeed"].fillna(0)
        pos_dfs[i]["yspeed"] = pos_dfs[i]["yspeed"].fillna(0)
        pos_dfs[i]["speed"] = np.sqrt(
            np.power(pos_dfs[i]["xspeed"].as_matrix(), 2) + np.power(pos_dfs[i]["yspeed"].as_matrix(), 2))
        pos_dfs[i]["xypos"] = np.sqrt(
            np.power(pos_dfs[i]["xpos"].as_matrix(), 2) + np.power(pos_dfs[i]["ypos"].as_matrix(), 2))

        pos_dfs[i]["xsmooth"] = x_pos_smoothed
        pos_dfs[i]["ysmooth"] = y_pos_smoothed

        from matplotlib import pyplot as plt  # 20 samples == 1 second
        plt.plot(range(len(y_pos[40:160])), y_pos[40:160])
        plt.plot(range(len(y_pos_smoothed[40:160])), y_pos_smoothed[40:160])
        plt.xlabel("time in 50 ms")
        #plt.show()

    print("Segmentation...")
    if segmentation_type == SEGMENTATION.MAX_SPEED_1_SEC:
        inputs = [([ev_dfs[i]], [pos_dfs[i]], i) for i in range(len(ev_dfs))]
        parallelize_stuff(inputs, segment_turning_point, simultaneous_processes = len(inputs))

    if segmentation_type == SEGMENTATION.SLIDING_WINDOW:
        inputs = [([ev_dfs[i]], [pos_dfs[i]], i) for i in range(len(ev_dfs))]
        parallelize_stuff(inputs, segment_window, simultaneous_processes = len(inputs))

    if segmentation_type == SEGMENTATION.LSTM_RAW:
        inputs = [([ev_dfs[i]], [pos_dfs[i]], i) for i in range(len(ev_dfs))]
        parallelize_stuff(inputs, segment_raw_shot_area, simultaneous_processes = len(inputs))
