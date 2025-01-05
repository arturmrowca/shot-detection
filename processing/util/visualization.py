from matplotlib import pyplot as plt


def visualize_pairwise(Xc, yc):
    """
    Visualize features pairwise
    """
    sub_Xp = Xc[yc == 1]
    sub_Xn = Xc[yc == 0]

    for feat_idx in range(Xc.shape[1]-1):
        feature_1_pos = sub_Xp[:, feat_idx]
        feature_1_neg = sub_Xn[:, feat_idx]

        for feat_idx_2 in range(Xc.shape[1]-1):
            feature_2_pos = sub_Xp[:, feat_idx_2]
            feature_2_neg = sub_Xn[:, feat_idx_2]

            plt.scatter(feature_1_neg, feature_2_neg, marker = "o", c="r")
            plt.scatter(feature_1_pos, feature_2_pos, marker = "o", c="g")
            plt.xlabel("feature %s" % str(feat_idx))
            plt.ylabel("feature %s" % str(feat_idx_2))
            plt.show()

def visualize_windows(Ws, ys, only_shot=False, compare=[], exclude = []):
    """
    Visualizes the given windows if available
    """
    i = -1
    for window in Ws:
        i += 1

        if ys[i] == 1:
            plt.title("Shot")
            cc = "r"
        else:
            plt.title("No Shot")
            cc = "b"
            if only_shot:continue

        if not compare:
            plt.plot(window["ts"], window["speed"], c=cc)
            if not "xsmooth" in exclude:plt.plot(window["ts"], window["xsmooth"])
            if not "ysmooth" in exclude:plt.plot(window["ts"], window["ysmooth"])
            if not "absxpos" in exclude:plt.plot(window["ts"], window["absxpos"])
            if not "absypos" in exclude:plt.plot(window["ts"], window["absypos"])
            plt.legend([a for a in ["speed", "xsmooth", "ysmooth", "absxpos", "absypos"] if not a in exclude])

            plt.show()

        else:
            for col in compare:
                t = window["ts"] - window["ts"].min()
                plt.plot(t, window[col], c=cc)

    if compare:
        plt.show()
