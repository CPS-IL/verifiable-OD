import math
from matplotlib import markers
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
import numpy as np
from scipy.optimize import curve_fit
import sys
from line_annotate import line_annotate
from constants import *

import warnings
warnings.filterwarnings("ignore", message="Covariance of the parameters could not be estimated")

scenarios = ["Simple", "Gap", "Incline"]

def beam_last_ground_index(d, h_g = 0):
    for i,b in enumerate(beam_angles):
        d_beam = (H_L - h_g) / math.tan(abs(b))
        if d_beam >= d:
            assert(i > 0)
            bi = i - 1
            assert (d >= (H_L - h_g) / math.tan(abs(beam_angles[bi])))
            assert (d <= (H_L - h_g) / math.tan(abs(beam_angles[bi + 1])))
            assert (d <= (H_L - h_g) / math.tan(abs(beam_angles[bi + 2])))
            return bi


def objective(x, a, b):
	return a * x + b


def find_max_line(X, res_th):
    curvefit_x = []
    curvefit_y = []

    for i,x in enumerate(X):
        if i == 0 or i == len(res_th1) - 1:
            continue

        if res_th[i-1] < res_th[i] and res_th[i] > res_th[i + 1]:
            curvefit_x.append(X[i])
            curvefit_y.append(res_th[i])

    while(1):
        assert len(curvefit_x) == len(curvefit_y)
        popt, _ = curve_fit(objective, curvefit_x, curvefit_y)
        a, b = popt

        found_error = False
        temp_x = []
        temp_y = []

        for i,x in enumerate(curvefit_x):
            if objective(x, a, b) <= curvefit_y[i]:
                temp_x.append(x)
                temp_y.append(curvefit_y[i])
            else:
                found_error = True

        if found_error == False:
            break
        else:
            if len (temp_x) < 2:
                break
            curvefit_x = temp_x
            curvefit_y = temp_y

        Y = []
        for x in X:
            Y.append(objective(x, a, b))

    return a, b, Y


def find_height_beam_incline(H_L, incline, ba, od, D_step):
    prev = H_L
    prev_oh = 0
    cur = 0
    for oh in np.arange(0, H_L, D_step/10):
        cur = abs((H_L - ((od + (oh / math.tan(incline))) * math.tan(abs(ba)))) - oh)
        if cur > prev:
            return prev_oh/math.sin(incline)
        prev = cur
        prev_oh = oh
    assert False


def plot_one(X, res_th1, res_th2, res_th3, filename):
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(4, 4))
    ax[0].plot(X, res_th1,color="blue",  label=r"$\alpha_{th}$" + " = {}$^o$".format(math.degrees(th1)))
    ax[1].plot(X, res_th2,color="purple", label=r"$\alpha_{th}$" + " = {}$^o$".format(math.degrees(th2)))
    ax[2].plot(X, res_th3,color="red",   label=r"$\alpha_{th}$" + " = {}$^o$".format(math.degrees(th3)))
    ax[0].set_ylim(0, y_lim)
    ax[1].set_ylim(0, y_lim)
    ax[2].set_ylim(0, y_lim)
    ax[0].legend(loc='upper left')
    ax[1].legend(loc='upper left')
    ax[2].legend(loc='upper left')
    ax[1].set_ylabel('Obstacle Length (m)')
    plt.xlabel('Obstacle Distance (m)')

    a, b, Y = find_max_line(X, res_th1)
    (l,) = ax[0].plot(X, Y,color="black", ls="--")
    line_annotate("y = {}x + {}".format(round(a,round_num), round(b,round_num)), l, line_annotate_start)
    ax[0].fill_between(X, res_th1, 75, color="forestgreen", alpha=alpha_color)

    a, b, Y = find_max_line(X, res_th2)
    (l,) = ax[1].plot(X, Y,color="black", ls="--")
    line_annotate("y = {}x + {}".format(round(a,round_num), round(b,round_num)), l, line_annotate_start)
    ax[1].fill_between(X, res_th2, 75, color="forestgreen", alpha=alpha_color)

    a, b, Y = find_max_line(X, res_th3)
    (l,) = ax[2].plot(X, Y,color="black", ls="--")
    line_annotate("y = {}x + {}".format(round(a,round_num), round(b,round_num)), l, line_annotate_start)
    ax[2].fill_between(X, res_th3, 75, color="forestgreen", alpha=alpha_color)

    plt.xticks(np.arange(0, D_max + 1, step=25))
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print("[INFO] Saved plot to", filename)


if "Simple" in scenarios:
    X = []
    Beam_H_2 = []
    Beam_H_3 = []

    res_th1 = []
    res_th2 = []
    res_th3 = []

    D_min = H_L / math.tan(abs(beam_angles[0])) # First point on ground assumption

    for od in np.arange(D_min + D_step, D_max + D_step, D_step):
        X.append(od)
        bi = beam_last_ground_index(od)
        height_beam_2 = H_L - (od * math.tan(abs(beam_angles[bi + 1])))
        height_beam_3 = H_L - (od * math.tan(abs(beam_angles[bi + 2])))
        assert(0 <= height_beam_2 < height_beam_3 <= H_L)
        Beam_H_2.append(height_beam_2)
        Beam_H_3.append(height_beam_3)

        delta_z = height_beam_2
        delta_x = od - (H_L / math.tan(abs(beam_angles[bi])))

        alpha = math.atan2(abs(delta_z), abs(delta_x))
        if alpha >= th1:
            res_th1.append(height_beam_2)
        else:
            res_th1.append(height_beam_3)

        if alpha >= th2:
            res_th2.append(height_beam_2)
        else:
            res_th2.append(height_beam_3)

        if alpha >= th3:
            res_th3.append(height_beam_2)
        else:
            res_th3.append(height_beam_3)

    plot_one(X, res_th1, res_th2, res_th3, "simple.pdf")


if "Gap" in scenarios:
    X = []
    Beam_H_2 = []
    Beam_H_3 = []

    res_th1 = []
    res_th2 = []
    res_th3 = []

    D_min = H_L / math.tan(abs(beam_angles[0])) # First point on ground assumption

    for od in np.arange(D_min + D_step, D_max + D_step, D_step):
        X.append(od)
        bi = beam_last_ground_index(od, h_gap)

        height_beam_2 = H_L - (od * math.tan(abs(beam_angles[bi + 1])))
        height_beam_3 = H_L - (od * math.tan(abs(beam_angles[bi + 2])))
        assert(0 <= height_beam_2 < height_beam_3 <= H_L)
        Beam_H_2.append(height_beam_2)
        Beam_H_3.append(height_beam_3)

        delta_z = height_beam_2
        delta_x = od - (H_L / math.tan(abs(beam_angles[bi])))

        alpha = math.atan2(abs(delta_z), abs(delta_x))
        if alpha >= th1:
            res_th1.append(height_beam_2 - h_gap)
        else:
            res_th1.append(height_beam_3 - h_gap)

        if alpha >= th2:
            res_th2.append(height_beam_2 - h_gap)
        else:
            res_th2.append(height_beam_3 - h_gap)

        if alpha >= th3:
            res_th3.append(height_beam_2 - h_gap)
        else:
            res_th3.append(height_beam_3 - h_gap)

    plot_one(X, res_th1, res_th2, res_th3, "gap.pdf")


if "Incline" in scenarios:
    X = []
    Beam_H_2 = []
    Beam_H_3 = []

    incline = math.radians(60)

    res_th1 = []
    res_th2 = []
    res_th3 = []

    D_min = H_L / math.tan(abs(beam_angles[0])) # First point on ground assumption

    for od in np.arange(D_min + D_step, D_max + D_step, D_step):
        bi = beam_last_ground_index(od)
        height_beam_2 = find_height_beam_incline(H_L, incline, beam_angles[bi + 1], od, D_step)
        height_beam_3 = find_height_beam_incline(H_L, incline, beam_angles[bi + 2], od, D_step)
        if height_beam_2 == -1 or height_beam_3 == -1:
            continue
        assert(0 <= height_beam_2 < height_beam_3)
        Beam_H_2.append(height_beam_2)
        Beam_H_3.append(height_beam_3)
        X.append(od)

        delta_z = height_beam_2
        delta_x = od - (H_L / math.tan(abs(beam_angles[bi])))

        alpha = math.atan2(abs(delta_z), abs(delta_x))
        if alpha >= th1:
            res_th1.append(height_beam_2)
        else:
            res_th1.append(height_beam_3)

        if alpha >= th2:
            res_th2.append(height_beam_2)
        else:
            res_th2.append(height_beam_3)

        if alpha >= th3:
            res_th3.append(height_beam_2)
        else:
            res_th3.append(height_beam_3)

    plot_one(X, res_th1, res_th2, res_th3, "incline.pdf")
