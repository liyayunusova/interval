import json
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from intvalpy import IntLinIncR2, Interval, Tol, precision
from intvalpy_fix import IntLinIncR2

precision.extendedPrecisionQ = True

def load_data(directory, side):
    values_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    loaded_data = []
    for i in range(8):
        loaded_data.append([])
        for j in range(1024):
            loaded_data[i].append([(values_x[i // 100], 0) for i in range(100 * len(values_x))])

    for offset, value_x in enumerate(values_x):
        data = {}
        with open(directory + "/" + str(value_x) + "lvl_side_" + side + "_fast_data.json", "rt") as f:
            data = json.load(f)
        for i in range(8):
            for j in range(1024):
                for k in range(len(data["sensors"][i][j])):
                    loaded_data[i][j][offset * 100 + k] = (value_x, data["sensors"][i][j][k])

    return loaded_data

# using Tol
def regression_type_1(points):
    x, y = zip(*points)
    # build intervals out of given points
    weights = [1 / 16384] * len(y)
    # we know that y_i = b_0 + b_1 * x_i
    # or, in other words
    # Y = X * b, where X is a matrix with row (x_i, 1), and b is a vector (b_1, b_0)
    X_mat = Interval([[[x_el, x_el], [1, 1]] for x_el in x])
    Y_vec = Interval([[y_el, weights[i]] for i, y_el in enumerate(y)], midRadQ=True)
    # find argmax for Tol
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat, Y_vec)
    updated = 0
    if tol_val < 0:
        # if Tol value is less than 0, we must iterate over all rows and add some changes to Y_vec, so Tol became 0
        for i in range(len(Y_vec)):
            X_mat_small = Interval([
                [[x[i], x[i]], [1, 1]]
            ])
            Y_vec_small = Interval([[y[i], weights[i]]], midRadQ=True)
            value = Tol.value(X_mat_small, Y_vec_small, b_vec)
            if value < 0:
                weights[i] = abs(y[i] - (x[i] * b_vec[0] + b_vec[1])) + 1e-8
                updated += 1

    Y_vec = Interval([[y_el, weights[i]] for i, y_el in enumerate(y)], midRadQ=True)
    # find argmax for Tol
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat, Y_vec)

    return b_vec, weights, updated

# using twin arithmetics
def regression_type_2(points):
    x, y = zip(*points)
    eps = 1 / 16384

    # first of all, lets build y_ex and y_in
    x_new = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    y_ex_up = [-float('inf')] * 11
    y_ex_down = [float('inf')] * 11
    y_in_up = [-float('inf')] * 11
    y_in_down = [float('inf')] * 11

    for i in range(len(x_new)):
        y_list = list(y[i * 100 : (i + 1) * 100])
        y_list.sort()
        y_in_down[i] = y_list[25] - eps
        y_in_up[i] = y_list[75] + eps
        y_ex_up[i] = min(y_list[75] + 1.5 * (y_list[75] - y_list[25]), y_list[-1])
        y_ex_down[i] = max(y_list[25] - 1.5 * (y_list[75] - y_list[25]), y_list[0])

    X_mat = []
    Y_vec = []
    for i in range(len(x_new)):
        x_el = x_new[i]
        # y_ex_up >= X_mat * b >= y_ex_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_ex_down[i], y_ex_up[i]])
        # y_in_up >= X_mat * b >= y_ex_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_ex_down[i], y_in_up[i]])
        # y_ex_up >= X_mat * b >= y_in_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_in_down[i], y_ex_up[i]])
        # y_in_up >= X_mat * b >= y_in_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_in_down[i], y_in_up[i]])

    # now we have matrix X * b = Y, but with some "additional" rows
    # we can walk over all rows and if some of them is less than 0, we can just remove it at all
    X_mat_interval = Interval(X_mat)
    Y_vec_interval = Interval(Y_vec)
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat_interval, Y_vec_interval)
    to_remove = []
    if tol_val < 0:
        # if Tol value is less than 0, we must iterate over all rows and add some changes to Y_vec, so Tol became 0
        for i in range(len(Y_vec)):
            X_mat_small = Interval([X_mat[i]])
            Y_vec_small = Interval([Y_vec[i]])
            value = Tol.value(X_mat_small, Y_vec_small, b_vec)
            if value < 0:
                to_remove.append(i)

        for i in sorted(to_remove, reverse=True):
            del X_mat[i]
            del Y_vec[i]

    X_mat_interval = Interval(X_mat)
    Y_vec_interval = Interval(Y_vec)
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat_interval, Y_vec_interval)

    vertices1 = IntLinIncR2(X_mat_interval, Y_vec_interval)
    vertices2 = IntLinIncR2(X_mat_interval, Y_vec_interval, consistency='tol')

    plt.xlabel("b0")
    plt.ylabel("b1")
    b_uni_vertices = []
    for v in vertices1:
        # если пересечение с ортантом не пусто
        if len(v) > 0:
            x, y = v[:, 0], v[:, 1]
            b_uni_vertices += [(x[i], y[i]) for i in range(len(x))]
            # plt.fill(x, y, linestyle='-', linewidth=1, color='gray', alpha=0.5, label="Uni")
            # plt.scatter(x, y, s=0, color='black', alpha=1)


    b_tol_vertices = []
    for v in vertices2:
        # если пересечение с ортантом не пусто
        if len(v) > 0:
            x, y = v[:, 0], v[:, 1]
            b_tol_vertices += [(x[i], y[i]) for i in range(len(x))]
            plt.fill(x, y, linestyle='-', linewidth=1, color='blue', alpha=0.3, label="Tol")
            plt.scatter(x, y, s=10, color='black', alpha=1)

    plt.scatter([b_vec[0]], [b_vec[1]], s=10, color='red', alpha=1, label="argmax Tol")
    plt.legend()
    return b_vec, (y_in_down, y_in_up), (y_ex_down, y_ex_up), to_remove, b_uni_vertices, b_tol_vertices
def build_plots(data, coord_x, coord_y):
    # method 1
    b_vec, rads, to_remove = regression_type_1(data)
    x, y = zip(*data)
    plt.figure()
    plt.title("Y(x) method 1 for " + str((coord_x, coord_y)))
    plt.scatter(x, y, label="medians")
    plt.plot([-0.5, 0.5], [b_vec[1] + b_vec[0] * -0.5, b_vec[1] + b_vec[0] * 0.5], label="Argmax Tol")
    plt.legend()
    print((coord_x, coord_y), 1, b_vec[0], b_vec[1], to_remove)

    plt.figure()
    plt.title("Y(x) - b_0*x - b_1 method 1 for " + str((coord_x, coord_y)))
    for i in range(len(y)):
        plt.plot([i, i], [y[i] - rads[i] - b_vec[1] - b_vec[0] * x[i],
                          y[i] + rads[i] - b_vec[1] - b_vec[0] * x[i]], color="k", zorder=1)
        plt.plot([i, i], [y[i] - 1 / 16384 - b_vec[1] - b_vec[0] * x[i],
                          y[i] + 1 / 16384 - b_vec[1] - b_vec[0] * x[i]], color="blue", zorder=2)
    # method 2
    plt.figure()
    plt.title("Uni and Tol method 2 for " + str((coord_x, coord_y)))
    b_vec2, y_in, y_ex, to_remove, b_uni_vertices, b_tol_vertices = regression_type_2(data)
    print((coord_x, coord_y), 2, b_vec2[0], b_vec2[1], len(to_remove))
    x2 = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    plt.figure()
    plt.title("Y(x) method 2 for " + str((coord_x, coord_y)))
    for i in range(len(x2)):
        plt.plot([x2[i], x2[i]], [y_ex[0][i], y_ex[1][i]], color="gray", zorder=1)
        plt.plot([x2[i], x2[i]], [y_in[0][i], y_in[1][i]], color="blue", zorder=2)

    plt.plot([-0.5, 0.5], [b_vec2[1] + b_vec2[0] * -0.5, b_vec2[1] + b_vec2[0] * 0.5], label="Argmax Tol", color="red",
             zorder=1000)

    x2 = [-3] + x2 + [3]

    for i in range(len(x2) - 1):
        x0 = x2[i]
        x1 = x2[i + 1]
        max_idx = 0
        min_idx = 0
        max_val = b_uni_vertices[0][1] + b_uni_vertices[0][0] * (x0 + x1) / 2
        min_val = b_uni_vertices[0][1] + b_uni_vertices[0][0] * (x0 + x1) / 2
        for j in range(len(b_uni_vertices)):
            val = b_uni_vertices[j][1] + b_uni_vertices[j][0] * (x0 + x1) / 2
            if max_val < val:
                max_idx = j
                max_val = val
            if min_val > val:
                min_idx = j
                min_val = val

        y0_low = b_uni_vertices[min_idx][1] + b_uni_vertices[min_idx][0] * x0
        y1_low = b_uni_vertices[min_idx][1] + b_uni_vertices[min_idx][0] * x1
        y0_hi = b_uni_vertices[max_idx][1] + b_uni_vertices[max_idx][0] * x0
        y1_hi = b_uni_vertices[max_idx][1] + b_uni_vertices[max_idx][0] * x1
        # plt.fill([x0, x1, x1, x0], [y0_low, y1_low, y1_hi, y0_hi], facecolor="lightgray", linewidth=0)

        max_idx = 0
        min_idx = 0
        max_val = b_tol_vertices[0][1] + b_tol_vertices[0][0] * (x0 + x1) / 2
        min_val = b_tol_vertices[0][1] + b_tol_vertices[0][0] * (x0 + x1) / 2
        for j in range(len(b_tol_vertices)):
            val = b_tol_vertices[j][1] + b_tol_vertices[j][0] * (x0 + x1) / 2
            if max_val < val:
                max_idx = j
                max_val = val
            if min_val > val:
                min_idx = j
                min_val = val

        y0_low = b_tol_vertices[min_idx][1] + b_tol_vertices[min_idx][0] * x0
        y1_low = b_tol_vertices[min_idx][1] + b_tol_vertices[min_idx][0] * x1
        y0_hi = b_tol_vertices[max_idx][1] + b_tol_vertices[max_idx][0] * x0
        y1_hi = b_tol_vertices[max_idx][1] + b_tol_vertices[max_idx][0] * x1
        plt.fill([x0, x1, x1, x0], [y0_low, y1_low, y1_hi, y0_hi], facecolor="lightblue", linewidth=0)

    plt.xlim((-0.6, 0.6))
    plt.ylim((-0.6, 0.6))
def amount_of_neg(all_data, coord_x, coord_y):
    x, y = zip(*all_data[coord_y][coord_x])
    eps = 1 / 16384

    # first of all, lets build y_ex and y_in
    x_new = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    y_ex_up = [-float('inf')] * 11
    y_ex_down = [float('inf')] * 11
    y_in_up = [-float('inf')] * 11
    y_in_down = [float('inf')] * 11

    for i in range(len(x_new)):
        y_list = list(y[i * 100: (i + 1) * 100])
        y_list.sort()
        y_in_down[i] = y_list[25] - eps
        y_in_up[i] = y_list[75] + eps
        y_ex_up[i] = min(y_list[75] + 1.5 * (y_list[75] - y_list[25]), y_list[-1])
        y_ex_down[i] = max(y_list[25] - 1.5 * (y_list[75] - y_list[25]), y_list[0])

    X_mat = []
    Y_vec = []
    for i in range(len(x_new)):
        x_el = x_new[i]
        # y_ex_up >= X_mat * b >= y_ex_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_in_down[i], y_in_up[i]])

    # now we have matrix X * b = Y, but with some "additional" rows
    # we can walk over all rows and if some of them is less than 0, we can just remove it at all
    X_mat_interval = Interval(X_mat)
    Y_vec_interval = Interval(Y_vec)
    to_remove = []
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat_interval, Y_vec_interval)
    # if Tol value is less than 0, we must iterate over all rows and add some changes to Y_vec, so Tol became 0
    for i in range(len(Y_vec)):
        X_mat_small = Interval([X_mat[i]])
        Y_vec_small = Interval([Y_vec[i]])
        value = Tol.value(X_mat_small, Y_vec_small, b_vec)
        if value < 0:
            to_remove.append(i)
    return len(to_remove)

if __name__ == "__main__":
    side_a_2 = load_data("bin/04_10_2024_074_068", "a")
    '''
    val = [0] * 8
    for i in range(8):
        val[i] = [0] * 1024
    for j in range(1024):
        for i in range(8):
            val[i][j] = amount_of_neg(side_a_1, j, i)
            print(i, j, val[i][j])
    '''
    #build_plots(side_a_1[0][0], 0, 0)
    #build_plots(side_a_1[3][73], 3, 73)

    # build_plots(side_a_2[0][0], 0, 0)
    build_plots(side_a_2[2][50], 2, 50)
    # build_plots(side_a_2[2][501], 2, 501)

    plt.show()