import json
import os
from itertools import combinations
import intvalpy as ip
import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt
from intvalpy import Interval, Tol, precision
from scipy.ndimage import label

from intvalpy_fix import IntLinIncR2
from scipy.optimize import linprog

precision.extendedPrecisionQ = True

channels = 8
cells = 1024

eps = 1 / (2 ** 14)


def draw_regression_line(x, b0, b1, ch, cell):
    plt.plot([x[0], x[-1]], [x[0] * b1 + b0, x[-1] * b1 + b0], color='deeppink', label=f'argmax for ({ch}, {cell})',
             linewidth=1)


def plot_lines_and_area(coefficients, x_values):
    # Преобразуем x_values в numpy.ndarray
    x_values = np.array(x_values)
    x_values = np.insert(x_values, 0, -1)
    x_values = np.append(x_values, 1)

    # Получаем сам массив коэффициентов (первый элемент из списка)
    coefficients = coefficients[0]

    # Массивы для хранения значений y для каждой прямой
    y_values_list = []

    # Массивы для хранения минимальных и максимальных значений y для каждой точки x
    y_min_values = []
    y_max_values = []

    # Строим все прямые
    for coef in coefficients:
        k, b = coef  # Распаковываем k и b
        y_values = k * x_values + b  # Уравнение прямой: y = kx + b

        y_values_list.append(y_values)  # Добавляем в список все y_values

    # Для каждой точки x находим минимум и максимум среди всех прямых
    for i in range(len(x_values)):
        # Извлекаем значения y для точки x[i] на всех прямых
        y_at_x = [y_values_list[j][i] for j in range(len(coefficients))]

        # Находим минимум и максимум для данной точки x
        y_min_values.append(np.min(y_at_x))
        y_max_values.append(np.max(y_at_x))

    # Преобразуем y_min_values и y_max_values в numpy массивы для удобства
    y_min_values = np.array(y_min_values)
    y_max_values = np.array(y_max_values)

    # Построение графиков прямых
    for coef in coefficients:
        k, b = coef
        y_values = k * x_values + b
        # plt.plot(x_values, y_values, label=f'y = {k:.4f}x + {b:.4f}')

    # Закрашиваем область между минимальными и максимальными значениями y
    plt.fill_between(x_values, y_min_values, y_max_values, color="skyblue", alpha=0.4)

    # Устанавливаем пределы осей с отступами
    x_min, x_max = np.min(x_values), np.max(x_values)
    y_min, y_max = np.min(y_min_values), np.max(y_max_values)
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_padding = 0.05 * x_range
    y_padding = 0.1 * y_range

    plt.xlim(x_min - x_padding, x_max + x_padding)
    plt.ylim(y_min - y_padding, y_max + y_padding)


def draw_information_set(X, Y, beta):
    vertices_tol = IntLinIncR2(X, Y, consistency='tol')
    vertices_uni = IntLinIncR2(X, Y)

    vertices_tol = [v for v in vertices_tol if len(v) > 0]
    vertices_uni = [v for v in vertices_uni if len(v) > 0]

    if len(vertices_tol) > 0:
        vertices_combined = np.vstack(vertices_tol)

        vertices_combined = np.concatenate([vertices_combined, [vertices_combined[0]]])

        plt.figure(figsize=(6, 6))
        plt.fill(vertices_combined[:, 0], vertices_combined[:, 1], color='gray',
                 alpha=0.5)
        plt.plot(vertices_combined[:, 0], vertices_combined[:, 1], color='black')  # Контур многоугольника

        plt.scatter(vertices_combined[:, 0], vertices_combined[:, 1], color='red', label='Vertices')
        plt.scatter([beta[0]], [beta[1]], color='black', label="Arg max Tol")
        vertices_combined = np.vstack(vertices_uni)

        vertices_combined = np.concatenate([vertices_combined, [vertices_combined[0]]])

        plt.fill(vertices_combined[:, 0], vertices_combined[:, 1], color='lightgray',
                 alpha=0.5)
        plt.plot(vertices_combined[:, 0], vertices_combined[:, 1], color='black')  # Контур многоугольника

        plt.scatter(vertices_combined[:, 0], vertices_combined[:, 1], color='red', label='Vertices')
        plt.legend()
        plt.title("Uni and Tol")
        plt.xlabel("b_1")
        plt.ylabel("b_0")
        plt.grid(True)
    else:
        print("Нет данных для построения многоугольника.")
    plt.show()
    return vertices_tol

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
def arg_max_tol_method(points, eps, ch, cell):
    print("Канал: " + str(ch) + " Ячейка: " + str(cell))
    x, y = zip(*points)
    y_low = list(np.array(y) - eps)
    y_up = list(np.array(y) + eps)
    weights = [eps] * len(y)

    X_mat = Interval([[[x_el, x_el], [1, 1]] for x_el in x])
    Y_vec = Interval([[y_el, weights[i]] for i, y_el in enumerate(y)], midRadQ=True)

    beta, tol, _, _, _ = Tol.maximize(X_mat, Y_vec)

    print("beta before", beta)
    print("tol before: " + str(tol))
    # draw_regression_line(x, beta[1], beta[0], ch, cell)
    # for i in range(len(y_low)):
    #     plt.plot([x[i], x[i]], [y_low[i], y_up[i]], color='darkslateblue')
    # plt.grid()
    # plt.legend()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    updated = 0
    if tol < 0:
        # if Tol value is less than 0, we must iterate over all rows and add some changes to Y_vec, so Tol became 0
        for i in range(len(Y_vec)):
            X_mat_small = Interval([
                [[x[i], x[i]], [1, 1]]
            ])
            Y_vec_small = Interval([[y[i], weights[i]]], midRadQ=True)
            value = Tol.value(X_mat_small, Y_vec_small, beta)
            if value < 0:
                weights[i] = abs(y[i] - (x[i] * beta[0] + beta[1])) + 1e-8
                updated += 1

    Y_vec = Interval([[y_el, weights[i]] for i, y_el in enumerate(y)], midRadQ=True)

    beta, tol, _, _, _ = Tol.maximize(X_mat, Y_vec)
    print("tol after: " + str(tol))
    print("beta after: ", beta)
    # draw_regression_line(x, beta[1], beta[0], ch, cell)
    # for i in range(len(y_low)):
    #     plt.plot([x[i], x[i]], [Y_vec[i].inf, Y_vec[i].sup], color='darkslateblue')
    # plt.grid()
    # plt.legend()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    # vertices = draw_information_set(X_mat, Y_vec, beta)
    # print(vertices)
    # plt.show()
    return beta


# using twin arithmetics
def twins_method(points, ch, cell):
    print("Канал: " + str(ch) + " Ячейка: " + str(cell))
    x, y = zip(*points)
    A = []
    B = []
    y_ex_up = [0] * 11
    y_ex_low = [0] * 11
    y_in_up = [0] * 11
    y_in_low = [0] * 11
    x_values = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for i in range(len(x_values)):
        # twin = find_twin(points[i])
        for n in range(4):
            A.append([[x_values[i], x_values[i]], [1, 1]])
        y_values = list(y[i * 100:(i + 1) * 100])
        y_values.sort()
        median = np.median(y_values)
        # внутренняя оценка
        y_in_up[i] = np.percentile(y_values, 75)
        y_in_low[i] = np.percentile(y_values, 25)
        iqr = y_in_up[i] - y_in_low[i]
        # внешняя оценка
        y_ex_up[i] = min(y_in_up[i] + 1.5 * iqr, max(y_values))
        y_ex_low[i] = max(y_in_low[i] - 1.5 * iqr, min(y_values))
        B.append([y_in_low[i], y_in_up[i]])
        B.append([y_ex_low[i], y_in_up[i]])
        B.append([y_in_low[i], y_ex_up[i]])
        B.append([y_ex_low[i], y_ex_up[i]])
    b_vec, tol_val = Tol.maximize(Interval(A), Interval(B))[0], Tol.maximize(Interval(A), Interval(B))[1]

    print(b_vec[1], b_vec[0])
    plt.plot([x[0], x[-1]], [x[0] * b_vec[0] + b_vec[1], x[-1] * b_vec[0] + b_vec[1]], color='deeppink',
             label=f'twins method for ({ch}, {cell})', linewidth=1)
    for i in range(len(x_values)):
        plt.plot([x_values[i], x_values[i]], [y_ex_low[i], y_ex_up[i]], color='orange')
        plt.plot([x_values[i], x_values[i]], [y_in_low[i], y_in_up[i]], color='blue')
    to_remove = []

    if tol_val < 0:
        # if Tol value is less than 0, we must iterate over all rows and add some changes to Y_vec, so Tol became 0
        for i in range(len(B)):
            X_mat_small = Interval([A[i]])
            Y_vec_small = Interval([B[i]])
            value = Tol.value(X_mat_small, Y_vec_small, b_vec)
            if value < 0:
                to_remove.append(i)

        for i in sorted(to_remove, reverse=True):
            del A[i]
            del B[i]

    X_mat_interval = Interval(A)
    Y_vec_interval = Interval(B)
    b_vec, tol_val = Tol.maximize(X_mat_interval, Y_vec_interval)[0], Tol.maximize(X_mat_interval, Y_vec_interval)[1]
    print(b_vec[1], b_vec[0])

    x_1 = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    vertices = draw_information_set(X_mat_interval, Y_vec_interval, b_vec)
    plot_lines_and_area(vertices, x_1)
    plt.plot([x[0], x[-1]], [x[0] * b_vec[0] + b_vec[1], x[-1] * b_vec[0] + b_vec[1]], color='black',
             label=f"twins method for ({ch}, {cell}) after correction", linewidth=1)
    for i in range(len(x_values)):
        plt.plot([x_values[i], x_values[i]], [y_ex_low[i], y_ex_up[i]], color='orange')
        plt.plot([x_values[i], x_values[i]], [y_in_low[i], y_in_up[i]], color='blue')
    plt.show()


if __name__ == "__main__":
    side_a_2 = load_data("bin/04_10_2024_074_068", "a")

    ch, cell = 0, 0
    arg_max_tol_method(side_a_2[ch][cell], eps, ch, cell)
    twins_method(side_a_2[ch][cell], ch, cell)

    ch, cell = 2, 50
    arg_max_tol_method(side_a_2[ch][cell], eps, ch, cell)
    twins_method(side_a_2[ch][cell], ch, cell)

    ch, cell = 2, 501
    arg_max_tol_method(side_a_2[ch][cell], eps, ch, cell)
    twins_method(side_a_2[ch][cell], ch, cell)
# 0 0,    0, 13,   2, 501
