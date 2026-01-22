import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
from scipy import stats


def mannwhitney(data, alpha=0.01):
    if hasattr(data, "values"):
        data = data.values.flatten()
    elif hasattr(data, "to_numpy"):
        data = data.to_numpy().flatten()
    else:
        data = np.array(data).flatten()
    n = len(data)
    n1 = n // 2
    n2 = n - n1
    sample1 = data[:n1]
    sample2 = data[n1:]
    stat, _ = stats.mannwhitneyu(sample1, sample2, alternative="two-sided")
    mu_U = n1 * n2 / 2
    sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    Z_stat = (stat - mu_U) / sigma_U
    print("Z вычисленное = ", Z_stat)
    z_critical = stats.norm.ppf(1 - alpha / 2)
    print("Z критическое = ", z_critical)
    if abs(Z_stat) < z_critical:
        return True
    return False


def gist(data, theta=None):
    if hasattr(data, "values"):
        data = data.values.flatten()
    elif hasattr(data, "to_numpy"):
        data = data.to_numpy().flatten()
    else:
        data = np.array(data).flatten()

    data_min = data.min()
    data_max = data.max()
    k = int(1 + math.log2(len(data)))
    bins_fixed = np.linspace(data_min, data_max, k + 1)
    plt.figure(figsize=(8, 6))
    counts, bins, _ = plt.hist(
        data,
        bins=bins_fixed,
        density=True,
        edgecolor="black",
        alpha=0.7,
        color="lightblue",
        label="Гистограмма",
    )
    counts = counts / counts.sum()
    for i in range(len(bins) - 1):
        x_center = bins[i] + (bins[i + 1] - bins[i]) / 2
        interval_label = f"[{bins[i]:.2f},\n{bins[i + 1]:.2f})"

        plt.text(
            x_center,
            -max(counts) * 0.08,
            interval_label,
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
        )

    if theta is not None:
        x_min = -1
        x_max = 2 * theta
        x = np.linspace(x_min - 1, x_max + 1, 1000)
        p = stats.uniform(x_min, x_max - x_min).pdf(x)
        plt.plot(
            x,
            p,
            "r-",
            linewidth=3,
            label=f"Равномерное распределение a = {-1}, b = {(2 * theta):.4f}",
        )
    plt.title("Гистограмма", fontsize=14)
    plt.ylabel("Частота", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig("gistogramm.png", dpi=300)

    return (counts, bins)


def seryas(data, median, alpha):
    signs = np.where(data >= median, "+", "-")
    n_plus = np.sum(signs == "+")
    n_minus = np.sum(signs == "-")
    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            runs += 1
    print(
        "Количество плюсов: ",
        n_plus,
        "\nКоличество минусов: ",
        n_minus,
        "\nКоличество серий: ",
        runs,
    )
    temp1 = 2 * n_plus * n_minus
    temp2 = n_plus + n_minus
    z1 = ((runs - (temp1 / temp2) - 1) - 0.5) / np.sqrt(
        (temp1 * (temp1 - temp2)) / (((temp2) ** 2) * (temp2 + 1))
    )
    print("Z вычисленное = ", z1)
    z2 = stats.norm.ppf(1 - alpha / 2)
    print("Z критическое = ", z2)
    if abs(z1) < z2:
        return True
    return False


def MMP(data):
    print("\nОмп-оценка для параметра θ = max(Xi) / 2")
    xMax = np.max(data, axis=0)[1]
    theta = xMax / 2
    print(f"\tmax(Xi) = {xMax} => θ = {theta}")
    return theta


def hy2_linear(data, theta, alpha, gist_data):
    counts, bins = gist_data

    # функция распределения
    a = -1
    b = 2 * theta

    def F(x):
        return (x - a) / (b - a)

    sum = 0
    probsum = 0
    for i in range(len(counts)):
        interval_probability = F(bins[i + 1]) - F(bins[i])
        probsum += interval_probability
        expected = len(data) * interval_probability
        actual = counts[i] * len(data)
        sum += (actual - expected) ** 2 / expected
    print(f"X^2 вычисленное = {sum}")

    # Количество степеней свободы = (количество столбцов) - (количество оцениваемых параметров) - 1
    dof = len(counts) - 1 - 1
    critical_value = stats.chi2.ppf(1 - alpha, dof)
    print("X^2 критическое = ", critical_value)

    return sum < critical_value


if __name__ == "__main__":
    df = pd.read_csv("data.csv", sep=";", header=None, usecols=[1])
    df[1] = df[1].str.replace(",", ".")
    df[1] = pd.to_numeric(df[1], errors="coerce")
    mean = df.mean()[1]
    print("Среднее значение: ", mean)
    var = df.var()[1]
    print("Дисперсия: ", var)
    median = df.median()[1]
    print("Медиана: ", median)
    skew = df.skew()[1]
    print(
        "Коэффицент ассиметрии: ", skew
    )  # Больше 0 значит много мелких значений и несколько очень больших
    kurtosis = df.kurt()[1]
    print("Эксцесс: ", kurtosis)
    (counts, bins) = gist(df)  # Распределение нормальное
    print("Критерий серий:")
    if seryas(df, median, alpha=0.05):
        print("Выборка случайна!")
    else:
        print("Выборка не случчайна!")

    # Находим θ_омп
    theta = MMP(df)

    print("Хи-квадрат Пирсона:")
    print(f"Гипотеза H0: X ~ R[-1, {(2 * theta):4f}]")
    if hy2_linear(df, theta, 0.1, (counts, bins)):
        print("H0 принимается")
    else:
        print("H0 отклоняется")

    gist(df, theta)

    print("Критерий Манна-Уитни:")
    if mannwhitney(df, alpha=0.01):
        print("Выборка однородна!")
    else:
        print("Выборка не однородна!")
