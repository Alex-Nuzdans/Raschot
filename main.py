import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import stats
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def mannwhitney(data,median,alpha=0.01):
    sample1 = data[data <= median]
    sample2 = data[data > median]
    stat,_ = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
    mu_U =len(sample1) * len(sample2) / 2
    sigma_U = np.sqrt(len(sample1) * len(sample2) * (len(sample1) + len(sample2) + 1) / 12)
    Z_stat = (stat - mu_U) / sigma_U
    z_critical = stats.norm.ppf(1 - alpha / 2)
    if abs(Z_stat) < z_critical:
        return True
    return False

def gist(data):
    if hasattr(data, 'values'):
        data = data.values.flatten()
    elif hasattr(data, 'to_numpy'):
        data = data.to_numpy().flatten()
    else:
        data = np.array(data).flatten()
    mean = np.mean(data)
    sigma_est = np.std(data)
    plt.figure(figsize=(8, 6))
    n, bins, patches = plt.hist(data, bins=30, density=True,
                                edgecolor='black', alpha=0.7,
                                color='lightblue', label='Гистограмма')
    data_min, data_max = data.min(), data.max()
    data_range = data_max - data_min
    x_min = data_min - 0.1 * data_range
    x_max = data_max + 0.1 * data_range
    x = np.linspace(x_min, x_max, 1000)
    p = stats.norm.pdf(x, mean, sigma_est)
    plt.plot(x, p, 'r-', linewidth=3,
             label=f'Нормальное распределение\nμ={mean:.2f}, σ={sigma_est:.2f}')
    plt.title('Гистограмма', fontsize=14)
    plt.xlabel('Значения', fontsize=12)
    plt.ylabel('Частота', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig("gistogramm.png", dpi=300)

def seryas(data,median,alpha):
    signs = np.where(data >= median, '+', '-')
    n_plus = np.sum(signs == '+')
    n_minus = np.sum(signs == '-')
    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            runs += 1
    temp1=2*n_plus*n_minus
    temp2=n_plus+n_minus
    z1=((runs-(temp1/temp2)-1)-0.5)/np.sqrt((temp1*(temp1-temp2))/(((temp2)**2)*(temp2+1)))
    z2 = stats.norm.ppf(1 - alpha/2)
    if abs(z1) < z2:
        return True
    return False

def hy2(data,alpha,mean):
    data = np.array(data)
    n = len(data)
    bins = int(1 + 3.322 * np.log10(n))
    bins = max(5, min(bins, 20))
    observed, bin_edges = np.histogram(data, bins=bins)
    std_est = np.std(data, ddof=0)
    probabilities = np.zeros(bins)
    for i in range(bins):
        prob = (stats.norm.cdf(bin_edges[i+1], loc=mean, scale=std_est) -
                stats.norm.cdf(bin_edges[i], loc=mean, scale=std_est))
        probabilities[i] = prob
    expected = probabilities * n
    i = 0
    while i < len(expected):
        if expected[i] < 5:
            if i == 0:  # объединяем с правым соседом
                expected[i + 1] += expected[i]
                observed[i + 1] += observed[i]
                expected = np.delete(expected, i)
                observed = np.delete(observed, i)
                bin_edges = np.delete(bin_edges, i + 1)
            elif i == len(expected) - 1:  # объединяем с левым соседом
                expected[i - 1] += expected[i]
                observed[i - 1] += observed[i]
                expected = np.delete(expected, i)
                observed = np.delete(observed, i)
                bin_edges = np.delete(bin_edges, i + 1)
            else:  # объединяем с правым соседом
                expected[i + 1] += expected[i]
                observed[i + 1] += observed[i]
                expected = np.delete(expected, i)
                observed = np.delete(observed, i)
                bin_edges = np.delete(bin_edges, i + 1)
        else:
            i += 1

    k = len(expected)
    chi2_stat = np.sum((observed - expected)**2 / expected)
    dof = k - 2 - 1
    critical_value = stats.chi2.ppf(1 - alpha, dof)
    if chi2_stat < critical_value:
        return True
    return False

def MMP(data,mean):
    sample_var_biased=data.var(ddof=0)[1]
    mu_mle = mean
    sigma_mle = np.sqrt(sample_var_biased)
    log_likelihood = np.sum(stats.norm.logpdf(data, loc=mu_mle, scale=sigma_mle))
    fitted_mu, fitted_sigma = stats.norm.fit(data)
    print(f"\nЛогарифмическая функция правдоподобия для оценок ММП: {log_likelihood:.4f}")
    print(f"  μ: {fitted_mu:.4f} (должно совпадать с выборочным средним)")
    print(f"  σ: {fitted_sigma:.4f} (должно совпадать с sqrt(смещенной дисперсии) = {sigma_mle:.4f})")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df=pd.read_csv("data.csv", sep=';', header=None,usecols=[1])
    df[1] = df[1].str.replace(',', '.')
    df[1] = pd.to_numeric(df[1], errors='coerce')
    mean=df.mean()[1]
    print("Среднее значение: ",mean)
    var=df.var()[1]
    print("Дисперсия: ",var)
    median=df.median()[1]
    print("Медиана: ",median)
    skew=df.skew()[1]
    print("Коэффицент ассиметрии: ",skew)#Больше 0 значит много мелких значений и несколько очень больших
    kurtosis=df.kurt()[1]
    print("Эксцесс: ",kurtosis)
    gist(df)#Распределение нормальное
    if seryas(df,median,alpha=0.05):
        print("Выборка случайна!")
    else:
        print("Выборка не случчайна!")
    MMP(df,mean)
    if hy2(df,alpha=0.1,mean=mean):
        print("Вид распределения соответствует!")
    else:
        print("Вид распределения не соответствует!")
    if mannwhitney(df,median,alpha=0.01):
        print("Выборка однородна!")
    else:
        print("Выборка не однородна!")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
