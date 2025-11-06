#!/Users/shiyawang/Documents/sy/project/pythonwork/.venv/bin/python
"""
Title: Implementation of ith-order test for information-based metrics proposed by Satoshi Hirose
Author: shiya wang
Date: Nov 2025
email: shaniasy.wang@gmail.com
"""

import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt


def find_ith_binom(N, alpha, r):
    for k in range(1, N + 1):
        i = binom.cdf(k - 1, N, 1 - r)
        if i > alpha:
            print(f"found largest i-th order {k - 1}")
            return k - 1
        print(f"{k}-th, with p {i}")


def simulate_D_acc_null(N, n_perm=1000, seed=555):
    """
    Generate distribution of null acc from label permutations with n_perm per subject for N subjects
    return numpy.ndarray with shape (N,n_perm)
    """
    rng = np.random.default_rng(seed=seed)
    mu = rng.normal(0.5, 0.05, N)
    var = rng.normal(0.2, 0.01, N)
    res = np.zeros((N, n_perm))
    for i, item in enumerate(zip(mu, var)):
        res[i] = rng.normal(item[0], item[1], n_perm)
    return np.clip(res, 0.1, 0.85)


def simulate_acc(N, seed=555):
    """
    Generate a right skewed acc with range [a,1.0]
    """
    a = 0.4
    rng = np.random.default_rng(seed=seed)
    x = rng.uniform(1.0, 2.0, N)
    y = np.log(x) + rng.normal(0, 0.1, N)
    return (y - np.min(y)) / (np.max(y) - np.min(y)) * (1 - a) + a


def plot_ai_L(lim=(0, 1.0)):
    """
    Plot a_i against L after setting the following values:
     alpha = 0.05
     r = 0.5
     N = 50
     i = 1,
     N_trial = 100
     P_correct=0.9
    """
    alpha = 0.05
    r = 0.8
    r0 = 0.5
    N = 50
    i = 19
    N_trial = 1000
    p_correct = 0.9
    p_chance = 0.5

    a_i = np.linspace(0, 1.0, N_trial)
    L = np.zeros(N_trial)
    T = np.zeros(N_trial)
    for ind, a in enumerate(a_i):
        P_a_chance = binom.cdf(np.ceil(a * N_trial).astype(int), N_trial, p_chance)
        Q = (1 - r0) * P_a_chance
        L[ind] = binom.cdf(i - 1, N, Q)
        T[ind] = a

    T_ = T[L < alpha][0]
    L_ = L[L < alpha][0]

    # plot
    plt.plot(T, L, "o-")
    plt.axhline(alpha)
    plt.text(lim[0], alpha, "0.05")
    plt.axvline(
        T_,
    )
    plt.title(f"T : {T_}")
    plt.xlim(*lim)
    plt.show()
    plt.close()

    # test
    P_a_lt_T = (1 - r) * binom.cdf(np.ceil(T_ * N_trial).astype(int), N_trial, p_chance) \
            + r * binom.cdf(np.ceil(T_ * N_trial).astype(int), N_trial, p_correct)

    P_sigf = binom.cdf(i - 1, N, P_a_lt_T)
    breakpoint()
    print(f"significant at {P_sigf}")


def search_ith_perm(imax, *args, h=0.05):
    """
    search ith order according to power using permuted acc
    """
    alpha, r0, p_chance, N, N_trial, perm_acc = args

    def set_T(i, r0, p_chance, N_trial):
        a_i = np.linspace(0, 1.0, N_trial)
        L = np.zeros(N_trial)
        T = np.zeros(N_trial)
        for ind, a in enumerate(a_i):
            P_a_chance = perm_acc[perm_acc < a].size / (perm_acc.shape[0] * perm_acc.shape[1])
            Q = (1 - r0) * P_a_chance
            L[ind] = binom.cdf(i - 1, N, Q)
            T[ind] = a

        return T[L < alpha][0]

    def cal_power(i, T, *args):
        r0, p_chance, N_trial, N, h = args
        rs = np.arange(r0, 1, h)
        p_corrs = np.arange(p_chance, 1, h)
        accum = 0
        accum_list = np.zeros((len(p_corrs), len(rs)))
        for j, r in enumerate(rs):
            for k, p_correct in enumerate(p_corrs):
                P1 = perm_acc[perm_acc < T].size / (perm_acc.shape[0] * perm_acc.shape[1])
                P_a_lt_T = (1 - r) * P1 \
                        + r * binom.cdf(np.ceil(T * N_trial).astype(int), N_trial, p_correct)
                P_ai_gt_T = binom.cdf(i - 1, N, P_a_lt_T)
                accum = accum + P_ai_gt_T
                accum_list[k, j] = P_ai_gt_T
        im = plt.imshow(accum_list, origin='lower', extent=[r0,1.0,p_chance,1.0])
        im.set_clim(0,1.0)
        plt.xlabel('p_correct')
        plt.ylabel('prevelance threshold r0')
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel("statistical power")
        plt.title(f'i-test-unif-perm at {i}-th order')
        plt.show()
        plt.close()
        return accum

    res = np.zeros(imax)
    for i in range(1, imax + 1):
        T = set_T(i, r0, p_chance, N_trial)
        res[i - 1] = cal_power(i, T, r0, p_chance, N_trial, N, h)
        print(f"i - {i}, T - {T}, power - {res[i - 1]}")

    return np.argmax(res) + 1


def search_ith_binom(imax, *args, h=0.05):
    """
    search ith order according to power using binomial distribution
    """
    alpha, r0, p_chance, N, N_trial = args

    def set_T(i, r0, p_chance, N_trial):
        a_i = np.linspace(0, 1.0, N_trial)
        L = np.zeros(N_trial)
        T = np.zeros(N_trial)
        for ind, a in enumerate(a_i):
            P_a_chance = binom.cdf(np.ceil(a * N_trial).astype(int), N_trial, p_chance)
            Q = (1 - r0) * P_a_chance
            L[ind] = binom.cdf(i - 1, N, Q)
            T[ind] = a

        return T[L < alpha][0]

    def cal_power(i, T, *args):
        r0, p_chance, N_trial, N, h = args
        rs = np.arange(r0, 1, h)
        p_corrs = np.arange(p_chance, 1, h)
        accum = 0
        accum_list = np.zeros((len(p_corrs), len(rs)))
        for j, r in enumerate(rs):
            for k, p_correct in enumerate(p_corrs):
                P_a_lt_T = (1 - r) * binom.cdf(np.ceil(T * N_trial).astype(int),N_trial, p_chance)\
                        + r * binom.cdf(np.ceil(T * N_trial).astype(int), N_trial, p_correct)
                P_ai_gt_T = binom.cdf(i - 1, N, P_a_lt_T)
                accum = accum + P_ai_gt_T
                accum_list[k, j] = P_ai_gt_T
        im=plt.imshow(accum_list,origin='lower',extent=[r0,1.0,p_chance,1.0])
        im.set_clim(0,1.0)
        plt.xlabel('p_correct')
        plt.ylabel('prevelance threshold r0')
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel("statistical power")
        plt.title(f"i-test-unif-binomial at {i}-th order")
        plt.show()
        plt.close()
        return accum

    res = np.zeros(imax)
    for i in range(1, imax + 1):
        T = set_T(i, r0, p_chance, N_trial)
        res[i - 1] = cal_power(i, T, r0, p_chance, N_trial, N, h)
        print(f"i - {i}, T - {T}, power - {res[i - 1]}")

    return np.argmax(res) + 1


def exhaust_imax(imax, *args):
    """
    test prevalence significance throughout all possible ith until imax
    """
    alpha, r0, p_chance, N_trial, N, perm_acc = args

    # 3. Perform i-test with iteration
    # 3.1 find a_i, ith smallest scores
    for i_unif in range(1, imax + 1):
        print(f"--- curent i is {i_unif}")
        acc_i = np.sort(acc)[i_unif - 1]
        print(f"acc_i: {acc_i}")

        # 3.2 calculate estimated probability P(a_n < a_i) in samples without label
        # information

        # Obtained from permutation test
        P_a_wol = perm_acc[perm_acc < acc_i].size / (perm_acc.shape[0] * perm_acc.shape[1])
        # breakpoint()

        # 5. calculate lower bound of the probability  P(a_n < a_i) in null
        # hypothesis (i.e. H0: prevalence <= r0)
        Q = (1 - r0) * P_a_wol

        # 6. compare lower bound  L with cricical value alpha
        L = binom.cdf(i_unif - 1, N, Q)
        # breakpoint()

        if L < alpha:
            print(f"Lower bound {L} < {alpha}, reject null, significant")
        else:
            print(f"Lower bound {L} > {alpha}, cannot reject null, not significant")

        ##Alternative: Obtrained from binomial distribution
        print(f"Alternatively: using binomial distribution")
        P_a_wol_binom = binom.cdf(
            (np.ceil(acc_i * N_trial)).astype(int), N_trial, p_chance
        )
        Q_binom = (1 - r0) * P_a_wol_binom
        L = binom.cdf(i_unif - 1, N, Q_binom)
        if L < alpha:
            print(f"binomial lower bound {L} < {alpha}, significant")
        else:
            print(f"binomial lower bound {L} > {alpha}, not significant")


def i_test_perm(ith, *args):
    """
    test prevalence significance using selected ith
    """
    alpha, r0, p_chance, N_trial, N, perm_acc, acc = args
    # 3. Perform i-test with ith
    acc_i = np.sort(acc)[ith - 1]
    print(f"acc_i: {acc_i}")

    # 3.2 calculate estimated probability P(a_n < a_i) in samples without label
    # information

    # Obtained from permutation test
    P_a_wol = perm_acc[perm_acc < acc_i].size / (perm_acc.shape[0] * perm_acc.shape[1])
    # breakpoint()

    # 5. calculate lower bound of the probability  P(a_n < a_i) in null
    # hypothesis (i.e. H0: prevalence <= r0)
    Q = (1 - r0) * P_a_wol

    # 6. compare lower bound  L with cricical value alpha
    L = binom.cdf(ith - 1, N, Q)
    # breakpoint()

    if L < alpha:
        print(f"Lower bound {L} < {alpha}, reject null, significant")
    else:
        print(f"Lower bound {L} > {alpha}, cannot reject null, not significant")

    print(f"Alternatively: using binomial distribution")
    P_a_wol_binom = binom.cdf((np.ceil(acc_i * N_trial)).astype(int), N_trial, p_chance)
    Q_binom = (1 - r0) * P_a_wol_binom
    L = binom.cdf(ith - 1, N, Q_binom)
    if L < alpha:
        print(f"binomial lower bound {L} < {alpha}, significant")
    else:
        print(f"binomial lower bound {L} > {alpha}, not significant")


if __name__ == "__main__":
    # 1. set critical value and prevalence thredhold
    N = 50
    alpha = 0.05
    r = 0.8
    r0 = 0.5  # theshold
    seed = 3284
    n_perm = 1000
    p_chance = 0.5
    p_correct = 0.9
    N_trial = 14

    ## Simulate data
    acc = simulate_acc(N, seed)
    plt.hist(acc, bins=10, density=True)
    plt.title(f"True accuracy of all {N} subjects")
    plt.show()
    plt.close()

    perm_acc = simulate_D_acc_null(N, n_perm=n_perm)
    plt.hist(perm_acc[3], bins=10, density=True)
    plt.title("permutation accuracy from one subject")
    plt.show()
    plt.close()

    # 2. Find imax
    imax = find_ith_binom(N, alpha, r0)
    breakpoint()

    ## DEBUG:test 1
    # plot_ai_L(lim=(0.40,0.60))
    # breakpoint()

    ## 3. Search for ith order
    ## manually set imax
    imax = 6
    #ith = search_ith_binom(imax, alpha, r0, p_chance, N, N_trial)
    ith = search_ith_perm(imax, alpha, r0, p_chance, N, N_trial, perm_acc)
    breakpoint()

    ## 4. Perform 2nd level test
    i_test_perm(ith, alpha, r0, p_chance, N_trial, N, perm_acc, acc)
    # exhaust_imax(imax, alpha, r0, p_chance, N_trial, N, perm_acc)
