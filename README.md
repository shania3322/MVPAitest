# Implementation of ith-order statistic (i-test) for fMRI decoding accuracy

## reference paper:

Hirose S. (2021). Valid and powerful second-level group statistics for decoding accuracy: Information prevalence inference using the i-th order statistic (i-test). NeuroImage, 242, 118456. https://doi.org/10.1016/j.neuroimage.2021.118456

## Implemented:

1. 
**i-test-unif-binom** : assuming whether a trial or a subject is decodable follows
a binomial distribution, probability of decoding scores lower than threshold
in trials without label information is built with a binomial distribution. 

```
P(scores < threshold | p_chance) = binomial([scores*threshold]*, N_trial, p)

[X]* indicates the largest integer less than X
```

2.
**i-test-unif-perm** : probability of decoding scores lowder than threshold in
trials without label information is built with permutation test for each
subject

```
P(scores < threshold | p_chance) = 1/(N*M)* sum(permute_scores_{n,m} < threshold))

N: number of subjects
M: number of permutations
```

## Example:

![binom](plot_i_test/example_i_test_binom.png)
*Example of using binomial distribution: statstic power with different pervalence value and probability of correctly classified*

![perm](plot_i_test/example_i_test_perm.png)
*Example of using permutation test: statstic power with different pervalence value and probability of
correctly classified*
