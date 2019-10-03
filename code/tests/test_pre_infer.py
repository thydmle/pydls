import infer.pre_infer as pre_infer
import numpy as np
import scipy
import scipy.integrate

d = []
for i in range(10):
    d.append(np.linspace(i, (i + 9), 10))

mean = []
for a in range(10):
    mean.append(d[a][8] - d[a][0])

sigma = []
for b in range(10):
    sigma.append(1e-1)

counter_distributions = [] #? not sure how to name distributions calculated here
for c in range(10):
    f1 = (1/(sigma[c] * np.sqrt(2*np.pi)))*np.exp(-0.5*((d[c] - mean[c])/sigma[c])**2)
    counter_distributions.append(f1)

counter_constants = []
for g in range(10):
    counter_constants.append(1/(scipy.integrate.trapz(counter_distributions[g], dx=d[0][1] - d[0][0])))


test_distributions = []
for e in range(10):
    test_distributions.append(pre_infer.generate_distribution(d[e], mean[e], sigma[e], 1))

test_constants = []
for f in range(10):
    test_constants.append(pre_infer.normalize(counter_distributions[f], 1, d[f][1] - d[f][0]))

for t in range(10):
    counter_distributions[t] = counter_constants[t] * counter_distributions[t]

test_bimodal_distributions = []
for f in range(10):
    test_bimodal_distributions.append(pre_infer.generate_bimodal_distribution(d[f], mean[f], sigma[f], mean[f] + 2,
                                                                              sigma[f] + 2, 1))

counter_bimodal_distributions = []
for f in range(10):
    f1 = 1 / (sigma[f] * np.sqrt(2 * np.pi**2)) * np.exp(-(d[f] - mean[f]) ** 2 / (2 * sigma[f] ** 2))
    f1 = f1 * pre_infer.normalize(f1, 1, d[f][1] - d[f][0])

    f2 = 1 / ((sigma[f] + 2) * np.sqrt(2 * np.pi**2)) * np.exp(-(d[f] - (mean[f] + 2)) ** 2 / (2 * (sigma[f] + 2)**2))
    f2 = f2 * pre_infer.normalize(f2, 1, d[f][1] - d[f][0])

    z = f1 + f2
    z = z * pre_infer.normalize(z, 1, d[f][1] - d[f][0])
    counter_bimodal_distributions.append(z)


def test_normalize():
    for h in range(10):
        assert counter_constants[h] == test_constants[h]


def test_distr_generator():
    for j in range(10):
        current_test = test_distributions[j]
        current_counter = counter_distributions[j]
        for k in range(10):
            assert current_test[k] == current_counter[k]


eta = np.linspace(1, 11, 11)
n = np.ones(11)
angle = np.full(11, fill_value=np.pi)
k_b = np.full(11, fill_value=1.39e-23)
t = np.full(11, fill_value=298)
lambda_0 = np.full(11, fill_value=630)

counter_gamma = []
for i in range(11):
    counter_gamma.append((16 * np.pi * (n[i]**2) * ((np.sin(angle[i] / 2))**2) * k_b[i] * t[i]) /
                         (3 * (lambda_0[i]**2) * eta[i]))

test_gamma = []
for i in range(11):
    test_gamma.append(pre_infer.calc_gamma(eta[i], n[i], angle[i], k_b[i], t[i], lambda_0[i]))


def test_calc_gamma():
    for y in range(11):
        assert test_gamma[y] == counter_gamma[y]


def test_bimodal_distribution():
    for g in range(10):
        current_test = test_bimodal_distributions[g]
        current_counter = counter_bimodal_distributions[g]
        for k in range(10):
            assert current_test[k] == current_counter[k]
