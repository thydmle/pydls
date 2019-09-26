# import sys
# sys.path.append("/Desktop/Personal/Research/dls/pydls/code/infer")
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


def test_normalize():
    for h in range(10):
        assert counter_constants[h] == test_constants[h]


def test_distr_generator():
    for j in range(10):
        current_test = test_distributions[j]
        current_counter = counter_distributions[j]
        for k in range(10):
            assert current_test[k] == current_counter[k]

