import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mean = [0, 0]
cov  = [[1, 0], [0, 1]]
size = 300

x = np.random.multivariate_normal(mean, cov, size)


mean = [0, 0]
cov  = [[1, 0.8], [0.8, 1]]
size = 300

y = np.random.multivariate_normal(mean, cov, size)


# sns.set()
#
# plt.figure()
# sns.scatterplot(x[:, 0], x[:, 1])
#
#
# plt.figure()
# sns.scatterplot(y[:, 0], y[:, 1])


plt.figure()
plt.plot(x[:, 0], x[:, 1], 'o')

plt.figure()
plt.plot(y[:, 0], y[:, 1], 'o')


plt.show()
