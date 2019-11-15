import numpy
import seaborn
import matplotlib.pyplot as plt

samples = numpy.random.rand(10240)
bootstrap_num = 256
means = []
for i in range(bootstrap_num):
    bootstrap_sample = numpy.random.choice(samples, samples.size, True)
    means.append(bootstrap_sample.mean())

seaborn.set()
seaborn.distplot(means, kde=True)
plt.show()
