import numpy
import seaborn as sns
import matplotlib.pyplot as plt
from bootstrapper import Bootstrapper

sample = numpy.random.randn(1000000)
b = Bootstrapper(bootstrap_count=1024)
means = b.run(sample, lambda array: array.mean())

sns.set()
sns.distplot(means, kde=True)
plt.show()
