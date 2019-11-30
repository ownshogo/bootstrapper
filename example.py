import numpy
import seaborn as sns
import matplotlib.pyplot as plt
from bootstrapper import Bootstrapper

sample = numpy.random.randn(1_000_000)
b = Bootstrapper()
means = b.run(lambda array: array.mean(), sample)

sns.set()
sns.distplot(means, kde=True)
plt.show()
