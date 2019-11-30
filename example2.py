import numpy
import seaborn as sns
import matplotlib.pyplot as plt
from bootstrapper import Bootstrapper

sample1 = numpy.random.randn(1_000)
sample2 = numpy.random.randn(1_000)
b = Bootstrapper()
mean_diffs = b.run(lambda s1, s2: s1.mean() - s2.mean(), sample1, sample2)

sns.set()
sns.distplot(mean_diffs, kde=True)
plt.show()
