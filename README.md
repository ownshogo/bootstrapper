# Bootstrapper
A Python library for [bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29).

## Basic usage
```python
# Import
import numpy
from bootstrapper import Bootstrapper

# Instantiation
# n_jobs         : How many CPUs are used. -1 to use all CPUs available.
# bootstrap_count: How many bootstrap samples are drawn.
b = Bootstrapper(n_jobs=2, bootstrap_count=1_000)

# Run bootstrap
x = numpy.random.power(0.1, 1_000)
mean_dist = b.run(lambda arr: arr.mean(), x)
```

See [examples](examples) for more detailed usages.

Documentations can be found [here](https://ownshogo.github.io/bootstrapper/).
