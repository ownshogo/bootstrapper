# bootstrapper
A Python library to [bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))

# Basic usage
```python
# Import
from bootstrapper import Bootstrapper

# Generate instance
# n_jobs         : How many CPUs are used. -1 to use all CPUs available.
# bootstrap_count: How many bootstrap samples are drawn.
b = Bootstrapper(n_jobs=2, bootstrap_count=1_000)

# Run bootstrap
x = numpy.random.randn(1_000)
mean_dist = b.run(lambda x: x.mean(), x)
```

See [examples](examples) for more detailed usages.
