import numpy
import joblib


class Bootstrapper:
    def __init__(self, n_jobs=-1, bootstrap_count=128):
        """Create an instance of Bootstrapper.

        Parameters
        ----------
        n_jobs: int
            How many process workers are used. -1 means the CPU counts available. Default to -1.
        bootstrap_count: positive int
            How many bootstrap samples are drawn. Default to 128.
        """
        if type(n_jobs) != int or (n_jobs < 1 and n_jobs != -1):
            raise ValueError('n_jobs must be positive integer or -1. {} was given'.format(n_jobs))
        if type(bootstrap_count) != int or bootstrap_count < 1:
            raise ValueError('bootstrap_count must be positive integer. {} was given.'.format(bootstrap_count))
        self.n_jobs = n_jobs
        self.bootstrap_count = bootstrap_count

    def run(self, function, *samples):
        """Run bootstrapping process using the given function.

        Parameters
        ----------
        function: callable
            The function applied to bootstrap sampling. Must take numpy 1d-arrays as its vararg.
        samples: numpy 1d arrays
            The samples to be bootstrapped.

        Returns
        -------
        numpy 1d-array
            The function output applied to each bootstrap sampling.
        """
        def bootstrap():
            bs = [numpy.random.choice(s, s.size, replace=True) for s in samples]
            return function(*bs)

        return joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(bootstrap)() for i in range(self.bootstrap_count))
