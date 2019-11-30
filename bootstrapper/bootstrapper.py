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
        self.n_jobs = n_jobs
        if bootstrap_count < 1:
            raise ValueError('bootstrap_count must be positive integer. {} was given.'.format(bootstrap_count))
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
