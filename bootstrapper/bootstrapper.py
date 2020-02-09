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

        rtn = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(bootstrap)() for _ in range(self.bootstrap_count))
        return numpy.array(rtn)

    def ci(self, function, confidence_level, method='quantile', *samples):
        """Calculate bootstrap confidence interval.

        Parameters
        ----------
        function: callable
            The function applied to bootstrap sampling. Must take numpy 1d-arrays as its vararg.
        confidence_level: float
            The confidence level of confidence interval. Must be in range (0, 1).
        method: str
            How to calculate confidence interval. Available values are 'quantile' and 'student'.
            'quantile' means a simple confidence interval using quantile as plug-in statistics.
            'student' is also called bootstrap-t, where Student's t statistics is used instead of naive quantile.
        samples: numpy 1d arrays
            The samples to be bootstrapped.

        Returns
        -------
        Tuple of (float, float, numpy 1d-array)
            The lower bound, higher bound of confidence interval, and bootstrap sample.
        """
        if confidence_level <= 0 or 1 <= confidence_level:
            raise ValueError('confidence_level must be in range (0, 1). {} was given.'.format(confidence_level))
        bs_samples = self.run(function, *samples)
        low_quantile = (1 - confidence_level) / 2
        high_quantile = 1 - low_quantile
        return numpy.quantile(bs_samples, low_quantile), numpy.quantile(bs_samples, high_quantile), bs_samples
