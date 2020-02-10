import numpy
import joblib


class Bootstrapper:
    """The entry point of all Bootstrapper features.

    :param n_jobs: How many process workers are used. -1 means the CPU counts available. Defaults to -1.
    :type n_jobs: int, optional

    :param bootstrap_count: How many bootstrap samples are drawn. Defaults to 10_000.
    :type bootstrap_count: int, optional

    :raise ValueError: When n_jobs is not positive integer or -1.
    :raise ValueError: When bootstrap_count is not positive integer.
    """
    def __init__(self, n_jobs=-1, bootstrap_count=10_000):
        if type(n_jobs) != int or (n_jobs < 1 and n_jobs != -1):
            raise ValueError('n_jobs must be positive integer or -1. {} was given'.format(n_jobs))
        if type(bootstrap_count) != int or bootstrap_count < 1:
            raise ValueError('bootstrap_count must be positive integer. {} was given.'.format(bootstrap_count))
        self.n_jobs = n_jobs
        self.bootstrap_count = bootstrap_count

    def run(self, function, *samples):
        """Run bootstrapping process using the given function.

        :param function: The function applied to bootstrap sampling. Must take numpy 1d-arrays as its vararg.
        :type function: callable

        :param samples: The samples to be bootstrapped.
        :type samples: numpy 1d arrays

        :return: The function output applied to each bootstrap sampling.
        :rtype: numpy 1d array
        """
        def bootstrap():
            bs = [numpy.random.choice(s, s.size, replace=True) for s in samples]
            return function(*bs)

        rtn = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(bootstrap)() for _ in range(self.bootstrap_count))
        return numpy.array(rtn)

    def ci(self, function, confidence_level, *samples):
        """Calculate bootstrap confidence interval.

        :param function: The function applied to bootstrap sampling. Must take numpy 1d-arrays as its vararg.
        :type function: callable

        :param confidence_level: The confidence level of confidence interval. Must be in range (0, 1).
        :type confidence_level: float

        :param samples: The samples to be bootstrapped.
        :type samples: numpy 1d arrays

        :return: The lower bound, higher bound of confidence interval, and bootstrap sample.
        :rtype: (float, float, numpy 1d-array)
        """
        if confidence_level <= 0 or 1 <= confidence_level:
            raise ValueError('confidence_level must be in range (0, 1). {} was given.'.format(confidence_level))
        bs_samples = self.run(function, *samples)
        low_quantile = (1 - confidence_level) / 2
        high_quantile = 1 - low_quantile
        return numpy.quantile(bs_samples, low_quantile), numpy.quantile(bs_samples, high_quantile), bs_samples

    def test_mean_diff(self, x, y):
        """Conduct a hypothesis test of mean difference between two samples.
        Null hypothesis: :math:`\\mu_x = \\mu_y`.
        Alternative hypothesis: :math:`\\mu_x > \\mu_y`.

        :param x: One of the two samples to test.
        :type x: numpy 1d array

        :param y: Another of the two samples to test.
        :type y: numpy 1d array

        :return: The p-value and bootstrap samples of mean differences under null hypothesis :math:`\\mu_x = \\mu_y`
        :rtype: (float, numpy 1d array)
        """

        sample_mean_diff = x.mean() - y.mean()
        pooled_mean = numpy.concatenate([x, y]).mean()
        xplus = x - x.mean() + pooled_mean
        yplus = y - y.mean() + pooled_mean
        boot_mean_diff = self.run(lambda s1, s2: s1.mean() - s2.mean(), xplus, yplus)
        pvalue = (boot_mean_diff >= sample_mean_diff).sum() / self.bootstrap_count
        return pvalue, boot_mean_diff
