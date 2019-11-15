import datetime

import numpy
import joblib
import seaborn
import matplotlib.pyplot as plt


def mean_from_bs(s):
    return numpy.random.choice(s, s.size, replace=True).mean()


samples = numpy.random.rand(10000000)
bootstrap_num = 256
start = datetime.datetime.now()
means = joblib.Parallel(n_jobs=-1)(joblib.delayed(mean_from_bs)(samples) for i in range(bootstrap_num))
end = datetime.datetime.now()
elapsed_sec = (end - start).total_seconds()
print('{} sec'.format(elapsed_sec))
