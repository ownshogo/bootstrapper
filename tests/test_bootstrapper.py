import unittest
from bootstrapper import Bootstrapper


class MyBootstrapInit(unittest.TestCase):
    def test_emtpy_argument(self):
        Bootstrapper()

    def test_full_argument(self):
        Bootstrapper(n_jobs=2, bootstrap_count=256)

    def test_illegal_n_jobs(self):
        with self.assertRaises(ValueError):
            Bootstrapper(n_jobs=0)
        with self.assertRaises(ValueError):
            Bootstrapper(n_jobs=-2)
        with self.assertRaises(ValueError):
            Bootstrapper(n_jobs=1.2)

    def test_illegal_bootstrap_count(self):
        with self.assertRaises(ValueError):
            Bootstrapper(bootstrap_count=0)
        with self.assertRaises(ValueError):
            Bootstrapper(bootstrap_count=-1)
        with self.assertRaises(ValueError):
            Bootstrapper(bootstrap_count=128.256)


if __name__ == '__main__':
    unittest.main()
