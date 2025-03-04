import unittest

# this will run all test cases
if __name__ == '__main__':
    test_suite = unittest.defaultTestLoader.discover('.', pattern='test_*')
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)