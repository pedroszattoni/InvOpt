"""
This script runs all scrips in the examples folder, and check if any exception
was raised.

Author: Pedro Zattoni Scroccaro
"""
import unittest
from os.path import dirname, abspath
import subprocess

path_to_examples = dirname(dirname(abspath(__file__))) + '\\examples\\'


def test_script(script_folder, script_name):
    script_path = script_folder + script_name
    try:
        subprocess.check_output(
            ['python', script_path], stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running script {script_name}: {e.output}")
        return False
    else:
        print(f"Script {script_name} ran successfully.")
        return True


class TestScript(unittest.TestCase):
    def test_binary_LP_consistent_data(self):
        script_name = 'discrete_consistent\\binary_LP_consistent_data.py'
        self.assertTrue(test_script(path_to_examples, script_name))

    def test_binary_LP_inconsistent_data(self):
        script_name = 'discrete\\binary_LP_inconsistent_data.py'
        self.assertTrue(test_script(path_to_examples, script_name))

    def test_LP(self):
        script_name = 'continuous_linear\\LP.py'
        self.assertTrue(test_script(path_to_examples, script_name))

    def test_QP(self):
        script_name = 'continuous_quadratic\\QP.py'
        self.assertTrue(test_script(path_to_examples, script_name))

    def test_MILP(self):
        script_name = 'mixed_integer_linear\\MILP.py'
        self.assertTrue(test_script(path_to_examples, script_name))

    def test_MIQP(self):
        script_name = 'mixed_integer_quadratic\\MIQP.py'
        self.assertTrue(test_script(path_to_examples, script_name))

    def test_first_order_methods(self):
        script_name = 'FOM\\first_order_methods.py'
        self.assertTrue(test_script(path_to_examples, script_name))

    def test_vrptw(self):
        script_name = 'FOM\\vrptw.py'
        self.assertTrue(test_script(path_to_examples, script_name))


if __name__ == '__main__':
    unittest.main()
