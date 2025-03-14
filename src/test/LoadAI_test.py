import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print(sys.path)
import time
from unittest import TestCase, main
from src.problems.loadAI import loadAI
import traceback

class CVRPTest(TestCase):
    def setUp(self):
        return super().setUp()

    def test_thirty_customers(self):
        try:
            instance = "loadAI_toy"
            problem = loadAI(
                os.path.join(os.path.dirname(__file__),
                             "assets/instances/", f"{instance}.csv"),
            )

            start = time.time()
            problem.solve()
            end = time.time()
            print(f"solving took {end-start} seconds")
        except:
            traceback.print_exc()


    def tearDown(self):
        return super().tearDown()


if __name__ == "__main__":
    main()
