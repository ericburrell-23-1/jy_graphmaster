import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print(sys.path)
import time
from unittest import TestCase, main
from src.problems.CVRP import CVRP
import traceback
import random

class CVRPTest(TestCase):
    def setUp(self):
        return super().setUp()

    def test_thirty_customers(self):
        try:
            instance = "NYC3"
            problem = CVRP(
                os.path.join(os.path.dirname(__file__),
                             "assets/instances/", f"{instance}.vrp"),
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
    random.seed(0)
    main()
