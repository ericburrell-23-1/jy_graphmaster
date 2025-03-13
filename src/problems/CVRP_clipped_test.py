import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print(sys.path)
import time
from unittest import TestCase, main
from src.problems.CVRP_clipped_LA_arc import CVRP_Clipped_LA_arc
import traceback
import random

class CVRPTest(TestCase):
    def setUp(self):
        return super().setUp()

    def test_thirty_customers(self):
        try:
            #instance = "100_customers"
            instance = "NYC3"
            problem = CVRP_Clipped_LA_arc(
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
    main()
