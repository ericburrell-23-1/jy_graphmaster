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
            instance = "loadAI_20cus"
            path = os.path.join(os.path.dirname(__file__),
                            "assets", "loadai_instances", f"{instance}.csv")
            problem = loadAI(
                path
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
