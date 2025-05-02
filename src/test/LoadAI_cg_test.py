import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print(sys.path)
import time
from unittest import TestCase, main
from src.problems.loadAI_cg import loadAI_cg
import traceback

class CVRPTest(TestCase):
    def setUp(self):
        return super().setUp()

    def test_thirty_customers(self):
        run_model=True
        
        #instance_name = "uhual_160"
        instance_name = "loadAI_20"
        path = os.path.join(os.path.dirname(__file__),
                        "assets", "loadai_instances", f"{instance_name}.csv")
        problem = loadAI_cg(\
            path,instance_name
        )
        if run_model ==True:
            start = time.time()
            problem.solve()
            end = time.time()
            print(f"solving took {end-start} seconds")



    def tearDown(self):
        return super().tearDown()


if __name__ == "__main__":
    main()
