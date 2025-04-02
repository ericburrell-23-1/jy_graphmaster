import cProfile
import io
import pstats
ENABLE_PROFILING = True
class TimeProfiler: 
    def __init__(self, output_filename):
        self.filename = output_filename
        self.filename = self.filename.split('.')[0]
        self.pr = cProfile.Profile()
    def __enter__(self):
        self.pr.enable()
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(self.pr, stream=s).sort_stats('cumtime')
        ps.print_stats()
        #with open(self.filename + '.txt', 'w+') as f:
        #    f.write(s.getvalue())