import os
import sys

class DamaskAutomate:
    def __init__(self):
        
        # Initialize parameters
        self.p1 = 0
        self.p2 = 0
        self.p3 = 0
        self.p3 = 0
       
        self.project_path = "/scratch/project_2004956"
        
    def run_simulation(self):
        # Run the subFFTCSC.sh file.
        try:
            os.system("sbatch subFFTCSC.sh")
        except:
            sys.stdout.write("Error with task process.")
        return
    
    def run_postprocess(self):
        # Run the subFFTCSCpost.sh file.
        try:
            os.system("sbatch subFFTCSC.sh")
        except:
            sys.stdout.write("Error with post processing.")
        return