import os
import numpy as np
import random
import shutil

class SIM:
    def __init__(
        self,
        tau_range = (100, 120, 1),
        taucs_range = (230, 250, 1),
        h0_range = (600, 800, 50),
        alpha_range = (3, 5, 1)
    ):
        # Suggested Parameter ranges.
        random.seed(69)
        self.tau_range = tau_range
        self.taucs_range = taucs_range
        self.h0_range = h0_range
        self.alpha_range = alpha_range

    def run(self, filename, wait):
        """
        Runs the simulation and postprocessing with a shell script. 
        code = 0 if success
        code = 1 if errors occurred
        """
        if wait:
            code = os.system(f"sh runsim.sh {filename}.geom {filename}.spectralOut {filename}.txt")
        else:
            code = os.system(f"sh runsim_no_wait.sh {filename}.geom {filename}.spectralOut {filename}.txt")
        return code
    
    def submit(self):
        # Run the simulation and postprocessing.
        code = os.system("sh combined_sim_postproc.sh")
        return code
    
    def edit_material_parameters(self, params):
        # Edit the material.config file.
        def tau0_edit(num):
            return f'tau0_slip               {num} {num}        # per family\n'

        def tausat_edit(num):
            return f'tausat_slip             {num} {num}       # per family\n'

        def h0_edit(num):
            return f'h0_slipslip             {num}\n'

        def a_edit(num):
            return f'a_slip                  {num}\n'

        path = './material.config'
        with open(path) as f:
            lines = f.readlines()

        lines[36] = tau0_edit(params[0])
        lines[37] = tausat_edit(params[1])
        lines[46] = h0_edit(params[2])
        lines[54] = a_edit(params[3])

        with open('material.config', 'w') as f:
            f.writelines(lines)
    
    def run_n_random_tests(self, n):
        """
        Runs n random simulations. Used when initializing a response surface.
        """
        error_code = 0
        simulations = 0

        while((error_code == 0) and simulations <= n):
            params = [np.arange(*self.tau_range),
                      np.arange(*self.taucs_range),
                      np.arange(*self.h0_range),
                      np.arange(*self.alpha_range)]
            self.edit_material_parameters(params)
            filename = '_'.join(str(p) for p in params)
            error_code = self.run(filename, wait=True)
            filename = filename + '.txt'
            shutil.copy2('postProc/512grains512_tensionX.txt', f'postProc/{filename}')
            print('Completed Simulation #', simulations)
            simulations += 1
            
    def run_single_test(self, params):
        """
        Runs a single simulation with 'params'. Used during optimization process.
        """
        self.edit_material_parameters(params)
        error_code = self.run(wait=True)
        filename = '_'.join(str(p) for p in params) + '.txt'
        shutil.copy2('postProc/512grains512_tensionX.txt', f'postProc/{filename}')
        print('Completed Simulation #', simulations)