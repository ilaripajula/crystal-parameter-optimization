import os
import numpy as np
import random
import shutil
from .preprocessing import *


class SIM:
    def __init__(
        self,
        param_range=None
    ):
        # Suggested Parameter ranges.
        self.param_range = param_range
        self.filename = 0
        self.filename2params = {}
        self.simulations = {}
        self.strain = None

    def submit_job(self):
        """
        Runs the simulation and postprocessing with a shell script. 
        code = 0 if success
        code = 1 if errors occurred
        """
        code = os.system(f'sh runsim.sh {self.filename}')
        return code
    
    def submit_array_jobs(self, start=None):
        """
        Run the simulation and postprocessing.
        Array jobs will submit multiple simulations up until filename:int.
        code = 0 if success
        code = 1 if errors occurred
        """
        if start:
            code = os.system(f'sh array_runsim.sh {self.filename} {start}')
        else:
            code = os.system(f'sh array_runsim.sh {self.filename} {1}')
        return code

    
    def make_new_job(self, params, path):
        shutil.copytree('./TEMPLATE/', path)
        self.filename2params[path] = params
        self.edit_material_parameters(params, path)
    
    def edit_material_parameters(self, params, job_path):
        # Edit the material.config file.
        def tau0_edit(num):
            return f'tau0_slip               {num} {num}        # per family\n'

        def tausat_edit(num):
            return f'tausat_slip             {num} {num}       # per family\n'

        def h0_edit(num):
            return f'h0_slipslip             {num}\n'

        def a_edit(num):
            return f'a_slip                  {num}\n'

        path = f'{job_path}/material.config'
        with open(path) as f:
            lines = f.readlines()

        lines[36] = tau0_edit(params[0])
        lines[37] = tausat_edit(params[1])
        lines[46] = h0_edit(params[2])
        lines[54] = a_edit(params[3])

        with open(f'{job_path}/material.config', 'w') as f:
            f.writelines(lines)
    
    def run_initial_simulations(self):
        """
        Runs N simulations according to get_grid().
        Used when initializing a response surface.
        """
        n_params = self.get_grid()
        for params in n_params:
            self.filename += 1
            path = f'./simulations/{str(self.filename)}'
            self.make_new_job(params, path)
        self.submit_array_jobs()
        self.strain = self.save_outputs()

    def run_single_test(self, params):
        """
        Runs a single simulation with 'params'.
        Used during optimization process.
        """
        self.filename += 1
        path = f'./simulations/{str(self.filename)}'
        self.make_new_job(params, path)
        self.submit_array_jobs(start=self.filename)
        self.save_single_output(path, params)
        
    def get_grid(self):
        points = []
        np.random.seed(450)
        for _ in range(30):
            tau = np.random.randint(low=self.param_range['tau']['low'], high=self.param_range['tau']['high'])
            taucs = np.random.randint(low=self.param_range['taucs']['low'], high=self.param_range['taucs']['high'])
            h0 = np.random.randint(low=self.param_range['h0']['low'], high=self.param_range['h0']['high'])
            alpha = np.random.randint(low=self.param_range['alpha']['low'], high=self.param_range['alpha']['high'])
            #alpha = np.round(np.random.uniform(low=3, high=4),1)
            points.append((tau, taucs, h0,alpha))
        return points
    
    def save_outputs(self):
        true_strains = []
        for (path, params) in self.filename2params.items():
            path2txt = f'{path}/postProc/'
            files = [f for f in os.listdir(path2txt) if os.path.isfile(os.path.join(path2txt, f))]
            processed = preprocess(f'{path2txt}/{files[0]}')
            true_strains.append(processed[0])
            self.simulations[params] = processed
        return np.array(true_strains).mean(axis=0).round(decimals=3) # Outputs the strain values for the simulations

    def save_single_output(self, path, params):
        path2txt = f'{path}/postProc/'
        files = os.listdir(path2txt)
        processed = preprocess(f'{path2txt}/{files[0]}')
        self.simulations[params] = processed