import pandas as pd
import numpy as np
import os 
import glob
import csv
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats

## Data preprocessing
def preprocess(path, to_csv=False):
    df = pd.read_csv(path, skiprows = 6, delimiter = "\t")
    #Main function that calls all other functions.
    strain = df["Mises(ln(V))"]
    stress = df["Mises(Cauchy)"]
    true_plastic_strain = getTruePlasticStrain(strain,stress)
    
    if to_csv:
        name_list = path.split('/')
        data = {
           "Trimmed Stress" : trimmedStress,
           "Trimmed Strain" : trimmedTrueStrain
        }
        newdf = pd.DataFrame(data)
        newdf.to_csv(name_list[-1] + ".csv", index=False)
    
    return (true_plastic_strain.to_numpy(), stress.to_numpy())


def getTruePlasticStrain(strain, stress):
    # Getting the slope
    slope = (stress[1] - stress[0]) / (strain[1] - strain[0])
    true_plastic_strain = strain - stress / slope
    return true_plastic_strain

def elastic(stress, strain, trueStrain):
    # Obtain the elastic stress and strain based on r-squared
    trimValue = np.arange(0.0005, 0.02, 0.00001)
    trimValue = np.flip(trimValue)
    r2 = 0.00
    elasticTrueStrain = trueStrain
    elasticStrain = strain
    elasticStress = stress
    while(r2 <= 0.985):
        for x in trimValue:
            val = np.argmax(strain >= x)
            elasticTrueStrain = trueStrain[1:val]
            elasticStrain = strain[1:val]
            elasticStress = stress[1:val]
            r2 = adjR(elasticStrain, elasticStress, 1)
    trimmedTrueStrain = trueStrain.drop(range(1, val))    
    trimmedStrain = strain.drop(range(1, val)) 
    trimmedStress = stress.drop(range(1, val)) 
    return elasticStress, elasticStrain, elasticTrueStrain, trimmedStress, trimmedStrain, trimmedTrueStrain

def plot(elasticStress, elasticStrain, trimmedStress, trimmedStrain, trimmedTrueStrain):
    plt.plot(trimmedStrain, trimmedStress, 'g', label = "SingleCrystalTest")
    plt.plot(trimmedTrueStrain, trimmedStress, 'b', label = "FlowCurve")
    plt.plot(elasticStrain, elasticStress, 'r', label = "Elastic")
    leg = plt.legend(loc = "upper right")
    plt.xlabel(xlabel = "Strain (mm)")
    plt.ylabel(ylabel = "Stress (MPa)")
    plt.xlim([0, 0.005])
    plt.figure(figsize = (6,6))
    plt.show()

def save_outputs(directory, save_to, save_csv=False):
    vr = {}
    for root, dirs, files in os.walk(directory):
        for i in files:
            if i.endswith('.txt'):
                processed = preprocess(root+i)
                params = tuple(float(p) for p in i[:-4].split('_'))
                vr[params] = processed
                if save_csv:
                    data = {
                       "Trimmed Stress" : processed[1],
                       "Trimmed Strain" : processed[0],
                    }
                    newdf = pd.DataFrame(data)
                    newdf.to_csv(f"{save_to}/{i[:-4]}.csv", index=False)
        break # Prevents os.walk from recurisvely going through folders
    return vr

def adjR(x, y, degree):
    results = []
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results = 1- (((1-(ssreg/sstot))*(len(y)-1))/(len(y)-degree-1))
    return results




def save_single_output(directory, file, save_csv=False):
    vr = {}
    processed = preprocess(directory+file)
    return processed