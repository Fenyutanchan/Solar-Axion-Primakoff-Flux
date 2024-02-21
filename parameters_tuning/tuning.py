import numpy as np
import pandas as pd

from scipy.optimize import minimize

flux_data = pd.read_csv("solar_axion_flux_data.csv")
photon_energy_list = np.array(flux_data["photon_energy"].values)
axion_mass_list = np.array(flux_data["axion_mass"].values)
flux_list = np.array(flux_data["flux"].values)

def supress_factor(photon_energy, axion_mass, γ):
    return np.clip((1 - (axion_mass / photon_energy)**γ), a_min=0, a_max=None)

def approx_flux(params):
    A, α, β, γ = params
    S = supress_factor(photon_energy_list, axion_mass_list, γ)
    return A * photon_energy_list**α * np.exp(-photon_energy_list/β) * S

def loss_fun(params):
    approx_flux_list = approx_flux(params)
    assert approx_flux_list.shape == flux_list.shape
    return np.sum((approx_flux_list - flux_list)**2)

init_guess = np.array([6.02, 2.481, 1.205, 1.7])

res = minimize(loss_fun, init_guess, method='Nelder-Mead')

print("Initial loss:", loss_fun(init_guess))
print("Optimal parameters:", res.x)
print("Loss:", res.fun)
