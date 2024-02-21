import sys
from os.path import dirname
sys.path.append(dirname(dirname(__file__)))

import numpy as np
import SolarAxionPrimakoffFlux as sapf

unit = sapf.units_in_MeV(1) # eV = 1
keV = unit.keV
eV = unit.eV

π = np.pi

solar_data = sapf.solar_model()

g10 = 1e-10 / unit.GeV
flux_unit = sapf.flux_unit(1e10, unit)

energy_length = 1000
mass_length = 5

photon_energy_list = np.linspace(.1*keV, 19*keV, energy_length)
axion_mass_list = np.linspace(0, 4*keV, mass_length)
# dΦ_dE_list = [[sapf.solar_axion_flux(E, solar_data, unit=unit, g_aγ=g10, axion_mass=ma) for E in photon_energy_list] for ma in axion_mass_list]

data = np.zeros((energy_length * mass_length, 3))

data_index = 0
for ma in axion_mass_list:
    for E in photon_energy_list:
        data[data_index, 0] = E / keV
        data[data_index, 1] = ma / keV
        data[data_index, 2] = sapf.solar_axion_flux(E, solar_data, unit=unit, g_aγ=g10, axion_mass=ma) / flux_unit
        data_index += 1
    # end for E
# end for ma

np.savetxt('solar_axion_flux_data.csv', data, delimiter=',', header="photon_energy,axion_mass,flux", comments="")
