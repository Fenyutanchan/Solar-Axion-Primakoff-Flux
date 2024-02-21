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

# photon_energy_list = np.linspace(.1*keV, 19*keV, energy_length)
axion_mass_list = np.linspace(0, 4*keV, mass_length)
# dΦ_dE_list = [[sapf.solar_axion_flux(E, solar_data, unit=unit, g_aγ=g10, axion_mass=ma) for E in photon_energy_list] for ma in axion_mass_list]

# data = np.zeros((energy_length * mass_length, 3))

normalized_factor = [7.2, 6.6, 4.8, 2.9, 1.5]

photon_energy_list = []
ma_list = []
dΦ_dE_list = []
# data_index = 0
for (ii, ma) in enumerate(axion_mass_list):
    E_over_keV = .1
    step_base = .1 * normalized_factor[ii] / 10
    while E_over_keV < 19:
        E = E_over_keV * keV
        flux = sapf.solar_axion_flux(E, solar_data, unit=unit, g_aγ=g10, axion_mass=ma) / flux_unit
        if flux == 0:
            E_over_keV += step_base
            continue

        photon_energy_list.append(E_over_keV)
        ma_list.append(ma / keV)
        dΦ_dE_list.append(flux)

        E_over_keV += step_base / (10 * flux)
    # end while

    # for E in photon_energy_list:
    #     data[data_index, 0] = E / keV
    #     data[data_index, 1] = ma / keV
    #     data[data_index, 2] = sapf.solar_axion_flux(E, solar_data, unit=unit, g_aγ=g10, axion_mass=ma) / flux_unit
    #     data_index += 1
    # # end for E
# end for ma

data = np.array([photon_energy_list, ma_list, dΦ_dE_list]).T

np.savetxt('solar_axion_flux_data.csv', data, delimiter=',', header="photon_energy,axion_mass,flux", comments="")
