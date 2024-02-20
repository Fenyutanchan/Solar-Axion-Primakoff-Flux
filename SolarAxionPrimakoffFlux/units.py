# Copyright (c) 2024 Quan-feng WU <wuquanfeng@ihep.ac.cn> and Xun-Jie Xu <xunjie.xu@ihep.ac.cn>
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from numpy import pi as π

class units_in_MeV:
    def __init__(self, MeV):
        self.__constant()
        self.set_MeV(MeV)
    # end __init__
    
    def __constant(self):
        self.hbar = self.ħ = self.c = self.k = 1
        self.h = 2 * π * self.ħ

        self.h_SI = 6.62607015e-34 # J s
        self.ħ_SI = self.h_SI / (2 * π)
        self.c_SI = 299792458
        self.k_SI = 1.380649e-23
        self.e_SI = 1.602176634e-19

        self.alpha_EM = self.α_EM = 1/137.03599908421
    # end __constant
    
    def set_MeV(self, MeV):
        self.MeV = MeV
        self.__set_other_units()
    # end set_MeV
    
    def __set_other_units(self):
        self.eV = 1e-6 * self.MeV
        self.keV = 1e-3 * self.MeV
        self.GeV = 1e3 * self.MeV
        # self.TeV = 1e3 * self.GeV

        self.J = self.eV / 1.602176634e-19

        self.m = (self.ħ * self.c / self.J) / (self.ħ_SI * self.c_SI)
        self.cm = 1e-2 * self.m
        # self.fm = 1e-15 * self.m

        self.s = self.c_SI * self.m

        self.kg = self.J / (self.m**2 / self.s**2)
        self.g = 1e-3 * self.kg

        self.K = self.k_SI * self.J
    # end __set_other_units
# end class units_in_MeV
        
def proton_mass(unit: units_in_MeV):
    return 938.27208816 * unit.MeV
# end proton_mass

def electron_mass(unit: units_in_MeV):
    return 0.51099895 * unit.MeV
# end electron_mass

def radius_of_Sun(unit: units_in_MeV):
    return 6.957e8 * unit.m
# end radius_of_Sun

def distance_to_Sun(unit: units_in_MeV):
    return 1.495978707e11 * unit.m
# end distance_to_Sun

def flux_unit(prefix, unit: units_in_MeV):
    return prefix / (unit.cm**2 * unit.s * unit.keV)
# end flux_unit

# unit_MeV = units_in_MeV(1)
