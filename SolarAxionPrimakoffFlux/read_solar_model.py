# Copyright (c) 2024 Quan-feng WU <wuquanfeng@ihep.ac.cn> and Xun-Jie Xu <xunjie.xu@ihep.ac.cn>
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from numpy import loadtxt
from numpy import interp
from .units import units_in_MeV, proton_mass

class solar_model:
    def __init__(self, data_file_path="data/struct_b16_agss09.dat"):
        self.data = loadtxt(data_file_path)
        self.heads = [
            "Mass", "Radius", "Temp", "Rho", "Pres", "Lumi", "H1", "He4", "He3", "C12",
            "C13", "N14", "N15", "O16", "O17", "O18", "Ne", "Na", "Mg", "Al", "Si", "P",
            "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni"
        ]
        self.head_dict = {head: i for i, head in enumerate(self.heads)}
    # end __init__
    
    def get_data(self, head, r):
        assert 0 <= r <= 1
        return interp(r, self.data[:, self.head_dict["Radius"]], self.data[:, self.head_dict[head]])
    # end get_data

    def Sun_density_profile(self, r, unit: units_in_MeV):
        return self.get_data("Rho", r) * unit.g / unit.cm**3
    # end Sun_density_profile

    def Sun_temperature_profile(self, r, unit: units_in_MeV):
        return self.get_data("Temp", r) * unit.K
    # end Sun_temperature_profile

    def Sun_proton_number_density_profile(self, r, unit: units_in_MeV):
        return self.Sun_density_profile(r, unit) / proton_mass(unit)
    # end Sun_proton_number_density_profile
# end class solar_model
