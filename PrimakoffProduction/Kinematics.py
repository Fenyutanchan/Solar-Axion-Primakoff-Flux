# Copyright (c) 2024 Quan-feng WU <wuquanfeng@ihep.ac.cn>
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from autograd import grad
from autograd.numpy import abs, isnan, sqrt
from warnings import warn

def λ(x, y, z):
    return (x - y - z)**2 - 4 * y * z
# end def λ

def center_of_mass_momentum_squared(s, m1, m2):
    return λ(s, m1**2, m2**2) / (4 * s)
# end def center_of_mass_momentum_squared

def center_of_mass_momentum(s, m1, m2):
    return sqrt(center_of_mass_momentum_squared(s, m1, m2))
# end def center_of_mass_momentum

# a + b -> 1 + 2
# s := (pa + pb)^2
# t := (pa - p1)^2
def dt_over_dCosθCM(s, ma, mb, m1, m2, cosθ):
    p_ini_squared = center_of_mass_momentum_squared(s, ma, mb)
    p_fin_squared = center_of_mass_momentum_squared(s, m1, m2)
    p_ini = sqrt(p_ini_squared)
    p_fin = sqrt(p_fin_squared)
    # Ea = sqrt(ma**2 + p_ini_squared)
    # E1 = sqrt(m1**2 + p_fin_squared)
    # t = ma**2 + m1**2 - 2 * (Ea * E1 - p_ini * p_fin * cosθ)
    return 2 * (p_ini * p_fin * cosθ)
# end def dt_over_dCosθ

class CenterOfMassFrame:
    def __init__(self, s, ma, mb, m1, m2):
        self.s = s
        self.ma = ma
        self.mb = mb
        self.m1 = m1
        self.m2 = m2

        self.initial_three_momentum_squared = center_of_mass_momentum_squared(self.s, self.ma, self.mb)
        self.initial_three_momentum_magnitude = center_of_mass_momentum(self.s, self.ma, self.mb)

        self.final_three_momentum_squared = center_of_mass_momentum_squared(self.s, self.m1, self.m2)
        self.final_three_momentum_magnitude = center_of_mass_momentum(self.s, self.m1, self.m2)

        self.Ea_CM_squared = self.ma**2 + self.initial_three_momentum_squared
        self.Ea_CM = sqrt(self.Ea_CM_squared)

        self.Eb_CM_squared = self.mb**2 + self.initial_three_momentum_squared
        self.Eb_CM = sqrt(self.Eb_CM_squared)

        self.E1_CM_squared = self.m1**2 + self.final_three_momentum_squared
        self.E1_CM = sqrt(self.E1_CM_squared)

        self.E2_CM_squared = self.m2**2 + self.final_three_momentum_squared
        self.E2_CM = sqrt(self.E2_CM_squared)
    # end def __init__

    def t(self, cosθ):
        return self.ma**2 + self.m1**2 - 2 * (self.Ea_CM * self.E1_CM - self.initial_three_momentum_magnitude * self.final_three_momentum_magnitude * cosθ)
    
    def dt_over_dcosθ(self, cosθ):
        g = grad(self.t)
        return g(cosθ)
    # end def dt_over_dcosθ
# end class CenterOfMassFrame

# Laboratory frame means pb = (mb, 0, 0, 0)^T.
class TransformCM2Lab:
    def __init__(self, s, ma, mb, m1, m2):
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.ma = ma
        self.mb = mb
    # end def __init__

    def Eb_CM_squared(self):
        pb_CM_squared = center_of_mass_momentum_squared(self.s, self.ma, self.mb)
        return self.mb**2 + pb_CM_squared
    # end def Eb_CM_squared
    
    def γ_squared(self):
        return self.Eb_CM_squared() / self.mb**2
    # end def γ_squared

    def β_squared(self):
        return 1 - 1 / self.γ_squared()
    # end def β_squared
# end class TransformCM2Lab
    
# θ is the angle between the direction of the initial particle a and the direction of the final particle 1.
class CosθCM2CosθLab(TransformCM2Lab):
    def __init__(self, s, ma, mb, m1, m2):
        super().__init__(s, ma, mb, m1, m2)
    # end def __init__

    def cosθLab(self, cosθCM):
        p_fin_CM_squared = center_of_mass_momentum_squared(self.s, self.m1, self.m2)
        p_fin_CM = sqrt(p_fin_CM_squared)
        E1_CM_squared = self.m1**2 + p_fin_CM_squared
        E1_CM = sqrt(E1_CM_squared)
        β_squared = self.β_squared()
        γ_squared = self.γ_squared()
        β = sqrt(β_squared)
        γ = sqrt(γ_squared)
        # sinθCM = sqrt(1 - cosθCM**2)
        sinθCM_squared = 1 - cosθCM**2
        # tanθLab = p_fin_CM * sinθCM / (γ * (-β * E1_CM + p_fin_CM * cosθCM))
        tanθLab_squared = p_fin_CM_squared * sinθCM_squared / (
            γ_squared * (β_squared * E1_CM_squared + p_fin_CM_squared * cosθCM**2 - 2 * β * E1_CM * p_fin_CM * cosθCM)
        )

        return sqrt(1 / (1 + tanθLab_squared))
    # end def cosθLab

    def dCosθLab_over_dCosθCM(self, cosθCM):
        g = grad(self.cosθLab)
        return g(cosθCM)
    # end def dCosθLab_over_dCosθCM
# end class CosθCM2CosθLab
    
class CosθLab2CosθCM(TransformCM2Lab):
    def __init__(self, s, ma, mb, m1, m2):
        super().__init__(s, ma, mb, m1, m2)
    # end def __init__

    def __check_result(self, cosθCM, cosθLab):
        γ = sqrt(self.γ_squared())
        β = sqrt(self.β_squared())
        p_fin_CM_squared = center_of_mass_momentum_squared(self.s, self.m1, self.m2)
        p_fin_CM = sqrt(p_fin_CM_squared)
        E1_CM = sqrt(self.m1**2 + p_fin_CM_squared)

        rslt = γ * (-β * E1_CM + p_fin_CM * cosθCM) - p_fin_CM * sqrt(1 - cosθCM**2) * cosθLab / sqrt(1 - cosθLab**2)
        if isnan(rslt):
            return abs(cosθCM - cosθLab)
        # end if
        return abs(rslt)
    # end def __check_result

    def cosθCM(self, cosθLab, error=1e-10):
        cosθLab_squared = cosθLab**2

        p_fin_CM_squared = center_of_mass_momentum_squared(self.s, self.m1, self.m2)
        p_fin_CM = sqrt(p_fin_CM_squared)
        E1_CM_squared = self.m1**2 + p_fin_CM_squared
        E1_CM = sqrt(E1_CM_squared)
        β_squared = self.β_squared()
        γ_squared = self.γ_squared()
        β = sqrt(β_squared)
        γ = sqrt(γ_squared)

        # print("p_fin_CM = {}".format(p_fin_CM))
        # print("E1_CM = {}".format(E1_CM))
        # print("β = {}".format(β))
        # print("γ = {}".format(γ))

        Δ = (
            p_fin_CM_squared**2 * cosθLab_squared**2 +
            p_fin_CM_squared * cosθLab_squared * (1 - cosθLab_squared) *
                (p_fin_CM_squared - E1_CM_squared * β_squared) * γ_squared
        )
        numerator = E1_CM * p_fin_CM * (1 - cosθLab_squared) * β * γ_squared
        denominator = p_fin_CM_squared * (γ_squared + cosθLab_squared * (1 - γ_squared))

        result1 = (numerator + sqrt(Δ)) / denominator
        result2 = (numerator - sqrt(Δ)) / denominator

        error1 = self.__check_result(result1, cosθLab)
        error2 = self.__check_result(result2, cosθLab)

        message = "cosθLab = {}, result1 = {}, error1 = {}, result2 = {}, error2 = {}".format(cosθLab, result1, error1, result2, error2)
        if abs(error1) < abs(error2):
            if abs(error1) > error:
                warn(message)
            # end if
            # assert abs(error1) < error, message
            return result1
        else:
            if abs(error2) > error:
                warn(message)
            # end if
            # assert abs(error2) < error, message
            return result2
    # end def cosθCM

    def dCosθCM_over_dCosθLab(self, cosθLab):
        g = grad(self.cosθCM)
        return g(cosθLab)
    # end def dCosθCM_over_dCosθLab
