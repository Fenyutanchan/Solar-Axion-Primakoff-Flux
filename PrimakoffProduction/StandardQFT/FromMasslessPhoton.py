# Copyright (c) 2024 Quan-feng WU <wuquanfeng@ihep.ac.cn>
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

class dσ_over_dt:
    def __init__(self):
        self.α = 1/137.035999084
        self.ZX = 1
        self.gOverΛ = 1
        self.FXt = lambda t: 1
        self.ma = 1
        self.mX = 1000
    # end def __init__

    def set_α(self, α):
        self.α = α
    # end def set_α
        
    def set_ZX(self, ZX):
        self.ZX = ZX
    # end def set_ZX

    def set_gOverΛ(self, gOverΛ):
        self.gOverΛ = gOverΛ
    # end def set_gOverΛ
        
    def set_FXt(self, FXt):
        self.FXt = FXt
    # end def set_FXt
        
    def set_ma(self, ma):
        self.ma = ma
    # end def set_ma
        
    def set_mX(self, mX):
        self.mX = mX
    # end def set_mX
        
    def __call__(self, s, t):
        coeff = -self.α * self.ZX**2 * self.gOverΛ**2 * self.FXt(t)**2 / 8
        numerator = self.ma**4 * self.mX**2 - self.ma**2 * t * (s + self.mX**2) + t * ((s - self.mX**2)**2 + s * t)
        denominator = t**2 * (s - self.mX**2)**2

        return coeff * numerator / denominator
    # end def __call__
# end class dσdt
    
if __name__ == "__main__":
    from numpy import sqrt

    import sys
    sys.path.append("..")
    from Kinematics import center_of_mass_momentum

    MeV = 1
    keV = 1e-3 * MeV

    diff_Xsection = dσ_over_dt()

    Eγ = 10 * MeV
    ma = 1 * MeV
    mX = 931 * MeV
    diff_Xsection.set_ma(ma)
    diff_Xsection.set_mX(mX)

    pin = Eγ * sqrt(mX / (2 * Eγ + mX))
    EX = sqrt(pin**2 + mX**2)
    s = (pin + EX)**2
    print((Eγ + mX)**2 - Eγ**2 - s)
    print(center_of_mass_momentum(s, 0, mX) - pin)
    pfi = center_of_mass_momentum(s, ma, EX)

    t = ma**2 - 2 * (pin * sqrt(pfi**2 + ma**2) - pin * pfi * -1)

    print(diff_Xsection(s, t))
# end if
