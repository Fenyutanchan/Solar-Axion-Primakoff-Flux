# Copyright (c) 2024 Quan-feng WU <wuquanfeng@ihep.ac.cn>
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
from numpy import cross, pi

class dσ_over_dCosθ:
    def __init__(self):
        self.α = 1/137.035999084
        self.gOverΛ = 1
        self.ZX = 1
        self.κ = 1
    # end def __init__
    
    def F(self, Δ):
        return self.ZX**2 * Δ**2 / (self.κ**2 + Δ**2)
    # end def F

    def set_α(self, α):
        self.α = α
    # end def set_α
    
    def set_gOverΛ(self, gOverΛ):
        self.gOverΛ = gOverΛ
    # end def set_gOverΛ
        
    def set_ZX(self, ZX):
        self.ZX = ZX
    # end def set_ZX
        
    def __call__(self, ω, cosθ):
        coeff = 2 * pi * self.ZX**2 * self.α * self.gOverΛ**2 / (32 * pi)
        numerator = 1 + cosθ
        denominator = 1 + 2 * (self.κ / (2 * ω))**2 - cosθ
        return coeff * numerator / denominator
    # end def __call__
# end class dσdCosθ
