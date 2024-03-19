# Copyright (c) 2024 Quan-feng WU <wuquanfeng@ihep.ac.cn> and Xun-Jie Xu <xunjie.xu@ihep.ac.cn>
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import pandas as pd

from torch import nn
from torch import optim

class Flux(nn.Module):
    def __init__(self, A=6.02, α=2.481, β=1.205, γ=1.7):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(A))
        self.α = nn.Parameter(torch.tensor(α))
        self.β = nn.Parameter(torch.tensor(β))
        self.γ = nn.Parameter(torch.tensor(γ))
    # end __init__

    def __get_device(self):
        return self.A.device
    # end __get_device

    def __suppress_factor(self, photon_energy, axion_mass, change_device=True):
        E = torch.tensor(photon_energy, device=self.__get_device()) if change_device else photon_energy
        ma = torch.tensor(axion_mass, device=self.__get_device()) if change_device else axion_mass
        return torch.clamp((1 - (ma / E)**self.γ), min=0)
    # end __suppress_factor
    
    def forward(self, photon_energy, axion_mass, change_device=True):
        E = torch.tensor(photon_energy, device=self.__get_device()) if change_device else photon_energy
        ma = torch.tensor(axion_mass, device=self.__get_device()) if change_device else axion_mass
        S = self.__suppress_factor(E, ma, change_device=False)
        return self.A * E**self.α * torch.exp(-E/self.β) * S
    # end forward
# end class Flux
    
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# end if-elif-else
    
flux_data = pd.read_csv("solar_axion_flux_data.csv")
photon_energy_list = torch.tensor(flux_data["photon_energy"].values, dtype=torch.float32, device=device)
axion_mass_list = torch.tensor(flux_data["axion_mass"].values, dtype=torch.float32, device=device)
flux_list = torch.tensor(flux_data["flux"].values, dtype=torch.float32, device=device)

model = Flux().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fun = nn.MSELoss()

for epoch in range(3000):
    approx_flux_list = model(photon_energy_list, axion_mass_list, change_device=False)
    loss = loss_fun(approx_flux_list, flux_list)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("Optimal parameters:")
print("A:", model.A.item())
print("α:", model.α.item())
print("β:", model.β.item())
print("γ:", model.γ.item())
