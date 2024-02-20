from numpy import exp, log, sqrt
from numpy import pi as π

from .read_solar_model import solar_model
from .units import units_in_MeV, proton_mass, electron_mass, radius_of_Sun, distance_to_Sun

from scipy.integrate import quad

def q_sqr(
    photon_energy,
    sign=+1, # or -1
    axion_mass=0,
    unit:units_in_MeV=units_in_MeV(1),
    mass="default"
):
    M = proton_mass(unit) if mass == "default" else mass
    ma = axion_mass
    E = photon_energy

    term1 = (2 * E**2 * M - ma**2 * (E + M))
    term2 = E * sqrt(
        4 * E**2 * M**2 -
        4 * ma**2 * M * (E + M) +
        ma**4
    )

    return (term1 + sign * term2) / (2 * E + M)
# end q_sqr

def total_cross_section(
    photon_energy,
    ks_sqr,
    Q_X=1,
    axion_mass=0,
    unit:units_in_MeV=units_in_MeV(1),
    g_aγ="default",
    mass="default"
):
    if ks_sqr == 0:
        return 0 # to do

    g = 1e-10 / unit.GeV if g_aγ == "default" else g_aγ
    M = proton_mass(unit) if mass == "default" else mass
    ma = axion_mass
    E = photon_energy

    q_minus_sqr = q_sqr(E, -1, ma, unit, M)
    q_plus_sqr = q_sqr(E, +1, ma, unit, M)

    L1_ma4 = 0 if ma == 0 else log(q_plus_sqr / q_minus_sqr) * ma**4
    L2 = log((q_plus_sqr + ks_sqr) / (q_minus_sqr + ks_sqr))

    coeff = unit.alpha_EM * g**2 * Q_X**2 / (128 * E**2 * M**2)
    line1 = (q_plus_sqr - q_minus_sqr) * (
        4 * ma**2 - 8 * E * M - 4 * M**2 + q_plus_sqr + q_minus_sqr - 2 * ks_sqr
    )
    line2 = 2 * L2 * (
        8 * E**2 * M**2 + ma**4 - 4 * E * ma**2 * M - 4 * ma**2 * M**2
    )
    line3 = 2 * L2 * (
        2 * ks_sqr * (2 * E * M + M**2 - ma**2) +
        2 * ma**4 * M**2 / ks_sqr + ks_sqr**2
    )
    line4 = -4 * L1_ma4 * M**2 / ks_sqr

    return coeff * (line1 + line2 + line3 + line4)
# end total_cross_section

def transiation_rate(
    photon_energy, normalized_solar_radius,
    solar_data:solar_model=solar_model(),
    axion_mass=0,
    unit:units_in_MeV=units_in_MeV(1),
    g_aγ="default",
    Debye_effect:bool=True
):
    r = normalized_solar_radius
    np = ne = solar_data.Sun_proton_number_density_profile(r, unit)
    g = 1e-10 / unit.GeV if g_aγ == "default" else g_aγ

    if Debye_effect:
        T = solar_data.Sun_temperature_profile(r, unit)
        ks_sqr = 4 * π * unit.α_EM * (np + ne) / T
    else:
        ks_sqr = 0
    # end if
        
    X_dict = {
        "e": {"mass": electron_mass(unit), "charge": -1, "density": ne},
        "p": {"mass": proton_mass(unit), "charge": +1, "density": np}
    }

    result = 0
    for X in X_dict:
        M = X_dict[X]["mass"]
        Q = X_dict[X]["charge"]
        n = X_dict[X]["density"]

        if axion_mass > sqrt(2 * photon_energy * M + M**2) - M:
            σ = 0
        else:
            σ = total_cross_section(
                photon_energy, ks_sqr,
                Q_X=Q, axion_mass=axion_mass,
                unit=unit, g_aγ=g, mass=M
            )

        result += n * σ
    return result
# end transiation_rate

def fine_Bose_Einstein_factor(E, T):
    if E/T > 709: # too large numerical number for numpy.exp
        return 0
    return 1 / (exp(E/T) - 1)
# end fine_Bose_Einstein_factor

def solar_axion_flux(
    photon_energy,
    solar_data:solar_model=solar_model(),
    axion_mass=0,
    unit:units_in_MeV=units_in_MeV(1),
    g_aγ="default",
    Debye_effect:bool=True
):
    if photon_energy < axion_mass:
        return 0
    
    R = radius_of_Sun(unit)
    D = distance_to_Sun(unit)
    local_T = lambda rr: solar_data.Sun_temperature_profile(rr, unit)

    coeff = photon_energy**2 / π**2
    integrand = lambda rr: (
        rr**2 * fine_Bose_Einstein_factor(photon_energy, local_T(rr)) * 
        transiation_rate(photon_energy, rr, solar_data,
            axion_mass=axion_mass, unit=unit,
            g_aγ=g_aγ, Debye_effect=Debye_effect
        )
    )
    integral, int_error = quad(integrand, 0, 1)

    return coeff * integral * R**3 / D**2
# end solar_axion_flux
