"""units module."""

# Units
from tokenize import TokenInfo

import pint
from pint.compat import tokenizer

import pkg_resources
energyplus_registry = pkg_resources.resource_filename(__name__, "energyplus_en.txt")


def underline_dash(input_string):
    """Enclose denominator with parentheses."""
    parts = []
    gen = tokenizer(input_string)
    for part in gen:
        parts.append(part)
        if part.string == "/":
            # put rest of string in parentheses
            parts.append(
                TokenInfo(type=53, string="(", start=(1, 0), end=(1, 1), line="(")
            )
            parts.extend(list(gen))
            parts.append(
                TokenInfo(type=53, string=")", start=(1, 0), end=(1, 1), line="(")
            )
    return "".join((s.string for s in parts))


def dash_to_mul(input_string):
    """Replace '-' with '*' in input_string."""
    return input_string.replace("-", "*")


class Token:
    def __init__(self, input_string):
        self.string = input_string


unit_registry = pint.UnitRegistry(preprocessors=[underline_dash, dash_to_mul])
unit_registry.load_definitions(energyplus_registry)
pint.set_application_registry(unit_registry)


IP_DEFAULT_CONVERSION = {
    unit_registry.parse_units("m3/s"): unit_registry.parse_units("ft3/min"),
    unit_registry.parse_units("W/K"): unit_registry.parse_units("Btu/h-F"),
    unit_registry.parse_units("kW"): unit_registry.parse_units("kBtuh/h"),
    unit_registry.parse_units("m2"): unit_registry.parse_units("ft2"),
    unit_registry.parse_units("m3"): unit_registry.parse_units("ft3"),
    unit_registry.parse_units("(kg/s)/W"): unit_registry.parse_units(
        "(lbm/sec)/(Btu/hr)"
    ),
    unit_registry.parse_units("1/K"): unit_registry.parse_units("1/F"),
    unit_registry.parse_units("1/m"): unit_registry.parse_units("1/ft"),
    unit_registry.parse_units("A/K"): unit_registry.parse_units("A/F"),
    unit_registry.parse_units("C"): unit_registry.parse_units("F"),
    unit_registry.parse_units("cm"): unit_registry.parse_units("in"),
    unit_registry.parse_units("cm2"): unit_registry.parse_units("inch2"),
    unit_registry.parse_units("deltaC"): unit_registry.parse_units("deltaF"),
    # unit_registry.parse_units("deltaJ/kg"): unit_registry.parse_units("deltaBtu/lb"),
    unit_registry.parse_units("g/GJ"): unit_registry.parse_units("lb/MWh"),
    unit_registry.parse_units("g/kg"): unit_registry.parse_units("grains/lb"),
    unit_registry.parse_units("g/MJ"): unit_registry.parse_units("lb/MWh"),
    unit_registry.parse_units("g/mol"): unit_registry.parse_units("lb/mol"),
    unit_registry.parse_units("g/m-s"): unit_registry.parse_units("lb/ft-s"),
    unit_registry.parse_units("g/m-s-K"): unit_registry.parse_units("lb/ft-s-F"),
    unit_registry.parse_units("GJ"): unit_registry("ton-hrs"),
    unit_registry.parse_units("J"): unit_registry.parse_units("Wh"),
    unit_registry.parse_units("J/K"): unit_registry.parse_units("Btu/F"),
    unit_registry.parse_units("J/kg"): unit_registry.parse_units("Btu/lb"),
    unit_registry.parse_units("J/kg-K"): unit_registry.parse_units("Btu/lb-F"),
    unit_registry.parse_units("J/kg-K2"): unit_registry.parse_units("Btu/lb-F2"),
    unit_registry.parse_units("J/kg-K3"): unit_registry.parse_units("Btu/lb-F3"),
    unit_registry.parse_units("J/m2-K"): unit_registry.parse_units("Btu/ft2-F"),
    unit_registry.parse_units("J/m3"): unit_registry.parse_units("Btu/ft3"),
    unit_registry.parse_units("J/m3-K"): unit_registry.parse_units("Btu/ft3-F"),
    unit_registry.parse_units("K"): unit_registry.parse_units("R"),
    unit_registry.parse_units("K/m"): unit_registry.parse_units("F/ft"),
    unit_registry.parse_units("kg"): unit_registry.parse_units("lb"),
    unit_registry.parse_units("kg/J"): unit_registry.parse_units("lb/Btu"),
    unit_registry.parse_units("kg/kg-K"): unit_registry.parse_units("lb/lb-F"),
    unit_registry.parse_units("kg/m"): unit_registry.parse_units("lb/ft"),
    unit_registry.parse_units("kg/m2"): unit_registry.parse_units("lb/ft2"),
    unit_registry.parse_units("kg/m3"): unit_registry.parse_units("lb/ft3"),
    unit_registry.parse_units("kg/m-s"): unit_registry.parse_units("lb/ft-s"),
    unit_registry.parse_units("kg/m-s-K"): unit_registry.parse_units("lb/ft-s-F"),
    unit_registry.parse_units("kg/m-s-K2"): unit_registry.parse_units("lb/ft-s-F2"),
    unit_registry.parse_units("kg/Pa-s-m2"): unit_registry.parse_units("lb/psi-s-ft2"),
    unit_registry.parse_units("kg/s"): unit_registry.parse_units("lb/s"),
    unit_registry.parse_units("kg/s2"): unit_registry.parse_units("lb/s2"),
    unit_registry.parse_units("kg/s-m"): unit_registry.parse_units("lb/s-ft"),
    unit_registry.parse_units("kJ/kg"): unit_registry.parse_units("Btu/lb"),
    unit_registry.parse_units("kPa"): unit_registry.parse_units("psi"),
    unit_registry.parse_units("L/day"): unit_registry.parse_units("pint/day"),
    unit_registry.parse_units("L/GJ"): unit_registry.parse_units("gal/kWh"),
    unit_registry.parse_units("L/kWh"): unit_registry.parse_units("pint/kWh"),
    unit_registry.parse_units("L/MJ"): unit_registry.parse_units("gal/kWh"),
    unit_registry.parse_units("lux"): unit_registry.parse_units("footcandles"),
    unit_registry.parse_units("m"): unit_registry.parse_units("ft"),
    unit_registry.parse_units("m/hr"): unit_registry.parse_units("ft/hr"),
    unit_registry.parse_units("m/s"): unit_registry.parse_units("ft/min"),
    unit_registry.parse_units("m/yr"): unit_registry.parse_units("inch/yr"),
    unit_registry.parse_units("m2"): unit_registry.parse_units("ft2"),
    unit_registry.parse_units("m2/m"): unit_registry.parse_units("ft2/ft"),
    unit_registry.parse_units("m2/person"): unit_registry.parse_units("ft2/person"),
    unit_registry.parse_units("m2/s"): unit_registry.parse_units("ft2/s"),
    unit_registry.parse_units("m2-K/W"): unit_registry.parse_units("ft2-F-hr/Btu"),
    unit_registry.parse_units("m3"): unit_registry.parse_units("ft3"),
    unit_registry.parse_units("m3/GJ"): unit_registry.parse_units("ft3/MWh"),
    unit_registry.parse_units("m3/hr"): unit_registry.parse_units("ft3/hr"),
    unit_registry.parse_units("m3/hr-m2"): unit_registry.parse_units("ft3/hr-ft2"),
    unit_registry.parse_units("m3/hr-person"): unit_registry.parse_units(
        "ft3/hr-person"
    ),
    unit_registry.parse_units("m3/kg"): unit_registry.parse_units("ft3/lb"),
    unit_registry.parse_units("m3/m2"): unit_registry.parse_units("ft3/ft2"),
    unit_registry.parse_units("m3/MJ"): unit_registry.parse_units("ft3/kWh"),
    unit_registry.parse_units("m3/person"): unit_registry.parse_units("ft3/person"),
    unit_registry.parse_units("m3/s"): unit_registry.parse_units("ft3/min"),
    unit_registry.parse_units("m3/s-m"): unit_registry.parse_units("ft3/min-ft"),
    unit_registry.parse_units("m3/s-m2"): unit_registry.parse_units("ft3/min-ft2"),
    unit_registry.parse_units("m3/s-person"): unit_registry.parse_units(
        "ft3/min-person"
    ),
    unit_registry.parse_units("m3/s-W"): unit_registry.parse_units("(ft3/min)/(Btu/h)"),
    unit_registry.parse_units("N-m"): unit_registry.parse_units("lbf-in"),
    unit_registry.parse_units("N-s/m2"): unit_registry.parse_units("lbf-s/ft2"),
    unit_registry.parse_units("Pa"): unit_registry.parse_units("psi"),
    unit_registry.parse_units("percent/K"): unit_registry.parse_units("percent/F"),
    unit_registry.parse_units("person/m2"): unit_registry.parse_units("person/ft2"),
    unit_registry.parse_units("s/m"): unit_registry.parse_units("s/ft"),
    unit_registry.parse_units("V/K"): unit_registry.parse_units("V/F"),
    unit_registry.parse_units("W"): unit_registry.parse_units("Btu/h"),
    unit_registry.parse_units("W/(m3/s)"): unit_registry.parse_units("W/(ft3/min)"),
    unit_registry.parse_units("W/K"): unit_registry.parse_units("Btu/h-F"),
    unit_registry.parse_units("W/m"): unit_registry.parse_units("Btu/h-ft"),
    unit_registry.parse_units("W/m2"): unit_registry.parse_units("Btu/h-ft2"),
    unit_registry.parse_units("W/m2"): unit_registry.parse_units("W/ft2"),
    unit_registry.parse_units("W/m2-K"): unit_registry.parse_units("Btu/h-ft2-F"),
    unit_registry.parse_units("W/m2-K2"): unit_registry.parse_units("Btu/h-ft2-F2"),
    unit_registry.parse_units("W/m-K"): unit_registry.parse_units("Btu-in/h-ft2-F"),
    unit_registry.parse_units("W/m-K2"): unit_registry.parse_units("Btu/h-F2-ft"),
    unit_registry.parse_units("W/m-K3"): unit_registry.parse_units("Btu/h-F3-ft"),
    unit_registry.parse_units("W/person"): unit_registry.parse_units("Btu/h-person"),
}

SI_DEFAULT_CONVERSION = {v: k for k, v in IP_DEFAULT_CONVERSION.items()}
