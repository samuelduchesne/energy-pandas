"""units module."""

# Units
import pint

unit_registry = pint.UnitRegistry()
unit_registry.define("m3 = 1 * meter ** 3 = m³")
unit_registry.define(
    "degree_Celsius = kelvin; offset: 273.15 = °C = C = celsius = degC = degreeC"
)
unit_registry.define(
    "degree_Fahrenheit = 5 / 9 * kelvin; offset: 233.15 + 200 / 9 = "
    "°F = F = fahrenheit = degF = degreeF"
)
unit_registry.define("ach = dimensionless")  # Air Changes per Hour
unit_registry.define("acr = 1 / hour")  # Air change rate
