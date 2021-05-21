import pytest
from numpy.testing import assert_almost_equal
from pandas import date_range

from energy_pandas import EnergyDataFrame, EnergySeries
from energy_pandas.units import dash_to_mul, underline_dash, unit_registry


@pytest.fixture()
def edf():
    frame = EnergyDataFrame({"Temp": range(0, 100)}, units="C", extrameta="this")
    yield frame

    # check that the name is passed to the slice
    assert frame["Temp"].name == "Temp"

    # check that the metadata is passed to the slice
    assert frame.extrameta == "this"


@pytest.fixture()
def edf_from_e_series():
    frame = EnergyDataFrame(
        {
            "Series 1 degC": EnergySeries.with_timeindex(range(0, 8760), units="degC"),
            "Series 2 degK": EnergySeries.with_timeindex(range(0, 8760), units="degK"),
        },
        extrameta="this",
    )
    yield frame

    # check that the name is passed to the slice
    assert frame["Series 1 degC"].name == "Series 1 degC"

    # check that the metadata is passed to the slice
    assert frame.extrameta == "this"


@pytest.fixture()
def es():
    datetimeindex = date_range(
        freq="H",
        start="{}-01-01".format("2018"),
        periods=100,
    )
    series = EnergySeries(
        range(0, 100), index=datetimeindex, name="Temp", units="C", extrameta="this"
    )
    yield series

    # check that the name is passed to the slice
    assert series.name == "Temp"

    # check that the metadata is passed to the slice
    assert series.extrameta == "this"


class TestEnergySeries:
    def test_units(self, es):
        """tests unit conversion"""
        assert es.to_units(to_units="degF").units == unit_registry.degF
        assert type(es.to_units(to_units="degF")) == EnergySeries

        # inplace
        es.to_units(to_units="degF", inplace=True)
        assert es.units == unit_registry.degF
        assert type(es) == EnergySeries

        # set attribute
        es.units = "degC"
        assert es.units == unit_registry.degC

    def test_meta_ops(self, es):
        """Operations keep units."""
        assert (es * 2).units == es.units
        assert (es - es).units == es.units

    def test_units_value(self, es):
        """test unit conversion."""
        assert (es.to_units(to_units="kelvin") == es + 273.15).all()

        # inplace
        original = es.copy()
        es.to_units(to_units="kelvin", inplace=True)
        assert es.equals(original + 273.15)

    def test_units_SI_system(self, es):
        """test unit conversion."""
        es.units = "W"
        print(es.to_si().units)

    def test_units_IP_system(self, es):
        """test unit conversion."""
        es.units = "W"
        print(es.to_ip().units)

    def test_units_ops(self, edf_from_e_series):
        a = edf_from_e_series["Series 1 degC"]
        b = edf_from_e_series["Series 2 degK"]

        c = a * b
        print(c.units)

    def test_normalize(self, es):
        """Tests normalization"""
        assert es.normalize().sum().sum() == 50
        assert type(es.normalize()) == EnergySeries
        es.normalize(inplace=True)

        # inplace
        assert es.sum().sum() == 50
        assert type(es) == EnergySeries

    def test_to_frame(self, es):
        """Check that a slice returns an EnergySeries."""
        assert type(es.to_frame(name="Temp")) == EnergyDataFrame
        assert es.to_frame()["Temp"].equals(es)

    def test_to_frame_units(self, es):
        """Check that a slice returns keeps the units."""
        assert es.to_frame()["Temp"].units == unit_registry.degC

    def test_repr(self, es):
        # check that a slice returns an EnergySeries
        print(es.__repr__())

    def test_monthly(self, es):
        assert es.monthly.extrameta == "this"
        print(es.monthly)

    def test_expanddim(self, es):
        """Tests when result has one higher dimension as the original"""
        # to_frame should return an EnergyDataFrame
        assert type(es.to_frame()) == EnergyDataFrame

        # Units of expandeddim frame should be same as EnergySeries
        assert es.to_frame().units == {"Temp": es.units}

        # Meta of expandeddim frame should be same as EnergySeries
        assert es.to_frame().extrameta == es.extrameta

    @pytest.mark.parametrize("kind", ["polygon", "surface", "contour"])
    def test_plot_3d(self, es, kind):
        fig, ax = es.plot3d(
            kind=kind,
            axis_off=False,
            cmap="Reds",
            show=True,
            save=True,
            edgecolors="grey",
            linewidths=0.01,
        )

    def test_plot_2d(self, es):
        fig, ax = es.plot2d(
            figsize=(8, 1),
            axis_off=True,
            cmap="Reds",
            show=True,
            save=False,
            colorbar=True,
            filename=es.name + "_heatmap",
        )

    def test_discretize(self, es):
        res = es.discretize_tsam(noTypicalPeriods=1)
        assert_almost_equal(res.sum(), 4235.070422535211, decimal=3)
        es.discretize_tsam(noTypicalPeriods=1, inplace=True)
        assert_almost_equal(res.sum(), 4235.070422535211, decimal=3)
        # check that the type is maintained
        assert type(es) == EnergySeries


class TestEnergyDataFrame:
    def test_units(self, edf):
        """test unit conversion."""
        assert edf.to_units(to_units="degF")["Temp"].units == unit_registry.degF
        assert type(edf.to_units(to_units="degF")) == EnergyDataFrame

        # inplace
        edf.to_units(to_units="degF", inplace=True)
        assert edf["Temp"].units == unit_registry.degF
        assert type(edf) == EnergyDataFrame

    def test_mixed_units(self, edf_from_e_series):
        """Check that units are kept on slices."""
        assert edf_from_e_series["Series 1 degC"].extrameta == "this"
        assert edf_from_e_series["Series 1 degC"].units == unit_registry.degC
        assert edf_from_e_series[["Series 1 degC"]].units == {
            "Series 1 degC": unit_registry.degC
        }

    def test_mixed_units_ops(self, edf_from_e_series):
        col_1 = edf_from_e_series.iloc[:, 0]
        col_2 = edf_from_e_series.iloc[:, 1]
        # todo: deal with mixed units magnitude
        assert (col_1 * col_2).units == "degree_Celsius"

    def test_mixed_units_convert(self, edf_from_e_series):
        assert edf_from_e_series.to_units("degR").units == {
            "Series 1 degC": unit_registry.degR,
            "Series 2 degK": unit_registry.degR,
        }

    def test_units_value(self, edf):
        """test unit conversion."""
        assert edf.to_units(to_units="kelvin").equals(edf + 273.15)

        # inplace
        original = edf.copy()
        edf.to_units(to_units="kelvin", inplace=True)
        assert edf.equals(original + 273.15)

    def test_normalize(self, edf):
        """Test normalization."""
        assert edf.normalize().sum().sum() == 50
        assert type(edf.normalize()) == EnergyDataFrame
        edf.normalize(inplace=True)

        # inplace
        assert edf.sum().sum() == 50
        assert type(edf) == EnergyDataFrame

    def test_slice(self, edf):
        # check that a slice returns an EnergySeries
        assert type(edf[["Temp"]]) == EnergyDataFrame
        assert type(edf["Temp"]) == EnergySeries

        # check that the name is passed to the slice
        with pytest.raises(AttributeError):
            # only EnergySeries have name
            assert edf[["Temp"]].name is None
        assert edf["Temp"].name == "Temp"

        # check that the metadata is passed to the slice
        assert edf[["Temp"]].extrameta == "this"
        assert edf["Temp"].extrameta == "this"

        # check that the slice keeps the units
        assert edf.units == {"Temp": edf["Temp"].units}

    def test_numeric_operations(self, edf):
        assert edf.mean(axis=1).units == "degree_Celsius"
        assert edf.sum(axis=1).units == "degree_Celsius"

    def test_repr(self, edf):
        # check that a slice returns an EnergySeries
        print(edf.__repr__())

    def test_discretize(self, edf_from_e_series):
        edf_from_e_series = edf_from_e_series.discretize_tsam(noTypicalPeriods=1)
        assert hasattr(edf_from_e_series, "agg")
        edf_from_e_series.discretize_tsam(noTypicalPeriods=1, inplace=True)
        assert hasattr(edf_from_e_series, "agg")
        # check that the type is maintained
        assert type(edf_from_e_series) == EnergyDataFrame

    def test_plot_2d(self):
        """Test plot2d with resolution higher than hours."""
        import numpy as np

        es = EnergySeries(
            np.random.randint(12, 36, size=(365 * 24 * 4,)),
            index=date_range("2018-01-01", periods=365 * 24 * 4, freq="15T"),
            units="degC",
        )
        fig, ax = es.plot2d(
            axis_off=False, cmap="Reds", show=True, save=True, extent="tight"
        )


class TestUnits:

    # (Quantity, unit, abbreviation)
    ENERGYPLUS_UNITS = [
        ("angular degrees", "degree", "deg"),
        ("Length", "meter", "m"),
        ("Area", "square meter", "m2"),
        ("Volume", "cubic meter", "m3"),
        ("Time", "seconds", "s"),
        ("Frequency", "Hertz", "Hz"),
        ("Temperature", "Celsius", "C"),
        ("absolute temperature", "Kelvin", "K"),
        ("temperature difference", "Kelvin", "deltaC"),
        ("speed", "meters per second", "m/s"),
        ("energy (or work)", "Joules", "J"),
        ("power", "Watts", "W"),
        ("mass", "kilograms", "kg"),
        ("force", "Newton", "N"),
        ("mass flow", "kilograms per second", "kg/s"),
        ("volume flow", "cubic meters per second", "m3/s"),
        ("pressure", "Pascals", "Pa"),
        ("pressure difference", "Pascals", "Pa"),
        ("specific enthalpy", "Joules per kilogram", "J/kg"),
        ("density", "kilograms per cubic meter", "kg/m3"),
        ("heat flux", "watts per square meter", "W/m2"),
        ("specific heat", "——-", "J/kg-K"),
        ("conductivity", "——-", "W/m-K"),
        ("diffusivity", "——-", "m2/s"),
        ("heat transfer coefficient", "——-", "W/m2-K"),
        ("R-value", "——-", "m2-K/W"),
        ("heating or cooling capacity", "Watts", "W"),
        ("electric potential", "volts", "V"),
        ("electric current", "Amperes", "A"),
        ("illuminace", "lux", "lx"),
        ("luminous flux", "lumen", "lm"),
        ("luminous intensity", "candelas", "cd"),
        ("luminance", "candelas per square meter", "cd/m2"),
        ("vapor diffusivity", "meters squared per second", "m2/s"),
        ("viscosity", "——-", "kg/m-s"),
        ("dynamic Viscosity", "——-", "N-s/m2"),
        ("thermal gradient coeff for moisture capacity", "——-", "kg/kg-K"),
        ("isothermal moisture capacity", "——-", "m3/kg"),
    ]
    ENERGYPLUS_UNITS_NAMES = [unit for _, _, unit in ENERGYPLUS_UNITS]

    def test_underline_dash(self):
        assert underline_dash("W/m-K") == "W/(m-K)"

    def test_dash_to_mul(self):
        assert dash_to_mul("W/m-K") == "W/m*K"

    @pytest.mark.parametrize("units", ENERGYPLUS_UNITS, ids=ENERGYPLUS_UNITS_NAMES)
    def test_unit_registry(self, units):
        quantity, unit, abbreviation = units
        parsed_units = unit_registry.parse_units(abbreviation)
        print(f"{parsed_units:~P}")
        assert parsed_units

    @pytest.mark.parametrize("units", ENERGYPLUS_UNITS, ids=ENERGYPLUS_UNITS_NAMES)
    def test_to_ip(self, units):
        """Test conversion to ip_units"""
        quantity, unit, abbreviation = units
        es = EnergySeries(range(0, 10), units=abbreviation)
        print(f"{es.to_ip().units}")

    @pytest.mark.parametrize("units", ENERGYPLUS_UNITS, ids=ENERGYPLUS_UNITS_NAMES)
    def test_to_si(self, units):
        quantity, unit, abbreviation = units
        es = EnergySeries(range(0, 10), units=abbreviation)
        print(f"{es.to_si().units}")
