import pytest
from numpy.testing import assert_almost_equal
from pandas import date_range, read_csv
from energy_pandas import EnergyDataFrame, EnergySeries
from energy_pandas.edf_utils import MultipleUnitsError
from energy_pandas.units import unit_registry


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
            save=True,
            show=True,
            axis_off=False,
            kind=kind,
            cmap="Reds",
            fig_width=4,
            fig_height=4,
            edgecolors="grey",
            linewidths=0.01,
        )

    def test_plot_2d(self, es):
        fig, ax = es.plot2d(
            axis_off=False,
            cmap="Reds",
            fig_height=2,
            fig_width=6,
            show=True,
            save=True,
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
        with pytest.raises(MultipleUnitsError):
            edf_from_e_series.to_units("degF")

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
            axis_off=False,
            cmap="Reds",
            fig_height=None,
            fig_width=8,
            show=True,
            save=True,
            extent="tight",
        )
