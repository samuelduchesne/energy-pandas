"""EnergySeries module."""

import copy
import logging
import warnings
from datetime import timedelta

from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from numpy import meshgrid, ndarray
from pandas import (
    DataFrame,
    DatetimeIndex,
    MultiIndex,
    Series,
    date_range,
    infer_freq,
    pivot_table,
    to_datetime,
    to_timedelta,
)
from pandas._libs.tslibs.offsets import to_offset
from pandas.core.generic import NDFrame
from pint import Quantity, Unit
from sklearn import preprocessing
from tsam import timeseriesaggregation as tsam

from .plotting import (
    _plot_poly_collection,
    _plot_surface,
    _setup_subplots,
    save_and_show,
)
from .units import IP_DEFAULT_CONVERSION, SI_DEFAULT_CONVERSION, unit_registry

log = logging.getLogger(__name__)


class EnergySeries(Series):
    """One-dimensional ndarray with axis labels (including time series) and units.

    Labels need not be unique but must be a hashable type. The object supports both
    integer- and label-based indexing and provides a host of methods for performing
    operations involving the index. Statistical methods from ndarray have been
    overridden to automatically exclude missing data (currently represented as NaN).

    Operations between Series (+, -, /, , *) align values based on their associated
    index values– they need not be the same length. The result index will be the
    sorted union of the two indexes.

    Units can be assigned to the data and unit conversion is possible with
    :meth:`to_units`. Units are not used in operations between series.

    """

    _metadata = [
        "bin_edges_",
        "bin_scaling_factors_",
        "base_year",
        "frequency",
        "units",
        "name",
    ]

    @property
    def _constructor(self):
        return EnergySeries

    @property
    def _constructor_expanddim(self):
        def f(*args, **kwargs):
            # adapted from https://github.com/pandas-dev/pandas/issues/19850#issuecomment-367934440

            finalize__ = EnergyDataFrame(*args, **kwargs).__finalize__(
                self, method="inherit"
            )
            finalize__.units = {self.name: self.units}
            return finalize__

        f._get_axis_number = super(EnergySeries, self)._get_axis_number

        return f

    def __init__(
        self,
        data=None,
        index=None,
        dtype=None,
        name=None,
        copy=False,
        units=None,
        **kwargs,
    ):
        """Initiate EnergySeries.

        Args:
            data (array-like, Iterable, dict, or scalar value): Contains data stored
                in Series.
            index (array-like or Index (1d)):  Values must be hashable and have the
                same length as `data`. Non-unique index values are allowed. Will
                default to RangeIndex (0, 1, 2, ..., n) if not provided. If both a
                dict and index sequence are used, the index will override the keys
                found in the dict.
            dtype (str, numpy.dtype, or ExtensionDtype, optional): Data type for the
                output Series. If not specified, this will be inferred from `data`.
                See the :ref:`user guide <basics.dtypes>` for more usages.
            name (str, optional): The name to give to the Series.
            copy (bool): Copy input data. Defaults to False
            units (str or Unit): The series units. Parsed as Pint units.
            **kwargs: Other keywords added as metadata.
        """
        super(EnergySeries, self).__init__(
            data=data, index=index, dtype=dtype, name=name, copy=copy
        )
        self.bin_edges_ = None
        self.bin_scaling_factors_ = None

        self.__set_units(units)

        for k, v in kwargs.items():
            EnergySeries._metadata.append(k)
            setattr(EnergySeries, k, v)

    def __get_units(self):
        """Get units for self."""
        return self._units

    def __set_units(self, value):
        """Set units on self."""
        if isinstance(value, str):
            self._units = unit_registry.parse_expression(value).units
        elif isinstance(value, (Unit, Quantity)):
            self._units = value
        elif value is None:
            self._units = unit_registry.parse_expression(value).units
        else:
            raise TypeError(f"Unit of type {type(value)}")

    units = property(__get_units, __set_units)

    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata from other to self."""
        if isinstance(other, NDFrame):
            for name in other.attrs:
                self.attrs[name] = other.attrs[name]
            # For subclasses using _metadata. Set known attributes and update list.
            for name in other._metadata:
                if name == "units":
                    if isinstance(other, EnergyDataFrame):
                        if len(set(getattr(other, "units").values())) == 1:
                            # if all the same units
                            setattr(self, name, *set(getattr(other, "units").values()))
                        else:
                            setattr(self, name, getattr(other, "units").get(self.name))
                    else:
                        setattr(self, name, getattr(other, "units"))
                elif name == "name":
                    pass
                else:
                    try:
                        object.__setattr__(self, name, getattr(other, name))
                    except AttributeError:
                        pass
                    if name not in self._metadata:
                        self._metadata.append(name)
        return self

    def __repr__(self):
        """Add units to repr."""
        result = super(EnergySeries, self).__repr__()
        return result + f", units:{self.units:~P}"

    @classmethod
    def with_timeindex(
        cls,
        data,
        base_year=2018,
        frequency="H",
        index=None,
        dtype=None,
        name=None,
        copy=False,
        units=None,
        **kwargs,
    ):
        """Initialize object with default DateTimeIndex.

        Uses :attr:`base_year` to build a DateTimeIndex of :attr:`frequency` and
        length `len(data)`.

        Args:
            data (array-like, Iterable, dict, or scalar value): Contains data stored
                in Series.
            base_year (int): The year to use in the DateTimeIndex.
            frequency (str or DateOffset): default 'H'
                Frequency strings can have multiples, e.g. '5H'. See
                :ref:`here <timeseries.offset_aliases>` for a list of
                frequency aliases.
            index (array-like or Index (1d)):  Values must be hashable and have the
                same length as `data`. Non-unique index values are allowed. Will
                default to RangeIndex (0, 1, 2, ..., n) if not provided. If both a
                dict and index sequence are used, the index will override the keys
                found in the dict.
            dtype (str, numpy.dtype, or ExtensionDtype, optional): Data type for the
                output Series. If not specified, this will be inferred from `data`.
                See the :ref:`user guide <basics.dtypes>` for more usages.
            name (str, optional): The name to give to the Series.
            copy (bool): Copy input data. Defaults to False
            units (str or Unit): The series units. Parsed as Pint units.
            **kwargs: Other keywords added as metadata.

        Returns:
            EnergySeries: An EnergySeries with a DateTimeIndex.
        """
        es = cls(
            data=data,
            index=index,
            dtype=dtype,
            name=name,
            copy=copy,
            units=units,
            **kwargs,
        )
        start_date = str(base_year) + "0101"
        newindex = date_range(start=start_date, freq=frequency, periods=len(es))
        es.index = newindex
        return es

    @classmethod
    def from_reportdata(
        cls,
        df,
        name=None,
        base_year=2018,
        units=None,
        normalize=False,
        sort_values=False,
        ascending=False,
        to_units=None,
        agg_func="sum",
    ):
        """Create a.

        Args:
            df (DataFrame):
            name:
            base_year:
            units:
            normalize (bool): Normalize between 0 and 1.
            sort_values:
            ascending:
            to_units (str): Convert original values to this unit. Dimensionality
                check performed by `pint`.
            agg_func (callable): The aggregation function to use in the case
                that multiple values have the same index value. If a function,
                must either work when passed a DataFrame or when passed to
                DataFrame.apply. For a DataFrame, can pass a dict, if the keys
                are DataFrame column names.

                Accepted Combinations are:
                    - string function name
                    - function
                    - list of functions
                    - dict of column names -> functions (or list of functions)
        """
        index = to_datetime(
            {
                "year": base_year,
                "month": df.Month,
                "day": df.Day,
                "hour": df.Hour,
                "minute": df.Minute,
            }
        )
        # Adjust timeindex by timedelta
        index -= df.Interval.apply(lambda x: timedelta(minutes=x))
        index = DatetimeIndex(index)
        # get data
        data = df.Value
        data.index = index
        units = [units] if units else set(df.Units)
        if len(units) > 1:
            raise ValueError("The DataFrame contains mixed units: {}".format(units))
        else:
            units = next(iter(units), None)
        # group data by index value (level=0) using the agg_func
        if agg_func:
            grouped_Data = data.groupby(level=0).agg(agg_func)
        else:
            df["DateTimeIndex"] = index
            grouped_Data = df.set_index(["DateTimeIndex", "Name"]).Value
        # Since we create the index, don't need to use .with_timeindex() constructor
        energy_series = cls(
            grouped_Data.values,
            name=name,
            units=units,
            index=grouped_Data.index,
            base_year=base_year,
        )
        if normalize:
            energy_series.normalize(inplace=True)
        if sort_values:
            energy_series.sort_values(ascending=ascending, inplace=True)
        if to_units and not normalize:
            energy_series.to_units(to_units, inplace=True)
        return energy_series

    def to_units(self, to_units=None, inplace=False):
        """returns the multiplier to convert units

        Args:
            to_units (str, pint.Unit):
            inplace (bool): If True, conversion is applied inplace.
        """
        cdata = unit_registry.Quantity(self.values, self.units).to(to_units)
        if inplace:
            self[:] = cdata.m
            self.__set_units(cdata.units)
        else:
            # create new instance using constructor
            result = self._constructor(data=cdata.m, index=self.index, copy=False)
            # Copy metadata over
            result.__finalize__(self)
            result.__set_units(to_units)
            return result

    def normalize(self, inplace=False):
        """Returns a normalized EnergySeries

        Args:
            inplace:
        """
        x = self.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x.reshape(-1, 1)).ravel()
        if inplace:
            # replace whole data with array
            self[:] = x_scaled
            # change units to dimensionless
            self.units = unit_registry.dimensionless
        else:
            # create new instance using constructor
            result = self._constructor(data=x_scaled, index=self.index, copy=False)
            # Copy metadata over
            result.__finalize__(self)
            return result

    def ldc_source(self, SCOPH=4, SCOPC=4):
        """Returns the Load Duration Curve from the source side of theoretical
        Heat Pumps

        Args:
            SCOPH: Seasonal COP in Heating
            SCOPC: Seasonal COP in Cooling

        Returns:
            (EnergySeries) Load Duration Curve
        """

        result = self.ldc.apply(
            lambda x: x * (1 - 1 / SCOPH) if x > 0 else x * (1 + 1 / SCOPC)
        )
        return result

    def source_side(self, SCOPH=None, SCOPC=None):
        """Returns the Source Side EnergySeries given a Seasonal COP. Negative
        values are considered like Cooling Demand.

        Args:
            SCOPH: Seasonal COP in Heating
            SCOPC: Seasonal COP in Cooling

        Returns:
            (EnergySeries) Load Duration Curve
        """
        if SCOPC or SCOPH:
            result = self.apply(
                lambda x: x * (1 - 1 / SCOPH) if SCOPH else x * (1 + 1 / SCOPC)
            )
            return result
        else:
            raise ValueError("Please provide a SCOPH or a SCOPC")

    def discretize_tsam(self, inplace=False, **kwargs):
        """Clusters time series data to typical periods. See
        :class:`tsam.timeseriesaggregation.TimeSeriesAggregation` for more info.

        Returns:
            EnergySeries:
        """
        try:
            import tsam.timeseriesaggregation as tsam
        except ImportError:
            raise ImportError("tsam is required for discretize_tsam()")
        if not isinstance(self.index, DatetimeIndex):
            raise TypeError("To use tsam, index of series must be a DateTimeIndex")

        timeSeries = self.to_frame()
        agg = tsam.TimeSeriesAggregation(timeSeries, **kwargs)

        agg.createTypicalPeriods()
        result = agg.predictOriginalData()
        if inplace:
            self.loc[:] = result.values.ravel()
        else:
            # create new instance using constructor
            result = self._constructor(
                data=result.values.ravel(), index=self.index, copy=False
            )
            # Copy metadata over
            result.__finalize__(self)
            return result

    def plot3d(
        self,
        kind="polygon",
        axis_off=True,
        cmap=None,
        figsize=None,
        show=True,
        view_angle=-60,
        save=False,
        close=False,
        dpi=300,
        file_format="png",
        axes=None,
        vmin=None,
        vmax=None,
        filename=None,
        time_steps_per_period=24,
        **kwargs,
    ):
        """Plot a 3D plot of the data.

        Args:
            kind (str): One of "polygon", "surface" or "contour".
            axis_off (bool): If True, no axis is plotted.
            cmap (str or Colormap): Colormap to select colors from. If string,
                load colormap with that name from matplotlib.
            figsize (tuple): Size of a figure object. A tuple (width, height) in inches.
            show (bool): whether to display the figure or not.
            view_angle (float): The angle view angle of the 3D plot.
            save (bool): whether to save the figure to disk or not.
            close (bool): close the figure (only if show equals False) to prevent
                display.
            dpi (int): the resolution of the image file if saving (Dots per inch)
            file_format (string): the format of the file to save (e.g., 'jpg',
                'png', 'svg').
            axes (Axes): An axes of the current figure.
            vmin (float): The data value that defines ``0.0`` in the normalization.
                Defaults to the min value of the dataset.
            vmax (float): The data value that defines ``1.0`` in the normalization.
            filename (string): the name of the file to save.
            time_steps_per_period (int): The number of discrete timesteps which
                describe one period.
            **kwargs:

        Returns:
            (tuple): tuple containing:

                fig: is the Matplotlib Figure object
                ax: a single axis object.
        """
        if self.empty:
            warnings.warn(
                "The EnergySeries you are attempting to plot is empty. "
                "Nothing has been displayed.",
                UserWarning,
            )
            return axes

        import matplotlib.pyplot as plt

        # noinspection PyUnresolvedReferences
        from mpl_toolkits.mplot3d import Axes3D

        if isinstance(self.index, MultiIndex):
            groups = self.groupby(level=0)
            nax = len(groups)
        else:
            nax = 1
            groups = [("unnamed", self)]

        # Set up plot
        fig, axes = plt.subplots(
            nax,
            1,
            subplot_kw=dict(projection="3d"),
            figsize=figsize,
            dpi=dpi,
        )
        if not isinstance(axes, ndarray):
            axes = [axes]

        for ax, (name, profile) in zip(axes, groups):
            values = profile.values

            vmin = values.min() if vmin is None else vmin
            vmax = values.max() if vmax is None else vmax

            if kind == "polygon":
                import tsam.timeseriesaggregation as tsam

                z, _ = tsam.unstackToPeriods(
                    profile, timeStepsPerPeriod=time_steps_per_period
                )
                nrows, ncols = z.shape

                xs = z.columns
                zs = z.index.values

                verts = []
                for i in zs:
                    ys = z.iloc[int(i), :]
                    verts.append([(xs[0], 0.0), *zip(xs, ys), (xs[-1], 0.0)])

                _plot_poly_collection(
                    ax,
                    verts,
                    zs,
                    cmap=cmap,
                    edgecolors=kwargs.get("edgecolors", None),
                    facecolors=kwargs.get("facecolors", None),
                    linewidths=kwargs.get("linewidths", None),
                )
            elif kind == "surface":
                import tsam.timeseriesaggregation as tsam

                z, _ = tsam.unstackToPeriods(
                    profile, timeStepsPerPeriod=time_steps_per_period
                )
                nrows, ncols = z.shape
                x = z.columns
                y = z.index.values

                x, y = meshgrid(x, y)
                # Remove edgecolors from kwargs
                kwargs.pop("edgecolors", None)
                _plot_surface(ax, x, y, z.values, cmap=cmap, **kwargs)
            elif kind == "contour":
                import tsam.timeseriesaggregation as tsam

                z, _ = tsam.unstackToPeriods(
                    profile, timeStepsPerPeriod=time_steps_per_period
                )
                nrows, ncols = z.shape
                x = z.columns
                y = z.index.values

                x, y = meshgrid(x, y)
                ax.contour3D(x, y, z.values, 150, cmap=cmap, **kwargs)
            else:
                raise NameError('plot kind "{}" is not supported'.format(kind))

            if filename is None:
                filename = "unnamed"

            # set the extent of the figure
            ax.set_xlim3d(-1, ncols)
            ax.set_xlabel(kwargs.get("xlabel", "hour of day"))
            ax.set_ylim3d(-1, nrows)
            ax.set_ylabel(kwargs.get("ylabel", "day of year"))
            ax.set_zlim3d(vmin, vmax)
            z_label = "{} [{:~P}]".format(
                self.name if self.name is not None else "Z",
                self.units,
            )
            ax.set_zlabel(z_label)

            # configure axis appearance
            xaxis = ax.xaxis
            yaxis = ax.yaxis
            zaxis = ax.zaxis

            xaxis.get_major_formatter().set_useOffset(False)
            yaxis.get_major_formatter().set_useOffset(False)
            zaxis.get_major_formatter().set_useOffset(False)

            # if axis_off, turn off the axis display set the margins to zero and
            # point the ticks in so there's no space around the plot
            if axis_off:
                ax.axis("off")
                ax.margins(0)
                ax.tick_params(which="both", direction="in")
                xaxis.set_visible(False)
                yaxis.set_visible(False)
                zaxis.set_visible(False)
                fig.canvas.draw()
            if view_angle is not None:
                ax.view_init(30, view_angle)
                ax.set_proj_type(kwargs.get("proj_type", "persp"))
                fig.canvas.draw()
        fig, axes = save_and_show(
            fig=fig,
            ax=axes,
            save=save,
            show=show,
            close=close,
            filename=filename,
            file_format=file_format,
            dpi=dpi,
            axis_off=axis_off,
            extent=None,
        )
        return fig, axes

    @property
    def p_max(self):
        if isinstance(self.index, MultiIndex):
            return self.groupby(level=0).max()
        else:
            return self.max()

    @property
    def monthly(self):
        if isinstance(self.index, DatetimeIndex):
            data = self.resample("M").mean()
            return self._constructor(
                data, index=data.index, frequency="M", units=self.units
            )
        else:
            return None

    @property
    def capacity_factor(self):
        max = self.max()
        mean = self.mean()
        return mean / max

    @property
    def bin_edges(self):
        return self.bin_edges_

    @property
    def time_at_min(self):
        """Return the index value where the min occurs."""
        return self.idxmin()

    @property
    def bin_scaling_factors(self):
        return self.bin_scaling_factors_

    @property
    def duration_scaling_factor(self):
        return list(map(tuple, self.bin_scaling_factors.values))

    @property
    def ldc(self):
        newdata = self.sort_values(ascending=False).reset_index(drop=True)
        return newdata.__finalize__(self)

    @property
    def nseries(self):
        if self.data.ndim == 1:
            return 1
        else:
            return self.data.shape[1]

    def to_si(self, inplace=False):
        """Convert self to SI units.

        Args:
            inplace (bool): If True, conversion is applied inplace.
        """
        try:
            si_units = SI_DEFAULT_CONVERSION[self.units]
        except KeyError:
            return self
        self.to_units(si_units, inplace=inplace)
        return self

    def to_ip(self, inplace=False):
        """Convert self to IP units (inch-pound).

        Args:
            inplace (bool): If True, conversion is applied inplace.
        """
        try:
            ip_units = IP_DEFAULT_CONVERSION[self.units]
        except KeyError:
            return self
        self.to_units(ip_units, inplace=inplace)
        return self

    def plot2d(
        self,
        periodlength=None,
        vmin=None,
        vmax=None,
        vcenter=None,
        axis_off=False,
        cmap="RdBu",
        figsize=None,
        show=True,
        save=False,
        close=False,
        dpi=300,
        file_format="png",
        colorbar=True,
        ax=None,
        filename="untitled",
        extent="tight",
        xlabel=None,
        ylabel=None,
        ax_title=True,
        **kwargs,
    ):
        """Plot rectangular data as a color-encoded 2d-matrix.

        Args:
            periodlength (int): The period length to show on the yaxis. By default,
                the period length will be calculated to represent one day of data on
                the yaxis and days on the xaxis.
            vmin (float): The data value that defines ``0.0`` in the normalization.
                Defaults to the min value of the dataset.
            vmax (float): The data value that defines ``1.0`` in the normalization.
                Defaults to the the max value of the dataset.
            vcenter (float): The data value that defines ``0.5`` in the
                normalization.
            axis_off (bool): If True, no axis is plotted.
            cmap (str or Colormap): Colormap to select colors from. If string,
                load colormap with that name from matplotlib.
            figsize (tuple): Size of a figure object. A tuple (width, height) in inches.
            show (bool): whether to display the figure or not.
            save (bool): whether to save the figure to disk or not.
            close (bool): close the figure (only if show equals False) to prevent
                display.
            dpi (int): the resolution of the image file if saving (Dots per inch)
            file_format (string): the format of the file to save (e.g., 'jpg',
                'png', 'svg').
            colorbar (bool): If True, plot the colorbar.
            ax (Axes): An axes of the current figure.
            filename (string): the name of the file to save.
            extent (str or `.Bbox`): Bounding box in inches: only the given portion of
                the figure is saved.  If 'tight', try to figure out the tight bbox of
                the figure.
            ylabel (str): Set the label for the y-axis.
            xlabel (str): Set the label for the x-axis.
            ax_title (bool or str): Title to use for the ax. If True, the Series name
                is used.
            **kwargs: Options to pass to matplotlib imshow method.

        Returns:
            (tuple): tuple containing:

                fig: is the Matplotlib Figure object
                ax: a single axis object.
        """
        if not ax:
            n = 1
            fig, axes = plt.subplots(nrows=n, ncols=1, figsize=figsize, dpi=dpi)
        else:
            fig = ax.get_figure()
            if figsize is not None:
                fig.set_size_inches(figsize)
            axes = ax

        freq = infer_freq(self.index[0:3])  # infer on first 3.
        offset = to_offset(freq)

        if periodlength is None:
            periodlength = 24 * 1 / (to_timedelta(offset).seconds / 3600)

        if ylabel is None:
            yperiod = (periodlength * offset).delta

            offset_n = f"{offset.n}-" if offset.n > 1 else ""
            ylabel = (
                f"{offset_n}{RESOLUTION_NAME[offset.name]} of "
                f"{RESOLUTION_NAME[yperiod.resolution_string][0:-1]}"
            )

        stacked, timeindex = tsam.unstackToPeriods(
            copy.deepcopy(self), int(periodlength)
        )
        if xlabel is None:
            xperiod = (periodlength * offset).delta
            xlabel = f"{RESOLUTION_NAME[xperiod.resolution_string]}"
        cmap = plt.get_cmap(cmap)
        if vcenter is not None:
            norm = TwoSlopeNorm(vcenter, vmin=vmin, vmax=vmax)
        else:
            norm = None
        im = axes.imshow(
            stacked.T,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            norm=norm,
            **kwargs,
        )
        axes.set_aspect("auto")
        axes.set_ylabel(ylabel)
        axes.set_xlabel(xlabel)

        if axis_off:
            axes.set_axis_off()

        # If True or string set ax_title.
        if ax_title:
            if not isinstance(ax_title, str):
                ax_title = f"{self.name}" if self.name is not None else None
            axes.set_title(ax_title)

        # add colorbar
        if colorbar is True:
            cbar = fig.colorbar(im, ax=axes)
            cbar.set_label(f"[{self.units:~P}]")

        fig, axes = save_and_show(
            fig, axes, save, show, close, filename, file_format, dpi, axis_off, extent
        )

        return fig, axes


RESOLUTION_NAME = dict(
    D="Days",
    H="Hours",
    T="Minutes",
    S="Seconds",
    L="Milliseconds",
    U="Microseconds",
    N="Nanoseconds",
)


class EnergyDataFrame(DataFrame):
    """EnergyDataFrame class.

    An EnergyDataFrame is a pandas.DataFrame that is adapted for building data (
    mostly time series of temperature, energy and power).

    Data structure also contains labeled axes (rows and columns). Arithmetic
    operations align on both row and column labels. Can be thought of as a
    dict-like container for Series objects. The primary energy-pandas data structure.

    """

    # temporary properties
    _internal_names = DataFrame._internal_names
    _internal_names_set = set(_internal_names)

    # normal properties
    _metadata = ["units", "name"]

    @property
    def _constructor(self):
        return EnergyDataFrame

    @property
    def _constructor_sliced(self):
        # return EnergySeries
        def f(*args, **kwargs):
            # adapted from https://github.com/pandas-dev/pandas/issues/13208#issuecomment-326556232
            return EnergySeries(*args, **kwargs).__finalize__(self, method="inherit")

        return f

    def __init__(
        self,
        data,
        units=None,
        index=None,
        columns=None,
        dtype=None,
        copy=True,
        **kwargs,
    ):
        super(EnergyDataFrame, self).__init__(
            data, index=index, columns=columns, dtype=dtype, copy=copy
        )

        # prepare units dict; holds series units.
        self.units = {}
        if isinstance(data, (dict, DataFrame)):
            # for each series, setattr `units` defined in each items or for whole df.
            for name, col in data.items():
                # ndarray (structured or homogeneous), Iterable, dict, or DataFrame
                self.units[name] = getattr(col, "units", self._parse_units(units))
        elif isinstance(data, EnergySeries):
            self.units[data.name] = data.units

        # for each extra kwargs, set as metadata of self.
        for k, v in kwargs.items():
            self._metadata.append(k)
            setattr(self, k, v)

    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata from other to self."""
        if isinstance(other, NDFrame):
            for name in other.attrs:
                self.attrs[name] = other.attrs[name]
            # For subclasses using _metadata. Set known attributes and update list.
            for name in other._metadata:
                try:
                    # for units, only keep units for each column name.
                    if name == "units":
                        object.__setattr__(
                            self,
                            name,
                            {
                                col: getattr(other, name).get(col)
                                for col in self.columns
                            },
                        )
                    else:
                        object.__setattr__(self, name, getattr(other, name))
                except AttributeError:
                    pass
                if name not in self._metadata:
                    self._metadata.append(name)
        return self

    @classmethod
    def from_reportdata(
        cls,
        df,
        name=None,
        base_year=2018,
        units=None,
        normalize=False,
        sort_values=False,
        to_units=None,
    ):
        """From a ReportData DataFrame"""
        # get data
        units = [units] if units else set(df.Units)
        if len(units) > 1:
            raise ValueError("The DataFrame contains mixed units: {}".format(units))
        else:
            units = next(iter(units), None)
        # group data by index value (level=0) using the agg_func
        grouped_Data = pivot_table(
            df, index="TimeIndex", columns=["KeyValue"], values=["Value"]
        ).droplevel(axis=1, level=0)
        df = pivot_table(
            df,
            index="TimeIndex",
            columns=None,
            values=["Month", "Day", "Hour", "Minute", "Interval"],
        )
        index = to_datetime(
            {
                "year": base_year,
                "month": df.Month,
                "day": df.Day,
                "hour": df.Hour,
                "minute": df.Minute,
            }
        )

        # Adjust timeindex by timedelta
        index -= df.Interval.apply(lambda x: timedelta(minutes=x))
        index = DatetimeIndex(index)
        grouped_Data.index = index
        # Since we create the index, use_timeindex must be false
        edf = cls(grouped_Data, units=units, index=grouped_Data.index, name=name)
        if to_units:
            edf.to_units(to_units=to_units, inplace=True)
        if normalize:
            edf.normalize(inplace=True)
        if sort_values:
            edf.sort_values(sort_values, inplace=True)
        return edf

    def _parse_units(self, value):
        if isinstance(value, str):
            return unit_registry.parse_expression(value).units
        elif isinstance(value, (Unit, Quantity)):
            return value
        elif value is None:
            return unit_registry.parse_expression(value).units
        else:
            raise TypeError(f"Unit of type {type(value)}")

    def to_units(self, to_units, inplace=False):
        """Returns the multiplier to convert units.

        Args:
            to_units (str or pint.Unit):
            inplace (bool): If True, conversion is applied inplace.

        Examples:
            >>> import energy_pandas as epd
            >>> edf = epd .EnergyDataFrame(
            >>>         {
            >>>             "Series 1": epd.EnergySeries(range(0, 8760), units="degC"),
            >>>             "Series 2": epd.EnergySeries(range(0, 8760), units="degK"),
            >>>             "Series 3": epd.EnergySeries(range(0, 8760), units="degR"),
            >>>         },
            >>> )
            >>> edf.to_units("degC").units
            {'Series 1': <Unit('degree_Celsius')>, 'Series 2': <Unit('degree_Celsius')>, 'Series 3': <Unit('degree_Celsius')>}

        """
        cdata = self.apply(
            lambda col: unit_registry.Quantity(col.values, col.units).to(to_units).m
        )
        if inplace:
            self[:] = cdata.values
            self.units = {
                col: u
                for col, u in zip(
                    self.columns, [unit_registry.Unit(to_units)] * len(self.columns)
                )
            }
        else:
            # create new instance using constructor
            result = self._constructor(
                data=cdata, index=self.index, columns=self.columns, copy=False
            )
            # Copy metadata over
            result.__finalize__(self)
            result.units = {
                col: u
                for col, u in zip(
                    self.columns, [unit_registry.Unit(to_units)] * len(self.columns)
                )
            }
            return result

    def normalize(self, inplace=False):
        x = self.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        if inplace:
            # replace whole data with array
            self[:] = x_scaled
            # change units to dimensionless
            self.units = {name: unit_registry.dimensionless for name in self.columns}
        else:
            # create new instance using constructor
            result = self._constructor(
                data=x_scaled, index=self.index, columns=self.columns, copy=False
            )
            # Copy metadata over
            result.__finalize__(self)
            return result

    def plot2d(
        self,
        subplots=True,
        periodlength=None,
        vmin=None,
        vmax=None,
        vcenter=None,
        axis_off=False,
        cmap="RdBu",
        figsize=None,
        show=True,
        save=False,
        close=False,
        dpi=300,
        file_format="png",
        colorbar=True,
        ax=None,
        filename="untitled",
        extent="tight",
        sharex=True,
        sharey=True,
        layout=None,
        layout_type="vertical",
        fig_title=None,
        **kwargs,
    ):
        """Plot rectangular data as a (or multiple) color-encoded 2d-matrix.

        Args:
            subplots (bool): If true, each column will be plotted as a separate subplot.
            periodlength (int): The period length to show on the yaxis. By default,
                the period length will be calculated to represent one day of data on
                the yaxis and days on the xaxis.
            vmin (float): The data value that defines ``0.0`` in the normalization.
                Defaults to the min value of the dataset.
            vmax (float): The data value that defines ``1.0`` in the normalization.
                Defaults to the the max value of the dataset.
            vcenter (float): The data value that defines ``0.5`` in the
                normalization.
            axis_off (bool): If True, no axis is plotted.
            cmap (str or Colormap): Colormap to select colors from. If string,
                load colormap with that name from matplotlib.
            figsize (tuple): Size of a figure object. A tuple (width, height) in inches.
            show (bool): whether to display the figure or not.
            save (bool): whether to save the figure to disk or not.
            close (bool): close the figure (only if show equals False) to prevent
                display.
            dpi (int): the resolution of the image file if saving (Dots per inch)
            file_format (string): the format of the file to save (e.g., 'jpg',
                'png', 'svg').
            colorbar (bool): If True, plot the colorbar.
            ax (Axes): An axes of the current figure.
            filename (string): the name of the file to save.
            extent (str or `.Bbox`): Bounding box in inches: only the given portion of
                the figure is saved.  If 'tight', try to figure out the tight bbox of
                the figure.
            sharex (bool): If True, the X axis will be shared amongst all subplots.
            sharey (bool): If True, the Y axis will be shared amongst all subplots.
            layout (tuple): Number of rows and columns of the subplot grid. If not
                specified, calculated from naxes and layout_type.
            layout_type (str): {'box', 'horizontal', 'vertical'}, default 'box'
                Specify how to layout the subplot grid.
            fig_title (str): The title of the figure.
            **kwargs: Options to pass to matplotlib plotting method.

        Returns:
            (tuple): tuple containing:

                fig: is the Matplotlib Figure object
                ax: can be either a single axis object or an array of axis objects if
                    more than one subplot was created.
        """
        nseries = self.nseries
        fig, axes = _setup_subplots(
            subplots, nseries, sharex, sharey, figsize, ax, layout, layout_type
        )
        cols = self.columns
        for ax, col in zip(axes, cols):
            self.loc[:, col].plot2d(
                periodlength=periodlength,
                vmin=vmin,
                vmax=vmax,
                vcenter=vcenter,
                axis_off=axis_off,
                cmap=cmap,
                show=False,
                save=False,
                close=False,
                dpi=dpi,
                file_format=file_format,
                colorbar=colorbar,
                ax=ax,
                filename=filename,
                extent=extent,
                **kwargs,
            )
        fig.suptitle(getattr(self, "name", fig_title), fontsize="large")
        fig.tight_layout()  # Adjust the padding between and around subplots.
        fig, axes = save_and_show(
            fig, axes, save, show, close, filename, file_format, dpi, axis_off, extent
        )

        return fig, axes

    @property
    def nseries(self):
        if self._data.ndim == 1:
            return 1
        else:
            return self._data.shape[0]

    def discretize_tsam(self, inplace=False, **kwargs):
        """Clusters time series data to typical periods. See
        :class:`tsam.timeseriesaggregation.TimeSeriesAggregation` for more info.

        Returns:
            EnergyDataFrame:
        """
        try:
            import tsam.timeseriesaggregation as tsam
        except ImportError:
            raise ImportError("tsam is required for discretize_tsam()")
        if not isinstance(self.index, DatetimeIndex):
            raise TypeError("To use tsam, index of series must be a " "DateTimeIndex")
        timeSeries = self.copy()
        agg = tsam.TimeSeriesAggregation(timeSeries, **kwargs)

        agg.createTypicalPeriods()
        result = agg.predictOriginalData()
        if inplace:
            self.loc[:] = result.values
        else:
            # create new instance using constructor
            result = self._constructor(
                data=result.values, index=self.index, columns=self.columns, copy=False
            )
            # Copy metadata over
            result.__finalize__(self)
            return result

    discretize_tsam.__doc__ = tsam.TimeSeriesAggregation.__init__.__doc__
