"""Plotting module."""

from __future__ import annotations

import logging
import os
import time

from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from numpy import asarray, ndarray
from pandas.plotting._matplotlib.tools import create_subplots, flatten_axes

logger = logging.getLogger(__name__)
log = logger.info


def save_and_show(
    fig: plt.Figure,
    ax: plt.Axes | list[plt.Axes] | None = None,
    save: bool = False,
    show: bool = True,
    close: bool = False,
    filename: str = "untitled",
    file_format: str = "png",
    dpi: int = 300,
    axis_off: bool = False,
    extent: str | plt.Bbox = "extent",
) -> tuple[plt.Figure, plt.Axes | list[plt.Axes]]:
    """Save a figure to disk and show it, as specified.

    Args:
        fig (matplotlib.figure.Figure): the figure.
        ax (matplotlib.axes.Axes or list(matplotlib.axes.Axes)): the axes.
        save (bool): whether to save the figure to disk or not.
        show (bool): whether to display the figure or not.
        close (bool): close the figure (only if show equals False) to prevent
            display.
        filename (string): the name of the file to save.
        file_format (string): the format of the file to save (e.g., 'jpg',
            'png', 'svg').
        dpi (int): the resolution of the image file if saving (Dots per inch).
        axis_off (bool): if True matplotlib axis was turned off by plot_graph so
            constrain the saved figure's extent to the interior of the axis.
        extent (str or `.Bbox`): Bounding box in inches: only the given portion of
            the figure is saved.  If 'tight', try to figure out the tight bbox of
            the figure.

    Returns:
        tuple: fig, ax
    """
    if save:
        start_time = time.time()

        # create the save folder if it doesn't already exist
        path_filename = os.path.join(os.extsep.join([filename, file_format]))
        if ax is None:
            ax = fig.get_axes()
        if not isinstance(ax, (ndarray, list)):
            ax = [ax]
        if file_format == "svg":
            for axx in ax:
                axx.patch.set_alpha(0.0)
            fig.patch.set_alpha(0.0)
            fig.savefig(
                path_filename,
                bbox_inches=0,
                format=file_format,
                facecolor=fig.get_facecolor(),
                transparent=True,
            )
        else:
            if extent is None:
                if len(ax) == 1:
                    if axis_off:
                        for axx in ax:
                            # if axis is turned off, constrain the saved
                            # figure's extent to the interior of the axis
                            extent = axx.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                else:
                    extent = "tight"
            fig.savefig(
                path_filename,
                dpi=dpi,
                bbox_inches=extent,
                format=file_format,
                facecolor=fig.get_facecolor(),
                transparent=True,
            )
        log(f"Saved the figure to disk in {time.time() - start_time:,.2f} seconds")

    # show the figure if specified
    if show:
        start_time = time.time()
        plt.show()
        # fig.show()
        log(f"Showed the plot in {time.time() - start_time:,.2f} seconds")
    # if show=False, close the figure if close=True to prevent display
    elif close:
        plt.close()

    return fig, ax


def _plot_poly_collection(ax, verts, zs=None, cmap=None, vmin=None, vmax=None, **kwargs):
    from matplotlib.collections import PolyCollection

    poly = PolyCollection(verts, **kwargs)
    if zs is not None:
        poly.set_array(asarray(zs))
        poly.set_cmap(cmap)
        poly.set_clim(vmin, vmax)

    ax.add_collection3d(poly, zs=zs, zdir="y")

    return poly


def _plot_surface(ax, x, y, z, cmap=None, **kwargs):
    """
    Args:
        ax:
        x:
        y:
        z:
        cmap:
        **kwargs:
    """
    if cmap is None:
        cmap = plt.get_cmap("gist_earth")

    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(z, cmap=plt.get_cmap(cmap), vert_exag=0.1, blend_mode="soft")
    surf = ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        facecolors=rgb,
        linewidth=0,
        antialiased=False,
        shade=False,
        **kwargs,
    )
    return surf


def _polygon_under_graph(xlist, ylist):
    """Construct vertex list defining polygon under the (xlist, ylist) line graph.

    Assumes the xs are in ascending order.

    Args:
        xlist:
        ylist:
    """
    return [(xlist[0], 0.0), *zip(xlist, ylist), (xlist[-1], 0.0)]


def _setup_subplots(
    subplots,
    nseries,
    sharex=False,
    sharey=False,
    figsize=None,
    ax=None,
    layout=None,
    layout_type="vertical",
):
    """Prepare the subplots."""
    if subplots:
        fig, axes = create_subplots(
            naxes=nseries,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            ax=ax,
            layout=layout,
            layout_type=layout_type,
        )
    else:
        if ax is None:
            fig = plt.figure(figsize=figsize)
            axes = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
            if figsize is not None:
                fig.set_size_inches(figsize)
            axes = ax

    axes = flatten_axes(axes)

    return fig, axes
