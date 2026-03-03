from itertools import cycle

import matplotlib as mpl


class PythonFigConfigs:
    """
    This class contains all the rcParams settings for different format styles of
    the ITM plotting style.

    Created by broeder: 2023/08/14
    """

    # Hard-coded constants
    CM = 1 / 2.54
    AXES_COLOR = "#262626"
    GRID_COLOR = "#dedede"
    GRID_LINESTYLE = (0, (1.5, 4.5))
    DEFAULT_COLORS = [
        "#0072bd",
        "#d95319",
        "#edb120",
        "#7e1f8e",
        "#77ac30",
        "#4dbeee",
        "#a2142f",
        "#e377c2",
        "#7f7f7f",
        "#8c564b",
    ]

    DEFAULTS = {
        # Sets background grid lines
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linestyle": GRID_LINESTYLE,
        # Changes thickness and color of the plot bounding box
        "axes.labelcolor": AXES_COLOR,
        "axes.edgecolor": AXES_COLOR,
        # Adjust the legend
        "legend.edgecolor": AXES_COLOR,
        "legend.borderaxespad": 0,
        "legend.borderpad": 0.3,  # Padding between content and legend box
        "legend.fancybox": False,
        "legend.framealpha": 1,
        "legend.handlelength": 1.75,  # Line length
        "legend.handletextpad": 0.3,  # Space to the colored lines
        "legend.labelspacing": 0.2,  # Vertical spacing between the entries
        "legend.loc": "upper right",
        "legend.frameon": True,
        "figure.dpi": 600,
        # Makes the figure transparent when saving
        "savefig.transparent": True,
        # Adjusts sizes, position and colors of the x- and y-ticks
        "xtick.top": True,
        "ytick.right": True,
        "xtick.color": AXES_COLOR,
        "ytick.color": AXES_COLOR,
        "xtick.major.size": 1.8,
        "ytick.major.size": 1.8,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.pad": 3,
        "ytick.major.pad": 1.5,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.minor.pad": 3,
        "ytick.minor.pad": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }

    # Dictionaries contain rcParams entries
    LATEX_CONFIG = {
        # Changes the fonts and uses LaTex to render them
        "font.family": "serif",
        "font.serif": "Latin Modern Roman",
        "font.size": 10,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        # "text.usetex": True,
        "text.latex.preamble": "lmodern",
        "axes.linewidth": 0.5,
        # "axes.prop_cycle": mpl.cycler(colors=DEFAULT_COLORS),
        "grid.linewidth": 0.5,
        "patch.linewidth": 0.5,
        # Changes the size of the plot lines
        "lines.linewidth": 0.5,
        "lines.markersize": 5,
    }

    POWERPOINT_CONFIG = {
        # Changes the fonts and uses LaTex to render them
        "font.family": "sans-serif",
        "font.serif": "Roboto",
        "font.size": 12,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        # "text.usetex": True,
        "text.latex.preamble": "lmodern",
        "axes.linewidth": 1,
        # "axes.prop_cycle": mpl.cycler(colors=DEFAULT_COLORS),
        "grid.linewidth": 1,
        "patch.linewidth": 1,
        # Changes the size of the plot lines
        "lines.linewidth": 1,
        "lines.markersize": 6,
    }

    @staticmethod
    def get_configs(format: str) -> dict | None:
        """
        Retrieves the rc settings for a specified figure format

        ### Arguments
        format (str): Choosing a predefined format for the figure. Options are
          - 'latexwide' ..... LaTeX wide plot (student template size)
          - 'latexnarrow' ... LaTeX narrow plot
          - 'word' .......... Word plot
          - 'powerpoint' .... PowerPoint standard size plot
          - 'poster' ........ Poster
          - 'diss' .......... LaTeX wide plot (dissertation template size)

        ### Returns
        config_dict (dict): A dict with specific rcParams settings
        """

        list_of_supported_formats = ["latexwide", "latexnarrow", "powerpoint"]
        if format not in list_of_supported_formats:
            raise ValueError(
                "Unsupported format. Format has to be one of the following {}".format(
                    list_of_supported_formats
                )
            )
        configs = None
        if format == "latexwide" or format == "latexnarrow":
            configs = dict(PythonFigConfigs.DEFAULTS, **PythonFigConfigs.LATEX_CONFIG)
        elif format == "powerpoint":
            configs = dict(
                PythonFigConfigs.DEFAULTS, **PythonFigConfigs.POWERPOINT_CONFIG
            )
        return configs

    @staticmethod
    def get_figsize(
        format: str, expl_width: float | None, expl_height: float | None
    ) -> tuple:
        """
        Gets the figure size in terms of inches.

        ### Arguments
        1. width (float): Width of the figure in cm
        2. height (float): Height of the figure in cm

        ### Returns
        figure_size (tuple): Figure size in inches
        """
        if format == "latexwide":
            width = 13.5
            height = 8
        elif format == "latexnarrow":
            width = 6
            height = 6
        elif format == "word":
            width = 13
            height = 10
        elif format == "powerpoint":
            width = 11.5
            height = 8
        elif format == "poster":
            width = 40
            height = 20
        elif format == "diss":
            width = 15.99
            height = 9.88
        else:
            raise ValueError("Unsupported format.")

        if expl_width is not None:
            width = expl_width
        if expl_height is not None:
            height = expl_height

        return (width * PythonFigConfigs.CM, height * PythonFigConfigs.CM)
