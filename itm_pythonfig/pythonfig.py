import matplotlib as mpl
import matplotlib.figure as mplf
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

from itm_pythonfig.pythonfig_configs import PythonFigConfigs

# for an example of axes and figure aspect/ratio/limits see https://stackoverflow.com/questions/32633322/changing-aspect-ratio-of-subplots-in-matplotlib


class PythonFig:
    """
    A class for the automatic setting of the plot parameter which follows the ITM
    plotting style.

    Created by broeder: 2023/08/14
    """

    def start_figure(
        self, format: str, width: float | None = None, height: float | None = None
    ) -> mplf.Figure:
        """
        Opens a plt figure with the passed (width, height) in cm while
        setting the plotting parameters according to a chosen preset.

        Created by broeder for Matplotlib Version 3.7.1

        ### Arguments
        1. format (str): Choosing a predefined format for the figure. Options are
          - 'latexwide' ..... LaTeX wide plot (student template size)
          - 'latexnarrow' ... LaTeX narrow plot
          - 'word' .......... Word plot
          - 'powerpoint' .... PowerPoint standard size plot
          - 'poster' ........ Poster
          - 'diss' .......... LaTeX wide plot (dissertation template size)
        2. width (float): Width of the figure in cm.
        3. height (float): Height of the figure in cm.

        ### Returns
        `matplotlib.figure.Figure`: An empty plt figure.

        ### Raises:
        ValueError: When unsupported format styles are given.
        """
        # Check if a supported format is requested.
        list_of_supported_formats = ["latexwide", "latexnarrow", "powerpoint"]
        if format not in list_of_supported_formats:
            raise ValueError(
                "Unsupported format. Format has to be one of the following {}".format(
                    list_of_supported_formats
                )
            )

        # Reset all old configuraitions
        plt.rcdefaults()

        # Update the rcParams with predefined settings
        configs = PythonFigConfigs.get_configs(format)
        assert configs is not None
        plt.rcParams.update(configs)

        # Create an empty figure
        fig = plt.figure(figsize=PythonFigConfigs.get_figsize(format, width, height))

        return fig

    def finish_figure(self, file_path=None, show_legend=False) -> None:
        """
        Completes a figure by adding a legend an allows to save the figure to disk.

        ### Arguments
            file_path (str, optional): The file in which to save the figure.
                When None is passed, the figure is not saved to disk. Defaults to None.
            show_legend (bool, optional): If a legend should be plotted. Defaults to False.
        """
        # plt.tight_layout(pad=0.15)
        plt.subplots_adjust(left=0.15)

        if show_legend:
            plt.legend().get_frame().set_linewidth(0.5)

        if file_path:
            plt.savefig(file_path, bbox_inches="tight", pad_inches=0, transparent=True)
