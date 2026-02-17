from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING, Callable, List, Tuple

import numpy as np
from anomaly_detectors.preprocessing import select_states_based_on_detection_interval
from itm_pythonfig.pythonfig import PythonFig
from matplotlib.animation import ArtistAnimation
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from visualization.warehouse_visualization import draw_warehouse

if TYPE_CHECKING:
    from robots.deployment_robot import Robot
    from robots.robot_swarm import RobotSwarm


class SwarmVisualization:

    def __init__(self, robot_swarm: RobotSwarm) -> None:
        self.robot_swarm = robot_swarm

    def _init_recorded_positions(
        self, min_time_stamp: int | None, max_time_stamp: int | None
    ):

        time_between_recordings = []
        max_timestamps = []
        assert self.robot_swarm.external_state_monitors is not None
        for sensor in self.robot_swarm.external_state_monitors:
            max_timestamps.append(list(sensor.time_stamps)[-1])
            time_between_recordings.append(sensor.pause_between_monitoring)
        self.min_recording_interval = min(time_between_recordings)
        self.n_recorded_time_steps = int(
            np.max(max_timestamps) / self.min_recording_interval
        )
        self.min_time_stamp = min_time_stamp if min_time_stamp is not None else 0
        self.max_time_stamp = (
            max_time_stamp if max_time_stamp else np.max(max_timestamps)
        )

        self.swarm_positions = np.zeros(
            shape=(self.robot_swarm.n_robots, self.n_recorded_time_steps + 1, 2)
        )

        for sensor in self.robot_swarm.external_state_monitors:
            recorded_pos = np.array(sensor.recorded_positions)[
                : self.n_recorded_time_steps
            ]
            recorded_time_stamps = np.array(sensor.time_stamps)[: len(recorded_pos)]
            ts_idx = (recorded_time_stamps / self.min_recording_interval).astype(int)
            # print(
            #     recorded_pos.shape,
            #     recorded_ts.shape,
            #     ts_idx.shape,
            #     self.swarm_positions.shape,
            # )
            self.swarm_positions[sensor.monitored_robot_id, ts_idx, :] = recorded_pos

        def fill_zeros_with_last_or_first(arr):
            tmp_idx = np.arange(arr.shape[1]).reshape(
                1, -1, 1
            )  # shape = (1, arr[1], 1)
            tmp1 = tmp_idx.repeat(arr.shape[0], axis=0)  # shape = (arr[0], arr[1], 1)

            tmp1[(arr == 0).all(axis=-1)] = self.n_recorded_time_steps
            first_non_zero_idx = np.min(tmp1, axis=1)  # shape = (1, arr[1])

            tmp2 = tmp_idx.repeat(arr.shape[0], axis=0)  # shape = (arr[0], arr[1], 1)
            tmp2[(arr == 0).all(axis=-1)] = 0
            tmp2[:, 0] = first_non_zero_idx
            tmp2 = tmp2.repeat(
                arr.shape[2], axis=-1
            )  # shape = (arr[0], arr[1], arr[2])
            prev_or_first_idx = np.maximum.accumulate(tmp2, axis=1)
            res = np.take_along_axis(arr, prev_or_first_idx, axis=1)

            return res

        self.swarm_positions = fill_zeros_with_last_or_first(self.swarm_positions)

        return

    def plot_run(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        plot_trajectory_as_line=True,
        n_action_samples_to_plot: int = 0,
        n_waypoints_to_plot: int = 0,
        min_time_stamp: int | None = None,
        max_time_stamp: int | None = None,
    ):
        if fig is None:
            pf = PythonFig()
            fig = pf.start_figure("latexwide", 13.7, 10)
        ax = self.adjust_axis(fig, ax)
        self._init_recorded_positions(min_time_stamp, max_time_stamp)

        # plot deployment area
        self.plot_area(fig, ax)

        if n_action_samples_to_plot > 0 or n_waypoints_to_plot > 0:

            self.plot_action_samples(
                n_action_samples_to_plot,
                n_waypoints_to_plot,
                ax,
                plot_trajectory_as_line=plot_trajectory_as_line,
            )
            if plot_trajectory_as_line:
                self.plot_motion_change(
                    ax, plot_trajectory_as_line, n_action_samples_to_plot
                )

        elif plot_trajectory_as_line:
            self.plot_positions(
                ax,
                plot_trajectory_as_line=plot_trajectory_as_line,
            )
            self.plot_motion_change(
                ax, plot_trajectory_as_line, n_action_samples_to_plot
            )
        else:
            self.plot_communicated_positions(
                ax,
                n_action_samples_to_plot=n_action_samples_to_plot,
            )
            self.plot_motion_change(
                ax, plot_trajectory_as_line, n_action_samples_to_plot
            )

        # plot the robots' target positions
        # self.plot_targets(ax)
        # legend and labels
        self.build_legend(ax)

        return fig

    def animate_run(
        self,
        folder_path: str | None = None,
        every_nth_img: int = 1,
        show_legend: bool = False,
        min_ts: int | None = None,
        max_ts: int | None = None,
    ):

        pf = PythonFig()
        fig = pf.start_figure("latexwide", 13.7, 10)
        ax = fig.gca()
        ax = self.adjust_axis(fig, ax)
        self._init_recorded_positions(min_ts, max_ts)
        # plt.ioff()

        self.plot_area(fig, ax)

        # legend and labels
        if show_legend:
            self.build_animation_legend(ax)

        images = self.collect_animation_images(fig, ax, folder_path, every_nth_img, pf)
        if folder_path != None:
            return

        assert type(fig) == Figure
        line_anim = ArtistAnimation(
            fig,
            images,
            interval=150,
            blit=True,
        )
        return line_anim

    def collect_animation_images(self, fig, ax, folder_path, every_nth_img, pf):
        if folder_path:
            os.makedirs(folder_path, exist_ok=True)

        frame_idx = 0
        images = []
        kwargs = {}
        kwargs["show_points"] = False
        kwargs["show_vertices"] = False
        # communicated_positions = self.robot_swarm.get_all_communicated_positions()

        for ts in np.arange(
            start=0, stop=self.n_recorded_time_steps + 1, step=every_nth_img
        ):

            robo_pics = self.plot_pos_per_ts(ts, ax)
            robo_targets = self.plot_targets_per_ts(ts, ax)

            t = self.show_time(ax, ts)

            img = robo_pics + robo_targets + [t]
            # plot static targets
            # self.plot_targets(ax)

            images.append(img)

            if folder_path:
                pf.finish_figure(
                    os.path.join(folder_path, f"frame_{frame_idx:04d}.png")
                )  # , show_legend=True)
            # fig.savefig(
            #     os.path.join(folder_path, f"frame_{frame_idx:04d}.png"),
            #     bbox_inches="tight",
            # )
            frame_idx += 1
            # Remove all artists in img from axes for next frame
            for artist in img:
                try:
                    artist.remove()
                except Exception:
                    pass

        return images

    def plot_ts(
        self,
        ts,
        fig,
        ax,
        plot_actions=True,
        scale_to_current_area=False,
        n_action_samples_to_plot=0,
        show_legend=False,
    ):
        start_positions = self.robot_swarm.get_all_communicated_positions()

        if fig is None:
            pf = PythonFig()
            fig = pf.start_figure("latexwide", 13.7, 10)
        ax = self.adjust_axis(fig, ax)
        ax.set(adjustable="datalim")

        self.plot_pos_per_ts(ts, ax)
        # self.plot_action_samples_per_ts(
        #     ax,
        #     ts,
        #     self.robot_swarm.anomaly_detector,
        #     n_action_samples_to_plot=n_action_samples_to_plot,
        # )

        # self.plot_targets(ax)
        # self.plot_spoofing_communication(ts, ax)
        # self.show_time(ax, ts)
        self.plot_area(fig, ax)

        if show_legend:
            self.build_legend(ax)

    def plot_area(
        self,
        fig,
        ax,
    ):
        assert self.robot_swarm.deployment_area is not None
        draw_warehouse(ax, warehouse=self.robot_swarm.deployment_area)
        return

    def plot_positions(
        self, ax, plot_trajectory_as_line: bool, plot_all_positions: bool = False
    ):
        start_ts = int(self.min_time_stamp / self.min_recording_interval)
        stop_ts = int(self.max_time_stamp / self.min_recording_interval)
        if plot_trajectory_as_line:  # first and and last timestep
            timesteps = np.array([start_ts, stop_ts])
        elif plot_all_positions:  # all timesteps
            timesteps = np.arange(stop_ts + 1)
        else:
            return

        for ts in timesteps:
            self.plot_pos_per_ts(ts, ax)

        return

    def plot_pos_per_ts(self, ts, ax) -> list:

        all_scatter_points_per_ts = []

        for robot in self.robot_swarm.swarm_robots:
            robot_pos = self.swarm_positions[robot.id][ts]
            # colored markers for robots
            facecolor, legend_marker, marker_size, marker_line, _ = self.get_markers(
                robot, approx_time_stamp=ts * self.min_recording_interval
            )
            rp = Circle(
                (robot_pos[0], robot_pos[1]),
                radius=0.29 / 2,
                edgecolor="black",
                facecolor=facecolor,
                linewidth=marker_line,
                zorder=6,
            )
            ax.add_patch(rp)
            all_scatter_points_per_ts.append(rp)

        return all_scatter_points_per_ts

    def plot_communicated_positions(self, ax, n_action_samples_to_plot):

        start_ts = int(self.min_time_stamp / self.robot_swarm.ts_communicate)
        stop_ts = int(self.max_time_stamp / self.robot_swarm.ts_communicate)
        communicated_pos = self.robot_swarm.get_all_communicated_positions()

        scatter_points = []
        for robot in self.robot_swarm.swarm_robots:
            robot_pos = communicated_pos[robot.id][start_ts:stop_ts]

            for p_i, pos in enumerate(robot_pos):
                approx_time_stamp = p_i * self.robot_swarm.ts_communicate
                facecolor, legend_marker, marker_size, marker_line, _ = (
                    self.get_markers(robot, approx_time_stamp=approx_time_stamp)
                )
                facecolor, _ = self.build_colormap(
                    facecolor, granularity=1, max_alpha=0.3
                )
                cp = Circle(
                    pos,
                    radius=0.29 / 2,
                    edgecolor="black",
                    facecolor=facecolor,
                    linewidth=marker_line,
                    zorder=6,
                )
                ax.add_patch(cp)
                scatter_points.append(cp)
        return scatter_points

    def plot_motion_change(
        self, ax: Axes, plot_trajectory_as_line: bool, n_action_samples_to_plot: int
    ) -> None:

        if n_action_samples_to_plot > 0:
            alpha = 0.5
        else:
            alpha = 1

        start_ts = int(self.min_time_stamp / self.min_recording_interval)
        stop_ts = int(self.max_time_stamp / self.min_recording_interval)

        for robot in self.robot_swarm.swarm_robots:
            if plot_trajectory_as_line:
                *_, linecolor = self.get_markers(robot, 0)
                arrowstyle = "-"
                shrinkB = 0

                robo_pos = self.swarm_positions[robot.id][start_ts:stop_ts]
            else:
                linecolor = "black"
                arrowstyle = "-|>"
                shrinkB = 0  # .9
                start_ts = int(self.min_time_stamp / self.robot_swarm.ts_communicate)
                stop_ts = int(self.max_time_stamp / self.robot_swarm.ts_communicate)
                robo_pos = self.robot_swarm.get_all_communicated_positions()[robot.id][
                    start_ts:stop_ts
                ]

            for start_pos, end_pos in zip(robo_pos[:-1], robo_pos[1:]):
                # plot robot trajectory
                arrow = FancyArrowPatch(
                    (start_pos[0], start_pos[1]),
                    (end_pos[0], end_pos[1]),
                    color=linecolor,
                    mutation_scale=3,
                    shrinkA=0,
                    shrinkB=shrinkB,
                    linewidth=0.3,
                    arrowstyle=arrowstyle,
                    zorder=4,
                    alpha=alpha,
                )
                _ = ax.add_patch(arrow)
        return

    def plot_action_samples(
        self,
        n_action_samples_to_plot: int,
        n_waypoints_to_plot: int,
        ax: Axes,
        plot_trajectory_as_line: bool,
    ) -> None:
        if n_action_samples_to_plot <= 0 and n_waypoints_to_plot <= 0:
            return
        if self.robot_swarm.anomaly_detector is None:
            print("please set an anomaly detector")
            return

        detection_interval = self.robot_swarm.anomaly_detector.detection_interval
        min_sample_ts = int(self.min_time_stamp / detection_interval)
        max_sample_ts = int(self.max_time_stamp / detection_interval)

        for sample_idx in range(min_sample_ts, max_sample_ts):
            approx_ts = detection_interval * sample_idx
            plot_position = (
                sample_idx == min_sample_ts
                or sample_idx == max_sample_ts - 1
                or not plot_trajectory_as_line
            )

            self.plot_action_samples_per_ts(
                ax,
                sample_idx,
                approx_ts,
                self.robot_swarm.anomaly_detector,
                n_action_samples_to_plot,
                n_waypoints_to_plot,
                plot_position,
            )
        return

    def plot_action_samples_per_ts(
        self,
        ax,
        sample_idx,
        approx_ts,
        anomaly_detector,
        n_action_samples_to_plot,
        n_waypoints_to_plot,
        plot_position: bool = True,
    ):
        if n_action_samples_to_plot <= 0 and n_waypoints_to_plot <= 0:
            return

        positions, action_samples, motion_change_2d, waypoints_2d, sample_log_prob = (
            anomaly_detector.sample_robot_actions(
                sample_idx=int(sample_idx),
                n_samples=max(n_action_samples_to_plot, n_waypoints_to_plot),
            )
        )
        for r, robot in enumerate(self.robot_swarm.swarm_robots):
            if plot_position:
                facecolor, legend_marker, marker_size, marker_line, _ = (
                    self.get_markers(robot, approx_time_stamp=approx_ts)
                )
                facecolor, _ = self.build_colormap(
                    facecolor, granularity=1, max_alpha=0.3
                )
                cp = Circle(
                    positions[r],
                    radius=0.29 / 2,
                    edgecolor="black",
                    facecolor=facecolor,
                    linewidth=marker_line,
                    zorder=6,
                )
                ax.add_patch(cp)

            for n in range(n_action_samples_to_plot):
                # if low_lp_mask[t, n]:
                arrow = FancyArrowPatch(
                    (positions[r, 0], positions[r, 1]),
                    (
                        positions[r, 0] + motion_change_2d[r, n, 0],
                        positions[r, 1] + motion_change_2d[r, n, 1],
                    ),
                    color="#515151",
                    mutation_scale=3,
                    linewidth=0.2,
                    shrinkA=0,
                    shrinkB=0,
                    arrowstyle="->",
                    zorder=7,
                    alpha=0.05,
                )
                _ = ax.add_patch(arrow)

            for n in range(n_waypoints_to_plot):
                # if low_lp_mask[t, n]:
                arrow = FancyArrowPatch(
                    (positions[r, 0], positions[r, 1]),
                    (
                        positions[r, 0] + waypoints_2d[r, n, 0],
                        positions[r, 1] + waypoints_2d[r, n, 1],
                    ),
                    color="#86A0A5",
                    mutation_scale=3,
                    linewidth=0.2,
                    shrinkA=0,
                    shrinkB=0,
                    arrowstyle="->",
                    zorder=7,
                    alpha=0.05,
                )
                _ = ax.add_patch(arrow)

        return

    def plot_targets(self, ax) -> None:

        for robot in self.robot_swarm.swarm_robots:
            current_targets = np.array(
                [state.current_target_pos for state in robot.state_history]
            )
            ax.scatter(
                current_targets[:, 0],
                current_targets[:, 1],
                color=robot.color,
                edgecolor="black",
                s=1,
                zorder=4,
            )
        return

    def plot_targets_per_ts(self, ts, ax) -> list:

        all_scatter_points_per_ts = []
        approx_time_stamp = ts * self.min_recording_interval
        for robot in self.robot_swarm.swarm_robots:
            state_time_stamps = np.array(
                [state.time_stamp for state in robot.state_history]
            )
            next_ts = np.where(state_time_stamps > approx_time_stamp)[0]
            if len(next_ts) > 0:
                ts_idx = next_ts[0]
                current_target = robot.state_history[ts_idx].current_target_pos
                ct = ax.scatter(
                    current_target[0],
                    current_target[1],
                    color=robot.color,
                    edgecolor=robot.color,
                    s=1,
                    zorder=5,
                    alpha=0.5,
                )
                all_scatter_points_per_ts.append(ct)
                adapted_target = robot.state_history[ts_idx].adapted_target_pos
                at = ax.scatter(
                    adapted_target[0],
                    adapted_target[1],
                    color=robot.color,
                    edgecolor="black",
                    s=1,
                    zorder=6,
                )
                all_scatter_points_per_ts.append(at)
        return all_scatter_points_per_ts

    def build_colormap(self, color, granularity, max_alpha=0.8):

        if granularity == 1:
            alphas = [max_alpha]
        else:
            alphas = np.zeros(shape=(granularity,))
            alphas = np.linspace(0, max_alpha, granularity)

        rgba = np.zeros(shape=(granularity, 4))
        rgba[:, :3] = to_rgb(color)
        rgba[:, -1] = alphas
        rgba_cmap = ListedColormap(rgba)

        rgb = np.ones(shape=(granularity, 3), dtype="float32")
        r, g, b, a = (
            rgba[:, 0],
            rgba[:, 1],
            rgba[:, 2],
            rgba[:, 3],
        )
        # a = np.asarray(a, dtype="float32")
        rgb[:, 0] = r * a + (1.0 - a)
        rgb[:, 1] = g * a + (1.0 - a)
        rgb[:, 2] = b * a + (1.0 - a)

        rgb_cmap = ListedColormap(rgb)
        rgb_cmap.set_under("white", alpha=0)

        return rgb, rgb_cmap

    def get_markers(self, robot: Robot, approx_time_stamp: float):

        color = robot.color

        if self.robot_swarm.is_anomal(robot, approx_time_stamp):
            marker = "s"
            markersize = 8
            linewidth = 0.4
            linecolor = color
        else:
            marker = "o"
            markersize = 8
            linewidth = 0.4
            linecolor = color

        return color, marker, markersize, linewidth, linecolor

    def build_legend(self, ax):

        legend_markers = []
        legend_labels = []

        # legend_markers.append(
        #     Line2D(
        #         [0, 0],
        #         [0, 0],
        #         marker="s",
        #         markeredgecolor="black",
        #         markerfacecolor="white",
        #         linestyle="",
        #         markeredgewidth=0.5,
        #     )
        # )
        # legend_labels.append("label anomalous")

        for robot in self.robot_swarm.swarm_robots:
            if (
                "swarm robots"  # robot.label_text
                not in legend_labels
                # and robot.label != 0
            ):
                legend_labels.append("swarm robots")
                legend_markers.append(
                    Line2D(
                        [0, 0],
                        [0, 0],
                        markeredgecolor="black",
                        markerfacecolor=robot.color,
                        marker="o",
                        linestyle="",
                        markeredgewidth=0.5,
                    )
                )
            # if robot.label_text not in legend_labels:
            #     legend_labels.append(f"{robot.label_text}")
            #     legend_markers.append(
            #         Line2D(
            #             [0, 0],
            #             [0, 0],
            #             markeredgecolor="black",
            #             markerfacecolor=robot.color,
            #             marker="o",
            #             linestyle="",
            #             markeredgewidth=0.5,
            #         )
            #     )
        legend = ax.legend(
            legend_markers,
            legend_labels,
            numpoints=1,
            facecolor="none",
            # loc="lower right",
            loc="lower left",
            bbox_to_anchor=(0, 1),
            ncol=5,
            handletextpad=0,
            columnspacing=0.75,
        )
        # legend.get_frame().set_linewidth(0.5)

        return legend

    def build_animation_legend(self, ax):

        legend_markers = []
        legend_labels = []

        # coverage area
        legend_markers.append(
            Line2D(
                [0, 0],
                [0, 0],
                linewidth=0,
                marker="_",
                color="#7878ff",
                linestyle="solid",
            )
        )
        legend_labels.append("area to monitor")

        legend_markers.append(
            Line2D(
                [0, 0],
                [0, 0],
                linewidth=0,
                marker="h",
                color="darkgrey",
                linestyle="solid",
            )
        )
        legend_labels.append("animals")

        legend_markers.append(
            Line2D(
                [0, 0],
                [0, 0],
                marker="s",
                markeredgecolor="black",
                markerfacecolor="white",
                linestyle="",
                markeredgewidth=0.5,
            )
        )
        legend_labels.append("label anomalous")

        for robot in self.robot_swarm.swarm_robots:
            # if (
            #     "swarm robots"  # robot.label_text
            #     not in legend_labels
            #     # and robot.label != 0
            # ):
            #     legend_labels.append("swarm robots")
            #     legend_markers.append(
            #         Line2D(
            #             [0, 0],
            #             [0, 0],
            #             markeredgecolor="black",
            #             markerfacecolor=robot.color,
            #             marker="o",
            #             linestyle="",
            #             markeredgewidth=0.5,
            #         )
            #     )
            if robot.label_text not in legend_labels:
                legend_labels.append(f"{robot.label_text}")
                legend_markers.append(
                    Line2D(
                        [0, 0],
                        [0, 0],
                        markeredgecolor="black",
                        markerfacecolor=robot.color,
                        marker="o",
                        linestyle="",
                        markeredgewidth=0.5,
                    )
                )
        legend = ax.legend(
            legend_markers,
            legend_labels,
            numpoints=1,
            facecolor="none",
            # loc="lower right",
            loc="lower left",
            bbox_to_anchor=(1, 0),
            ncol=1,
            handletextpad=0,
            # columnspacing=0.75,
        )
        # legend.get_frame().set_linewidth(0.5)

        return legend

    def show_time(self, ax, ts):

        approx_time_stamp = ts * self.min_recording_interval
        ti = np.round(
            approx_time_stamp,
            decimals=2,
        )

        pad_pts = 5
        pad_pixels = pad_pts * ax.figure.dpi / 72
        pad_frac_x = pad_pixels / ax.bbox.width
        pad_frac_y = pad_pixels / ax.bbox.height

        tx = ax.text(
            x=1 - pad_frac_x - 0.03,
            y=0 - pad_frac_y - 0.01,
            s=f"t = {ti} s",
            ha="right",
            va="top",
            fontsize=8,
            zorder=6,
            transform=ax.transAxes,
            bbox=dict(facecolor="none", edgecolor="black", pad=pad_pts),
        )
        return tx

    def adjust_axis(self, fig, ax):

        if ax is None:
            ax = fig.gca()
        # else: ax.clear()
        ax.set_xticks([], labels=None)
        ax.set_yticks([], labels=None)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set(aspect=1)  # , adjustable="datalim")

        return ax
