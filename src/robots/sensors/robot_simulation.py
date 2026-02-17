from __future__ import annotations

import abc
import multiprocessing
import threading
from multiprocessing import Array, Value
from typing import TYPE_CHECKING

import lcm
import numpy as np
from helper_functions import l2_norm
from lcm_types.itmessage import vector_t
from scipy.integrate import solve_ivp
from warehouse_env.warehouse import Warehouse


class RobotSimulation(metaclass=abc.ABCMeta):

    def __init__(self, id: int, use_lcm: bool, ts_control: float, **kwargs) -> None:
        """Simulates a hardware robot.

        Args:
            use_lcm (bool): use LCM to get information about the target velocity.
            ts_control (float): A new target velocity is set every ts_control seconds.
        """
        self.id = id
        self.use_lcm = use_lcm
        self.ts_control = ts_control
        self.wheel_is_lagging = False

        if self.use_lcm:
            self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
            self.seq_number_u = 0
            lcs = self.lc.subscribe(f"/robot{self.id}/u", self._lcm_handler)
            lcs.set_queue_capacity(1)

            self.listen_thread = threading.Thread(target=self._listen, daemon=True)
        return

    def trigger_movement(self, vel: np.ndarray) -> None:
        """Trigger the execution of a velocity instruction vel.

        Args:
            vel (np.array): Target velocity.
        """
        self._simulate_movement(vel)
        return

    def introduce_wheel_lag(self, wheel_is_lagging: bool) -> None:
        self.wheel_is_lagging = wheel_is_lagging
        if self.wheel_is_lagging:
            self.lagging_wheel = np.random.choice(4)
            self.wheel_lag = 0.01
        return

    @abc.abstractmethod
    def _simulate_movement(self, vel: np.ndarray) -> None:
        raise NotImplementedError()

    def _listen(self) -> None:
        while self.is_listening:
            self.lc.handle_timeout(int(3 * self.ts_control * 1000))
        print("simulation: stop listening")

    def _lcm_handler(self, _: str, msg: bytes) -> None:
        """Executes the last velocity instruction that has been received via LCM.

        Args:
            msg (vector_t): Message containing the target velocity.
        """
        assert type(msg) == vector_t
        msg = vector_t.decode(msg)
        # 3 dim: 1st x, 2nd y, 3rd rotation
        # print('Received message on channel "%s"' % channel)
        # print("   value = %s" % str(msg.value))
        # print("")
        if msg.seq_number > self.seq_number_u:
            self.seq_number_u = msg.seq_number
            vel = np.array(msg.value)[:2]
            self._simulate_movement(vel)
        return

    def start(self):

        if self.use_lcm and not self.listen_thread.is_alive():
            print("robot simulation start listening")
            self.is_listening = True
            self.listen_thread.start()
        return

    def stop(self):
        if self.use_lcm and self.listen_thread.is_alive():
            self.is_listening = False
            # self.listen_thread.join()
            # create a new thread for the next simulation
            self.listen_thread = threading.Thread(target=self._listen, daemon=True)
        return

    @abc.abstractmethod
    def initialize_robot_position(self, pos: np.ndarray):
        raise NotImplementedError()

    @abc.abstractmethod
    def initialize_robot_position_in_formation(
        self, formation_center: np.ndarray, swarm_size: int
    ) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def initialize_robot_position_randomly(
        self, deployment_area: object, swarm_size: int
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_robot_state(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_current_velocity(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def set_load(self, load: float) -> None:
        raise NotImplementedError()


class Hera(RobotSimulation):
    def __init__(
        self,
        id: int,
        use_lcm: bool,
        ts_control: float,  # 0.2 seconds
        **kwargs,
    ) -> None:

        super().__init__(id, use_lcm, ts_control, **kwargs)

        # self.hera_state = [
        #     phi1,      # Rad1
        #     phi2,
        #     phi3,
        #     phi1dot,   # Winkelgeschwindigkeit Rad 1
        #     phi2dot,
        #     phi3dot,
        #     phiR,      # Rotationswinkel Roboterchassis
        #     x_c,       # Position x in Inertialsystem
        #     y_c]       # Position y in Inertialsystem
        self.hera_state = Array("f", list(np.zeros(9)), lock=True)
        self.current_velocity = Array("f", [0, 0], lock=True)

        # PID controller sampling time in seconds
        # DO NOT CHANGE: PID controller is adapted to this value
        self.ts_PID = 0.01
        if self.ts_PID > self.ts_control:
            print("PID controller sampling time larger than control sampling time.")
            self.ts_PID = self.ts_control
        assert (
            self.ts_control % self.ts_PID < 1e-10
        ), "ts_control must be a multiple of ts_PID"

        self.robot_radius = 0.29 / 2
        self.wheel_radius = 0.03  # r
        self.kinematic_radius = 0.115  # R
        # maximum moment
        self.m_max = 0.05
        # maximum velocity in meters per second
        self.max_vel = 0.4
        # additional load on the robot in kg
        self.load = 0

        self.num_substeps = int(self.ts_control / self.ts_PID)

        self.kp = 20 / 1000
        self.ki = 1 / 1000
        self.kd = 0

        self.error_sum = np.zeros(shape=(4))
        self.error_old = np.zeros(shape=(4))
        return

    def _simulate_movement(self, vel: np.ndarray) -> None:

        # limit control input to maximum admissible velocity (reduce velocity for both axes to maintain the direction of vel)
        max_abs_vel = np.max(np.abs(vel))
        if max_abs_vel > self.max_vel:
            vel = vel * (self.max_vel / max_abs_vel)

        # inner sampling loop for PID control, sampling time self.ts_PID
        for substep in range(self.num_substeps):

            with self.hera_state.get_lock():
                current_state = np.array(self.hera_state)
            # auxiliary PI control of robot propulsion forces with limited propulsion forces
            # formula and robot state numbering (*.x) as in dissertation of Henrik
            # desired velocity Rvcx and Rvcy in robot system
            v_des_x_r = vel[0] * np.cos(current_state[6]) + vel[1] * np.sin(
                current_state[6]
            )
            v_des_y_r = -vel[0] * np.sin(current_state[6]) + vel[1] * np.cos(
                current_state[6]
            )  # x[6]: phiR
            phi_r_dot_des = 0
            # Zshg Roboterschwerpunktsbewegung und Radbewegungen -> wie schnell sollen sich Räder drehen
            # (4.5) - (4.8)
            omega_wheel_des = np.array(
                [
                    (1 / self.wheel_radius)
                    * (v_des_x_r - v_des_y_r - self.kinematic_radius * phi_r_dot_des),
                    (1 / self.wheel_radius)
                    * (-v_des_x_r - v_des_y_r - self.kinematic_radius * phi_r_dot_des),
                    (1 / self.wheel_radius)
                    * (-v_des_x_r + v_des_y_r - self.kinematic_radius * phi_r_dot_des),
                    (1 / self.wheel_radius)
                    * (v_des_x_r + v_des_y_r - self.kinematic_radius * phi_r_dot_des),
                ]
            )
            if self.wheel_is_lagging:
                omega_wheel_des[self.lagging_wheel] *= self.wheel_lag

            # Fehler zu tatsächlicher Radbewegung, x(3): phi_1
            error_cur = omega_wheel_des - np.array(
                [
                    current_state[3],
                    current_state[4],
                    current_state[5],
                    current_state[3] + current_state[5] - current_state[4],
                ]
            )
            error_sum_temp = self.error_sum + error_cur
            temp_u_moment_val = (
                self.kp * error_cur
                + self.ki * self.ts_PID * error_sum_temp
                + self.kd * (error_cur - self.error_old) / self.ts_PID
            )
            # Index welcher Momente überschreiten Maximum unter Bedingung der richtigen Richtung
            anti_windup_selector = (
                (temp_u_moment_val > self.m_max) | (temp_u_moment_val < -self.m_max)
            ) & ((temp_u_moment_val * error_cur) > 0)

            # Regelfehler von Motoren aufsummieren, die nicht m_max überschreiten
            self.error_sum = self.error_sum + error_cur * np.logical_not(
                anti_windup_selector
            )

            self.u_moment = (
                self.kp * error_cur
                + self.ki * self.ts_PID * self.error_sum
                + self.kd * (error_cur - self.error_old) / self.ts_PID
            )
            # limit moments
            self.u_moment = np.clip(self.u_moment, a_min=-self.m_max, a_max=self.m_max)
            self.error_old = error_cur
            # Motorregler regelt 100 x pro Sekunde
            stv_prev = 0  # simTimeVec(t-1)
            t0 = stv_prev + self.ts_PID * substep
            t1 = stv_prev + self.ts_PID * (substep + 1)

            fun = lambda t, y: self._sim_ode_fun(y)
            res = solve_ivp(
                fun,
                [t0, t1],
                y0=current_state,
                method="RK45",
                t_eval=None,
                dense_output=False,
                events=None,
                vectorized=False,
                args=None,
                rtol=1e-6,
                atol=1e-8,
                max_step=5e-3,
            )

            with self.hera_state.get_lock():
                self.hera_state[:] = res.y[:, -1]

        return

    def _sim_ode_fun(self, x):

        # Hera model parameters:
        m_r = 0.075
        m_c = 2.3 - 4 * m_r + self.load

        # estimates:
        I_y = 0.5 * m_r * self.wheel_radius**2
        I_i = (m_r / 12) * (3 * self.wheel_radius**2 + 0.03**2)
        I_c = 0.5 * m_c * 0.145**2

        # no external forces
        applied_force = np.zeros(shape=(2, 1))
        # can be used for contact forces, e.g., between robot and an object

        dxdt = self._hera_model_rhs(
            x,
            self.u_moment,
            applied_force,
            m_c,
            m_r,
            self.wheel_radius,
            self.kinematic_radius,
            I_y,
            I_i,
            I_c,
        )
        return dxdt

    def _hera_model_rhs(self, x, Ma, Fa, m_c, m_r, r, R, I_y, I, I_c):
        """
        Ma  : vector, [4,1], Motor moments
        Fa  : vector of applied forces, [2,1], inertial frame of reference (e.g.
            through contact)
        m_c: mass of the robot chassis (without wheels)
        m_r: mass of one wheel
        r  : wheel radius
        R  : radius from chassis center to projection of wheel contact point
        I_y: moment of inertia about the y-axis of the wheel coordinate system
            ('normal' rolling wheel rotation)
        I  : moment of inertia about other two principal axes (assuming wheel to
            be symmetric in that regard)
        I_c: moment of inertia of the chassis, rotation around the z-axis
            ('turning' / yaw)
        """

        dxdt = np.zeros(shape=(9))

        m_s = m_c + 4 * m_r  # p.60, m_t
        I_0 = I_c + 4 * I + 4 * m_r * R**2  # p.60, J_t
        A = (m_s * (r**3)) / (8 * R * (m_s * (r**2)) / 4 + I_y)  # (4.21)
        B = 1 / (2 * (m_s * r**2 / 4 + I_y))  # (4.21)
        C = 1 / (2 * (I_0 * r**2 / (4 * R**2)) + I_y)  # (4.22)

        phi1dot = x[3]
        phi2dot = x[4]
        phi3dot = x[5]
        phi_R = x[6]

        dxdt[0:3] = x[3:6]

        # äußere wirkende Kraft (unwichtig, da Fa = 0)
        F_rot_x = np.cos(phi_R) * Fa[0] + np.sin(phi_R) * Fa[1]
        F_rot_y = -np.sin(phi_R) * Fa[0] + np.cos(phi_R) * Fa[1]

        # from equations of motion
        temp0 = (A / 2) * (phi1dot + phi3dot)
        temp1 = B + C / 2
        temp2 = C / 2 - B
        temp3 = C / 2
        temp4 = B * r / 2

        dxdt[3] = (
            temp0 * (-phi1dot + 2 * phi2dot - phi3dot)
            + temp1 * Ma[0]
            + temp3 * (Ma[1] + Ma[3])
            + temp2 * Ma[2]
            + temp4 * (F_rot_x - F_rot_y).item()
        )
        dxdt[4] = (
            temp0 * (phi3dot - phi1dot)
            + temp1 * Ma[1]
            + temp2 * Ma[3]
            + temp3 * (Ma[0] + Ma[2])
            - temp4 * (F_rot_x + F_rot_y).item()
        )
        dxdt[5] = (
            temp0 * (phi1dot - 2 * phi2dot + phi3dot)
            + temp1 * Ma[2]
            + temp2 * Ma[0]
            + temp3 * (Ma[1] + Ma[3])
            + temp4 * (F_rot_y - F_rot_x).item()
        )

        # from kinematics
        dxdt[6] = -(r / (2 * R)) * (phi1dot + phi3dot)
        Rv_C_x = (r / 2) * (phi1dot - phi2dot)
        Rv_C_y = (r / 2) * (phi3dot - phi2dot)
        dxdt[7] = np.cos(phi_R) * Rv_C_x - np.sin(phi_R) * Rv_C_y
        dxdt[8] = np.sin(phi_R) * Rv_C_x + np.cos(phi_R) * Rv_C_y

        with self.current_velocity.get_lock():
            self.current_velocity[:] = np.array([dxdt[7], dxdt[8]])
        return dxdt

    def initialize_robot_position(self, pos: np.ndarray):
        with self.hera_state.get_lock():
            self.hera_state[-2:] = pos
        return

    def initialize_robot_position_in_formation(
        self, formation_center: np.ndarray, swarm_size: int
    ) -> None:
        """Initialize the robot position within the deployment area as part of a circular formation.

        Parameters
        ----------
        deployment_area : object
        """
        margin = 0.7  # between robots
        formation_radius = np.sqrt((margin * swarm_size) / (2 * np.pi))

        # xy coordinates sampled on a circle around (0,0)
        pos_on_circle = 2 * np.pi / swarm_size * self.id
        pos = (
            np.array([np.cos(pos_on_circle), np.sin(pos_on_circle)]) * formation_radius
        )
        pos = pos + formation_center

        # add noise, verify that the noisy position is within the deployment area
        max_noise = 0.1
        noisy_pos = pos + np.random.normal(0, max_noise, 2)

        self.initialize_robot_position(noisy_pos)
        return

    def initialize_robot_position_randomly(
        self, deployment_area: object, swarm_size: int
    ) -> None:
        """Randomly initialize the robot position such that it lies within the deployment area.

        Parameters
        ----------
        deployment_area : object
        """
        assert type(deployment_area) is Warehouse
        all_start_points = deployment_area.waypoints
        waypoints_per_robot = int(len(all_start_points) / swarm_size)

        random_waypoint = np.random.choice(
            all_start_points[
                self.id * waypoints_per_robot : (self.id + 1) * waypoints_per_robot
            ]
        )
        start_pos = deployment_area.get_coords(random_waypoint)
        self.initialize_robot_position(start_pos)

        return

    def get_robot_state(self) -> np.ndarray:
        with self.hera_state.get_lock():
            current_state = np.array(self.hera_state)
        return current_state

    def get_current_velocity(self) -> np.ndarray:
        with self.current_velocity.get_lock():
            current_velocity = np.array(self.current_velocity)
        return current_velocity

    def set_load(self, load: float) -> None:
        self.load = load
        return
