from manim import (
    ThreeDScene,
    ThreeDAxes,
    Sphere,
    Text,
    WHITE,
    BLUE,
    RED,
    YELLOW,
    DEGREES,
    Arrow3D,
    VMobject,
    UpdateFromAlphaFunc,
    smooth,
    linear,
    Mobject,
)
import numpy as np
import pickle
import os


class BlochSphereTrajectory(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()

        sphere = Sphere(radius=1.5, color=BLUE, fill_opacity=0.1, resolution=(10, 10))

        ket_0 = Text("|0⟩").move_to([0, 0, 1.7]).scale(1.2).set_color(WHITE)
        ket_1 = Text("|1⟩").move_to([0, 0, -1.7]).scale(1.2).set_color(WHITE)

        with open(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "trajectories",
                "ramsey_1.pickle",
            ),
            "rb",
        ) as f:
            datadict = pickle.load(f)
        c_timetrace = datadict["c_timetrace"]
        phase = np.mod(datadict["phase_decimate"], 2 * np.pi)
        counterphase = np.exp(-1j * phase)
        ae = c_timetrace[0, :] * counterphase
        ag = c_timetrace[2, :] * counterphase

        # t_vals = np.linspace(0, 2 * np.pi, 100)
        x_vals = np.real(np.conj(ag) * ae + ag * np.conj(ae))
        y_vals = np.real(1j * (np.conj(ag) * ae - ag * np.conj(ae)))
        z_vals = np.real(ae * np.conj(ae) - ag * np.conj(ag))

        vector = Arrow3D(
            start=[0, 0, 0],
            end=[x_vals[0], y_vals[0], z_vals[0]],
            color=RED,
            stroke_width=4,
        )

        trail = VMobject(color=YELLOW)
        trail.set_points_as_corners([vector.get_end(), vector.get_end()])

        def update_vector(mob: Mobject, alpha: float):
            index = int(alpha * (len(x_vals) - 1))  # Trova il frame attuale
            new_pos = np.array([x_vals[index], y_vals[index], z_vals[index]])
            mob.put_start_and_end_on([0, 0, 0], new_pos)

            trail.add_points_as_corners([new_pos])

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes, sphere, ket_0, ket_1, vector, trail)

        self.wait(1)
        self.play(
            UpdateFromAlphaFunc(
                vector, update_vector, run_time=5, rate_func=smooth
            )  # possibly use linear
        )

        self.wait(2)
