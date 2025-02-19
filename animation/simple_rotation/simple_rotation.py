from manim import *


class BlochSphereRotation(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(x_length=5, y_length=5, z_length=5)

        sphere = Sphere(radius=2, color=BLUE, fill_opacity=0.1, resolution=(6, 6))

        ket_0 = (
            MathTex(r"|0\rangle")
            .move_to([0, 0, 3])
            .scale(1.2)
            .set_color(WHITE)
            .rotate(90 * DEGREES, [1, 0, 0])
            .rotate(135 * DEGREES, [0, 0, 1])
        )
        ket_1 = (
            MathTex(r"|1\rangle")
            .move_to([0, 0, -3])
            .scale(1.2)
            .set_color(WHITE)
            .rotate(90 * DEGREES, [1, 0, 0])
            .rotate(135 * DEGREES, [0, 0, 1])
        )

        vector = Arrow3D(start=[0, 0, 0], end=[0, 0, 2], color=RED)

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        self.add(axes, sphere, vector, ket_0, ket_1)

        self.play(
            Rotate(vector, angle=2 * PI, axis=[1, 1, 0], about_point=ORIGIN, run_time=5)
        )

        self.wait(2)
