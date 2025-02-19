from manim import *
import numpy as np


class BlochSphereTrajectory(ThreeDScene):
    def construct(self):
        # Imposta gli assi 3D
        axes = ThreeDAxes()

        # Crea la sfera di Bloch
        sphere = Sphere(radius=1.5, color=BLUE, fill_opacity=0.1, resolution=(10, 10))

        # Etichette |0⟩ e |1⟩ (senza LaTeX per evitare problemi)
        ket_0 = Text("|0⟩").move_to([0, 0, 1.7]).scale(1.2).set_color(WHITE)
        ket_1 = Text("|1⟩").move_to([0, 0, -1.7]).scale(1.2).set_color(WHITE)

        # Definiamo la traiettoria simulata
        t_vals = np.linspace(0, 2 * np.pi, 100)
        x_vals = np.sin(t_vals)
        y_vals = np.zeros_like(t_vals)
        z_vals = np.cos(t_vals)

        # Crea il vettore iniziale
        vector = Arrow3D(
            start=[0, 0, 0],
            end=[x_vals[0], y_vals[0], z_vals[0]],
            color=RED,
            stroke_width=4,
        )

        # Creiamo una scia della traiettoria con almeno un punto iniziale
        trail = VMobject(color=YELLOW)
        trail.set_points_as_corners(
            [vector.get_end(), vector.get_end()]
        )  # ✅ FIX: inizializziamo con 2 punti uguali

        # Funzione di aggiornamento per il vettore
        def update_vector(mob, alpha):
            index = int(alpha * (len(t_vals) - 1))  # Trova il frame attuale
            new_pos = np.array([x_vals[index], y_vals[index], z_vals[index]])
            mob.put_start_and_end_on([0, 0, 0], new_pos)

            # ✅ FIX: Aggiungiamo punti invece di rimpiazzare l'oggetto
            trail.add_points_as_corners([new_pos])

        # Anima l'evoluzione del vettore lungo la traiettoria
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes, sphere, ket_0, ket_1, vector, trail)
        self.play(
            UpdateFromAlphaFunc(vector, update_vector, run_time=5, rate_func=smooth)
        )

        # Mantiene la scena
        self.wait(2)
