import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Base class for mechanical systems
class MechanicalSystem:
    def __init__(self, initial_conditions, time_span):
        self.initial_conditions = initial_conditions
        self.time_span = time_span
    
    def equations(self, y, t):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def simulate(self):
        time_points = np.linspace(0, self.time_span, 1000)
        solution = odeint(self.equations, self.initial_conditions, time_points)
        return time_points, solution

# Spring-Mass-Damper System
class SpringMassDamper(MechanicalSystem):
    def __init__(self, m, c, k, initial_conditions, time_span):
        super().__init__(initial_conditions, time_span)
        self.m = m
        self.c = c
        self.k = k
    
    def equations(self, y, t):
        x, v = y
        dxdt = v
        dvdt = (-self.c * v - self.k * x) / self.m
        return [dxdt, dvdt]

# Simple Pendulum
class SimplePendulum(MechanicalSystem):
    def __init__(self, l, g, initial_conditions, time_span):
        super().__init__(initial_conditions, time_span)
        self.l = l
        self.g = g
    
    def equations(self, y, t):
        theta, omega = y
        dthetadt = omega
        domegadt = -(self.g / self.l) * np.sin(theta)
        return [dthetadt, domegadt]

# Rotational Inertia System
class RotationalInertia(MechanicalSystem):
    def __init__(self, I, torque, initial_conditions, time_span):
        super().__init__(initial_conditions, time_span)
        self.I = I
        self.torque = torque
    
    def equations(self, y, t):
        theta, omega = y
        dthetadt = omega
        domegadt = self.torque / self.I
        return [dthetadt, domegadt]

# Function to plot the results
def plot_results(system, title):
    time_points, solution = system.simulate()
    position = solution[:, 0]
    velocity = solution[:, 1]

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_points, position)
    plt.title(f'{title} - Position vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')

    plt.subplot(2, 1, 2)
    plt.plot(time_points, velocity)
    plt.title(f'{title} - Velocity vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')

    plt.tight_layout()
    plt.show()

%run MechanicalSystem.ipynb

class MechanicalSystemTest:
    def __init__(self):
        self.systems = []

    def add_system(self, system, title):
        self.systems.append((system, title))

    def run_tests(self):
        for system, title in self.systems:
            plot_results(system, title)

# Main script to create instances, add them to the test, and run the tests
if __name__ == "__main__":
    test = MechanicalSystemTest()

    # Spring-Mass-Damper System
    spring_mass_damper = SpringMassDamper(m=1.0, c=0.5, k=2.0, initial_conditions=[1.0, 0.0], time_span=20)
    test.add_system(spring_mass_damper, "Spring-Mass-Damper System")

    # Simple Pendulum
    simple_pendulum = SimplePendulum(l=1.0, g=9.81, initial_conditions=[np.pi / 4, 0.0], time_span=20)
    test.add_system(simple_pendulum, "Simple Pendulum")

    # Rotational Inertia System
    rotational_inertia = RotationalInertia(I=1.0, torque=1.0, initial_conditions=[0.0, 0.0], time_span=20)
    test.add_system(rotational_inertia, "Rotational Inertia System")

    # Run all tests
    test.run_tests()
