import openmm as mm
import openmm.app as app
from openmm import unit
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a system with two particles
system = mm.System()

# Add two particles to the system
system.addParticle(mass=1.0 * unit.amu)
system.addParticle(mass=1.0 * unit.amu)

# Set up nonbonded force (which includes electrostatics)
nonbonded_force = mm.NonbondedForce()
system.addForce(nonbonded_force)

# Add particles to the nonbonded force
# Parameters: charge, sigma, epsilon
nonbonded_force.addParticle(1.0, 1.0, 0.0)  # Particle 1: positive charge
nonbonded_force.addParticle(-1.0, 1.0, 0.0)  # Particle 2: negative charge

# Create a Langevin integrator
integrator = mm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)

# Create a simulation context
platform = mm.Platform.getPlatformByName('Reference')
context = mm.Context(system, integrator, platform)

# Set initial positions (in nanometers)
initial_positions = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0]
]) * unit.nanometers

context.setPositions(initial_positions)

# Run simulation and collect data
n_steps = 1000
positions = []
energies = []

for step in range(n_steps):
    integrator.step(1)
    state = context.getState(getPositions=True, getEnergy=True)
    positions.append(state.getPositions(asNumpy=True).value_in_unit(unit.nanometers))
    energies.append(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))

# Convert positions to numpy array for easier manipulation
positions = np.array(positions)

# Plot trajectory
fig = plt.figure(figsize=(12, 5))

# 3D trajectory plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(positions[:, 0, 0], positions[:, 0, 1], positions[:, 0, 2], label='Particle 1')
ax1.plot(positions[:, 1, 0], positions[:, 1, 1], positions[:, 1, 2], label='Particle 2')
ax1.set_xlabel('X (nm)')
ax1.set_ylabel('Y (nm)')
ax1.set_zlabel('Z (nm)')
ax1.set_title('Particle Trajectories')
ax1.legend()

# Energy plot
ax2 = fig.add_subplot(122)
ax2.plot(energies)
ax2.set_xlabel('Simulation Step')
ax2.set_ylabel('Potential Energy (kJ/mol)')
ax2.set_title('System Potential Energy')

plt.tight_layout()
plt.show()