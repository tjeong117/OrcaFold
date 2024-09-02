import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def generate_random_protein_backbone(n_residues, step_size=3.8):
    """Generate a random protein backbone structure."""
    phi = np.random.uniform(-np.pi, np.pi, n_residues)
    psi = np.random.uniform(-np.pi, np.pi, n_residues)

    x, y, z = [0], [0], [0]
    for i in range(1, n_residues):
        dx = step_size * np.cos(phi[i]) * np.sin(psi[i])
        dy = step_size * np.sin(phi[i]) * np.sin(psi[i])
        dz = step_size * np.cos(psi[i])
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
        z.append(z[-1] + dz)

    return np.array(x), np.array(y), np.array(z)


# Generate a random protein backbone
n_residues = 100
x, y, z = generate_random_protein_backbone(n_residues)

# Set up the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Initialize an empty line
line, = ax.plot([], [], [], color='b', alpha=0.7, linewidth=2)
point, = ax.plot([], [], [], 'ro', markersize=8)

# Set axis limits
ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))
ax.set_zlim(min(z), max(z))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Protein Backbone Structure Formation')


def init():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    return line, point


def animate(i):
    line.set_data(x[:i], y[:i])
    line.set_3d_properties(z[:i])
    point.set_data(x[i - 1:i], y[i - 1:i])
    point.set_3d_properties(z[i - 1:i])
    ax.view_init(elev=10., azim=i)
    return line, point


# Create the animation
anim = FuncAnimation(fig, animate, init_func=init, frames=n_residues,
                     interval=50, blit=False, repeat=True)

plt.tight_layout()
plt.show()

# Uncomment the following line to save the animation as a gif
# anim.save('protein_folding.gif', writer='pillow', fps=30)