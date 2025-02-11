"""
This script simulates a flock of boids and generates an animation of their movement.
The simulation applies basic flocking behaviors like alignment, cohesion, and separation.

The boids' movement is visualized in a 2D grid, and the resulting animation is saved as a GIF file in the current working directory.

Imports:
    - numpy as np: Provides numerical operations, including random number generation and array manipulation.
    - matplotlib.pyplot as plt: Used for creating the plot and visualization of the simulation.
    - matplotlib.animation as animation: Used for creating and saving animations.
    - os: Used to handle file paths and retrieve the current working directory.

Functions:
    - interface() -> dict: Prompts the user for simulation parameters (number of boids, flocking strength, grid shape, and visual preferences).
    - init_boids(N) -> np.ndarray: Initializes an array of N boids with random positions, velocities, and zero acceleration.
    - flock(boids, strengths) -> np.ndarray: Applies flocking behaviors (alignment, cohesion, separation) to the boids.
    - step(boids, flock_strengths) -> np.ndarray: Advances the simulation one time step, applying movement and boundary conditions.
    - run_sim(N, flock_strengths, steps) -> list: Runs the boid simulation for the specified number of steps and returns the positions at each step.
    - create_animation(frames, grid_shape, high_contrast=False) -> animation.FuncAnimation: Creates an animation of the boid simulation.
    - main() -> None: Main function that coordinates the simulation setup, execution, animation creation, and saving.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def init_boids(N):
    """
    Initializes an array of N boids with random positions, velocities, and zero acceleration.

    Args:
        N (int): The number of boids to initialize.

    Returns:
        np.ndarray: A structured NumPy array containing N boids, where each boid has:
                    - 'pos' : (float, float) -> The x and y coordinates, randomly initialized in the range [-1, 1].
                    - 'vel' : (float, float) -> The velocity components (vx, vy), randomly initialized in the range [-0.1, 0.1]*100/N.
                    - 'acc' : (float, float) -> The acceleration components (ax, ay), initialized to (0,0).
    """
    
    pos = np.random.uniform(low=-1, high=1, size=(N,2))
    vel = np.random.uniform(low=-0.1, high=0.1, size=(N,2))*100/N
    acc = np.zeros((N,2))

    # Define a structured array with named fields
    dtype = [('pos', '2float'), ('vel', '2float'), ('acc', '2float')]
    boids = np.zeros(N, dtype=dtype)

    boids['pos'] = pos
    boids['vel'] = vel
    boids['acc'] = acc

    return boids

def flock(boids, strengths):
    """
    Applies flocking behavior to the boids based on alignment, cohesion, and separation.

    Args:
        boids (np.ndarray): A structured NumPy array of boids, where each boid has:
                            - 'pos' (float, float): The x and y coordinates.
                            - 'vel' (float, float): The velocity components (vx, vy).
                            - 'acc' (float, float): The acceleration components (ax, ay).
        strengths (tuple):  Strength coefficients for the flocking rules:
                            - strengths[0]: Alignment strength.
                            - strengths[1]: Cohesion strength.
                            - strengths[2]: Separation strength.

    Returns:
        np.ndarray: Updated array of boids with new acceleration values based on flocking behavior.
    """
    positions = np.array([boid["pos"] for boid in boids])
    velocities = np.array([boid["vel"] for boid in boids])

    # Compute pairwise distances between boids
    dist_matrix = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)

    # Define neighborhoods (excluding self)
    neighbors_mask = dist_matrix < 0.1
    np.fill_diagonal(neighbors_mask, False)

    for i in range(len(boids)):
        neighbors = np.where(neighbors_mask[i])[0]
        if len(neighbors) > 0:
            # Alignment: Steer towards the average velocity of neighbors
            avg_velocity = np.mean(velocities[neighbors], axis=0)
            align_force = strengths[0] * (avg_velocity - velocities[i])

            # Cohesion: Steer towards the average position of neighbors
            avg_position = np.mean(positions[neighbors], axis=0)
            cohesion_force = strengths[1] * (avg_position - positions[i]) / 25

            # Separation: Steer away from nearby boids
            diff = positions[i] - positions[neighbors]
            distances = np.clip(dist_matrix[i, neighbors][:, np.newaxis], 0.01, None)
            separation_force = strengths[2] * np.sum(diff / distances, axis=0)

            # Update acceleration
            boids[i]["acc"] = align_force + cohesion_force + separation_force

    return boids

def step(boids, flock_strengths):
    """
    Advances the simulation by one time step, applying flocking behavior and enforcing continuous boundaries.

    Args:
        boids (np.ndarray): A structured NumPy array of boids, where each boid has:
                            - 'pos' (float, float): The x and y coordinates.
                            - 'vel' (float, float): The velocity components (vx, vy).
                            - 'acc' (float, float): The acceleration components (ax, ay).
        flock_strengths (tuple): Strength coefficients for the flocking rules:
                                 - flock_strengths[0]: Alignment strength.
                                 - flock_strengths[1]: Cohesion strength.
                                 - flock_strengths[2]: Separation strength.

    Returns:
        np.ndarray: Updated array of boids after one simulation step, with new positions and velocities.
    """
    # Apply flocking behavior
    boids = flock(boids, flock_strengths)
    
    # Integrate equations of motion
    boids["pos"] += boids["vel"]
    boids["vel"] += boids["acc"]
    
    # Enforce continuous boundary conditions
    mask_x_left = boids["pos"][:, 0] < -1.0
    mask_x_right = boids["pos"][:, 0] > 1.0
    mask_y_bottom = boids["pos"][:, 1] < -1.0
    mask_y_top = boids["pos"][:, 1] > 1.0

    # Teleport boids to the opposite side when crossing boundaries
    boids["pos"][:, 0][mask_x_left] = 1.0  
    boids["pos"][:, 0][mask_x_right] = -1.0
    boids["pos"][:, 1][mask_y_bottom] = 1.0
    boids["pos"][:, 1][mask_y_top] = -1.0

    return boids

def run_sim(N, flock_strengths, steps):
    """
    Runs the boid simulation for a given number of time steps.

    Args:
        N (int): The number of boids in the simulation.
        flock_strengths (tuple): Strength coefficients for the flocking rules:
                                 - flock_strengths[0]: Alignment strength.
                                 - flock_strengths[1]: Cohesion strength.
                                 - flock_strengths[2]: Separation strength.
        steps (int): The number of time steps to run the simulation.

    Returns:
        list of np.ndarray: A list of NumPy arrays, where each array contains the positions of all boids at a given time step.
    """
    boids = init_boids(N)
    frames = []

    for _ in range(steps):
        frames.append(np.copy(boids["pos"]))  # Store positions for visualization
        boids = step(boids, flock_strengths)  # Update boid positions and velocities

    return frames

def create_animation(frames, grid_shape, high_contrast=False):
    """
    Creates an animated visualization of the boid simulation on a discrete grid.

    Args:
        frames (list): A list of NumPy arrays containing boid positions at each time step.
        grid_shape (tuple): The shape of the grid (n, m) used for visualization.
        high_contrast (bool, optional): If True, uses a high-contrast (black and white) colormap. Defaults to False.

    Returns:
        matplotlib.animation.FuncAnimation: An animation object displaying the boid movement over time.
    """
    n, m = grid_shape
    grid = np.ones((n, m))
    boid_rad = 2  # Boid visualization radius

    def continuous_to_grid(pos, n, m):
        """
        Converts continuous (x, y) coordinates into discrete grid indices.

        Args:
            pos (np.ndarray): Array of shape (N, 2) with boid positions in the range [-1, 1].
            n (int): Number of rows in the grid.
            m (int): Number of columns in the grid.

        Returns:
            tuple of np.ndarray: Two arrays containing row and column indices in the grid.
        """
        x, y = pos[:, 0], pos[:, 1]
        c = ((x + 1) / 2) * (m - 1)  # Normalize x to grid columns
        r = ((1 - y) / 2) * (n - 1)  # Normalize y to grid rows
        c = np.clip(np.round(c), 0, m - 1).astype(int)
        r = np.clip(np.round(r), 0, n - 1).astype(int)

        return r, c

    # Initialize grid with the first frame
    r, c = continuous_to_grid(frames[0], n, m)
    N = frames[0].shape[0]
    grid[r, c] = 0

    fig, ax = plt.subplots()

    cmap = "Purples" if not high_contrast else "gray"
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, origin="upper")

    def update(i, frames):
        """
        Updates the grid for each frame in the animation.

        Args:
            i (int): The current frame index.
            frames (list of np.ndarray): The list of boid positions over time.

        Returns:
            list: Updated image array.
        """
        grid = np.ones((n, m))
        r, c = continuous_to_grid(frames[i], n, m)

        for j in range(N):
            ri, ci = r[j], c[j]
            if high_contrast:
                grid[ri, ci] = 0
            else:
                y, x = np.indices((n, m))
                distances = np.sqrt((x - ci)**2 + (y - ri)**2)
                normalized_distances = np.clip(distances / boid_rad, 0, 1)
                grid[distances <= boid_rad] = normalized_distances[distances <= boid_rad] + 0.2

        im.set_array(grid)
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), fargs=(frames,), interval=1)
    return anim

def interface():
    """
    Provides a user interface to set simulation parameters.

    Prompts the user to customize parameters or use default values. 
    Ensures valid inputs through error handling.

    Returns:
        dict: A dictionary containing simulation parameters:
              - 'N' (int): Number of boids (default: 200, max: 2000).
              - 'strength_mult' (tuple): Strength multipliers for (alignment, cohesion, separation).
              - 'grid_shape' (tuple): Grid dimensions (rows, columns).
              - 'high_contrast' (bool): Whether to use high-contrast visuals.
    """
    # Default values
    params = {
        "N": 200,
        "strength_mult": (1.0, 1.0, 1.0),
        "grid_shape": (100, 100),
        "high_contrast": False,
    }

    # Custom parameter input
    set_params = input("Set custom parameters? (Y/N): ").strip().lower()
    if set_params in ["y", "yes"]:
        # Number of boids
        while True:
            try:
                N = int(input("Number of Boids (<= 2000): ").strip())
                if 0 < N <= 2000:
                    params["N"] = N
                    break
                else:
                    print("Please enter a number between 1 and 2000.")
            except ValueError:
                print("Invalid input. Enter an integer.")

        # Strength multipliers
        while True:
            try:
                strength_mult = tuple(
                    map(float, input("Flocking strength multipliers (alignment, cohesion, separation): ").strip().split(","))
                )
                if len(strength_mult) == 3:
                    params["strength_mult"] = strength_mult
                    break
                else:
                    print("Please enter three comma-separated values.")
            except ValueError:
                print("Invalid input. Enter three numerical values separated by commas.")

        # Grid shape
        while True:
            try:
                grid_shape = tuple(
                    map(int, input("Grid shape (rows, columns): ").strip().split(","))
                )
                if len(grid_shape) == 2 and all(i > 0 for i in grid_shape):
                    params["grid_shape"] = grid_shape
                    break
                else:
                    print("Please enter two positive integers separated by a comma.")
            except ValueError:
                print("Invalid input. Enter two positive integers separated by a comma.")

    # High contrast visuals
    select_high_contrast = input("High contrast visuals? (Y/N): ").strip().lower()
    params["high_contrast"] = select_high_contrast in ["y", "yes"]

    return params

def main():
    """
    Runs the boid simulation and saves the animation to a GIF file.

    - Prompts the user for simulation parameters.
    - Runs the simulation for 1000 time steps.
    - Creates an animation of the boid movement.
    - Saves the animation as 'boids.gif' using PillowWriter.
    
    The animation is displayed before saving.
    """
    params = interface()
    flock_strengths = np.array([0.25, 0.15, 0.00015]) * np.array(params["strength_mult"])
    
    frames = run_sim(params["N"], flock_strengths, 2000)
    anim = create_animation(frames, params["grid_shape"], params["high_contrast"])
    
    plt.show()  # Display animation before saving
    
    # Get current working directory
    current_directory = os.getcwd()
    print("Current working directory:", current_directory)
    
    # Save animation as GIF using PillowWriter in the current directory
    writergif = animation.PillowWriter(fps=30)
    print("Saving animation...")
    
    # Save the file in the current directory
    filename = os.path.join(current_directory, 'boids.gif')
    anim.save(filename, writer=writergif)
    
    print(f"Animation saved as {filename}!")
    
    plt.close()

if __name__ == "__main__":
    main()
