# Grid-Boid-NumPy

A NumPy-based simulation of [Boids](https://en.wikipedia.org/wiki/Boids) flocking behavior, visualized on a discrete grid. 

## Flocking Rules  

Boids follow three simple rules to exhibit emergent flocking behavior:  

1. **Separation** – Avoid crowding neighbors.  
2. **Alignment** – Match the average heading of nearby Boids.  
3. **Cohesion** – Move toward the center of the local group.  

## Implementation  

The Boids are initialized with random positions and velocities. At each time step each Boid considers all others within a radius $r$, computing:  

- The **center of mass** for cohesion.  
- The **average velocity** for alignment.  
- A **repulsion force** for **separation**, which is inversely proportional to distance from nearby Boids.  

These forces are applied in a continuous 2D space $\\{(x,y) \in \mathbb{R} \mid -1 \leq x \leq 1, -1 \leq y \leq 1\\}$ with periodic boundary conditions. The system is then projected onto a discrete grid $\\{(i,j) \in \mathbb{N} \mid 0 \leq i < n, 0 \leq j < m\\}$. The simulation uses vectorized NumPy operations for efficiency. 

The simulation generates an animated Pyplot visualization and saves it as a file `boids.gif`.

## Visualization

- **Default Visualization**: A smooth circular gradient around each Boid's grid cell, using the `Purples` colormap:
  
![boids_purp](https://github.com/user-attachments/assets/004ae386-33ba-4293-b46e-6dc83e9903c0)

- **Optional Visualization**: A high-contrast black-and-white grid with no gradient, providing a clearer view of the Boids' movements:

![boids](https://github.com/user-attachments/assets/fe18f1f5-57df-4b6b-9514-823501e6c587)

User can toggle between these visualization modes by specifying a flag when running the simulation.

## Technologies

This project is built using the following technologies:  

- **[NumPy](https://numpy.org/) (v1.26.4)**
- **[Matplotlib](https://matplotlib.org/) (v3.9.2)** 

## How to Run

### 1. Install Dependencies

First, install the required dependencies:
```
pip install -r requirements.txt
```

### 2. Run the Application

Then, run the program with the following command:
```
python3 main.py
```

### 3. Customize Parameters (Optional)

You can customize the number of boids, strength multipliers for flocking behavior (alignment, cohesion, separation), and grid shape via an interactive interface. After running the simulation, the animation will be generated and saved in the same directory.
