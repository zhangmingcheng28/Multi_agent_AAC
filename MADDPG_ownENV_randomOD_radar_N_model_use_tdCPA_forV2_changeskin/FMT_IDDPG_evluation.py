#_________________________________________________________evaluations__________________________________________________
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import distance
import time
import matplotlib
import pickle
import os


# Function to check if a point is inside an obstacle
def is_in_obstacle(point, obstacle_pos, size):
    ox, oy = obstacle_pos
    return (point[0] - ox) ** 2 + (point[1] - oy) ** 2 <= size ** 2


# Function to check if a point is inside a nearby aircraft (treated as a dynamic obstacle)
def is_nearby_aircraft(point, aircraft_pos, avoidance_radius=10):
    return np.linalg.norm(np.array(point) - np.array(aircraft_pos)) < avoidance_radius


# Function to find the nearest neighbor nodes to a node
def nearest_neighbors(nodes, target_node, radius):
    distances = distance.cdist(nodes, [target_node])
    return np.where(distances <= radius)[0]


# Function to generate random nodes, avoiding obstacles and nearby aircraft
def generate_random_nodes(num_nodes, x_lim, y_lim, obstacle_pos, obstacle_size, other_aircraft_positions):
    nodes = []
    while len(nodes) < num_nodes:
        x = random.uniform(x_lim[0], x_lim[1])
        y = random.uniform(x_lim[0], x_lim[1])
        if not is_in_obstacle([x, y], obstacle_pos, obstacle_size) and \
                not any(is_nearby_aircraft([x, y], ac_pos) for ac_pos in other_aircraft_positions):
            nodes.append([x, y])
    return np.array(nodes)


# Function to check if there is a collision-free path between two points
def is_collision_free(node1, node2, obstacle_pos, obstacle_size, other_aircraft_positions, step_size=0.05):
    steps = int(np.linalg.norm(np.array(node2) - np.array(node1)) / step_size)
    for i in range(steps):
        interp = node1 + (i / steps) * (np.array(node2) - np.array(node1))
        if is_in_obstacle(interp, obstacle_pos, obstacle_size) or \
                any(is_nearby_aircraft(interp, ac_pos) for ac_pos in other_aircraft_positions):
            return False
    return True


# Simulate obstacle moving 1 unit per timestep
def dynamic_obstacle_movement(start_pos, end_pos, t, obstacle_speed=1):
    total_distance = np.linalg.norm(end_pos - start_pos)
    direction = (end_pos - start_pos) / total_distance
    step_distance = obstacle_speed  # Move 1 unit per timestep
    return start_pos + direction * step_distance * t


# Saving the simulation results to pickle
def save_simulation_results(aircraft_trajectories, obstacle_trajectory):
    # Get the current system time to create a unique filename
    current_time = time.strftime("%Y%m%d-%H%M%S")

    # Create a dictionary to store the results
    simu_results = {
        'aircraft_trajectories': aircraft_trajectories,
        'obstacle_trajectory': obstacle_trajectory,
    }

    # Define the directory path and filename
    directory = r'D:/FMT_vs_IDDPG/data'
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = f'simu_results_{current_time}.pkl'
    file_path = os.path.join(directory, filename)

    # Save the results into a pickle file at the specified path
    with open(file_path, 'wb') as file:
        pickle.dump(simu_results, file)

    print(f"Simulation results saved to {file_path}")


# Replan the path at each timestep, considering the dynamic obstacle and other aircraft as obstacles
def plan_path(current_pos, goal, num_nodes, x_lim, y_lim, obstacle_pos, obstacle_size, other_aircraft_positions,
              radius):
    nodes = generate_random_nodes(num_nodes, x_lim, y_lim, obstacle_pos, obstacle_size, other_aircraft_positions)
    nodes = np.vstack([current_pos, nodes, goal])  # Include current and goal in the set of nodes

    tree = {0: None}  # node_index: parent_node_index
    open_set = {0}  # Set of nodes to be explored
    closed_set = set()  # Set of explored nodes

    while open_set:
        current_node = min(open_set, key=lambda node: np.linalg.norm(nodes[node] - goal))
        open_set.remove(current_node)
        closed_set.add(current_node)

        if np.linalg.norm(nodes[current_node] - goal) <= radius:
            tree[len(nodes) - 1] = current_node
            break

        neighbors = nearest_neighbors(nodes, nodes[current_node], radius)
        for neighbor in neighbors:
            if neighbor not in closed_set and is_collision_free(nodes[current_node], nodes[neighbor], obstacle_pos,
                                                                obstacle_size, other_aircraft_positions,
                                                                step_size=0.05):
                if neighbor not in tree:
                    tree[neighbor] = current_node
                    open_set.add(neighbor)

    # Extract the path from the tree
    path = []
    node = len(nodes) - 1  # Goal node
    while node is not None:
        path.append(nodes[node])
        node = tree.get(node)

    return path[::-1]


# Dynamic FMT* algorithm implementation for multiple aircraft with delayed entry and collision avoidance
def dynamic_flight_path_multiple(aircraft_dict_initial, aircraft_dict_late, num_nodes, x_lim, y_lim, obstacle_start,
                                 obstacle_end, t_max, obstacle_speed=1, radius=10, late_timestep=10):
    all_flown_trajectories = {}  # Dictionary to store flown trajectories for each aircraft
    obstacle_trajectory = []  # List to store the obstacle's movement

    # Initialize positions for the initial aircraft
    current_positions = {name: OD[0] for name, OD in aircraft_dict_initial.items()}
    flown_trajectories = {name: [OD[0]] for name, OD in aircraft_dict_initial.items()}

    t = 0  # Timestep counter

    while t < t_max:
        t += 1

        # Move the obstacle by 1 unit per timestep
        obstacle_pos = dynamic_obstacle_movement(obstacle_start, obstacle_end, t, obstacle_speed)
        obstacle_trajectory.append(obstacle_pos)

        # Add the second group of aircraft at the specified timestep (e.g., timestep 10)
        if t == late_timestep:
            current_positions.update({name: OD[0] for name, OD in aircraft_dict_late.items()})
            flown_trajectories.update({name: [OD[0]] for name, OD in aircraft_dict_late.items()})

        # Track progress to ensure we don't have an endless loop
        has_progress = False  # Track whether any aircraft made progress this timestep

        # Loop through each aircraft
        for name, (start, goal) in {**aircraft_dict_initial, **aircraft_dict_late}.items():
            if name not in current_positions:  # If aircraft has been removed after reaching its goal
                continue

            current_pos = current_positions[name]
            if np.linalg.norm(current_pos - goal) > 0:  # Check if the aircraft has reached the goal

                # List of positions of other aircraft (excluding the current aircraft)
                other_aircraft_positions = [pos for other_name, pos in current_positions.items() if other_name != name]

                # Replan the path based on the updated obstacle position and nearby aircraft
                current_path = plan_path(current_pos, goal, num_nodes, x_lim, y_lim, obstacle_pos, obstacle_size=20,
                                         other_aircraft_positions=other_aircraft_positions, radius=radius)

                # Aircraft moves to the next node in the path
                if len(current_path) > 1:
                    next_node = current_path[1]
                    if not np.array_equal(current_pos, next_node):  # Ensure there's actual progress
                        current_positions[name] = next_node
                        flown_trajectories[name].append(next_node)
                        has_progress = True

            # If the aircraft has reached its goal, remove it from current_positions
            if np.linalg.norm(current_positions[name] - goal) <= 1e-3:
                print(f"Aircraft {name} has reached its goal and is removed from the simulation.")
                current_positions.pop(name)

        # If no aircraft made progress and all have not reached their goals, break to prevent infinite loop
        if not has_progress and t >= t_max / 2:  # Add a safeguard to force stopping after half max timesteps
            print("No progress detected for several timesteps. Breaking to avoid infinite loop.")
            break

        # Break the loop when all aircraft have reached their goals
        if all(np.linalg.norm(current_positions[name] - goal) <= 1e-3 for name in current_positions):
            print(f"All aircraft reached their goals by timestep {t}.")
            break

    return flown_trajectories, obstacle_trajectory


# Visualization of the dynamic trajectories for multiple aircraft
def plot_dynamic_trajectories(aircraft_trajectories, obstacle_trajectory, obstacle_size, x_lim, y_lim):
    plt.figure()

    # Plot flown trajectories for each aircraft
    for name, trajectory in aircraft_trajectories.items():
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], linestyle=':', label=f'{name} trajectory')
        plt.scatter(trajectory[0, 0], trajectory[0, 1], marker='^', color='green', s=50)  # Start marker
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], marker='*', color='red', s=50)  # Goal marker

    # Add text labels for the starting positions of AR1, AR2, AR3
    plt.text(20, 180, 'AR1', fontsize=10, ha='right')
    plt.text(20, 102, 'AR2', fontsize=10, ha='right')
    plt.text(20, 30, 'AR3', fontsize=10, ha='right')

    # Plot direct lines for each AR with width of 10 units
    plt.plot([20, 170], [180, 20], linestyle='-', color='purple', linewidth=10, alpha=0.1)  # AR1
    plt.plot([20, 185], [102, 102], linestyle='-', color='green', linewidth=10, alpha=0.1)  # AR2
    plt.plot([20, 170], [30, 180], linestyle='-', color='red', linewidth=10, alpha=0.1)  # AR3

    # Plot obstacle's trajectory
    obstacle_trajectory = np.array(obstacle_trajectory)
    plt.plot(obstacle_trajectory[:, 0], obstacle_trajectory[:, 1], 'r--')

    # Plot obstacle at each timestep
    for i, obs in enumerate(obstacle_trajectory):
        if i % 10 == 0:  # plot circle every 5 timesteps
            circle = plt.Circle(obs, obstacle_size, color='red', fill=True, alpha=0.3)
            plt.gca().add_patch(circle)

    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('Length (nm)')
    plt.ylabel('Width (nm)')

    # Save the figure to the specified path
    figure_directory = r'D:/FMT_vs_IDDPG/figure'
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)  # Create the directory if it doesn't exist

    current_time = time.strftime("%Y%m%d-%H%M%S")  # Current system time for unique naming
    figure_path = os.path.join(figure_directory, f"dynamic_trajectories_{current_time}.png")

    plt.savefig(figure_path)

    plt.show()


# Performance Evaluation
def evaluate_performance(aircraft_trajectories, obstacle_trajectory, obstacle_size, od_dict, computation_times):
    total_aircraft = len(aircraft_trajectories)
    loss_of_separation_aircraft = 0
    loss_of_separation_obstacle = 0
    reached_aircraft = 0
    total_distance_ratio = 0

    # 1. Loss of separation rate between aircraft trajectories
    for t in range(max(len(traj) for traj in aircraft_trajectories.values())):  # Iterate over timesteps
        aircraft_positions_at_t = []
        for traj in aircraft_trajectories.values():
            if t < len(traj):  # Get aircraft positions at time t
                aircraft_positions_at_t.append(traj[t])

        for i in range(len(aircraft_positions_at_t)):
            for j in range(i + 1, len(aircraft_positions_at_t)):
                dist = np.linalg.norm(np.array(aircraft_positions_at_t[i]) - np.array(aircraft_positions_at_t[j]))
                if dist < 10:
                    loss_of_separation_aircraft += 1

    # 2. Loss of separation between aircraft and obstacle
    for t in range(len(obstacle_trajectory)):  # Iterate over timesteps
        obstacle_pos = obstacle_trajectory[t]
        for traj in aircraft_trajectories.values():
            if t < len(traj):
                aircraft_pos = traj[t]
                dist = np.linalg.norm(np.array(aircraft_pos) - np.array(obstacle_pos))
                if dist < 10:
                    loss_of_separation_obstacle += 1

    # 3. Goal reach rate
    for ac, (start, goal) in od_dict.items():
        final_position = aircraft_trajectories[ac][-1]
        if np.linalg.norm(np.array(final_position) - np.array(goal)) <= 1e-3:  # Close enough to the goal
            reached_aircraft += 1

    goal_reach_rate = reached_aircraft / total_aircraft

    # 4. Distance ratio: actual flight distance / shortest distance
    for ac, (start, goal) in od_dict.items():
        shortest_distance = np.linalg.norm(np.array(goal) - np.array(start))  # Straight line distance
        actual_flight_distance = 0
        for i in range(1, len(aircraft_trajectories[ac])):
            actual_flight_distance += np.linalg.norm(
                np.array(aircraft_trajectories[ac][i]) - np.array(aircraft_trajectories[ac][i - 1]))

        distance_ratio = actual_flight_distance / shortest_distance if shortest_distance != 0 else 0
        total_distance_ratio += distance_ratio

    avg_distance_ratio = total_distance_ratio / total_aircraft

    # 5. Average computation time for each timestep
    # Using the actual number of timesteps (total_used_timesteps) when all aircraft have reached their goals or terminated
    total_used_timesteps = t  # 't' represents the actual number of timesteps used in the simulation
    total_computation_time = sum(computation_times)
    avg_computation_time = total_computation_time / total_used_timesteps if total_used_timesteps != 0 else 0

    # Return the results
    results = {
        'loss_of_separation_aircraft_rate': loss_of_separation_aircraft / (
                    total_aircraft * max(len(traj) for traj in aircraft_trajectories.values())),
        'loss_of_separation_obstacle_rate': loss_of_separation_obstacle / (total_aircraft * len(obstacle_trajectory)),
        'goal_reach_rate': goal_reach_rate,
        'average_distance_ratio': avg_distance_ratio,
        'average_computation_time': avg_computation_time
    }

    return results


# Main Execution
if __name__ == "__main__":
    # Parameters
    num_nodes = 2000  # Sampled nodes from start to end
    radius = 10  # Connection threshold of nearby node
    x_lim = [0, 200]
    y_lim = [0, 200]

    # Define initial aircraft with OD locations (appearing at timestep 1)
    aircraft_dict_initial = {
        'AC1': (np.array([20, 180]), np.array([170, 20])),
        'AC2': (np.array([20, 102]), np.array([185, 102])),
        'AC3': (np.array([20, 30]), np.array([170, 180])),
    }

    # Define later aircraft with OD locations (appearing at timestep 10)
    aircraft_dict_late = {
        'AC4': (np.array([20, 180]), np.array([170, 20])),
        'AC5': (np.array([20, 102]), np.array([185, 102])),
        'AC6': (np.array([20, 30]), np.array([170, 180])),
    }

    # Obstacle moves from one point to another within boundary
    obstacle_start = np.array([135, 160])
    obstacle_end = np.array([80, 25])

    t_max = 80  # Manually adjust the number of time steps
    late_timestep = 10  # Time step at which late aircraft appear
    obstacle_speed = 2.5
    obstacle_size = 30

    # Run the dynamic FMT* algorithm for multiple aircraft with delayed entry and collision avoidance
    start_time = time.time()  # Start timing
    aircraft_trajectories, obstacle_trajectory = dynamic_flight_path_multiple(
        aircraft_dict_initial, aircraft_dict_late, num_nodes, x_lim, y_lim, obstacle_start, obstacle_end, t_max,
        obstacle_speed, radius, late_timestep=late_timestep
    )
    end_time = time.time()  # End timing

    computation_times = [end_time - start_time]  # Example: Can be updated per timestep in the main loop

    # Display total simulation time
    print(f"Total simulation time: {sum(computation_times):.2f} seconds")

    # Define the OD dict to evaluate performance
    od_dict = {**aircraft_dict_initial, **aircraft_dict_late}

    # Evaluations
    performance_results = evaluate_performance(aircraft_trajectories, obstacle_trajectory, obstacle_size, od_dict,
                                               computation_times)
    print(f"Performance Results: {performance_results}")

    # Plot the result
    matplotlib.use('Qt5Agg')
    plot_dynamic_trajectories(aircraft_trajectories, obstacle_trajectory, obstacle_size, x_lim, y_lim)

    # Save the results
    save_simulation_results(aircraft_trajectories, obstacle_trajectory)
