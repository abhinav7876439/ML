import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import networkx as nx
import logging
import os
import math


def read_tsp_file(file_path):
    """
    Read the TSP file, extract coordinates, and compute the edge matrix
    dynamically for EUC_2D or read it directly for EXPLICIT formats.
    """
    with open(file_path, 'r') as file:
        lines = [line.replace(':', '').strip() for line in file.readlines()]

        # Extract specifications
        specifications = {}
        for line in lines:
            if line in ['EDGE_WEIGHT_SECTION', 'NODE_COORD_SECTION',
                        'DISPLAY_DATA_SECTION']:
                break
            item = line.split()
            if len(item) > 1:
                specifications[item[0]] = item[1:]

        num_cities = int(specifications['DIMENSION'][0])

        # Default to EXPLICIT if EDGE_WEIGHT_TYPE is not provided
        edge_weight_type = specifications.get('EDGE_WEIGHT_TYPE', ['EXPLICIT'])[
            0]

        # 1. Extract coordinates if available
        coordinates = None
        coord_idx = -1
        if 'NODE_COORD_SECTION' in lines:
            coord_idx = lines.index('NODE_COORD_SECTION')
        elif 'DISPLAY_DATA_SECTION' in lines:
            coord_idx = lines.index('DISPLAY_DATA_SECTION')

        if coord_idx != -1:
            coordinates = {}
            for line in lines[coord_idx + 1:]:
                if line in ['EOF', 'EDGE_WEIGHT_SECTION',
                            'DISPLAY_DATA_SECTION'] or not line:
                    break
                item = line.split()
                if len(item) >= 3:
                    # 0-indexed node IDs
                    coordinates[int(item[0]) - 1] = (
                    float(item[1]), float(item[2]))

        # 2. Create edge matrix based on the weight type
        edge_matrix = [[0] * num_cities for _ in range(num_cities)]

        if edge_weight_type == 'EUC_2D':
            if not coordinates:
                raise ValueError(
                    f"File {file_path} specifies EUC_2D but contains no coordinate section.")

            for i in range(num_cities):
                for j in range(num_cities):
                    if i != j:
                        # Standard TSPLIB rounding for Euclidean distance
                        dx = coordinates[i][0] - coordinates[j][0]
                        dy = coordinates[i][1] - coordinates[j][1]
                        edge_matrix[i][j] = int(
                            round(math.sqrt(dx ** 2 + dy ** 2)))

        else:  # EXPLICIT
            index = lines.index('EDGE_WEIGHT_SECTION')
            flat_data = [
                int(value) for line in lines[index + 1:]
                if line and line.strip() not in ['DISPLAY_DATA_SECTION',
                                                 'NODE_COORD_SECTION', 'EOF']
                for value in line.split() if value.lstrip('-').isdigit()
            ]

            format_type = \
            specifications.get('EDGE_WEIGHT_FORMAT', ['FULL_MATRIX'])[0]
            idx = 0

            if format_type == 'LOWER_DIAG_ROW':
                for i in range(num_cities):
                    for j in range(i + 1):
                        edge_matrix[i][j] = edge_matrix[j][i] = flat_data[idx]
                        idx += 1
            elif format_type == 'UPPER_DIAG_ROW':
                for i in range(num_cities):
                    for j in range(i, num_cities):
                        edge_matrix[i][j] = edge_matrix[j][i] = flat_data[idx]
                        idx += 1
            else:  # FULL_MATRIX
                for i in range(num_cities):
                    edge_matrix[i] = [flat_data[idx + j] for j in
                                      range(num_cities)]
                    idx += num_cities

    return specifications, edge_matrix, coordinates, num_cities


def tsp_model(edge_matrix, num_cities):
    """ Create a model with degree constraints """
    model = gp.Model('DFJ_TSP_Edge')
    n = num_cities

    # Symmetric TSP, only need (i, j) where i < j
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Binary decision variables
    x = model.addVars(edges, vtype=GRB.BINARY, name='x')

    # Objective: Minimize total distance
    model.setObjective(
        gp.quicksum(edge_matrix[i][j] * x[i, j] for (i, j) in edges),
        GRB.MINIMIZE
    )

    # Constraint: Each city must have exactly 2 edges connected to it
    model.addConstrs(
        (gp.quicksum(x[min(i, j), max(i, j)] for j in range(n) if i != j) == 2
         for i in range(n)),
        name="Degree_Constraints"
    )

    model.update()
    return model, x, edges


def find_subtour(edges, n):
    """ Given a set of edges, find the shortest subtour """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    subtours = list(nx.connected_components(G))
    if len(subtours) == 1:
        return None
    else:
        return min(subtours, key=len)


def plotting(selected_edges, n, filepath, coordinates=None, subtour=None,
             x_values=None):
    """ Plot the graph and save it to a PNG file """
    plt.figure(figsize=(10, 10))

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(selected_edges)

    pos = coordinates if coordinates else nx.spring_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color="skyblue")
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Draw selected edges in black
    nx.draw_networkx_edges(G, pos, edgelist=selected_edges, edge_color="black",
                           width=2)

    # Highlight subtour edges in red, if any
    if subtour:
        subtour_edges = [(i, j) for i in subtour for j in subtour if
                         i < j and (i, j) in selected_edges]
        nx.draw_networkx_edges(G, pos, edgelist=subtour_edges, edge_color="red",
                               width=2)

    # Highlight fractional edges in dark blue
    if x_values is not None:
        fractional_edges = [(i, j) for (i, j), value in x_values.items() if
                            0 < value < 1]
        nx.draw_networkx_edges(G, pos, edgelist=fractional_edges,
                               edge_color="darkblue", style='dashed', width=2)

    plt.title("TSP Support Graph Visualization")

    # Save the figure to the specified file path and close it to free memory
    plt.savefig(filepath, format='png', bbox_inches='tight')
    plt.close()


def subtour_elimination(model, where):
    """ Callback function to include lazy constraints and trigger image saving """

    if where == gp.GRB.Callback.MIPNODE:
        model._node_counter += 1

    # Apply cuts at MIPSOL or occasionally at MIPNODE
    if where == gp.GRB.Callback.MIPSOL or (
            model._node_counter >= model._node_threshold and where == gp.GRB.Callback.MIPNODE):

        x_values = {}
        if where == gp.GRB.Callback.MIPSOL:
            # Check if integer solution is a new best
            current_obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            if model._best_ip_with_cuts is None or current_obj < model._best_ip_with_cuts:
                model._best_ip_with_cuts = current_obj
            x_values = model.cbGetSolution(model._x)

        elif where == gp.GRB.Callback.MIPNODE:
            if model.cbGet(
                    gp.GRB.Callback.MIPNODE_STATUS) == gp.GRB.Status.OPTIMAL:
                x_values = model.cbGetNodeRel(model._x)
            else:
                return

                # Filter edges based on threshold
        selected_edges = [(i, j) for i, j in model._edges if
                          x_values[i, j] >= 0.001]

        # Find shortest subtour
        subtour = find_subtour(selected_edges, model._n)

        # Generate filename and save the plot
        img_filename = os.path.join(model._img_dir,
                                    f"support_graph_{model._plot_counter:04d}.png")
        plotting(selected_edges, model._n, img_filename,
                 coordinates=model._coordinates, subtour=subtour,
                 x_values=x_values)
        model._plot_counter += 1  # Increment the image sequence counter

        # Apply lazy constraint if subtour found
        if subtour is not None:
            model.cbLazy(
                gp.quicksum(model._x[i, j] for i in subtour for j in subtour if
                            i < j) <= len(subtour) - 1
            )
            model._cut_counter += 1
            logging.info(
                f"Added subtour elimination constraint {model._cut_counter} for subset {subtour}")

        model._node_counter = 0


def solve_model(model, x, num_cities, edges, coordinates, img_dir):
    """ Set parameters, bind callback variables, and optimize """
    model.setParam('LazyConstraints', 1)
    model.setParam('presolve', 0)
    model.setParam('Cuts', 0)
    model.setParam('Heuristics', 0)
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', 120)

    # Attach required data directly to the model
    model._coordinates = coordinates
    model._x = x
    model._n = num_cities
    model._edges = edges
    model._img_dir = img_dir  # Pass the directory to the callback
    model._plot_counter = 1  # Track the image sequence number
    model._cut_counter = 0
    model._node_counter = 0
    model._node_threshold = 500
    model._best_ip_with_cuts = None

    model.optimize(subtour_elimination)

    best_lp_bound = model.objBound
    nodes_explored = model.nodeCount
    lazy_cuts = model._cut_counter
    gap = None
    elapsed_time = model.Runtime

    if model.status == GRB.OPTIMAL:
        tour_edges = [(i, j) for i, j in model._edges if
                      model._x[i, j].x >= 0.5]

        # Final visualization
        final_img_path = os.path.join(img_dir, "final_optimal_tour.png")
        plotting(tour_edges, num_cities, final_img_path,
                 coordinates=coordinates)

        if find_subtour(tour_edges, num_cities) is None:
            gap = model.MIPGap
        else:
            logging.warning(
                "No valid integer solution found; returning the best LP bound.")
    elif model.status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED]:
        logging.error("Model is infeasible or unbounded.")
    else:
        if model._best_ip_with_cuts:
            gap = abs(
                model.objBound - model._best_ip_with_cuts) / model._best_ip_with_cuts
        else:
            logging.info("No best ip found so far.")

    return best_lp_bound, model._best_ip_with_cuts, nodes_explored, lazy_cuts, gap, elapsed_time


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    

    # file_folder = "STSP"
    # output_base_folder = "output_images"

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Build absolute paths relative to the script's location
    file_folder = os.path.join(script_dir, "STSP")
    output_base_folder = os.path.join(script_dir, "output_images")

    if not os.path.exists(file_folder):
        logging.error(
            f"Directory '{file_folder}' not found. Please create it and add .tsp files.")
    else:
        tsp_files = [f for f in os.listdir(file_folder) if f.endswith('.tsp')]

        for file_name in tsp_files:
            file_path = os.path.join(file_folder, file_name)

            # Create a dedicated output directory for this specific TSP instance
            instance_name = os.path.splitext(file_name)[0]
            instance_img_dir = os.path.join(output_base_folder, instance_name)
            os.makedirs(instance_img_dir, exist_ok=True)

            specifications, edge_matrix, coordinates, num_cities = read_tsp_file(
                file_path)
            model, x, edges = tsp_model(edge_matrix, num_cities)

            # Pass the image directory to the solver
            results = solve_model(model, x, num_cities, edges, coordinates,
                                  instance_img_dir)

            logging.info(f"Results for {file_name}: {results}")
            model.write(f"example_{instance_name}.lp")
            logging.info(f"Visualizations saved in: {instance_img_dir}")
