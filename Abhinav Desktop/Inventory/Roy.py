import networkx as nx
import matplotlib.pyplot as plt

# Small subset of inventory for visualization
max_i = 3
states_subset = [(i, a_prev) for i in range(max_i+1) for a_prev in [0,1]]

G = nx.DiGraph()

# Build transitions (simplified: only deterministic connectivity)
for i in range(max_i+1):
    for a_prev in [0,1]:
        for action in [0,1]:
            state = (i, a_prev)
            # next states based on up/down/stay
            next_up = (min(i+1, max_i), action)
            next_down = (max(i-1,0), action)
            next_stay = (i, action)
            # add edges
            G.add_edge(state, next_up, label='up')
            G.add_edge(state, next_down, label='down')
            G.add_edge(state, next_stay, label='stay')

# Position nodes in 2D: x=inventory, y=a_prev
pos = {(i,a): (i*2, a*2) for i in range(max_i+1) for a in [0,1]}

plt.figure(figsize=(10,6))
nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightblue', font_size=10)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
plt.title("Transition diagram (i, a_prev) with switch_cost=0")
plt.show()
