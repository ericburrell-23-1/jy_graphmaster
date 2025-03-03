import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
from collections import defaultdict
from src.common.pgm_approach import PGM_appraoch
from src.common.state import State
class Visulizer:

    def __init__(self, this_pgm:PGM_appraoch):
        self.pgm = this_pgm
        self.rez_states_minus = this_pgm.rez_states_minus
        self.index_to_graph = this_pgm.index_to_graph
        self.pgm_graph_2_rmp_graph = this_pgm.pgm_graph_2_rmp_graph
    def _create_plot(self,figsize=(20, 10), title="State-Action Graph", graph_id=None):
        
        
        # Create a directed graph
        G = nx.DiGraph()

        # Filter states by graph_id if specified
        if graph_id is not None:
            states_to_use:list[State] = [s for s in self.rez_states_minus if s.l_id == graph_id]
        else:
            states_to_use:list[State] = list(self.rez_states_minus)
        
        # Dictionary to store the original state objects by state_id for later reference
        state_objects = {}
        
        # Add states as nodes with their attributes
        for state in states_to_use:
            # Extract the state vector value for labeling
            state_vec_value = state.state_vec.toarray()[0, 0]
            
            G.add_node(state.state_id, 
                    node=state.node,
                    l_id=state.l_id,
                    state_vec_value=state_vec_value)
            
            # Store original state object for edge creation
            state_objects[state.state_id] = state
        
        # Add action connections from flow constraints
        # Collect all potential state pairs that could form an edge
        edges_to_add = []
        
        # For each graph in the model
        for g_id in self.index_to_graph:
            # Skip if we're only visualizing a specific graph and this isn't it
            if graph_id is not None and g_id != graph_id:
                continue
                
            g = self.index_to_graph[g_id]
            
            # Skip if this graph has no RMP representation
            if not hasattr(self, 'pgm_graph_2_rmp_graph') or g not in self.pgm_graph_2_rmp_graph:
                continue
            
            rmp_graph = self.pgm_graph_2_rmp_graph[g]
            
            # For each equivalence class in the RMP graph
            for eq_class in rmp_graph.equiv_class_2_s1_s2_pairs:
                # For each state pair in this equivalence class
                for s1, s2 in rmp_graph.equiv_class_2_s1_s2_pairs[eq_class]:
                    # Add an edge from s1 to s2 if their state_ids are in our graph
                    if s1.state_id in G.nodes and s2.state_id in G.nodes:
                        edges_to_add.append((s1.state_id, s2.state_id, eq_class))
        
        # Add all collected edges to the graph
        for s1_id, s2_id, eq_class in edges_to_add:
            G.add_edge(s1_id, s2_id, eq_class=eq_class)
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Position nodes in a horizontal layout based on their node values
        # Group states by node value
        nodes_by_level = defaultdict(list)
        for node_id, node_data in G.nodes(data=True):
            node_value = node_data['node']
            nodes_by_level[node_value].append(node_id)
        
        # Position nodes horizontally by their node value, with multiple states per level stacked vertically
        pos = {}
        
        # Custom sorting for node values to place source (-1) at far left and sink (-2) at far right
        sorted_levels = []
        
        # First add source node (-1) if it exists
        if -1 in nodes_by_level:
            sorted_levels.append(-1)
        
        # Then add all other nodes except sink in ascending order
        for level in sorted([l for l in nodes_by_level.keys() if l not in [-1, -2]]):
            sorted_levels.append(level)
        
        # Finally add sink node (-2) if it exists
        if -2 in nodes_by_level:
            sorted_levels.append(-2)
        
        for i, level in enumerate(sorted_levels):
            nodes = nodes_by_level[level]
            # Sort nodes at this level by their state_id for consistent positioning
            nodes.sort()
            
            # Position each node at this level
            for j, node_id in enumerate(nodes):
                # Horizontal position determined by node value
                x = i
                # Vertical position spaces nodes at the same level
                # Center nodes vertically
                y = j - (len(nodes) - 1) / 2
                pos[node_id] = (x, y)
        
        # Draw nodes (states)
        nx.draw_networkx_nodes(G, pos, 
                            node_size=2000, 
                            node_color='lightblue',
                            alpha=0.8,
                            edgecolors='black',
                            ax=ax)
        
        # Draw edges (actions)
        nx.draw_networkx_edges(G, pos,
                            width=1.5,
                            alpha=0.7,
                            edge_color='gray',
                            arrowsize=20,
                            connectionstyle='arc3,rad=0.1',  # Curved edges for better visibility
                            ax=ax)
        
        # Add state labels with only node number and state_vec value (no state ID)
        labels = {node: f"node:{data['node']}\ncap:{int(data['state_vec_value'])}" 
                for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)
        
        # Remove axis
        ax.set_axis_off()
        
        # Add title
        graph_title = title
        if graph_id is not None:
            graph_title += f" (Graph {graph_id})"
        plt.title(graph_title, fontsize=15)
        
        # Add legend
        state_patch = mpatches.Patch(color='lightblue', label='States')
        action_patch = mpatches.Patch(color='gray', label='Actions')
        plt.legend(handles=[state_patch, action_patch], loc='upper right')
        
        plt.tight_layout()
        return fig
    def plot_graph(self):
        for l_id in self.pgm.index_to_graph:
            this_fig = self._create_plot(graph_id=l_id)
            plt.figure(this_fig.number)
            plt.show()