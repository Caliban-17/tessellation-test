import os
import networkx as nx
import matplotlib.pyplot as plt

def build_tree_graph(root):
    """
    Walks through the directory starting from root and builds a directed graph.
    
    Parameters:
        root (str): The root directory to start building the tree.
    
    Returns:
        networkx.DiGraph: A directed graph representing the directory tree.
    """
    G = nx.DiGraph()
    for dirpath, dirnames, filenames in os.walk(root):
        # Ensure the current directory is added.
        G.add_node(dirpath)
        for dirname in dirnames:
            child = os.path.join(dirpath, dirname)
            G.add_edge(dirpath, child)
        for filename in filenames:
            child = os.path.join(dirpath, filename)
            G.add_edge(dirpath, child)
    return G

def visualize_tree(root):
    """
    Builds and visualizes the directory tree for the given root directory.
    
    Parameters:
        root (str): The root directory.
    """
    tree = build_tree_graph(root)
    
    # Attempt to use graphviz layout for a hierarchical tree view.
    try:
        pos = nx.nx_pydot.graphviz_layout(tree, prog='dot')
    except Exception as e:
        print("Graphviz layout not available, using spring layout instead:", e)
        pos = nx.spring_layout(tree)
    
    plt.figure(figsize=(12, 8))
    nx.draw(tree, pos, with_labels=True, node_size=500, font_size=8)
    plt.title(f"Directory Tree for '{root}'")
    plt.show()

if __name__ == "__main__":
    # Replace '.' with the path to your project if needed
    visualize_tree('.')