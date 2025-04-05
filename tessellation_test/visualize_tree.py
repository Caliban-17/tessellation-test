import os
import networkx as nx
import matplotlib.pyplot as plt
import platform

def build_tree_graph(root):
    """
    Walks through the directory starting from root and builds a directed graph
    representing the file structure, using relative paths for node names.

    Parameters:
        root (str): The root directory to start building the tree.

    Returns:
        networkx.DiGraph: A directed graph representing the directory tree.
                          Node names are relative paths from the root.
    """
    G = nx.DiGraph()
    abs_root = os.path.abspath(root)
    root_name = os.path.basename(abs_root) # Keep root name simple

    # Add root node first
    G.add_node(root_name)

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # Determine parent node name based on relative path
        rel_path_from_root = os.path.relpath(dirpath, root)
        if rel_path_from_root == '.':
            parent_node_name = root_name
        else:
            parent_node_name = os.path.join(root_name, rel_path_from_root.replace(os.sep, '/')) # Use forward slash for consistency

        # Add current directory node if not root (already added)
        if parent_node_name != root_name:
            # Ensure the node exists (it should if walk is top-down)
             if not G.has_node(parent_node_name): G.add_node(parent_node_name)
             # Add edge from its parent directory
             grandparent_rel_path = os.path.dirname(rel_path_from_root)
             if grandparent_rel_path == '': # Parent is root
                 grandparent_node_name = root_name
             else:
                 grandparent_node_name = os.path.join(root_name, grandparent_rel_path.replace(os.sep, '/'))

             # Add edge if grandparent exists
             if G.has_node(grandparent_node_name):
                 G.add_edge(grandparent_node_name, parent_node_name)

        # Process subdirectories
        dirs_to_remove = []
        for dirname in dirnames:
            # Exclude VCS and cache directories
            if dirname == '__pycache__' or dirname.startswith('.'):
                 dirs_to_remove.append(dirname)
                 continue
            child_node_name = os.path.join(parent_node_name, dirname).replace(os.sep, '/')
            G.add_node(child_node_name)
            G.add_edge(parent_node_name, child_node_name)
        # Remove excluded directories from further traversal
        for d in dirs_to_remove:
            dirnames.remove(d)

        # Process files
        for filename in filenames:
            # Exclude cache/compiled files and hidden files
            if filename == "__pycache__" or filename.endswith(".pyc") or filename.startswith('.'):
                continue
            child_node_name = os.path.join(parent_node_name, filename).replace(os.sep, '/')
            G.add_node(child_node_name)
            G.add_edge(parent_node_name, child_node_name)

    # Clean up labels for display (use basename for non-root nodes)
    labels = {node: os.path.basename(node) if node != root_name else root_name for node in G.nodes()}

    return G, labels

def visualize_tree(root="."):
    """
    Builds and visualizes the directory tree for the given root directory.

    Parameters:
        root (str): The root directory (defaults to current directory).
    """
    if not os.path.isdir(root):
        print(f"Error: Root directory '{root}' not found.")
        return

    tree, labels = build_tree_graph(root)
    if not tree:
        print(f"Could not build tree for '{root}'.")
        return

    plt.figure(figsize=(18, 12)) # Adjusted figure size

    # Attempt to use graphviz layout for a hierarchical tree view.
    pos = None
    try:
        # Use 'dot' layout engine for hierarchical structure
        pos = nx.nx_pydot.graphviz_layout(tree, prog='dot')
        print("Using Graphviz 'dot' layout for hierarchical view.")
    except ImportError:
         print("Warning: PyDot not found (pip install pydot). Graphviz layout unavailable.")
    except Exception as e:
        # Catch other potential errors (like Graphviz not found)
        # Provide specific hint for common OS
        system = platform.system()
        install_hint = ""
        if system == "Linux":
            install_hint = "Try 'sudo apt-get install graphviz' or 'sudo yum install graphviz'."
        elif system == "Darwin": # macOS
            install_hint = "Try 'brew install graphviz'."
        elif system == "Windows":
            install_hint = "Download from graphviz.org and ensure it's in your system PATH."
        print(f"Warning: Graphviz layout failed ({e}). {install_hint}")

    # Use spring layout if graphviz failed or wasn't attempted
    if pos is None:
        print("Using NetworkX 'spring' layout as fallback.")
        pos = nx.spring_layout(tree, k=0.3, iterations=50, seed=42)

    # Draw the graph
    nx.draw(tree, pos,
            labels=labels, # Use cleaned labels
            with_labels=True,
            node_size=3000,
            node_color='lightblue',
            font_size=9,
            font_weight='bold',
            edge_color='gray',
            arrows=False) # Trees typically don't use arrows

    plt.title(f"Project Directory Tree Structure\nRoot: '{os.path.abspath(root)}'", size=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Automatically detect the likely project root relative to the script location
    script_dir = os.path.dirname(__file__)
    # Assumes script is in 'tessellation_test' or the root containing it
    if os.path.basename(script_dir).startswith('tessellation_test'):
        project_root_path = os.path.abspath(os.path.join(script_dir, '..'))
    elif os.path.exists(os.path.join(script_dir, 'tessellation_test')):
         project_root_path = os.path.abspath(script_dir)
    else:
         project_root_path = '.' # Default to current if structure unclear

    print(f"Visualizing directory tree starting from: {project_root_path}")
    visualize_tree(project_root_path)