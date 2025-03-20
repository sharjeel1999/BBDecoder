import graphviz

# Parameters for the graph
graph_name = "DetailedGraph"
graph_attr = {"rankdir": "TB"}  # Top-to-Bottom layout
node_attr = {"shape": "box", "style": "filled", "fontcolor": "black"}
edge_attr = {"color": "black"}  # Edge style
filename = "detailed_graph"

# Create an empty graph
visual_graph = graphviz.Digraph(
    name=graph_name, engine="dot",
    graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr,
    filename=filename
)

# Add nodes with more details
visual_graph.node("Input", label="Input Layer\nShape: [1, 10]\nColor: lightblue", shape="ellipse", color="lightblue")
visual_graph.node("Hidden1", label="Hidden Layer 1\nType: Linear\nUnits: 64\nActivation: ReLU", color="lightgreen")
visual_graph.node("Hidden2", label="Hidden Layer 2\nType: Linear\nUnits: 32\nActivation: ReLU", color="yellow")
visual_graph.node("Output", label="Output Layer\nShape: [1, 2]\nType: Softmax", shape="ellipse", color="orange")

# Add edges to represent connections
visual_graph.edge("Input", "Hidden1", label="Weights\nShape: [10, 64]")
visual_graph.edge("Hidden1", "Hidden2", label="Weights\nShape: [64, 32]")
visual_graph.edge("Hidden2", "Output", label="Weights\nShape: [32, 2]")

# Render the graph
visual_graph.render("detailed_graph", format="png", cleanup=True)
