from torchvision.models import resnet50, vit_b_16

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

model = resnet50()


