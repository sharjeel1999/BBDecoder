from graphviz import Digraph

from graphviz import Digraph
import inspect
import dis

def list_layers(model):
    for name, module in model.named_children():
        print(f"Layer Index: {module.index}, Layer Name: {module.name}")

def create_graphviz_graph(model, input_tensor):
    visual_graph = Digraph(
        name="ComputationGraph",
        format="png",
        graph_attr={"rankdir": "TB"},  # Top-to-Bottom layout
        node_attr={"shape": "box", "style": "filled", "fontcolor": "black"},
    )

    # Track nodes and connections
    previous_layer = None

    for name, module in model.named_children():
        # Add the layer as a node
        layer_name = f"{name}\n({module.__class__.__name__})"  # Add layer details
        visual_graph.node(name, label=layer_name, color="lightblue")

        # Add edge from the previous layer
        if previous_layer is not None:
            visual_graph.edge(previous_layer, name)

        previous_layer = name  # Update for next connection

    # Render the graph
    visual_graph.render("computation_graph", format="png", cleanup=True)
    print("Graph saved to computation_graph.png")


# create_graphviz_graph(model, x)
