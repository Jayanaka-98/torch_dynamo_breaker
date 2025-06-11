import torch
from random import randint
from torch.fx.passes.graph_drawer import FxGraphDrawer
import inspect
from itertools import count
import os


graph_counter = count()


def save_fx_graph(gm, name_prefix="graph"):
    dot_graph = FxGraphDrawer(gm, f"{name_prefix}_{next(graph_counter)}")
    png = dot_graph.get_dot_graph().create_png()
    # Get the filename of the calling script
    calling_file = inspect.stack()[-1].filename

    name_prefix = os.path.splitext(os.path.basename(calling_file))[0]
    print(f">>>>>Saving FX graph to {name_prefix}")
    # Set the save location to the same directory as the calling script
    save_dir = os.path.dirname(calling_file)
    save_path = os.path.join(
        save_dir, f"_{name_prefix}_{graph_counter}_{randint(0, 100)}.png"
    )
    with open(save_path, "wb") as f:
        f.write(png)


def inspect_backend(gm, sample_inputs):
    print("Calling Backend")
    print(gm.print_readable())
    save_fx_graph(gm, "main")

    for name, mod in gm.named_modules():
        if name != "" and isinstance(mod, torch.fx.GraphModule):
            print(f"Found subgraph : {name}")
            save_fx_graph(mod, f"{name}")

    return gm.forward
