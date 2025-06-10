import:py torch;
import:py from torch {tensor}
# import:py torch.fx;

# @torch.compile
can f(x: tensor, y: tensor) -> tensor {
    try {
        return torch.sin(x) + torch.cos(y);
    } except Exception as e {
        print("Error in f:", e);
        return torch.zeros_like(x);
    }
}

with entry {
    x = torch.randn(10);
    y = torch.randn(10);
    traced_f = torch.fx.symbolic_trace(f);
    print(traced_f.graph);
    traced_f(x, y);
}
