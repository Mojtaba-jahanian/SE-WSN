from pathlib import Path
import numpy as np
import networkx as nx
import pickle


def load_xdados(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Expected data file at {path} (place Xdados.txt there).")
    data = np.loadtxt(path)
    return data  # shape: (num_snapshots, num_nodes)


def build_grid_coordinates(num_nodes: int):
    # Assign nodes to positions on a regular 2D grid
    side = int(np.sqrt(num_nodes))
    if side * side < num_nodes:
        side += 1
    coords = {}
    idx = 0
    for r in range(side):
        for c in range(side):
            if idx < num_nodes:
                coords[idx] = np.array([r, c], dtype=float)
                idx += 1
    return coords


def main():
    this_file = Path(__file__).resolve()
    data_root = this_file.parents[1]
    raw_path = data_root / "raw" / "Xdados.txt"
    out_dir = data_root / "processed" / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    mat = load_xdados(raw_path)
    num_snapshots, num_nodes = mat.shape
    print(f"Loaded Xdados: {num_snapshots} snapshots, {num_nodes} nodes")

    coords = build_grid_coordinates(num_nodes)
    comm_radius = 1.6

    for t in range(num_snapshots):
        values_t = mat[t]

        G = nx.Graph()
        for i in range(num_nodes):
            G.add_node(int(i), value=float(values_t[i]))

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                d = np.linalg.norm(coords[i] - coords[j])
                if d <= comm_radius:
                    w = float(1.0 / (1.0 + d))
                    G.add_edge(int(i), int(j), weight=w)

        out_path = out_dir / f"graph_{t:05d}.gpickle"
        with open(out_path, "wb") as f:
            pickle.dump(G, f)

        if (t + 1) % 500 == 0:
            print(f"... built {t+1}/{num_snapshots} graphs")

    print(f"Finished building graph sequence in: {out_dir}")


if __name__ == "__main__":
    main()
