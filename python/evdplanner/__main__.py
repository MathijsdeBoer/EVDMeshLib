from pathlib import Path

from evdplanner.rs import Mesh

root = Path(r"S:\E_ResearchData\evdplanner\Samples")

n_triangles = []

for subdir in root.iterdir():
    print(subdir.name)
    if subdir.is_file():
        continue

    mesh_path = subdir / "mesh_ventricles.stl"
    if not mesh_path.exists():
        print(f"Mesh not found: {mesh_path}")
        continue

    mesh = Mesh.load(str(mesh_path))
    n_triangles.append(mesh.num_triangles)

print(f"Min: {min(n_triangles)}")
print(f"Max: {max(n_triangles)}")
print(f"Mean: {sum(n_triangles) / len(n_triangles)}")
