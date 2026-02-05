import ray
import os

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

try:
    # Try connecting to an existing cluster
    ray.init(address="auto")
    print("Connected to existing cluster.")
except Exception as e:
    print(f"Could not connect to auto ({e}). Starting local.")
    # Start a local instance
    ray.init()

print("\n--- Ray Nodes ---")
nodes = ray.nodes()
for node in nodes:
    print(f"NodeID: {node['NodeID']}")
    print(f"  Alive: {node['Alive']}")
    print(f"  Resources: {node.get('Resources')}")

print("\n--- Cluster Resources ---")
print(ray.cluster_resources())
