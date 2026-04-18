"""
scripts/inspect_onnx.py
Inspects ONNX model graph to find the correct node names for INT8 exclusion.
Run: python scripts/inspect_onnx.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnx
from pathlib import Path

MODELS_TO_CHECK = ["triage", "misinfo", "sentiment"]

for name in MODELS_TO_CHECK:
    onnx_path = Path(f"models/onnx/{name}/model.onnx")
    if not onnx_path.exists():
        print(f"[{name}] model.onnx NOT FOUND")
        continue

    model = onnx.load(str(onnx_path))

    print(f"\n{'='*60}")
    print(f"[{name}] MatMul nodes in graph:")
    print(f"{'='*60}")

    # Print all initializer shapes (weight tensors)
    weight_shapes = {init.name: list(init.dims) for init in model.graph.initializer}

    matmul_count = 0
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        matmul_count += 1
        for inp in node.input:
            shape = weight_shapes.get(inp, None)
            if shape is not None:
                print(f"  Node: '{node.name}' | Weight: '{inp}' | Shape: {shape}")

    print(f"  Total MatMul nodes: {matmul_count}")