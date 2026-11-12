"""Create a test-only identity ONNX model and build a dynamic-batch TensorRT engine."""
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--trtexec", default="/usr/src/tensorrt/bin/trtexec")
    parser.add_argument("--onnx-only", action="store_true", help="Validate and save the CPU fixture without building an engine")
    args = parser.parse_args()
    if min(args.width, args.height, args.capacity) < 1:
        parser.error("dimensions and capacity must be positive")
    import onnx
    from onnx import TensorProto, helper
    args.output.mkdir(parents=True, exist_ok=True)
    model_path, engine_path = args.output / "identity.onnx", args.output / "identity.engine"
    shape = ["batch", 3, args.height, args.width]
    graph = helper.make_graph([helper.make_node("Identity", ["images"], ["identity"])], "motion-identity",
                              [helper.make_tensor_value_info("images", TensorProto.FLOAT, shape)],
                              [helper.make_tensor_value_info("identity", TensorProto.FLOAT, shape)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    if args.onnx_only:
        print(model_path)
        return
    tail = f"3x{args.height}x{args.width}"
    subprocess.run([args.trtexec, f"--onnx={model_path}", f"--saveEngine={engine_path}",
                    f"--minShapes=images:1x{tail}", f"--optShapes=images:{args.capacity}x{tail}",
                    f"--maxShapes=images:{args.capacity}x{tail}", "--skipInference"], check=True)
    if not engine_path.is_file():
        raise RuntimeError("trtexec did not produce the identity engine")
    print(engine_path)


if __name__ == "__main__":
    main()
