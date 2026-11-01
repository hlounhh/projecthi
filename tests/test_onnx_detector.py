"""ONNX trust boundaries and delegated session updates without optional runtimes."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from ai.onnx_detector import _load, _reject_external_data, _replace_session, verify_onnx_bundle
from ai.pose_detector import POSE_SHA256


def write_manifest(directory, **updates):
    graph = directory / "model.onnx"
    graph.write_bytes(b"verified test bytes")
    manifest = {"schema_version": 1, "task": "pose", "image_size": 416,
                "source_sha256": POSE_SHA256, "models": {"onnx": "model.onnx"},
                "files": {"model.onnx": hashlib.sha256(graph.read_bytes()).hexdigest()}}
    manifest.update(updates)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return graph


def test_rejects_modified_graph_and_wrong_export_profile(tmp_path):
    graph = write_manifest(tmp_path)
    assert verify_onnx_bundle(tmp_path, "pose", 416) == graph
    with pytest.raises(ValueError, match="does not match"):
        verify_onnx_bundle(tmp_path, "pose", 320)
    with pytest.raises(ValueError, match="does not match"):
        verify_onnx_bundle(tmp_path, "detect", 416)
    graph.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        verify_onnx_bundle(tmp_path, "pose", 416)


def test_manifest_cannot_escape_bundle(tmp_path):
    write_manifest(tmp_path, files={"model.onnx": "0", "../outside.bin": "0"})
    # Put traversal first so failure cannot merely be a graph hash mismatch.
    data = json.loads((tmp_path / "manifest.json").read_text())
    data["files"] = {"../outside.bin": "0", "model.onnx": data["files"]["model.onnx"]}
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid export manifest path"):
        verify_onnx_bundle(tmp_path, "pose", 416)


@pytest.mark.parametrize("confidence", [False, float("nan"), float("inf"), 0, 1.1, "0.4"])
def test_invalid_confidence_fails_before_optional_imports(confidence):
    with pytest.raises(ValueError, match="Confidence"):
        _load(None, "pose", confidence, 416, 2)


def test_replaces_actual_backend_session():
    backend = SimpleNamespace(session="original")
    wrapper = SimpleNamespace(backend=backend)
    yolo = SimpleNamespace(predictor=SimpleNamespace(model=wrapper))
    _replace_session(yolo, "two-thread-session")
    assert backend.session == "two-thread-session"
    assert not hasattr(wrapper, "session")


@pytest.mark.parametrize("placement", ["initializer", "constant", "subgraph", "sparse", "function"])
def test_rejects_external_tensors_everywhere_before_runtime_loading(placement):
    onnx = pytest.importorskip("onnx")
    tensor = onnx.TensorProto(name="weight", data_type=onnx.TensorProto.FLOAT, dims=[1])
    tensor.data_location = onnx.TensorProto.EXTERNAL
    tensor.external_data.add(key="location", value="../outside.bin")
    model = onnx.helper.make_model(onnx.helper.make_graph([], "test", [], []))
    if placement == "initializer":
        model.graph.initializer.append(tensor)
    elif placement == "constant":
        model.graph.node.append(onnx.helper.make_node("Constant", [], ["out"], value=tensor))
    elif placement == "subgraph":
        graph = onnx.helper.make_graph([], "nested", [], [], initializer=[tensor])
        model.graph.node.append(onnx.helper.make_node("Loop", [], [], body=graph))
    elif placement == "sparse":
        model.graph.sparse_initializer.add().values.CopyFrom(tensor)
    else:
        function = model.functions.add(name="test", domain="local")
        function.attribute_proto.append(onnx.helper.make_attribute("weight", tensor))
    with pytest.raises(ValueError, match="External ONNX"):
        _reject_external_data(model)