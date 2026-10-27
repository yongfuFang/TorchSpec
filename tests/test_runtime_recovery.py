"""CPU-only regression tests; Ray scheduling is mocked, not cluster-tested."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runtime(monkeypatch):
    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(relative):
        name = "_runtime_test_" + relative.replace("/", "_").removesuffix(".py")
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    ray = stub(
        "ray",
        remote=lambda cls: cls,
        get=lambda value: value,
        get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: "control-node"),
    )
    queue = MagicMock()
    stub("ray.util.queue", Queue=queue)
    stub("ray.util.scheduling_strategies", NodeAffinitySchedulingStrategy=SimpleNamespace)
    stub("torchspec.utils.logging", logger=MagicMock())
    stub("torchspec.data.utils", length_grouped_order=MagicMock())
    stub("torchspec.training.data_fetcher", TrainSample=object)
    stub("torchspec.utils.memory", estimate_tensor_bytes=MagicMock())
    stub("torchspec.utils.types", InferenceInput=object, InferenceOutput=object)
    stub("torchspec.ray.ray_actor", RayActor=SimpleNamespace(get_node_ip=lambda: "127.0.0.1"))
    helpers = load("torchspec/transfer/mooncake/helpers.py")
    monkeypatch.setitem(sys.modules, "torchspec.transfer.mooncake.helpers", helpers)
    env = load("torchspec/utils/env.py")
    monkeypatch.setitem(sys.modules, "torchspec.utils.env", env)
    return SimpleNamespace(load=load, stub=stub, queue=queue, ray=ray, env=env)


@pytest.mark.parametrize("values", [None, (1.5, 20.0, 180.0), (0.0, 0.0, 0.0)])
def test_retry_settings_reach_runtime_config(runtime, values):
    config_type = runtime.load("torchspec/config/mooncake_config.py").MooncakeConfig
    fields = ("get_retry_wait_seconds", "get_retry_log_interval_seconds", "get_retry_max_wait_seconds")
    args = SimpleNamespace()
    if values is not None:
        for key, value in zip(fields, values):
            setattr(args, "mooncake_" + key, value)
    config = config_type.from_flat_args(args)
    expected = values if values is not None else (0.5, 10.0, 60.0)
    assert tuple(getattr(config, key) for key in fields) == expected


