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


def test_pythonpath_forwarding(runtime, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert "PYTHONPATH" not in runtime.env.get_torchspec_env_vars()
    monkeypatch.setenv("PYTHONPATH", "/shared/checkout:/shared/dependencies")
    assert runtime.env.get_torchspec_env_vars()["PYTHONPATH"] == "/shared/checkout:/shared/dependencies"


def test_train_and_eval_queues_stay_with_controller(runtime):
    controller_type = runtime.load("torchspec/controller/training_controller.py").AsyncTrainingController
    controller = controller_type(SimpleNamespace(per_dp_rank_batch_size=1), dp_size=2)
    assert len(controller.train_queues) == len(controller.eval_queues) == 2
    assert runtime.queue.call_count == 4
    for call in runtime.queue.call_args_list:
        strategy = call.kwargs["actor_options"]["scheduling_strategy"]
        assert strategy.node_id == "control-node"
        assert strategy.soft is False


@pytest.mark.parametrize("existing_controller", [False, True])
def test_manager_stays_with_driver(runtime, existing_controller):
    controller_factory, manager_factory = MagicMock(), MagicMock()
    runtime.stub("torchspec.controller.training_controller", AsyncTrainingController=controller_factory)
    runtime.stub("torchspec.controller.inference_manager", AsyncInferenceManager=manager_factory)
    setup = runtime.load("torchspec/controller/setup.py")
    supplied = MagicMock() if existing_controller else None
    group = MagicMock()
    args = SimpleNamespace(training_num_nodes=1, training_num_gpus_per_node=2, per_dp_rank_batch_size=1)
    controller, manager = setup.setup_async_training_with_engines(args, group, object(), [], supplied)
    strategy = manager_factory.options.call_args.kwargs["scheduling_strategy"]
    assert strategy.node_id == "control-node"
    assert strategy.soft is False
    assert manager is manager_factory.options.return_value.remote.return_value
    if existing_controller:
        assert controller is supplied
        controller_factory.options.assert_not_called()
    else:
        assert controller is controller_factory.options.return_value.remote.return_value
    group.set_train_queues.assert_called_once()
    group.set_eval_queues.assert_called_once()
