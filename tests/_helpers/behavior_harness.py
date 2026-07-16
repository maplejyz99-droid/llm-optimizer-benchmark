import argparse
import importlib.util
import shlex
import sys
import tempfile
import types
import uuid
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"


@contextmanager
def patched_modules(replacements):
    sentinel = object()
    previous = {}
    for name, module in replacements.items():
        previous[name] = sys.modules.get(name, sentinel)
        sys.modules[name] = module
    try:
        yield
    finally:
        for name, old_module in previous.items():
            if old_module is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


@contextmanager
def isolated_modules(*prefixes):
    """Restore the exact module cache for project-owned prefixes on exit."""
    prefixes = tuple(prefixes)

    def matches(name):
        return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)

    previous = {name: module for name, module in sys.modules.items() if matches(name)}
    for name in list(sys.modules):
        if matches(name):
            sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if matches(name):
                sys.modules.pop(name, None)
        sys.modules.update(previous)


def load_module_from_path(path, replacements=None, package=None):
    module_name = f"_behavior_{path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
        submodule_search_locations=[str(path.parent)] if package else None,
    )
    module = importlib.util.module_from_spec(spec)
    with patched_modules(replacements or {}):
        spec.loader.exec_module(module)
    return module


def make_fake_distributed_module():
    module = types.ModuleType("distributed")
    module.registered_backends = lambda: [None, "nccl"]
    return module


def load_base_config_module():
    return load_module_from_path(
        SRC_ROOT / "config" / "base.py",
        replacements={"distributed": make_fake_distributed_module()},
    )


def make_base_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_format", default="base", choices=["base"])
    return parser


def parse_base_args(argv):
    parser = make_base_parser()
    base_config = load_base_config_module()
    namespace, remainder = parser.parse_known_args(argv)
    args = base_config.parse_args(parser, remainder, namespace)
    return args, parser


def _continued_shell_commands(script_text):
    commands = []
    current = []
    for raw_line in script_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            if current:
                commands.append(" ".join(current))
                current = []
            continue
        if stripped.endswith("\\"):
            current.append(stripped[:-1].strip())
            continue
        current.append(stripped)
        commands.append(" ".join(current))
        current = []
    if current:
        commands.append(" ".join(current))
    return commands


def extract_main_argv_from_script(path):
    return [command["argv"] for command in extract_main_commands_from_script(path)]


def extract_main_commands_from_script(path):
    commands = []
    for command in _continued_shell_commands(Path(path).read_text()):
        tokens = shlex.split(command)
        if not tokens:
            continue
        try:
            main_index = next(
                idx
                for idx, token in enumerate(tokens)
                if token in {"./src/main.py", "src/main.py"}
            )
        except StopIteration:
            continue
        commands.append(
            {
                "launcher": tokens[:main_index],
                "main": tokens[main_index],
                "argv": tokens[main_index + 1 :],
            }
        )
    return commands


class FakeParam:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape

    def numel(self):
        total = 1
        for value in self.shape:
            total *= value
        return total

    def __repr__(self):
        return f"FakeParam({self.name!r}, shape={self.shape!r})"


class FakeWeight:
    def __init__(self, size):
        self._size = size

    def numel(self):
        return self._size


class FakeTinyModel:
    def __init__(self):
        self._params = [
            ("transformer.wte.weight", FakeParam("transformer.wte.weight", (4, 4))),
            ("block.attn.weight", FakeParam("block.attn.weight", (4, 4))),
            ("block.mlp.weight", FakeParam("block.mlp.weight", (4, 4))),
            ("lm_head.weight", FakeParam("lm_head.weight", (4, 4))),
        ]
        self.lm_head = types.SimpleNamespace(weight=FakeWeight(16))
        self.transformer = types.SimpleNamespace(
            wte=types.SimpleNamespace(weight=FakeWeight(16))
        )

    def to(self, device):
        self.device = device
        return self

    def parameters(self):
        return [param for _, param in self._params]

    def named_parameters(self):
        return list(self._params)

    def get_parameter_group_specs(self, config):
        return [
            {"params": ["block.attn.weight"], "lr": config.lr},
            {"params": ["block.mlp.weight"], "lr": config.lr * 0.5},
        ]

    def get_num_params(self, non_embedding=True):
        if non_embedding:
            return 32
        return sum(param.numel() for _, param in self._params)

    def __repr__(self):
        return "FakeTinyModel()"


class FakeBackend:
    def __init__(self, args, world_size=1):
        self.args = args
        self.world_size = world_size
        self.finalized = False
        self.finalize_count = 0
        self.barrier_count = 0

    def transform_model(self, model):
        return model

    def get_adjusted_args_for_process(self, args):
        return args

    def get_world_size(self):
        return self.world_size

    def is_master_process(self):
        return True

    def get_raw_model(self, model):
        return model

    def translate_model_parameter_name_for_node(self, parameter_name):
        return [parameter_name]

    def barrier(self):
        self.barrier_count += 1

    def broadcast_object(self, value, src=0):
        return value

    def all_gather_object(self, value):
        return [value] * self.get_world_size()

    def finalize(self):
        self.finalized = True
        self.finalize_count += 1


class FakeOptimizer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.state = {}
        self.param_groups = self._normalize_param_groups(args[0] if args else [])

    def _normalize_param_groups(self, params):
        if isinstance(params, list) and params and isinstance(params[0], dict):
            return [dict(group) for group in params]
        return [{"params": list(params) if isinstance(params, (list, tuple)) else params}]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.kwargs!r})"


class FakeMuonOptimizer(FakeOptimizer):
    def _normalize_param_groups(self, params):
        return [
            {
                "params": list(params) if isinstance(params, (list, tuple)) else params,
                "lr": self.kwargs.get("lr"),
                "adamw_lr": self.kwargs.get("adamw_lr"),
                "adamw_wd": self.kwargs.get("adamw_wd"),
            }
        ]


class FakeNewtonMuon(FakeMuonOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attached_model = None

    def attach_preconditioner(self, model):
        self.attached_model = model


class FakeScheduler:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return f"{self.__class__.__name__}({self.kwargs!r})"


class FakeCombinedScheduler(FakeScheduler):
    pass


def make_fake_torch_module():
    torch = types.ModuleType("torch")

    class AdamW(FakeOptimizer):
        pass

    class SGD(FakeOptimizer):
        pass

    class Muon(FakeOptimizer):
        pass

    class OneCycleLR(FakeScheduler):
        pass

    class LambdaLR(FakeScheduler):
        pass

    torch.optim = types.SimpleNamespace(
        AdamW=AdamW,
        SGD=SGD,
        Muon=Muon,
        lr_scheduler=types.SimpleNamespace(OneCycleLR=OneCycleLR, LambdaLR=LambdaLR),
    )
    torch.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            matmul=types.SimpleNamespace(allow_tf32=False),
            enable_flash_sdp=lambda value: None,
            enable_mem_efficient_sdp=lambda value: None,
            enable_math_sdp=lambda value: None,
        ),
        cudnn=types.SimpleNamespace(allow_tf32=False),
    )
    torch.cuda = types.SimpleNamespace(set_device=lambda device: None)
    torch.manual_seed = lambda seed: None
    torch.device = lambda device: device
    return torch


def make_optimizer_module(*class_names):
    module = types.ModuleType("optimizer_stub")
    for class_name in class_names:
        setattr(module, class_name, type(class_name, (FakeOptimizer,), {}))
    return module


def make_fake_main_replacements(capture):
    numpy_module = types.ModuleType("numpy")
    numpy_module.random = types.SimpleNamespace(seed=lambda seed: None)

    config_module = types.ModuleType("config")
    config_module.registered_formats = lambda: ["base"]
    config_module.parse_args_with_format = lambda format, base_parser, args, namespace: namespace

    wandb_module = types.ModuleType("wandb")

    def wandb_init(**kwargs):
        capture["wandb_init_kwargs"] = kwargs

    wandb_module.init = wandb_init
    wandb_module.define_metric = lambda *args, **kwargs: None
    wandb_module.log = lambda *args, **kwargs: None

    distributed_module = types.ModuleType("distributed")

    def make_backend_from_args(args):
        backend = FakeBackend(args, world_size=capture.get("world_size", 1))
        capture["backend"] = backend
        return backend

    distributed_module.make_backend_from_args = make_backend_from_args
    distributed_module.registered_backends = lambda: [None, "nccl"]

    data_utils_module = types.ModuleType("data.utils")
    data_utils_module.DataReader = object
    data_utils_module.get_dataset = lambda args: {}

    models_utils_module = types.ModuleType("models.utils")

    def get_model(args):
        capture["get_model_calls"] = capture.get("get_model_calls", 0) + 1
        return FakeTinyModel()

    models_utils_module.get_model = get_model

    optim_base_module = types.ModuleType("optim.base")

    def fake_train(**kwargs):
        capture["train_kwargs"] = kwargs
        if "train_error" in capture:
            raise capture["train_error"]
        return {"captured": True}

    optim_base_module.train = fake_train

    muon_module = types.ModuleType("optim.muon")
    muon_module.CombinedScheduler = FakeCombinedScheduler
    muon_module.DistributedMuon = type("DistributedMuon", (FakeOptimizer,), {})
    muon_module.Muon = type("Muon", (FakeMuonOptimizer,), {})

    newton_module = types.ModuleType("optim.newton_muon")
    newton_module.NewtonMuon = FakeNewtonMuon

    experimental_module = types.ModuleType("optim.experimental")
    softeq_muon_module = types.ModuleType("optim.experimental.softeq_muon")
    softeq_muon_module.SoftEqK2000Muon = type(
        "SoftEqK2000Muon", (FakeMuonOptimizer,), {}
    )

    scion_module = types.ModuleType("optim.scion")
    scion_module.Scion = type("Scion", (FakeOptimizer,), {})
    scion_module.ScionLight = type("ScionLight", (FakeOptimizer,), {})
    scion_module.scion_partitions = lambda group_specs, model, args: group_specs

    schedule_module = types.ModuleType("optim.schedule")
    schedule_module.cos_inf_schedule = lambda **kwargs: (lambda step: 1.0)
    schedule_module.wsd_schedule = lambda **kwargs: (lambda step: 1.0)

    run_manifest_module = types.ModuleType("run_manifest")
    run_manifest_module.build_data_manifest = lambda dataset, sources: {
        "schema_version": 1,
        "dataset": dataset,
        "semantics_id": "flat-next-token-v1",
        "artifacts": {},
        "identity": "fake-data-identity",
    }

    def build_run_manifest(args, data_manifest, **kwargs):
        manifest = {
            "schema_version": 1,
            "run_identity": "fake-run-identity",
            "data": data_manifest,
            "metric_semantics": {
                "validation_accuracy": "next_token_accuracy",
            },
        }
        capture["run_manifest"] = manifest
        return manifest

    run_manifest_module.build_run_manifest = build_run_manifest
    run_manifest_module.collect_runtime_identity = lambda: {
        "python": {"version": "3.10.19"},
        "packages": {},
    }

    def sanitized_config(args):
        capture["sanitized_config_calls"] = capture.get("sanitized_config_calls", 0) + 1
        sanitized = {}
        safe_notify_keys = {
            "notify_interval",
            "notify_method",
        }
        for key, value in vars(args).items():
            if key.startswith("notify_") and key not in safe_notify_keys:
                sanitized[key] = None if value is None else "<redacted>"
            else:
                sanitized[key] = value
        return sanitized

    run_manifest_module.sanitized_config = sanitized_config

    def ensure_compatible_manifest(path, manifest):
        capture.setdefault("lifecycle_events", []).append("manifest_checked")
        if "manifest_error" in capture:
            raise capture["manifest_error"]

    run_manifest_module.ensure_compatible_manifest = ensure_compatible_manifest

    def write_json_atomic(path, payload):
        capture.setdefault("atomic_json_writes", []).append((Path(path), payload))

    run_manifest_module.write_json_atomic = write_json_atomic

    return {
        "numpy": numpy_module,
        "torch": make_fake_torch_module(),
        "wandb": wandb_module,
        "config": config_module,
        "distributed": distributed_module,
        "data": types.ModuleType("data"),
        "data.utils": data_utils_module,
        "models": types.ModuleType("models"),
        "models.utils": models_utils_module,
        "optim": types.ModuleType("optim"),
        "optim.adafactor": make_optimizer_module("Adafactor"),
        "optim.ademamix": make_optimizer_module("AdEMAMix"),
        "optim.adopt": make_optimizer_module("ADOPT"),
        "optim.base": optim_base_module,
        "optim.cadamw": make_optimizer_module("CAdamW"),
        "optim.lamb": make_optimizer_module("Lamb"),
        "optim.lion": make_optimizer_module("Lion"),
        "optim.mars": make_optimizer_module("MARS"),
        "optim.magma": make_optimizer_module("MagmaAdamW", "MagmaMuon"),
        "optim.muon": muon_module,
        "optim.newton_muon": newton_module,
        "optim.prodigy": make_optimizer_module("Prodigy"),
        "optim.schedule": schedule_module,
        "optim.schedulefree": make_optimizer_module("AdamWScheduleFree", "SGDScheduleFree"),
        "optim.scion": scion_module,
        "optim.sign": make_optimizer_module("Signum"),
        "optim.soap": make_optimizer_module("SOAP"),
        "optim.experimental": experimental_module,
        "optim.experimental.softeq_muon": softeq_muon_module,
        "optim.sophia": make_optimizer_module("SophiaG"),
        "run_manifest": run_manifest_module,
    }


def load_main_with_fakes(capture):
    module = load_module_from_path(
        SRC_ROOT / "main.py",
        replacements=make_fake_main_replacements(capture),
    )
    module.get_data_readers = lambda args: {"train": object(), "val": object()}
    return module


def make_args_for_main(extra_argv):
    argv = [
        "--device",
        "cpu",
        "--dataset",
        "fineweb",
        "--iterations",
        "10",
        "--warmup_steps",
        "2",
        "--results_base_folder",
        tempfile.mkdtemp(prefix="llmopt-behavior-"),
    ]
    argv.extend(extra_argv)
    return parse_base_args(argv)
