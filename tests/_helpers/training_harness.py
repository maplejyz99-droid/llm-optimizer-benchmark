import importlib
import sys
import types
from contextlib import contextmanager, nullcontext
from pathlib import Path

from tests._helpers.behavior_harness import REPO_ROOT, SRC_ROOT, patched_modules


class FakeScalar:
    def __init__(self, value, events=None, label=None):
        self.value = float(value)
        self.events = events
        self.label = label

    def __truediv__(self, other):
        return FakeScalar(self.value / other, self.events, self.label)

    def __lt__(self, other):
        other_value = other.value if isinstance(other, FakeScalar) else other
        return self.value < other_value

    def backward(self):
        if self.events is not None:
            self.events.append(("loss.backward", self.label))

    def detach(self):
        return self

    def cpu(self):
        return self

    def item(self):
        return self.value

    def __repr__(self):
        return f"FakeScalar({self.value!r})"


class FakeTensorList:
    def __init__(self, values):
        self.values = values

    def mean(self):
        if not self.values:
            return FakeScalar(0.0)
        raw_values = [
            value.value if isinstance(value, FakeScalar) else float(value)
            for value in self.values
        ]
        return FakeScalar(sum(raw_values) / len(raw_values))


class FakeLogits:
    def __init__(self, events):
        self.events = events

    def view(self, *shape):
        self.events.append(("logits.view", shape))
        return self

    def size(self, dim=None):
        return 4 if dim == -1 else (1, 4)


class FakeTrainingModel:
    def __init__(self, events, parameter_source="model.parameters"):
        self.events = events
        self.forward_calls = []
        self.parameter_source = parameter_source

    def train(self):
        self.events.append(("model.train",))

    def eval(self):
        self.events.append(("model.eval",))

    def parameters(self):
        return [self.parameter_source]

    def zero_grad(self):
        self.events.append(("model.zero_grad",))

    def generate_from_string(self, prefix, max_new_tokens, temperature, top_k):
        self.events.append(
            (
                "model.generate_from_string",
                prefix,
                max_new_tokens,
                temperature,
                top_k,
            )
        )
        return f"{prefix}<generated>"

    def __call__(self, x, targets=None, moe=False, **kwargs):
        self.forward_calls.append({"x": x, "targets": targets, "moe": moe, **kwargs})
        self.events.append(("model.forward", x, targets, moe, dict(kwargs)))
        if kwargs.get("get_logits"):
            return {"logits": FakeLogits(self.events)}
        return {"loss": FakeScalar(2.0, self.events, "train"), "aux_losses": {}}


class FakeDDPModel:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        return getattr(self.module, name)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def parameters(self):
        return ["ddp.wrapper.parameters"]


class FakeOptimizerForTrain:
    def __init__(self, events, lr=0.01):
        self.events = events
        self.param_groups = [{"lr": lr}]

    def step(self, **kwargs):
        self.events.append(("opt.step", dict(kwargs)))

    def zero_grad(self, set_to_none=False):
        self.events.append(("opt.zero_grad", set_to_none))

    def train(self):
        self.events.append(("opt.train",))

    def eval(self):
        self.events.append(("opt.eval",))

    def update_hessian(self):
        self.events.append(("opt.update_hessian",))

    def update_last_grad(self):
        self.events.append(("opt.update_last_grad",))


class FakePreconditionedOptimizerForTrain(FakeOptimizerForTrain):
    def __init__(self, events, lr=0.01, precond_flag=True):
        super().__init__(events, lr=lr)
        self.precond_flag = precond_flag

    def precond_flag_for_step(self):
        self.events.append(("opt.precond_flag_for_step",))
        return self.precond_flag


class FakeSchedulerForTrain:
    def __init__(self, events):
        self.events = events

    def step(self):
        self.events.append(("scheduler.step",))


class FakeDataReader:
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.num_tokens = 10_000
        self.steps = []
        self.batch_count = 0

    def set_step(self, step):
        self.steps.append(step)
        self.events.append((f"{self.name}.set_step", step))

    def num_batches(self):
        self.events.append((f"{self.name}.num_batches",))
        return 3


class FakeDistributedBackendForTrain:
    def __init__(self, events, master=True, world_size=1):
        self.events = events
        self.master = master
        self.world_size = world_size

    def is_master_process(self):
        return self.master

    def get_world_size(self):
        return self.world_size

    def get_raw_model(self, model):
        return model

    def get_context_for_microstep_forward(
        self, model, microstep_idx, gradient_accumulation_steps
    ):
        self.events.append(
            ("microstep_context", microstep_idx, gradient_accumulation_steps)
        )
        return nullcontext()


def make_training_cfg(**overrides):
    defaults = {
        "compile": False,
        "device": "cpu",
        "dtype": "float32",
        "resume_from": None,
        "iterations": 1,
        "acc_steps": 1,
        "sequence_length": 8,
        "batch_size": 2,
        "opt": "adamw",
        "scheduler": "cos",
        "moe": False,
        "grad_clip": 1.0,
        "warmup_steps": 0,
        "eval_interval": 100,
        "eval_batches": 2,
        "full_eval_at": [],
        "permanent_ckpt_interval": 0,
        "latest_ckpt_interval": 0,
        "weight_average": False,
        "wa_horizon": 1,
        "wa_interval": 1,
        "wa_use_temp_dir": True,
        "wa_dtype": "float32",
        "exponential_weight_average": False,
        "ewa_interval": 1,
        "ewa_decay": 0.99,
        "ewa_after_warmup": False,
        "log_dynamics": False,
        "dynamics_logger_cfg": "",
        "results_base_folder": "",
        "wandb": False,
        "log_interval": 0,
        "log_parameter_norms": False,
        "norm_order": 2,
        "eval_seq_prefix": "none",
        "plot_router_logits": False,
        "gn_inner_iters": 1,
        "gn_inner_wd": 0.0,
        "gn_log_inner_steps": False,
        "gn_linesearch": False,
        "gn_ls_range": [1.0],
        "sophia_bs": 1,
        "precondition_frequency": 10,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def make_fake_torch_for_train(events):
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torch.float64 = "float64"
    torch.compile = lambda model: model
    torch.amp = types.SimpleNamespace(autocast=lambda **kwargs: nullcontext())

    def clip_grad_norm_(parameters, max_norm):
        events.append(("clip_grad_norm", tuple(parameters), max_norm))
        return FakeScalar(0.25)

    def tensor(value, device=None):
        if isinstance(value, list):
            return FakeTensorList(value)
        return FakeScalar(value)

    torch.tensor = tensor
    torch.nn = types.SimpleNamespace(
        parallel=types.SimpleNamespace(DistributedDataParallel=FakeDDPModel),
        utils=types.SimpleNamespace(clip_grad_norm_=clip_grad_norm_),
        functional=types.SimpleNamespace(
            cross_entropy=lambda *args, **kwargs: FakeScalar(1.0, events, "sampled")
        ),
    )
    torch.distributions = types.SimpleNamespace(
        Categorical=lambda logits: types.SimpleNamespace(sample=lambda: logits)
    )
    return torch


def make_training_replacements(capture):
    events = capture["events"]

    yaml_module = types.ModuleType("yaml")
    yaml_module.safe_load = lambda stream: {}

    logger_pkg = types.ModuleType("logger")
    logger_module = types.ModuleType("logger.logger")

    class FakeDynamicsLogger:
        def __init__(self, *args, **kwargs):
            events.append(("dynamics_logger.init",))
            self.iteration = 0

    logger_module.DynamicsLogger = FakeDynamicsLogger

    notify_module = types.ModuleType("notify")
    notify_module.maybe_notify = lambda cfg, **kwargs: events.append(
        ("maybe_notify", kwargs)
    )

    weight_averaging_module = types.ModuleType("optim.weight_averaging")

    class FakeWeightAverager:
        def __init__(self, model, **kwargs):
            save_dir = kwargs["save_dir"]
            events.append(
                (
                    "WeightAverager.init",
                    {
                        "horizon": kwargs["horizon"],
                        "interval": kwargs["interval"],
                        "save_dir": (
                            None if save_dir is None else Path(save_dir).as_posix()
                        ),
                        "dtype": kwargs["dtype"],
                        "count": kwargs["count"],
                    },
                )
            )

        def step(self, model, is_master):
            events.append(("weight_averager.step", is_master))

    class FakeExponentialWeightAverager:
        def __init__(self, model, **kwargs):
            events.append(
                (
                    "ExponentialWeightAverager.init",
                    {
                        "interval": kwargs["interval"],
                        "decay": kwargs["decay"],
                        "warmup": kwargs["warmup"],
                        "dtype": kwargs["dtype"],
                    },
                )
            )

        def step(self, model, is_master):
            events.append(("ewa.step", is_master))

    def fake_eval_wa(curr_iter, *args, **kwargs):
        events.append(("eval_wa", curr_iter, kwargs.get("full_eval")))

    def fake_eval_ewa(curr_iter, *args, **kwargs):
        events.append(("eval_ewa", curr_iter, kwargs.get("full_eval")))

    weight_averaging_module.WeightAverager = FakeWeightAverager
    weight_averaging_module.ExponentialWeightAverager = FakeExponentialWeightAverager
    weight_averaging_module.eval_wa = fake_eval_wa
    weight_averaging_module.eval_ewa = fake_eval_ewa

    gn_module = types.ModuleType("optim.gn")
    gn_module.clone_param_dict = lambda model: {}
    gn_module.clone_param_dict_from_named_params = lambda params: {}
    gn_module.current_param_dict = lambda model: {}
    gn_module.sub_param_dict = lambda current, initial: {}
    def fake_line_search_over_direction(**kwargs):
        events.append(
            (
                "line_search_over_direction",
                len(kwargs["batches"]),
                tuple(kwargs["ls_range"]),
            )
        )
        return 1.0, {}

    gn_module.line_search_over_direction = fake_line_search_over_direction

    def compute_gn_step(**kwargs):
        events.append(("compute_gn_step", kwargs["mode"]))
        metrics = types.SimpleNamespace(
            loss=1.5,
            base_loss=1.25,
            gradient_norm=0.5,
            param_norm=2.0,
        )
        return [], metrics

    gn_module.compute_gn_step = compute_gn_step

    utils_module = types.ModuleType("optim.utils")

    def fake_get_batch(reader, device):
        events.append(("get_batch", reader.name, device))
        reader.batch_count += 1
        if capture.get("unique_batches"):
            return f"{reader.name}_x_{reader.batch_count}", f"{reader.name}_y_{reader.batch_count}"
        return f"{reader.name}_x", f"{reader.name}_y"

    def fake_eval(
        model,
        val_reader,
        device,
        max_num_batches,
        ctx,
        moe,
        get_router_logits,
        cfg,
    ):
        events.append(("eval", max_num_batches, moe, get_router_logits))
        return (
            0.75,
            1.25,
            2.5,
            capture.get("eval_aux_losses", {}),
            capture.get("router_logits", []),
        )

    def fake_save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir):
        events.append(("save_checkpoint", curr_iter, Path(ckpt_dir).as_posix()))
        events.append(("save_checkpoint_scheduler_is_none", curr_iter, scheduler is None))

    def fake_save_worker_state(ckpt_dir):
        events.append(("save_worker_state", Path(ckpt_dir).as_posix()))

    utils_module.get_batch = fake_get_batch
    utils_module.eval = fake_eval
    utils_module.save_checkpoint = fake_save_checkpoint
    utils_module.save_worker_state = fake_save_worker_state
    def fake_load_checkpoint(model, opt, scheduler, ckpt_path, device):
        events.append(
            (
                "load_checkpoint",
                Path(ckpt_path).as_posix(),
                device,
                scheduler is None,
            )
        )
        return capture.get("checkpoint_iter", 0)

    def fake_load_worker_state(ckpt_dir):
        events.append(("load_worker_state", Path(ckpt_dir).as_posix()))

    def fake_extend_onecycle_total_steps(scheduler, iterations):
        events.append(("extend_onecycle_total_steps", scheduler is None, iterations))

    utils_module.load_checkpoint = fake_load_checkpoint
    utils_module.load_worker_state = fake_load_worker_state
    utils_module.extend_onecycle_total_steps = fake_extend_onecycle_total_steps
    def fake_get_parameter_norms(model, order):
        events.append(("get_parameter_norms", order))
        return 9.0

    def fake_log_prodigy_lr(opt):
        events.append(("log_prodigy_lr",))
        return [0.123]

    utils_module.get_parameter_norms = fake_get_parameter_norms
    utils_module.log_prodigy_lr = fake_log_prodigy_lr
    def fake_visualize_routing(router_logits, cfg):
        events.append(("visualize_routing", tuple(router_logits)))
        return capture.get("routing_logs", {})

    utils_module.visualize_routing = fake_visualize_routing

    wandb_module = types.ModuleType("wandb")
    wandb_module.log = lambda logs: events.append(("wandb.log", logs))
    wandb_module.Table = lambda columns: types.SimpleNamespace(
        add_data=lambda *args: events.append(("wandb.Table.add_data", args))
    )
    wandb_module.run = types.SimpleNamespace(name="fake-run")

    return {
        "torch": make_fake_torch_for_train(events),
        "yaml": yaml_module,
        "logger": logger_pkg,
        "logger.logger": logger_module,
        "notify": notify_module,
        "optim.weight_averaging": weight_averaging_module,
        "optim.gn": gn_module,
        "optim.utils": utils_module,
        "wandb": wandb_module,
    }


@contextmanager
def loaded_training_base_module(capture):
    sentinel = object()
    old_base = sys.modules.pop("optim.base", sentinel)
    old_path = list(sys.path)
    sys.path.insert(0, str(SRC_ROOT))
    try:
        with patched_modules(make_training_replacements(capture)):
            module = importlib.import_module("optim.base")
            yield module
    finally:
        sys.modules.pop("optim.base", None)
        if old_base is not sentinel:
            sys.modules["optim.base"] = old_base
        sys.path[:] = old_path


def make_train_components(
    events,
    master=True,
    world_size=1,
    use_ddp=False,
    use_precond_optimizer=False,
    precond_flag=True,
):
    parameter_source = "ddp.module.parameters" if use_ddp else "model.parameters"
    raw_model = FakeTrainingModel(events, parameter_source=parameter_source)
    model = FakeDDPModel(raw_model) if use_ddp else raw_model
    optimizer = (
        FakePreconditionedOptimizerForTrain(events, precond_flag=precond_flag)
        if use_precond_optimizer
        else FakeOptimizerForTrain(events)
    )
    scheduler = FakeSchedulerForTrain(events)
    backend = FakeDistributedBackendForTrain(
        events, master=master, world_size=world_size
    )
    datareaders = {
        "train": FakeDataReader("train", events),
        "val": FakeDataReader("val", events),
    }
    return model, optimizer, scheduler, backend, datareaders
