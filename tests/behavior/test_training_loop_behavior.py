import random
import unittest
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from tests._helpers.training_harness import (
    loaded_training_base_module,
    make_train_components,
    make_training_cfg,
)


def event_names(events):
    return [event[0] for event in events]


def assert_in_order(test_case, names, expected):
    cursor = 0
    for name in names:
        if cursor < len(expected) and name == expected[cursor]:
            cursor += 1
    test_case.assertEqual(cursor, len(expected), names)


def clip_grad_events(events):
    return [event for event in events if event[0] == "clip_grad_norm"]


USE_DEFAULT_SCHEDULER = object()


class TrainingLoopBehaviorTest(unittest.TestCase):
    def run_train(
        self,
        cfg,
        checkpoint_iter=0,
        scheduler_override=USE_DEFAULT_SCHEDULER,
        master=True,
        world_size=1,
        rank=0,
        use_ddp=False,
        use_precond_optimizer=False,
        precond_flag=True,
        capture_overrides=None,
        events=None,
    ):
        events = [] if events is None else events
        capture = {"events": events, "checkpoint_iter": checkpoint_iter}
        if capture_overrides:
            capture.update(capture_overrides)
        model, opt, scheduler, backend, datareaders = make_train_components(
            events,
            master=master,
            world_size=world_size,
            rank=rank,
            use_ddp=use_ddp,
            use_precond_optimizer=use_precond_optimizer,
            precond_flag=precond_flag,
        )
        if scheduler_override is not USE_DEFAULT_SCHEDULER:
            scheduler = scheduler_override
        with loaded_training_base_module(capture) as training_base:
            with redirect_stdout(StringIO()):
                stats = training_base.train(
                    model=model,
                    opt=opt,
                    datareaders=datareaders,
                    scheduler=scheduler,
                    exp_dir=Path("/tmp/llmopt-behavior-exp"),
                    distributed_backend=backend,
                    cfg=cfg,
                )
        return events, stats

    def test_raw_final_eval_uses_positive_cap_without_affecting_periodic_eval(self):
        events = []
        capture = {"events": events, "checkpoint_iter": 0}
        cfg = make_training_cfg(
            iterations=10,
            eval_batches=2,
            final_eval_batches=3,
        )
        _model, _opt, _scheduler, _backend, readers = make_train_components(events)
        readers["val"].num_batches = lambda: 11

        with loaded_training_base_module(capture) as training_base:
            self.assertEqual(
                training_base._get_eval_batch_count(10, readers["val"], cfg, False),
                3,
            )
            self.assertEqual(
                training_base._get_eval_batch_count(4, readers["val"], cfg, True),
                3,
            )
            self.assertEqual(
                training_base._get_eval_batch_count(4, readers["val"], cfg, False),
                2,
            )

    def test_normal_optimizer_one_step_order_is_preserved(self):
        cfg = make_training_cfg(iterations=1, scheduler="cos", eval_interval=100)
        events, stats = self.run_train(cfg)
        names = event_names(events)

        assert_in_order(
            self,
            names,
            [
                "get_batch",
                "microstep_context",
                "model.forward",
                "loss.backward",
                "clip_grad_norm",
                "opt.step",
                "scheduler.step",
                "opt.zero_grad",
            ],
        )
        self.assertEqual(stats["completed_iterations"], 1)
        self.assertEqual(events.count(("scheduler.step",)), 1)

    def test_gradient_accumulation_microsteps_run_before_single_optimizer_step(self):
        cfg = make_training_cfg(
            iterations=1,
            acc_steps=3,
            scheduler="cos",
            eval_interval=100,
        )
        events, stats = self.run_train(cfg)
        names = event_names(events)
        train_batch_events = [
            event for event in events if event == ("get_batch", "train", "cpu")
        ]
        microstep_events = [
            event for event in events if event[0] == "microstep_context"
        ]
        train_forward_events = [
            event
            for event in events
            if event[0] == "model.forward" and event[1] == "train_x"
        ]

        self.assertEqual(len(train_batch_events), 3)
        self.assertEqual(
            microstep_events,
            [
                ("microstep_context", 0, 3),
                ("microstep_context", 1, 3),
                ("microstep_context", 2, 3),
            ],
        )
        self.assertEqual(len(train_forward_events), 3)
        self.assertEqual(events.count(("loss.backward", "train")), 3)
        self.assertEqual(events.count(("opt.step", {})), 1)
        assert_in_order(
            self,
            names,
            [
                "get_batch",
                "microstep_context",
                "model.forward",
                "loss.backward",
                "get_batch",
                "microstep_context",
                "model.forward",
                "loss.backward",
                "get_batch",
                "microstep_context",
                "model.forward",
                "loss.backward",
                "opt.step",
            ],
        )
        self.assertEqual(stats["completed_iterations"], 1)

    def test_scheduler_none_skips_scheduler_step_but_keeps_optimizer_step(self):
        cfg = make_training_cfg(
            iterations=1,
            scheduler="none",
            eval_interval=100,
        )
        events, stats = self.run_train(cfg, scheduler_override=None)
        names = event_names(events)

        self.assertIn(("opt.step", {}), events)
        self.assertIn(("opt.zero_grad", True), events)
        self.assertNotIn("scheduler.step", names)
        self.assertEqual(stats["completed_iterations"], 1)

    def test_schedulefree_train_mode_runs_before_standard_optimizer_step(self):
        cfg = make_training_cfg(
            iterations=1,
            opt="sf-adamw",
            scheduler="none",
            eval_interval=100,
        )
        events, stats = self.run_train(cfg, scheduler_override=None)
        names = event_names(events)

        assert_in_order(self, names, ["opt.train", "opt.step", "opt.zero_grad"])
        self.assertEqual(stats["completed_iterations"], 1)

    def test_precondition_flag_is_read_once_and_passed_to_each_microstep(self):
        cfg = make_training_cfg(
            iterations=1,
            acc_steps=2,
            scheduler="none",
            eval_interval=100,
        )
        events, stats = self.run_train(
            cfg,
            scheduler_override=None,
            use_precond_optimizer=True,
            precond_flag=True,
        )
        train_forward_events = [
            event
            for event in events
            if event[0] == "model.forward" and event[1] == "train_x"
        ]

        self.assertEqual(events.count(("opt.precond_flag_for_step",)), 1)
        self.assertEqual(len(train_forward_events), 2)
        self.assertEqual(
            [event[4]["precond_flag"] for event in train_forward_events],
            [True, True],
        )
        self.assertEqual(stats["completed_iterations"], 1)

    def test_final_eval_tokens_include_all_accumulation_microsteps(self):
        cfg = make_training_cfg(
            iterations=2,
            acc_steps=3,
            eval_interval=100,
            wandb=True,
        )
        events, stats = self.run_train(cfg)
        eval_logs = [
            event[1]
            for event in events
            if event[0] == "wandb.log" and "iter" in event[1]
        ]

        self.assertEqual([logs["iter"] for logs in eval_logs], [0, 2])
        self.assertEqual(eval_logs[-1]["tokens"], 96)
        self.assertEqual(stats["completed_iterations"], 2)

    def test_train_logging_uses_last_microstep_loss_after_accumulation(self):
        cfg = make_training_cfg(
            iterations=1,
            acc_steps=3,
            eval_interval=100,
            log_interval=1,
            wandb=True,
        )
        events, stats = self.run_train(cfg)
        train_logs = [
            event[1]
            for event in events
            if event[0] == "wandb.log" and "train/loss" in event[1]
        ]

        self.assertEqual(len(train_logs), 1)
        self.assertEqual(train_logs[0]["train/loss"], 2.0)
        self.assertEqual(events.count(("loss.backward", "train")), 3)
        self.assertEqual(stats["completed_iterations"], 1)

    def test_sophiag_hessian_uses_last_microstep_batch_after_accumulation(self):
        cfg = make_training_cfg(
            iterations=1,
            opt="sophiag",
            acc_steps=3,
            scheduler="cos",
            eval_interval=100,
            precondition_frequency=1,
        )
        events, stats = self.run_train(
            cfg,
            capture_overrides={"unique_batches": True},
        )
        train_forward_events = [
            event
            for event in events
            if event[0] == "model.forward" and event[1].startswith("train_x_")
        ]
        regular_forwards = [
            event for event in train_forward_events if "get_logits" not in event[4]
        ]
        sampled_forwards = [
            event for event in train_forward_events if event[4].get("get_logits")
        ]

        self.assertEqual([event[1] for event in regular_forwards], [
            "train_x_1",
            "train_x_2",
            "train_x_3",
        ])
        self.assertEqual(len(sampled_forwards), 1)
        self.assertEqual(sampled_forwards[0][1], "train_x_3")
        self.assertEqual(sampled_forwards[0][2], "train_y_3")
        self.assertIn(("opt.update_hessian",), events)
        self.assertEqual(stats["completed_iterations"], 1)

    def test_mars_accumulation_finishes_before_last_grad_update(self):
        cfg = make_training_cfg(
            iterations=1,
            opt="mars",
            acc_steps=2,
            scheduler="cos",
            eval_interval=100,
        )
        events, stats = self.run_train(
            cfg,
            capture_overrides={"unique_batches": True},
        )
        names = event_names(events)
        train_forward_events = [
            event
            for event in events
            if event[0] == "model.forward" and event[1].startswith("train_x_")
        ]

        self.assertEqual([event[1] for event in train_forward_events], [
            "train_x_1",
            "train_x_2",
        ])
        assert_in_order(
            self,
            names,
            [
                "model.forward",
                "loss.backward",
                "model.forward",
                "loss.backward",
                "opt.step",
                "scheduler.step",
                "opt.zero_grad",
                "opt.update_last_grad",
            ],
        )
        self.assertEqual(events.count(("loss.backward", "train")), 2)
        self.assertEqual(stats["completed_iterations"], 1)

    def test_normal_optimizer_grad_clip_uses_model_parameters_without_ddp(self):
        cfg = make_training_cfg(iterations=1, scheduler="cos", eval_interval=100)
        events, stats = self.run_train(cfg)

        self.assertEqual(
            clip_grad_events(events),
            [("clip_grad_norm", ("model.parameters",), 1.0)],
        )
        self.assertEqual(stats["completed_iterations"], 1)

    def test_normal_optimizer_grad_clip_uses_module_parameters_with_ddp(self):
        cfg = make_training_cfg(iterations=1, scheduler="cos", eval_interval=100)
        events, stats = self.run_train(cfg, use_ddp=True)

        self.assertEqual(
            clip_grad_events(events),
            [("clip_grad_norm", ("ddp.module.parameters",), 1.0)],
        )
        self.assertEqual(stats["completed_iterations"], 1)

    def test_gn_grad_clip_uses_model_parameters_without_ddp(self):
        cfg = make_training_cfg(
            iterations=1,
            opt="gn-prox",
            scheduler="cos",
            eval_interval=100,
            gn_inner_iters=2,
        )
        events, stats = self.run_train(cfg)

        self.assertEqual(
            clip_grad_events(events),
            [("clip_grad_norm", ("model.parameters",), 1.0)] * 2,
        )
        self.assertEqual(stats["completed_iterations"], 1)

    def test_gn_grad_clip_uses_module_parameters_with_ddp(self):
        cfg = make_training_cfg(
            iterations=1,
            opt="gn-prox",
            scheduler="cos",
            eval_interval=100,
            gn_inner_iters=2,
        )
        events, stats = self.run_train(cfg, use_ddp=True)

        self.assertEqual(
            clip_grad_events(events),
            [("clip_grad_norm", ("ddp.module.parameters",), 1.0)] * 2,
        )
        self.assertEqual(stats["completed_iterations"], 1)

    def test_gn_rejects_multiple_ranks_before_data_or_optimizer_side_effects(self):
        cfg = make_training_cfg(
            iterations=1,
            opt="gn-prox",
            expected_world_size=2,
        )
        events = []

        with self.assertRaisesRegex(RuntimeError, "single-rank"):
            self.run_train(cfg, world_size=2, use_ddp=True, events=events)

        names = event_names(events)
        self.assertNotIn("get_batch", names)
        self.assertNotIn("opt.step", names)
        self.assertNotIn("save_checkpoint", names)

    def test_grad_clip_zero_skips_clip_but_keeps_optimizer_step(self):
        cfg = make_training_cfg(
            iterations=1,
            scheduler="cos",
            eval_interval=100,
            grad_clip=0.0,
        )
        events, stats = self.run_train(cfg, use_ddp=True)
        names = event_names(events)

        self.assertEqual(clip_grad_events(events), [])
        self.assertIn("opt.step", names)
        self.assertEqual(stats["completed_iterations"], 1)

    def test_eval_runs_at_initial_and_final_iterations(self):
        cfg = make_training_cfg(iterations=1, eval_interval=100, wandb=True)
        events, stats = self.run_train(cfg)
        wandb_logs = [
            event[1]
            for event in events
            if event[0] == "wandb.log" and "iter" in event[1]
        ]

        self.assertEqual([logs["iter"] for logs in wandb_logs], [0, 1])
        self.assertIn("val/loss", wandb_logs[0])
        self.assertIn("final-val/loss", wandb_logs[1])
        self.assertEqual(stats["completed_iterations"], 1)

    def test_full_eval_uses_final_wandb_keys_and_all_val_batches(self):
        cfg = make_training_cfg(
            iterations=2,
            eval_interval=100,
            full_eval_at=[1],
            wandb=True,
        )
        events, stats = self.run_train(cfg)
        eval_events = [event for event in events if event[0] == "eval"]
        wandb_logs = [
            event[1]
            for event in events
            if event[0] == "wandb.log" and "iter" in event[1]
        ]

        self.assertEqual(
            eval_events,
            [
                ("eval", 2, False, False),
                ("eval", 3, False, False),
                ("eval", 3, False, False),
            ],
        )
        self.assertEqual([logs["iter"] for logs in wandb_logs], [0, 1, 2])
        self.assertIn("val/loss", wandb_logs[0])
        self.assertIn("final-val/loss", wandb_logs[1])
        self.assertIn("final-val/loss", wandb_logs[2])
        self.assertEqual(events.count(("val.num_batches",)), 2)
        self.assertEqual(stats["completed_iterations"], 2)

    def test_master_only_eval_and_weight_average_eval_finish_before_barrier(self):
        cfg = make_training_cfg(
            iterations=0,
            eval_interval=1,
            exponential_weight_average=True,
        )
        events, _stats = self.run_train(cfg)

        assert_in_order(self, event_names(events), ["eval", "eval_ewa", "barrier"])
        self.assertEqual(events.count(("barrier",)), 1)

    def test_evaluation_restores_training_rng_state(self):
        cfg = make_training_cfg(iterations=0, eval_interval=1)
        rng_state = {"value": 17}

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            random.seed(123)
            np.random.seed(456)
            expected_python = random.Random(123).random()
            expected_numpy = np.random.RandomState(456).random_sample()

            self.run_train(
                cfg,
                capture_overrides={
                    "eval_consumes_rng": True,
                    "torch_rng_state": rng_state,
                },
            )

            self.assertEqual(rng_state["value"], 17)
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(np.random.random(), expected_numpy)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def test_master_eval_failure_is_synchronized_before_all_ranks_raise(self):
        cfg = make_training_cfg(iterations=0, eval_interval=1, expected_world_size=2)
        events = []

        with self.assertRaisesRegex(RuntimeError, "synthetic eval failure"):
            self.run_train(
                cfg,
                world_size=2,
                events=events,
                capture_overrides={
                    "eval_error": RuntimeError("synthetic eval failure"),
                },
            )

        names = event_names(events)
        self.assertIn("all_gather_object", names)
        self.assertNotIn("barrier", names)
        self.assertNotIn("get_batch", names)

    def test_non_master_eval_and_log_returns_without_eval_side_effects(self):
        events = []
        capture = {"events": events, "checkpoint_iter": 0}
        model, opt, _scheduler, backend, datareaders = make_train_components(
            events, master=False
        )
        cfg = make_training_cfg(wandb=True, eval_seq_prefix="Hello")

        with loaded_training_base_module(capture) as training_base:
            with redirect_stdout(StringIO()):
                result = training_base.eval_and_log(
                    tokens=0,
                    curr_iter=0,
                    epoch=0.0,
                    model=model,
                    val_reader=datareaders["val"],
                    type_ctx=nullcontext(),
                    distributed_backend=backend,
                    cfg=cfg,
                    opt=opt,
                )

        self.assertEqual(result, (None, None, None))
        self.assertEqual(events, [])

    def test_eval_and_log_sets_model_eval_and_restores_train_for_normal_optimizer(self):
        events = []
        capture = {"events": events, "checkpoint_iter": 0}
        model, opt, _scheduler, backend, datareaders = make_train_components(events)
        cfg = make_training_cfg(
            iterations=10,
            eval_interval=2,
            opt="adamw",
            wandb=False,
        )

        with loaded_training_base_module(capture) as training_base:
            with redirect_stdout(StringIO()):
                result = training_base.eval_and_log(
                    tokens=20,
                    curr_iter=2,
                    epoch=0.0,
                    model=model,
                    val_reader=datareaders["val"],
                    type_ctx=nullcontext(),
                    distributed_backend=backend,
                    cfg=cfg,
                    opt=opt,
                )
        names = event_names(events)

        self.assertEqual(result, (1.25, 2.5, 0.75))
        assert_in_order(self, names, ["model.eval", "val.set_step", "eval", "model.train"])
        self.assertNotIn("opt.eval", names)

    def test_eval_and_log_sets_schedulefree_optimizer_eval_before_eval(self):
        events = []
        capture = {"events": events, "checkpoint_iter": 0}
        model, opt, _scheduler, backend, datareaders = make_train_components(events)
        cfg = make_training_cfg(
            iterations=10,
            eval_interval=2,
            opt="sf-adamw",
            wandb=True,
        )

        with loaded_training_base_module(capture) as training_base:
            with redirect_stdout(StringIO()):
                result = training_base.eval_and_log(
                    tokens=20,
                    curr_iter=2,
                    epoch=0.0,
                    model=model,
                    val_reader=datareaders["val"],
                    type_ctx=nullcontext(),
                    distributed_backend=backend,
                    cfg=cfg,
                    opt=opt,
                )
        names = event_names(events)

        self.assertEqual(result, (1.25, 2.5, 0.75))
        assert_in_order(
            self,
            names,
            ["model.eval", "opt.eval", "val.set_step", "eval", "wandb.log", "model.train"],
        )

    def test_schedulefree_eval_sets_optimizer_eval_and_train_step_restores_train(self):
        cfg = make_training_cfg(
            iterations=1,
            opt="sf-adamw",
            scheduler="none",
            eval_interval=100,
        )
        events, stats = self.run_train(cfg, scheduler_override=None)
        names = event_names(events)

        assert_in_order(
            self,
            names,
            [
                "model.eval",
                "opt.eval",
                "eval",
                "model.train",
                "get_batch",
                "opt.train",
                "opt.step",
            ],
        )
        self.assertEqual(stats["completed_iterations"], 1)

    def test_generated_text_table_uses_existing_wandb_and_prefix_gate(self):
        cfg = make_training_cfg(
            iterations=0,
            wandb=True,
            eval_seq_prefix="Hello",
            eval_interval=100,
        )
        events, stats = self.run_train(cfg)
        generated_logs = [
            event[1]
            for event in events
            if event[0] == "wandb.log"
            and any(key.startswith("generated-text-") for key in event[1])
        ]

        self.assertIn(
            ("model.generate_from_string", "Hello", 40, 0.9, None), events
        )
        self.assertIn(("wandb.Table.add_data", (0, 2.5, "Hello<generated>")), events)
        self.assertEqual(len(generated_logs), 1)
        self.assertEqual(stats["completed_iterations"], 0)

        cfg = make_training_cfg(
            iterations=0,
            wandb=False,
            eval_seq_prefix="Hello",
            eval_interval=100,
        )
        events, stats = self.run_train(cfg)
        names = event_names(events)

        self.assertNotIn("model.generate_from_string", names)
        self.assertNotIn("wandb.Table.add_data", names)
        self.assertEqual(stats["completed_iterations"], 0)

    def test_eval_wandb_logs_include_aux_and_router_metrics(self):
        cfg = make_training_cfg(
            iterations=0,
            eval_interval=100,
            wandb=True,
            moe=True,
            plot_router_logits=True,
        )
        events, stats = self.run_train(
            cfg,
            capture_overrides={
                "eval_aux_losses": {"val/moe_aux": 0.42},
                "router_logits": ["router-a", "router-b"],
                "routing_logs": {"router/load_balance": 0.7},
            },
        )
        eval_logs = [
            event[1]
            for event in events
            if event[0] == "wandb.log" and "iter" in event[1]
        ]

        self.assertEqual(eval_logs, [
            {
                "tokens": 0,
                "iter": 0,
                "final-val/loss": 1.25,
                "final-val/perplexity": 2.5,
                "final-val/acc": 0.75,
                "val/moe_aux": 0.42,
                "router/load_balance": 0.7,
            }
        ])
        self.assertIn(("eval", 3, True, True), events)
        self.assertIn(("visualize_routing", ("router-a", "router-b")), events)
        self.assertEqual(stats["completed_iterations"], 0)

    def test_generated_text_gate_skips_non_multiple_non_final_eval(self):
        events = []
        capture = {"events": events, "checkpoint_iter": 0}
        model, opt, _scheduler, backend, datareaders = make_train_components(events)
        cfg = make_training_cfg(
            iterations=10,
            eval_interval=2,
            wandb=True,
            eval_seq_prefix="Hello",
        )

        with loaded_training_base_module(capture) as training_base:
            with redirect_stdout(StringIO()):
                result = training_base.eval_and_log(
                    tokens=20,
                    curr_iter=2,
                    epoch=0.0,
                    model=model,
                    val_reader=datareaders["val"],
                    type_ctx=nullcontext(),
                    distributed_backend=backend,
                    cfg=cfg,
                    opt=opt,
                )
        names = event_names(events)
        generated_logs = [
            event[1]
            for event in events
            if event[0] == "wandb.log"
            and any(key.startswith("generated-text-") for key in event[1])
        ]

        self.assertEqual(result, (1.25, 2.5, 0.75))
        self.assertNotIn("model.generate_from_string", names)
        self.assertNotIn("wandb.Table.add_data", names)
        self.assertEqual(generated_logs, [])

    def test_generated_text_gate_runs_at_final_iteration(self):
        events = []
        capture = {"events": events, "checkpoint_iter": 0}
        model, opt, _scheduler, backend, datareaders = make_train_components(events)
        cfg = make_training_cfg(
            iterations=6,
            eval_interval=2,
            wandb=True,
            eval_seq_prefix="Hello",
        )

        with loaded_training_base_module(capture) as training_base:
            with redirect_stdout(StringIO()):
                result = training_base.eval_and_log(
                    tokens=60,
                    curr_iter=6,
                    epoch=0.0,
                    model=model,
                    val_reader=datareaders["val"],
                    type_ctx=nullcontext(),
                    distributed_backend=backend,
                    cfg=cfg,
                    opt=opt,
                )
        generated_logs = [
            event[1]
            for event in events
            if event[0] == "wandb.log"
            and any(key.startswith("generated-text-") for key in event[1])
        ]

        self.assertEqual(result, (1.25, 2.5, 0.75))
        self.assertIn(("model.generate_from_string", "Hello", 40, 0.9, None), events)
        self.assertIn(("wandb.Table.add_data", (6, 2.5, "Hello<generated>")), events)
        self.assertEqual(len(generated_logs), 1)

    def test_weight_averager_initialization_and_resume_use_checkpoint_state(self):
        cfg = make_training_cfg(
            resume_from="/tmp/resume-source",
            iterations=6,
            eval_interval=100,
            weight_average=True,
            wa_horizon=7,
            wa_interval=3,
            wa_use_temp_dir=False,
            wa_dtype="float64",
        )
        events, stats = self.run_train(
            cfg,
            checkpoint_iter=5,
            capture_overrides={
                "checkpoint_training_state": {
                    "iteration": 5,
                    "train_reader_step": 5,
                    "substep": 5,
                },
            },
        )

        self.assertIn(
            (
                "WeightAverager.init",
                {
                    "horizon": 7,
                    "interval": 3,
                    "save_dir": "/tmp/llmopt-behavior-exp/avgs",
                    "dtype": "float64",
                    "count": 0,
                },
            ),
            events,
        )
        load_events = [event for event in events if event[0] == "load_checkpoint"]
        self.assertEqual(load_events[0][4], ("wa",))
        self.assertEqual(events.count(("weight_averager.step", True)), 1)
        self.assertEqual(stats["completed_iterations"], 1)

    def test_exponential_weight_averager_initialization_uses_warmup_and_dtype(self):
        for ewa_after_warmup, expected_warmup in ((True, 6), (False, 0)):
            with self.subTest(ewa_after_warmup=ewa_after_warmup):
                cfg = make_training_cfg(
                    iterations=1,
                    eval_interval=100,
                    exponential_weight_average=True,
                    ewa_interval=4,
                    ewa_decay=0.88,
                    ewa_after_warmup=ewa_after_warmup,
                    warmup_steps=6,
                    wa_dtype="float64",
                )
                events, stats = self.run_train(cfg)

                self.assertIn(
                    (
                        "ExponentialWeightAverager.init",
                        {
                            "interval": 4,
                            "decay": 0.88,
                            "warmup": expected_warmup,
                            "dtype": "float64",
                        },
                    ),
                    events,
                )
                self.assertEqual(events.count(("ewa.step", True)), 1)
                self.assertEqual(stats["completed_iterations"], 1)

    def test_weight_average_eval_and_step_timing_is_preserved(self):
        cfg = make_training_cfg(
            iterations=2,
            eval_interval=1,
            full_eval_at=[2],
            weight_average=True,
            wa_interval=1,
        )
        events, stats = self.run_train(cfg)
        wa_eval_events = [event for event in events if event[0] == "eval_wa"]
        wa_step_events = [
            event for event in events if event[0] == "weight_averager.step"
        ]

        self.assertEqual(wa_eval_events, [("eval_wa", 2, True)])
        self.assertEqual(wa_step_events, [("weight_averager.step", True)] * 2)
        self.assertEqual(stats["completed_iterations"], 2)

    def test_exponential_weight_average_eval_and_step_timing_is_preserved(self):
        cfg = make_training_cfg(
            iterations=2,
            eval_interval=1,
            full_eval_at=[1],
            exponential_weight_average=True,
        )
        events, stats = self.run_train(cfg)
        ewa_eval_events = [event for event in events if event[0] == "eval_ewa"]
        ewa_step_events = [event for event in events if event[0] == "ewa.step"]

        self.assertEqual(
            ewa_eval_events,
            [
                ("eval_ewa", 0, False),
                ("eval_ewa", 1, True),
                ("eval_ewa", 2, False),
            ],
        )
        self.assertEqual(ewa_step_events, [("ewa.step", True)] * 2)
        self.assertEqual(stats["completed_iterations"], 2)

    def test_weight_averager_steps_pass_non_master_flag(self):
        cfg = make_training_cfg(
            iterations=1,
            eval_interval=100,
            weight_average=True,
            exponential_weight_average=True,
        )
        events, stats = self.run_train(cfg, master=False)

        self.assertIn(("weight_averager.step", False), events)
        self.assertIn(("ewa.step", False), events)
        self.assertEqual(stats["completed_iterations"], 1)

    def test_train_log_interval_writes_wandb_train_metrics_on_master(self):
        cfg = make_training_cfg(
            iterations=1,
            eval_interval=100,
            log_interval=1,
            wandb=True,
        )
        events, stats = self.run_train(cfg)
        train_logs = [
            event[1]
            for event in events
            if event[0] == "wandb.log" and "train/loss" in event[1]
        ]
        notify_events = [event for event in events if event[0] == "maybe_notify"]

        self.assertEqual(len(train_logs), 1)
        train_log = train_logs[0]
        self.assertEqual(train_log["tokens"], 0)
        self.assertEqual(train_log["iter"], 1)
        self.assertEqual(train_log["train/loss"], 2.0)
        self.assertAlmostEqual(train_log["train/perplexity"], 2.71828**2.0)
        self.assertEqual(train_log["lr"], 0.01)
        self.assertIn("iter_dt", train_log)
        self.assertIn("wall_clock/elapsed_seconds", train_log)
        self.assertIn("train/step_seconds_total", train_log)
        self.assertEqual(train_log["max_grad_norm"], 0.25)
        self.assertEqual(train_log["mean_grad_norm"], 0.25)
        self.assertEqual(len(notify_events), 1)
        self.assertEqual(notify_events[0][1]["curr_iter"], 1)
        self.assertEqual(notify_events[0][1]["train_loss"], 2.0)
        self.assertEqual(notify_events[0][1]["val_loss"], 1.25)
        self.assertEqual(notify_events[0][1]["val_pp"], 2.5)
        self.assertEqual(notify_events[0][1]["val_acc"], 0.75)
        self.assertEqual(notify_events[0][1]["lr"], 0.01)
        self.assertEqual(notify_events[0][1]["run_name"], "llmopt-behavior-exp")
        self.assertEqual(stats["completed_iterations"], 1)

    def test_summary_stats_record_train_and_validation_metrics_with_semantics(self):
        cfg = make_training_cfg(
            iterations=1,
            eval_interval=100,
            log_interval=0,
            metric_semantics={
                "validation_loss": "next_token_cross_entropy",
                "validation_accuracy": "token_accuracy_epfl_flat_eotpad_v1",
            },
        )
        _events, stats = self.run_train(cfg)

        self.assertEqual(stats["train_loss"], [2.0])
        self.assertEqual(stats["val_loss"], [1.25, 1.25])
        self.assertEqual(stats["val_pp"], [2.5, 2.5])
        self.assertEqual(stats["val_acc"], [0.75, 0.75])
        self.assertEqual(
            stats["train_records"],
            [{"iteration": 1, "tokens": 16, "loss": 2.0}],
        )
        self.assertEqual(
            stats["validation_records"],
            [
                {
                    "iteration": 0,
                    "tokens": 0,
                    "loss": 1.25,
                    "perplexity": 2.5,
                    "token_accuracy": 0.75,
                },
                {
                    "iteration": 1,
                    "tokens": 16,
                    "loss": 1.25,
                    "perplexity": 2.5,
                    "token_accuracy": 0.75,
                },
            ],
        )
        self.assertEqual(
            stats["metric_semantics"]["validation_accuracy"],
            "token_accuracy_epfl_flat_eotpad_v1",
        )

    def test_log_interval_zero_skips_train_wandb_but_still_notifies_master(self):
        cfg = make_training_cfg(
            iterations=1,
            eval_interval=100,
            log_interval=0,
            wandb=True,
        )
        events, stats = self.run_train(cfg)
        train_logs = [
            event[1]
            for event in events
            if event[0] == "wandb.log" and "train/loss" in event[1]
        ]
        notify_events = [event for event in events if event[0] == "maybe_notify"]

        self.assertEqual(train_logs, [])
        self.assertEqual(len(notify_events), 1)
        self.assertEqual(notify_events[0][1]["curr_iter"], 1)
        self.assertIsNone(notify_events[0][1]["train_loss"])
        self.assertEqual(notify_events[0][1]["val_loss"], 1.25)
        self.assertEqual(notify_events[0][1]["lr"], 0.01)
        self.assertEqual(stats["completed_iterations"], 1)

    def test_non_master_skips_train_logging_and_notification(self):
        cfg = make_training_cfg(
            iterations=1,
            eval_interval=100,
            log_interval=1,
            wandb=True,
        )
        events, stats = self.run_train(cfg, master=False)
        names = event_names(events)

        self.assertNotIn("wandb.log", names)
        self.assertNotIn("maybe_notify", names)
        self.assertEqual(stats["completed_iterations"], 1)

    def test_prodigy_and_parameter_norm_logging_are_preserved(self):
        cfg = make_training_cfg(
            iterations=1,
            eval_interval=100,
            log_interval=1,
            wandb=True,
            opt="prodigy",
            log_parameter_norms=True,
            norm_order=1,
        )
        events, stats = self.run_train(cfg)
        train_logs = [
            event[1]
            for event in events
            if event[0] == "wandb.log" and "train/loss" in event[1]
        ]

        self.assertIn(("log_prodigy_lr",), events)
        self.assertIn(("get_parameter_norms", 1), events)
        self.assertEqual(len(train_logs), 1)
        self.assertEqual(train_logs[0]["effective_lr"], 0.123)
        self.assertEqual(train_logs[0]["model_norm"], 9.0)
        self.assertEqual(stats["completed_iterations"], 1)

    def test_sophiag_step_and_hessian_update_order_is_preserved(self):
        cfg = make_training_cfg(
            iterations=1,
            opt="sophiag",
            scheduler="cos",
            eval_interval=100,
            precondition_frequency=1,
            sophia_bs=3,
            sequence_length=8,
        )
        events, stats = self.run_train(cfg)
        names = event_names(events)

        self.assertIn(("opt.step", {"bs": 24}), events)
        assert_in_order(
            self,
            names,
            [
                "opt.step",
                "scheduler.step",
                "opt.zero_grad",
                "model.forward",
                "loss.backward",
                "opt.update_hessian",
                "opt.zero_grad",
                "model.zero_grad",
            ],
        )
        self.assertIn(("loss.backward", "sampled"), events)
        self.assertEqual(events.count(("opt.zero_grad", True)), 2)
        self.assertEqual(events.count(("scheduler.step",)), 1)
        self.assertEqual(stats["completed_iterations"], 1)

    def test_mars_step_and_last_grad_update_order_is_preserved(self):
        cfg = make_training_cfg(
            iterations=1,
            opt="mars",
            scheduler="cos",
            eval_interval=100,
        )
        events, stats = self.run_train(cfg)
        names = event_names(events)

        assert_in_order(
            self,
            names,
            [
                "opt.step",
                "scheduler.step",
                "opt.zero_grad",
                "opt.update_last_grad",
            ],
        )
        self.assertIn(("opt.step", {}), events)
        self.assertEqual(events.count(("opt.zero_grad", True)), 1)
        self.assertEqual(events.count(("opt.update_last_grad",)), 1)
        self.assertEqual(events.count(("scheduler.step",)), 1)
        self.assertEqual(stats["completed_iterations"], 1)

    def test_gn_inner_loop_line_search_and_scheduler_order_is_preserved(self):
        for opt_name, mode in (("gn-prox", "prox"), ("gn-full", "full")):
            with self.subTest(opt=opt_name):
                cfg = make_training_cfg(
                    iterations=1,
                    opt=opt_name,
                    scheduler="cos",
                    eval_interval=100,
                    gn_inner_iters=2,
                    gn_log_inner_steps=True,
                    gn_linesearch=True,
                    gn_ls_range=[0.5, 1.0],
                    wandb=True,
                )
                events, stats = self.run_train(cfg)
                names = event_names(events)
                compute_events = [
                    event for event in events if event[0] == "compute_gn_step"
                ]
                train_batches = [
                    event
                    for event in events
                    if event[0] == "get_batch" and event[1] == "train"
                ]
                gn_inner_logs = [
                    event[1]
                    for event in events
                    if event[0] == "wandb.log"
                    and "train/gn_inner_step" in event[1]
                ]

                self.assertEqual(compute_events, [("compute_gn_step", mode)] * 2)
                self.assertEqual(len(train_batches), 4)
                self.assertIn(("line_search_over_direction", 2, (0.5, 1.0)), events)
                self.assertEqual(
                    [log["train/gn_inner_step"] for log in gn_inner_logs],
                    [0, 1],
                )
                assert_in_order(
                    self,
                    names,
                    [
                        "compute_gn_step",
                        "opt.step",
                        "opt.zero_grad",
                        "compute_gn_step",
                        "opt.step",
                        "opt.zero_grad",
                        "line_search_over_direction",
                        "scheduler.step",
                    ],
                )
                self.assertEqual(events.count(("scheduler.step",)), 1)
                self.assertEqual(stats["completed_iterations"], 1)

    def test_gn_checkpoint_and_resume_use_all_consumed_inner_batches(self):
        initial_cfg = make_training_cfg(
            iterations=1,
            opt="gn-prox",
            gn_inner_iters=2,
            gn_linesearch=True,
            latest_ckpt_interval=1,
        )
        initial_events, _stats = self.run_train(initial_cfg)
        saved_positions = [
            event[2]["train_reader_step"]
            for event in initial_events
            if event[0] == "save_checkpoint_metadata"
        ]
        self.assertEqual(saved_positions, [0, 4])

        resumed_cfg = make_training_cfg(
            resume_from="/tmp/resume-source",
            iterations=2,
            opt="gn-prox",
            gn_inner_iters=2,
            gn_linesearch=True,
        )
        resumed_events, resumed_stats = self.run_train(
            resumed_cfg,
            checkpoint_iter=1,
            capture_overrides={
                "checkpoint_training_state": {
                    "iteration": 1,
                    "train_reader_step": 4,
                    "substep": 4,
                },
            },
        )

        self.assertIn(("train.set_step", 4), resumed_events)
        resumed_batches = [
            event
            for event in resumed_events
            if event[0] == "get_batch" and event[1] == "train"
        ]
        self.assertEqual(len(resumed_batches), 4)
        self.assertEqual(resumed_stats["completed_iterations"], 1)

    def test_checkpoint_intervals_save_at_initial_and_final_iterations(self):
        cfg = make_training_cfg(
            iterations=1,
            scheduler="none",
            permanent_ckpt_interval=1,
            latest_ckpt_interval=1,
        )
        events, stats = self.run_train(cfg)
        save_events = [event for event in events if event[0] == "save_checkpoint"]

        self.assertEqual(
            save_events,
            [
                ("save_checkpoint", 0, "/tmp/llmopt-behavior-exp/ckpts/0"),
                ("save_checkpoint", 0, "/tmp/llmopt-behavior-exp/ckpts/latest"),
                ("save_checkpoint", 1, "/tmp/llmopt-behavior-exp/ckpts/1"),
                ("save_checkpoint", 1, "/tmp/llmopt-behavior-exp/ckpts/latest"),
            ],
        )
        self.assertEqual(stats["completed_iterations"], 1)

        metadata_events = [
            event for event in events if event[0] == "save_checkpoint_metadata"
        ]
        self.assertEqual(
            [event[2] for event in metadata_events],
            [
                {"iteration": 0, "train_reader_step": 0, "substep": 0},
                {"iteration": 0, "train_reader_step": 0, "substep": 0},
                {"iteration": 1, "train_reader_step": 1, "substep": 1},
                {"iteration": 1, "train_reader_step": 1, "substep": 1},
            ],
        )
        self.assertTrue(all(event[3] == "fake-run-identity" for event in metadata_events))
        self.assertTrue(all(event[4] == 1 for event in metadata_events))

    def test_checkpoint_waits_for_main_and_all_rank_worker_states(self):
        cfg = make_training_cfg(
            iterations=0,
            permanent_ckpt_interval=1,
            latest_ckpt_interval=0,
        )
        events, _stats = self.run_train(cfg)

        assert_in_order(
            self,
            event_names(events),
            ["save_checkpoint", "barrier", "save_worker_state", "barrier"],
        )

    def test_checkpoint_failures_are_synchronized_before_the_next_collective(self):
        cfg = make_training_cfg(
            iterations=0,
            permanent_ckpt_interval=1,
            latest_ckpt_interval=0,
            expected_world_size=2,
        )

        main_events = []
        with self.assertRaisesRegex(RuntimeError, "synthetic main save failure"):
            self.run_train(
                cfg,
                world_size=2,
                events=main_events,
                capture_overrides={
                    "save_checkpoint_error": RuntimeError(
                        "synthetic main save failure"
                    ),
                },
            )
        assert_in_order(
            self,
            event_names(main_events),
            ["save_checkpoint", "all_gather_object"],
        )
        self.assertNotIn("save_worker_state", event_names(main_events))
        self.assertNotIn("barrier", event_names(main_events))

        worker_events = []
        with self.assertRaisesRegex(RuntimeError, "synthetic worker save failure"):
            self.run_train(
                cfg,
                world_size=2,
                events=worker_events,
                capture_overrides={
                    "save_worker_state_error": RuntimeError(
                        "synthetic worker save failure"
                    ),
                },
            )
        assert_in_order(
            self,
            event_names(worker_events),
            [
                "save_checkpoint",
                "all_gather_object",
                "barrier",
                "save_worker_state",
                "all_gather_object",
            ],
        )
        self.assertEqual(worker_events.count(("barrier",)), 1)

    def test_zero_iterations_evaluates_but_does_not_train(self):
        cfg = make_training_cfg(iterations=0, eval_interval=100)
        events, stats = self.run_train(cfg)
        names = event_names(events)

        self.assertIn("eval", names)
        self.assertNotIn("get_batch", names)
        self.assertNotIn("opt.step", names)
        self.assertEqual(stats["completed_iterations"], 0)

    def test_no_resume_starts_from_zero_without_resume_side_effects(self):
        cfg = make_training_cfg(iterations=1, acc_steps=3, eval_interval=100)
        events, stats = self.run_train(cfg)
        names = event_names(events)

        self.assertNotIn("load_checkpoint", names)
        self.assertNotIn("load_worker_state", names)
        self.assertNotIn("extend_onecycle_total_steps", names)
        self.assertIn(("train.set_step", 0), events)
        self.assertEqual(stats["completed_iterations"], 1)

    def test_resume_loads_checkpoint_worker_state_and_extends_scheduler_before_training(self):
        cfg = make_training_cfg(
            resume_from="/tmp/resume-source",
            iterations=2,
            acc_steps=3,
            eval_interval=100,
        )
        events, stats = self.run_train(
            cfg,
            checkpoint_iter=1,
            capture_overrides={
                "checkpoint_training_state": {
                    "iteration": 1,
                    "train_reader_step": 7,
                    "substep": 7,
                },
                "worker_training_state": {
                    "iteration": 1,
                    "train_reader_step": 7,
                    "substep": 7,
                },
            },
        )
        names = event_names(events)

        assert_in_order(
            self,
            names,
            [
                "load_checkpoint",
                "load_worker_state",
                "extend_onecycle_total_steps",
                "train.set_step",
                "get_batch",
            ],
        )
        self.assertEqual(
            events[0],
            (
                "load_checkpoint",
                "/tmp/resume-source/main.pt",
                "cpu",
                False,
                (),
                "fake-run-identity",
                1,
                False,
                True,
            ),
        )
        self.assertEqual(
            events[1],
            (
                "load_worker_state",
                "/tmp/resume-source",
                True,
                0,
                1,
                "fake-snapshot-id",
                False,
            ),
        )
        self.assertEqual(events[2], ("extend_onecycle_total_steps", False, 2))
        self.assertIn(("train.set_step", 7), events)
        self.assertEqual(stats["completed_iterations"], 1)

    def test_resume_passes_none_scheduler_through_resume_helpers(self):
        cfg = make_training_cfg(
            resume_from="/tmp/resume-source",
            iterations=1,
            eval_interval=100,
            scheduler="none",
        )
        events, stats = self.run_train(
            cfg,
            checkpoint_iter=0,
            scheduler_override=None,
            capture_overrides={
                "checkpoint_training_state": {
                    "iteration": 0,
                    "train_reader_step": 0,
                    "substep": 0,
                },
            },
        )

        self.assertEqual(
            events[0],
            (
                "load_checkpoint",
                "/tmp/resume-source/main.pt",
                "cpu",
                True,
                (),
                "fake-run-identity",
                1,
                False,
                True,
            ),
        )
        self.assertEqual(
            events[1],
            (
                "load_worker_state",
                "/tmp/resume-source",
                True,
                0,
                1,
                "fake-snapshot-id",
                False,
            ),
        )
        self.assertEqual(events[2], ("extend_onecycle_total_steps", True, 1))
        self.assertIn(("train.set_step", 0), events)
        self.assertEqual(stats["completed_iterations"], 1)

    def test_standard_legacy_resume_requires_explicit_opt_in_and_uses_old_formula(self):
        cfg = make_training_cfg(
            resume_from="/tmp/resume-source",
            iterations=2,
            acc_steps=3,
            allow_legacy_checkpoint_resume=True,
        )
        events, stats = self.run_train(
            cfg,
            checkpoint_iter=1,
            capture_overrides={"checkpoint_format_version": 0},
        )

        self.assertIn(("train.set_step", 3), events)
        self.assertIn(
            ("load_worker_state", "/tmp/resume-source", True, 0, 1, None, True),
            events,
        )
        self.assertEqual(stats["completed_iterations"], 1)

    def test_gn_legacy_resume_is_rejected_even_with_explicit_opt_in(self):
        cfg = make_training_cfg(
            resume_from="/tmp/resume-source",
            iterations=2,
            opt="gn-prox",
            gn_inner_iters=2,
            allow_legacy_checkpoint_resume=True,
        )
        events = []

        with self.assertRaisesRegex(RuntimeError, "GN.*legacy"):
            self.run_train(
                cfg,
                checkpoint_iter=1,
                capture_overrides={"checkpoint_format_version": 0},
                events=events,
            )

        self.assertNotIn("get_batch", event_names(events))

    def test_versioned_resume_rejects_missing_or_inconsistent_reader_position(self):
        for training_state, message in (
            ({"iteration": 1}, "train_reader_step"),
            (
                {"iteration": 1, "train_reader_step": 4, "substep": 5},
                "does not match",
            ),
        ):
            with self.subTest(training_state=training_state):
                cfg = make_training_cfg(
                    resume_from="/tmp/resume-source",
                    iterations=2,
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    self.run_train(
                        cfg,
                        checkpoint_iter=1,
                        capture_overrides={
                            "checkpoint_training_state": training_state,
                        },
                    )

    def test_checkpoint_saves_pass_none_scheduler_when_train_receives_none(self):
        cfg = make_training_cfg(
            iterations=0,
            scheduler="none",
            permanent_ckpt_interval=1,
            latest_ckpt_interval=1,
        )
        events, stats = self.run_train(cfg, scheduler_override=None)
        scheduler_shape_events = [
            event for event in events if event[0] == "save_checkpoint_scheduler_is_none"
        ]

        self.assertEqual(
            scheduler_shape_events,
            [
                ("save_checkpoint_scheduler_is_none", 0, True),
                ("save_checkpoint_scheduler_is_none", 0, True),
            ],
        )
        self.assertEqual(stats["completed_iterations"], 0)


if __name__ == "__main__":
    unittest.main()
