import unittest
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path

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
        use_ddp=False,
        use_precond_optimizer=False,
        precond_flag=True,
        capture_overrides=None,
    ):
        events = []
        capture = {"events": events, "checkpoint_iter": checkpoint_iter}
        if capture_overrides:
            capture.update(capture_overrides)
        model, opt, scheduler, backend, datareaders = make_train_components(
            events,
            master=master,
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

    def test_weight_averager_initialization_uses_cfg_and_resume_count(self):
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
        events, stats = self.run_train(cfg, checkpoint_iter=5)

        self.assertIn(
            (
                "WeightAverager.init",
                {
                    "horizon": 7,
                    "interval": 3,
                    "save_dir": "/tmp/llmopt-behavior-exp/avgs",
                    "dtype": "float64",
                    "count": 5,
                },
            ),
            events,
        )
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
        events, stats = self.run_train(cfg, checkpoint_iter=1)
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
            ("load_checkpoint", "/tmp/resume-source/main.pt", "cpu", False),
        )
        self.assertEqual(events[1], ("load_worker_state", "/tmp/resume-source"))
        self.assertEqual(events[2], ("extend_onecycle_total_steps", False, 2))
        self.assertIn(("train.set_step", 3), events)
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
        )

        self.assertEqual(
            events[0],
            ("load_checkpoint", "/tmp/resume-source/main.pt", "cpu", True),
        )
        self.assertEqual(events[1], ("load_worker_state", "/tmp/resume-source"))
        self.assertEqual(events[2], ("extend_onecycle_total_steps", True, 1))
        self.assertIn(("train.set_step", 0), events)
        self.assertEqual(stats["completed_iterations"], 1)

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
