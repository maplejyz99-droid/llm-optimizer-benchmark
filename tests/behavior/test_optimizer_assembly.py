import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

from tests._helpers.behavior_harness import make_args_for_main, load_main_with_fakes


class OptimizerAssemblyBehaviorTest(unittest.TestCase):
    def _run_main(self, argv):
        capture = {}
        main_module = load_main_with_fakes(capture)
        args, parser = make_args_for_main(argv)
        with redirect_stdout(StringIO()):
            main_module.main(args, parser)
        train_kwargs = capture["train_kwargs"]
        return train_kwargs["opt"], train_kwargs["scheduler"], train_kwargs["cfg"]

    def test_adamw_uses_torch_adamw_with_training_lr(self):
        opt, scheduler, cfg = self._run_main(["--opt", "adamw", "--lr", "0.002"])

        self.assertEqual(type(opt).__name__, "AdamW")
        self.assertEqual(opt.kwargs["lr"], 0.002)
        self.assertEqual(opt.kwargs["betas"], (cfg.beta1, cfg.beta2))
        self.assertEqual(opt.kwargs["weight_decay"], cfg.weight_decay)
        self.assertEqual(type(scheduler).__name__, "OneCycleLR")

    def test_gn_uses_inner_adamw_and_zero_weight_decay(self):
        opt, scheduler, cfg = self._run_main(
            ["--opt", "gn-prox", "--gn_inner_lr", "0.004", "--gn_inner_wd", "0.3"]
        )

        self.assertEqual(type(opt).__name__, "AdamW")
        self.assertEqual(opt.kwargs["lr"], 0.004)
        self.assertEqual(opt.kwargs["betas"], (cfg.gn_inner_b1, cfg.gn_inner_b2))
        self.assertEqual(opt.kwargs["weight_decay"], 0.0)
        self.assertEqual(type(scheduler).__name__, "OneCycleLR")

    def test_muon_uses_combined_scheduler(self):
        opt, scheduler, cfg = self._run_main(
            ["--opt", "muon", "--lr", "0.003", "--muon_lr_factor", "0.02"]
        )

        self.assertEqual(type(opt).__name__, "Muon")
        self.assertEqual(opt.kwargs["lr"], 0.02)
        self.assertEqual(opt.kwargs["adamw_lr"], 0.003)
        self.assertEqual(opt.kwargs["adamw_betas"], (cfg.beta1, cfg.beta2))
        self.assertEqual(type(scheduler).__name__, "FakeCombinedScheduler")

    def test_llama_muon_keeps_configured_matrix_lr_without_mup_scaling(self):
        opt, _, cfg = self._run_main(
            [
                "--model",
                "llama",
                "--opt",
                "muon",
                "--muon_lr_factor",
                "0.03",
                "--n_embd",
                "1024",
                "--scale_base_model",
                "256",
            ]
        )

        self.assertEqual(type(opt).__name__, "Muon")
        self.assertEqual(opt.kwargs["lr"], 0.03)
        self.assertEqual(cfg.model, "llama")

    def test_mup_llama_muon_scales_matrix_lr_by_width_multiplier(self):
        opt, _, cfg = self._run_main(
            [
                "--model",
                "mup_llama",
                "--opt",
                "muon",
                "--muon_lr_factor",
                "0.03",
                "--n_embd",
                "1024",
                "--scale_base_model",
                "256",
            ]
        )

        self.assertEqual(type(opt).__name__, "Muon")
        self.assertEqual(cfg.model, "mup_llama")
        self.assertEqual(cfg.n_embd / cfg.scale_base_model, 4.0)
        self.assertEqual(opt.kwargs["lr"], 0.03 / 4.0)

    def test_softeq_k2000_muon_uses_combined_scheduler_and_adamw_backup(self):
        opt, scheduler, cfg = self._run_main(
            [
                "--opt",
                "softeq-k2000-muon",
                "--lr",
                "0.003",
                "--muon_lr_factor",
                "0.02",
                "--momentum",
                "0.91",
            ]
        )

        self.assertEqual(type(opt).__name__, "SoftEqK2000Muon")
        self.assertEqual(opt.kwargs["lr"], 0.02)
        self.assertEqual(opt.kwargs["momentum"], 0.91)
        self.assertEqual(opt.kwargs["weight_decay"], cfg.weight_decay)
        self.assertEqual(opt.kwargs["adamw_lr"], 0.003)
        self.assertEqual(opt.kwargs["adamw_betas"], (cfg.beta1, cfg.beta2))
        self.assertEqual(type(scheduler).__name__, "FakeCombinedScheduler")

    def test_mup_llama_softeq_k2000_muon_scales_matrix_lr_by_width_multiplier(self):
        opt, _, cfg = self._run_main(
            [
                "--model",
                "mup_llama",
                "--opt",
                "softeq-k2000-muon",
                "--muon_lr_factor",
                "0.03",
                "--n_embd",
                "1024",
                "--scale_base_model",
                "256",
            ]
        )

        self.assertEqual(type(opt).__name__, "SoftEqK2000Muon")
        self.assertEqual(cfg.n_embd / cfg.scale_base_model, 4.0)
        self.assertEqual(opt.kwargs["lr"], 0.03 / 4.0)

    def test_logged_parameter_counts_use_model_contract_for_tied_embeddings(self):
        main_module = load_main_with_fakes({})

        class TiedLlamaLikeModel:
            def __init__(self):
                shared_weight = SimpleNamespace(numel=lambda: 64)
                self.lm_head = SimpleNamespace(weight=shared_weight)
                self.transformer = SimpleNamespace(
                    wte=SimpleNamespace(weight=shared_weight)
                )

            def get_num_params(self, non_embedding=True):
                return 100

        params_cnt, nonemb_param_cnt = main_module.get_logged_parameter_counts(
            TiedLlamaLikeModel()
        )

        self.assertEqual(params_cnt, 100)
        self.assertEqual(nonemb_param_cnt, 100)

    def test_log_optimizer_groups_prints_effective_adamw_backup_lr(self):
        main_module = load_main_with_fakes({})

        class Param:
            shape = (2, 2)

        muon_param = Param()
        backup_param = Param()
        opt = SimpleNamespace(
            param_groups=[
                {
                    "params": [muon_param, backup_param],
                    "lr": 0.02,
                    "adamw_lr_ratio": 0.1,
                    "weight_decay": 0.1,
                }
            ],
            state={
                muon_param: {"use_muon": True},
                backup_param: {"use_muon": False},
            },
        )
        output = StringIO()

        with redirect_stdout(output):
            main_module.log_optimizer_groups(
                opt,
                {
                    id(muon_param): "block.weight",
                    id(backup_param): "norm.weight",
                },
                "test",
            )

        rendered = output.getvalue()
        self.assertIn("muon: block.weight shape=(2, 2) lr=0.02", rendered)
        self.assertIn("adamw_backup: norm.weight shape=(2, 2) lr=0.002", rendered)

    def test_newton_muon_attaches_preconditioner_and_uses_combined_scheduler(self):
        opt, scheduler, cfg = self._run_main(
            [
                "--opt",
                "newton-muon",
                "--lr",
                "0.003",
                "--muon_lr_factor",
                "0.01",
                "--newton_muon_precond_every",
                "16",
            ]
        )

        self.assertEqual(type(opt).__name__, "FakeNewtonMuon")
        self.assertEqual(opt.kwargs["lr"], 0.01)
        self.assertEqual(opt.kwargs["adamw_lr"], 0.003)
        self.assertEqual(opt.kwargs["precond_every"], 16)
        self.assertIsNotNone(opt.attached_model)
        self.assertEqual(type(scheduler).__name__, "FakeCombinedScheduler")

    def test_newton_muon_rejects_mup_llama_model_before_training(self):
        capture = {}
        main_module = load_main_with_fakes(capture)
        args, parser = make_args_for_main(["--opt", "newton-muon", "--model", "mup_llama"])

        with self.assertRaisesRegex(ValueError, "only supports --model llama"):
            with redirect_stdout(StringIO()):
                main_module.main(args, parser)
        self.assertNotIn("train_kwargs", capture)

    def test_newton_muon_rejects_distributed_backend_before_remote_work(self):
        capture = {}
        main_module = load_main_with_fakes(capture)
        args, parser = make_args_for_main(["--opt", "newton-muon", "--distributed_backend", "nccl"])

        with self.assertRaisesRegex(ValueError, "only supports single-device"):
            with redirect_stdout(StringIO()):
                main_module.main(args, parser)
        self.assertNotIn("train_kwargs", capture)

    def test_softeq_k2000_muon_rejects_unsupported_model_before_training(self):
        capture = {}
        main_module = load_main_with_fakes(capture)
        args, parser = make_args_for_main(["--opt", "softeq-k2000-muon", "--model", "base"])

        with self.assertRaisesRegex(ValueError, "only supports --model llama or mup_llama"):
            with redirect_stdout(StringIO()):
                main_module.main(args, parser)
        self.assertNotIn("train_kwargs", capture)

    def test_softeq_k2000_muon_rejects_moe_before_training(self):
        capture = {}
        main_module = load_main_with_fakes(capture)
        args, parser = make_args_for_main(["--opt", "softeq-k2000-muon", "--moe"])

        with self.assertRaisesRegex(ValueError, "does not support MoE"):
            with redirect_stdout(StringIO()):
                main_module.main(args, parser)
        self.assertNotIn("train_kwargs", capture)

    def test_sophia_and_mars_keep_special_constructor_parameters(self):
        sophia_opt, _, sophia_cfg = self._run_main(["--opt", "sophiag", "--sophia_rho", "0.08"])
        mars_opt, _, mars_cfg = self._run_main(["--opt", "mars", "--mars_lr", "0.007"])

        self.assertEqual(type(sophia_opt).__name__, "SophiaG")
        self.assertEqual(sophia_opt.kwargs["rho"], 0.08)
        self.assertEqual(sophia_opt.kwargs["betas"], (sophia_cfg.beta1, sophia_cfg.beta2))
        self.assertEqual(type(mars_opt).__name__, "MARS")
        self.assertEqual(mars_opt.kwargs["lr"], 0.007)
        self.assertEqual(mars_opt.kwargs["betas"], (mars_cfg.mars_beta1, mars_cfg.mars_beta2))
        self.assertEqual(mars_opt.kwargs["lr_1d"], mars_cfg.lr)

    def test_cadamw_keeps_cautious_constructor_parameters(self):
        opt, _, cfg = self._run_main(
            ["--opt", "cadamw", "--lr", "0.004", "--cautious_xi", "0.6"]
        )

        self.assertEqual(type(opt).__name__, "CAdamW")
        self.assertEqual(opt.kwargs["lr"], 0.004)
        self.assertEqual(opt.kwargs["betas"], (cfg.beta1, cfg.beta2))
        self.assertEqual(opt.kwargs["weight_decay"], cfg.weight_decay)
        self.assertEqual(opt.kwargs["eps"], 1e-8)
        self.assertEqual(opt.kwargs["cautious_xi"], 0.6)

    def test_soap_keeps_preconditioner_constructor_parameters(self):
        opt, _, cfg = self._run_main(
            [
                "--opt",
                "soap",
                "--shampoo_beta",
                "0.91",
                "--precondition_frequency",
                "13",
                "--max_precond_dim",
                "2048",
                "--merge_dims",
                "True",
                "--precondition_1d",
                "True",
                "--normalize_grads",
                "True",
                "--soap_data_format",
                "channels_last",
                "--correct_bias",
                "True",
            ]
        )

        self.assertEqual(type(opt).__name__, "SOAP")
        self.assertEqual(opt.kwargs["lr"], cfg.lr)
        self.assertEqual(opt.kwargs["betas"], (cfg.beta1, cfg.beta2))
        self.assertEqual(opt.kwargs["shampoo_beta"], 0.91)
        self.assertEqual(opt.kwargs["precondition_frequency"], 13)
        self.assertEqual(opt.kwargs["max_precond_dim"], 2048)
        self.assertIs(opt.kwargs["merge_dims"], True)
        self.assertIs(opt.kwargs["precondition_1d"], True)
        self.assertIs(opt.kwargs["normalize_grads"], True)
        self.assertEqual(opt.kwargs["data_format"], "channels_last")
        self.assertIs(opt.kwargs["correct_bias"], True)

    def test_magma_optimizers_keep_magma_constructor_parameters(self):
        for opt_name, class_name in (
            ("adamw-magma", "MagmaAdamW"),
            ("muon-magma", "MagmaMuon"),
        ):
            with self.subTest(opt=opt_name):
                opt, scheduler, cfg = self._run_main(
                    [
                        "--opt",
                        opt_name,
                        "--magma_survival_p",
                        "0.7",
                        "--magma_tau",
                        "1.8",
                        "--magma_beta",
                        "0.83",
                        "--magma_scope",
                        "attn-mlp",
                    ]
                )

                self.assertEqual(type(opt).__name__, class_name)
                self.assertEqual(opt.kwargs["magma_survival_p"], 0.7)
                self.assertEqual(opt.kwargs["magma_tau"], 1.8)
                self.assertEqual(opt.kwargs["magma_beta"], 0.83)
                self.assertGreater(len(opt.kwargs["magma_param_ids"]), 0)
                if opt_name == "muon-magma":
                    self.assertEqual(opt.kwargs["adamw_lr"], cfg.lr)
                    self.assertEqual(type(scheduler).__name__, "FakeCombinedScheduler")
                else:
                    self.assertEqual(opt.kwargs["lr"], cfg.lr)

    def test_distributed_muon_keeps_group_specs_and_muon_parameters(self):
        opt, _, cfg = self._run_main(
            [
                "--opt",
                "d-muon",
                "--lr",
                "0.006",
                "--momentum",
                "0.87",
                "--nesterov",
                "True",
                "--muon_ns_steps",
                "8",
            ]
        )

        self.assertEqual(type(opt).__name__, "DistributedMuon")
        self.assertEqual(opt.kwargs["lr"], 0.006)
        self.assertEqual(opt.kwargs["momentum"], 0.87)
        self.assertIs(opt.kwargs["nesterov"], True)
        self.assertEqual(opt.kwargs["ns_steps"], 8)
        self.assertEqual(opt.kwargs["adamw_betas"], (cfg.beta1, cfg.beta2))
        self.assertEqual(opt.kwargs["weight_decay"], cfg.weight_decay)
        self.assertIsInstance(opt.args[0][0], dict)

    def test_ademamix_and_lion_keep_constructor_parameters(self):
        ademamix_opt, _, ademamix_cfg = self._run_main(
            [
                "--opt",
                "ademamix",
                "--adema_beta3",
                "0.82",
                "--adema_alpha",
                "3.2",
                "--adema_beta3_warmup",
                "11",
                "--adema_alpha_warmup",
                "22",
            ]
        )
        lion_opt, _, lion_cfg = self._run_main(["--opt", "lion", "--lr", "0.004"])

        self.assertEqual(type(ademamix_opt).__name__, "AdEMAMix")
        self.assertEqual(
            ademamix_opt.kwargs["betas"],
            (ademamix_cfg.beta1, ademamix_cfg.beta2, 0.82),
        )
        self.assertEqual(ademamix_opt.kwargs["alpha"], 3.2)
        self.assertEqual(ademamix_opt.kwargs["beta3_warmup"], 11)
        self.assertEqual(ademamix_opt.kwargs["alpha_warmup"], 22)
        self.assertEqual(type(lion_opt).__name__, "Lion")
        self.assertEqual(lion_opt.kwargs["lr"], 0.004)
        self.assertEqual(lion_opt.kwargs["betas"], (lion_cfg.beta1, lion_cfg.beta2))

    def test_schedulefree_optimizers_keep_schedulefree_parameters(self):
        for opt_name, class_name in (
            ("sf-adamw", "AdamWScheduleFree"),
            ("sf-sgd", "SGDScheduleFree"),
        ):
            with self.subTest(opt=opt_name):
                opt, _, cfg = self._run_main(
                    [
                        "--opt",
                        opt_name,
                        "--scheduler",
                        "none",
                        "--schedulefree_r",
                        "0.04",
                        "--weight_lr_power",
                        "1.6",
                        "--warmup_steps",
                        "3",
                    ]
                )

                self.assertEqual(type(opt).__name__, class_name)
                self.assertEqual(opt.kwargs["lr"], cfg.lr)
                self.assertEqual(opt.kwargs["warmup_steps"], 3)
                self.assertEqual(opt.kwargs["r"], 0.04)
                self.assertEqual(opt.kwargs["weight_lr_power"], 1.6)
                if opt_name == "sf-adamw":
                    self.assertEqual(opt.kwargs["betas"], (cfg.beta1, cfg.beta2))
                else:
                    self.assertEqual(opt.kwargs["momentum"], cfg.momentum)

    def test_schedulefree_optimizers_reject_external_scheduler(self):
        capture = {}
        main_module = load_main_with_fakes(capture)
        args, parser = make_args_for_main(["--opt", "sf-adamw", "--scheduler", "cos"])

        with self.assertRaisesRegex(ValueError, "require --scheduler none"):
            with redirect_stdout(StringIO()):
                main_module.main(args, parser)
        self.assertNotIn("train_kwargs", capture)

    def test_sign_optimizers_keep_momentum_contract(self):
        signsgd_opt, _, signsgd_cfg = self._run_main(
            ["--opt", "signsgd", "--momentum", "0.76", "--dampening", "0.03"]
        )
        signum_opt, _, signum_cfg = self._run_main(
            ["--opt", "signum", "--momentum", "0.76", "--dampening", "0.03"]
        )

        self.assertEqual(type(signsgd_opt).__name__, "Signum")
        self.assertEqual(signsgd_opt.kwargs["momentum"], 0.0)
        self.assertEqual(signsgd_opt.kwargs["dampening"], 0.03)
        self.assertEqual(signsgd_opt.kwargs["weight_decay"], signsgd_cfg.weight_decay)
        self.assertIs(signsgd_opt.kwargs["sign_update"], True)
        self.assertEqual(type(signum_opt).__name__, "Signum")
        self.assertEqual(signum_opt.kwargs["momentum"], 0.76)
        self.assertEqual(signum_opt.kwargs["dampening"], 0.03)
        self.assertEqual(signum_opt.kwargs["weight_decay"], signum_cfg.weight_decay)
        self.assertIs(signum_opt.kwargs["sign_update"], True)

    def test_adaptive_optimizers_keep_constructor_parameters(self):
        prodigy_opt, _, prodigy_cfg = self._run_main(
            [
                "--opt",
                "prodigy",
                "--prodigy_beta3",
                "0.97",
                "--prodigy_decouple",
                "True",
                "--prodigy_use_bias_correction",
                "True",
                "--prodigy_safeguard_warmup",
                "True",
                "--prodigy_fsdp_in_use",
                "True",
            ]
        )
        adopt_opt, _, _ = self._run_main(
            ["--opt", "adopt", "--adopt_eps", "2e-6", "--adopt_decouple", "True"]
        )
        adafactor_opt, _, adafactor_cfg = self._run_main(
            ["--opt", "adafactor", "--adafactor_decay_rate", "-0.7"]
        )
        lamb_opt, _, lamb_cfg = self._run_main(
            ["--opt", "lamb", "--lamb_use_bias_correction", "True"]
        )

        self.assertEqual(type(prodigy_opt).__name__, "Prodigy")
        self.assertEqual(prodigy_opt.kwargs["betas"], (prodigy_cfg.beta1, prodigy_cfg.beta2))
        self.assertEqual(prodigy_opt.kwargs["beta3"], 0.97)
        self.assertIs(prodigy_opt.kwargs["decouple"], True)
        self.assertIs(prodigy_opt.kwargs["use_bias_correction"], True)
        self.assertIs(prodigy_opt.kwargs["safeguard_warmup"], True)
        self.assertIs(prodigy_opt.kwargs["fsdp_in_use"], True)
        self.assertEqual(type(adopt_opt).__name__, "ADOPT")
        self.assertEqual(adopt_opt.kwargs["eps"], 2e-6)
        self.assertIs(adopt_opt.kwargs["decouple"], True)
        self.assertEqual(type(adafactor_opt).__name__, "Adafactor")
        self.assertEqual(adafactor_opt.kwargs["decay_rate"], -0.7)
        self.assertEqual(adafactor_opt.kwargs["beta1"], adafactor_cfg.beta1)
        self.assertEqual(adafactor_opt.kwargs["clip_threshold"], 1.0)
        self.assertEqual(type(lamb_opt).__name__, "Lamb")
        self.assertEqual(lamb_opt.kwargs["betas"], (lamb_cfg.beta1, lamb_cfg.beta2))
        self.assertIs(lamb_opt.kwargs["adam"], False)
        self.assertIs(lamb_opt.kwargs["bias_correction"], True)

    def test_scion_variants_use_partitioned_groups(self):
        for opt_name, class_name in (("scion", "Scion"), ("scion-light", "ScionLight")):
            with self.subTest(opt=opt_name):
                opt, _, cfg = self._run_main(["--opt", opt_name, "--momentum", "0.82"])

                self.assertEqual(type(opt).__name__, class_name)
                self.assertEqual(opt.kwargs["lr"], cfg.lr)
                self.assertEqual(opt.kwargs["momentum"], 0.82)
                self.assertIsInstance(opt.args[0][0], dict)

    def test_torch_muon_branch_keeps_pytorch_muon_parameters(self):
        opt, _, cfg = self._run_main(
            [
                "--opt",
                "muon-pytorch",
                "--lr",
                "0.006",
                "--momentum",
                "0.86",
                "--nesterov",
                "True",
                "--muon_ns_steps",
                "6",
            ]
        )

        self.assertEqual(type(opt).__name__, "Muon")
        self.assertEqual(opt.kwargs["lr"], 0.006)
        self.assertEqual(opt.kwargs["momentum"], 0.86)
        self.assertIs(opt.kwargs["nesterov"], True)
        self.assertEqual(opt.kwargs["ns_steps"], 6)
        self.assertEqual(opt.kwargs["ns_coefficients"], (3.4445, -4.775, 2.0315))
        self.assertEqual(opt.kwargs["eps"], 1e-7)
        self.assertIsNone(opt.kwargs["adjust_lr_fn"])
        self.assertEqual(cfg.opt, "muon-pytorch")

    def test_scheduler_none_returns_none(self):
        _, scheduler, _ = self._run_main(["--opt", "adamw", "--scheduler", "none"])

        self.assertIsNone(scheduler)

    def test_cos_inf_and_wsd_use_lambda_lr_for_standard_optimizers(self):
        for scheduler_name in ("cos_inf", "wsd"):
            with self.subTest(scheduler=scheduler_name):
                _, scheduler, _ = self._run_main(
                    ["--opt", "adamw", "--scheduler", scheduler_name]
                )

                self.assertEqual(type(scheduler).__name__, "LambdaLR")

    def test_muon_family_uses_combined_scheduler_for_non_none_schedulers(self):
        for opt_name in ("muon", "muon-magma", "newton-muon", "softeq-k2000-muon"):
            for scheduler_name in ("cos", "cos_inf", "wsd"):
                with self.subTest(opt=opt_name, scheduler=scheduler_name):
                    _, scheduler, _ = self._run_main(
                        ["--opt", opt_name, "--scheduler", scheduler_name]
                    )

                    self.assertEqual(type(scheduler).__name__, "FakeCombinedScheduler")

    def test_gn_scheduler_max_lr_contract_is_stable(self):
        _, scheduler, cfg = self._run_main(
            ["--opt", "gn-prox", "--lr", "0.006", "--gn_inner_lr", "0.004"]
        )

        self.assertEqual(type(scheduler).__name__, "OneCycleLR")
        self.assertEqual(scheduler.kwargs["max_lr"], [cfg.lr, cfg.lr * 0.5])
        self.assertEqual(scheduler.kwargs["anneal_strategy"], cfg.scheduler)


if __name__ == "__main__":
    unittest.main()
