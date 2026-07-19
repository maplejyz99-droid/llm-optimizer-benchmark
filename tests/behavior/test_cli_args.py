import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from tests._helpers.behavior_harness import parse_base_args


class CliArgumentBehaviorTest(unittest.TestCase):
    def test_default_training_arguments_are_stable(self):
        args, _ = parse_base_args([])

        self.assertEqual(args.config_format, "base")
        self.assertEqual(args.opt, "adamw")
        self.assertEqual(args.model, "llama")
        self.assertEqual(args.dataset, "slimpajama")
        self.assertEqual(args.scheduler, "cos")
        self.assertEqual(args.lr, 1e-3)
        self.assertEqual(args.weight_decay, 1e-1)
        self.assertEqual(args.beta1, 0.9)
        self.assertEqual(args.beta2, 0.95)
        self.assertEqual(args.device, "cuda:0")
        self.assertEqual(args.iterations, 15000)
        self.assertEqual(args.warmup_steps, 3000)
        self.assertIsNone(args.final_eval_batches)
        self.assertIsNone(args.final_eval_tokens)
        self.assertIs(args.save_final_model, False)
        self.assertEqual(args.sophia_estimator_mode, "legacy_last_microbatch")
        self.assertIs(args.sophia_verify_rank_state, False)
        self.assertEqual(args.distributed_control_timeout_seconds, 86400)
        self.assertIs(args.from_dense, True)
        self.assertEqual(args.results_base_folder, "./exps")

    def test_datasets_dir_uses_portable_environment_default_and_cli_precedence(self):
        with patch.dict(
            "os.environ", {"LLMOPT_DATASETS_DIR": "/data/shared/llmopt"}
        ):
            from_environment, _ = parse_base_args([])
            from_cli, _ = parse_base_args(
                ["--datasets_dir", "/data/explicit/fineweb-30B"]
            )

        self.assertEqual(from_environment.datasets_dir, "/data/shared/llmopt")
        self.assertEqual(from_cli.datasets_dir, "/data/explicit/fineweb-30B")

    def test_dataset_download_requires_explicit_opt_in(self):
        default_args, _ = parse_base_args([])
        opted_in_args, _ = parse_base_args(["--allow_dataset_download"])

        self.assertIs(default_args.allow_dataset_download, False)
        self.assertIs(opted_in_args.allow_dataset_download, True)

    def test_final_eval_caps_must_be_positive_and_mutually_exclusive(self):
        batch_args, _ = parse_base_args(["--final_eval_batches", "17"])
        token_args, _ = parse_base_args(["--final_eval_tokens", "4096"])
        self.assertEqual(batch_args.final_eval_batches, 17)
        self.assertEqual(token_args.final_eval_tokens, 4096)

        for flag in ("--final_eval_batches", "--final_eval_tokens"):
            for value in ("0", "-1"):
                with self.subTest(flag=flag, value=value):
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        parse_base_args([flag, value])

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_base_args(
                ["--final_eval_batches", "17", "--final_eval_tokens", "4096"]
            )

    def test_sophia_estimator_and_final_model_flags_are_explicit(self):
        args, _ = parse_base_args(
            [
                "--sophia_estimator_mode",
                "global_accum",
                "--sophia_verify_rank_state",
                "True",
                "--save_final_model",
                "True",
            ]
        )

        self.assertEqual(args.sophia_estimator_mode, "global_accum")
        self.assertIs(args.sophia_verify_rank_state, True)
        self.assertIs(args.save_final_model, True)

    def test_distributed_control_timeout_must_be_positive(self):
        args, _ = parse_base_args(["--distributed_control_timeout_seconds", "123"])
        self.assertEqual(args.distributed_control_timeout_seconds, 123)

        for value in ("0", "-1"):
            with self.subTest(value=value):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_base_args(["--distributed_control_timeout_seconds", value])

    def test_periodic_evaluation_controls_must_be_positive(self):
        for flag in ("--eval_interval", "--eval_batches"):
            for value in ("0", "-1"):
                with self.subTest(flag=flag, value=value):
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        parse_base_args([flag, value])

    def test_from_dense_keeps_bare_flag_compatibility_and_accepts_explicit_bool(self):
        bare, _ = parse_base_args(["--from_dense"])
        explicit_true, _ = parse_base_args(["--from_dense", "True"])
        explicit_false, _ = parse_base_args(["--from_dense", "False"])

        self.assertIs(bare.from_dense, True)
        self.assertIs(explicit_true.from_dense, True)
        self.assertIs(explicit_false.from_dense, False)

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_base_args(["--from_dense", "not-a-bool"])

    def test_model_options_reject_only_silent_combinations(self):
        supported_argvs = [
            ["--model", "base", "--parallel_block"],
            ["--model", "mup_gpt", "--parallel_block"],
            ["--model", "base", "--moe", "--moe_routing", "expert_choice"],
            ["--model", "mup_gpt", "--moe", "--moe_routing", "expert_choice"],
        ]
        for argv in supported_argvs:
            with self.subTest(argv=argv):
                parse_base_args(argv)

        unsupported_argvs = [
            ["--mlp_dim_exp_factor", "2"],
            ["--model", "llama", "--parallel_block"],
            ["--model", "mup_llama", "--parallel_block"],
            ["--model", "llama", "--moe", "--moe_routing", "expert_choice"],
            ["--model", "mup_llama", "--moe", "--moe_routing", "expert_choice"],
            ["--model", "base", "--moe_routing", "expert_choice"],
        ]
        for argv in unsupported_argvs:
            with self.subTest(argv=argv):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_base_args(argv)

    def test_optimizer_choices_are_stable(self):
        _, parser = parse_base_args([])
        opt_action = next(action for action in parser._actions if "--opt" in action.option_strings)

        self.assertEqual(
            opt_action.choices,
            [
                "adamw",
                "gn-prox",
                "gn-full",
                "cadamw",
                "adamw-magma",
                "sgd",
                "muon",
                "newton-muon",
                "muon-magma",
                "softeq-k2000-muon",
                "soap",
                "ademamix",
                "lion",
                "sf-adamw",
                "sf-sgd",
                "signsgd",
                "signum",
                "prodigy",
                "sophiag",
                "adopt",
                "mars",
                "adafactor",
                "lamb",
                "scion",
                "scion-light",
                "d-muon",
                "muon-pytorch",
            ],
        )

    def test_newton_muon_arguments_are_parseable(self):
        args, _ = parse_base_args(
            [
                "--opt",
                "newton-muon",
                "--model",
                "llama",
                "--device",
                "cpu",
                "--muon_lr_factor",
                "0.01",
                "--newton_muon_precond_every",
                "32",
                "--newton_muon_precond_ewma",
                "0.95",
                "--newton_muon_precond_init_diag",
                "1e-3",
                "--newton_muon_precond_ridge_mult",
                "0.2",
                "--newton_muon_precond_eps",
                "1e-8",
            ]
        )

        self.assertEqual(args.opt, "newton-muon")
        self.assertEqual(args.model, "llama")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.muon_lr_factor, 0.01)
        self.assertEqual(args.newton_muon_precond_every, 32)
        self.assertEqual(args.newton_muon_precond_ewma, 0.95)

    def test_string_boolean_arguments_parse_true_false_and_reject_other_text(self):
        boolean_flags = [
            "auto_resume",
            "allow_legacy_checkpoint_resume",
            "merge_dims",
            "precondition_1d",
            "normalize_grads",
            "correct_bias",
            "nesterov",
            "prodigy_decouple",
            "prodigy_use_bias_correction",
            "prodigy_safeguard_warmup",
            "prodigy_fsdp_in_use",
            "mars_is_approx",
            "lamb_use_bias_correction",
            "adopt_decouple",
            "bias",
        ]

        for flag in boolean_flags:
            with self.subTest(flag=flag, value="True"):
                args, _ = parse_base_args([f"--{flag}", "True"])
                self.assertIs(getattr(args, flag), True)

            with self.subTest(flag=flag, value="False"):
                args, _ = parse_base_args([f"--{flag}", "False"])
                self.assertIs(getattr(args, flag), False)

            with self.subTest(flag=flag, value="not-a-bool"):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_base_args([f"--{flag}", "not-a-bool"])

    def test_unimplemented_public_options_fail_before_training(self):
        unsupported_argvs = [
            ["--tokenizer", "mistral"],
            ["--resume_from_swa", "checkpoint.pt"],
            ["--clipping_type", "local"],
            ["--clip_eta", "0.5"],
            ["--n_kv_head", "4"],
        ]

        for argv in unsupported_argvs:
            with self.subTest(argv=argv):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_base_args(argv)


if __name__ == "__main__":
    unittest.main()
