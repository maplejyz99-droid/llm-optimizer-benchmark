import io
import unittest
from contextlib import redirect_stderr

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
        self.assertEqual(args.results_base_folder, "./exps")

    def test_final_eval_batch_cap_must_be_positive_when_set(self):
        args, _ = parse_base_args(["--final_eval_batches", "17"])
        self.assertEqual(args.final_eval_batches, 17)

        for value in ("0", "-1"):
            with self.subTest(value=value):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_base_args(["--final_eval_batches", value])

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
                "softeq-k2000-muon",
                "muon-magma",
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
