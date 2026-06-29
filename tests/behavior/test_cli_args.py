import unittest

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
        self.assertEqual(args.results_base_folder, "./exps")

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


if __name__ == "__main__":
    unittest.main()
