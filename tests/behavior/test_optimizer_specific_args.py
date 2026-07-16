import unittest

from tests._helpers.behavior_harness import parse_base_args


class OptimizerSpecificCliArgumentBehaviorTest(unittest.TestCase):
    def parse(self, *argv):
        args, _ = parse_base_args(list(argv))
        return args

    def test_cadamw_specific_argument_is_parseable(self):
        args = self.parse("--opt", "cadamw", "--cautious_xi", "0.7")

        self.assertEqual(args.opt, "cadamw")
        self.assertEqual(args.cautious_xi, 0.7)

    def test_soap_specific_arguments_are_parseable(self):
        args = self.parse(
            "--opt",
            "soap",
            "--shampoo_beta",
            "0.92",
            "--precondition_frequency",
            "12",
            "--max_precond_dim",
            "4096",
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
        )

        self.assertEqual(args.opt, "soap")
        self.assertEqual(args.shampoo_beta, 0.92)
        self.assertEqual(args.precondition_frequency, 12)
        self.assertEqual(args.max_precond_dim, 4096)
        self.assertIs(args.merge_dims, True)
        self.assertIs(args.precondition_1d, True)
        self.assertIs(args.normalize_grads, True)
        self.assertEqual(args.soap_data_format, "channels_last")
        self.assertIs(args.correct_bias, True)

    def test_ademamix_specific_arguments_are_parseable(self):
        args = self.parse(
            "--opt",
            "ademamix",
            "--adema_beta3",
            "0.81",
            "--adema_alpha",
            "3.5",
            "--adema_beta3_warmup",
            "120",
            "--adema_alpha_warmup",
            "240",
        )

        self.assertEqual(args.opt, "ademamix")
        self.assertEqual(args.adema_beta3, 0.81)
        self.assertEqual(args.adema_alpha, 3.5)
        self.assertEqual(args.adema_beta3_warmup, 120)
        self.assertEqual(args.adema_alpha_warmup, 240)

    def test_mars_specific_arguments_are_parseable(self):
        args = self.parse(
            "--opt",
            "mars",
            "--mars_lr",
            "0.004",
            "--mars_beta1",
            "0.91",
            "--mars_beta2",
            "0.997",
            "--mars_type",
            "mars-shampoo",
            "--mars_vr_gamma",
            "0.031",
            "--mars_is_approx",
            "False",
        )

        self.assertEqual(args.opt, "mars")
        self.assertEqual(args.mars_lr, 0.004)
        self.assertEqual(args.mars_beta1, 0.91)
        self.assertEqual(args.mars_beta2, 0.997)
        self.assertEqual(args.mars_type, "mars-shampoo")
        self.assertEqual(args.mars_vr_gamma, 0.031)
        self.assertIs(args.mars_is_approx, False)

    def test_prodigy_specific_arguments_are_parseable(self):
        args = self.parse(
            "--opt",
            "prodigy",
            "--prodigy_beta3",
            "0.999",
            "--prodigy_decouple",
            "True",
            "--prodigy_use_bias_correction",
            "True",
            "--prodigy_safeguard_warmup",
            "True",
            "--prodigy_fsdp_in_use",
            "True",
        )

        self.assertEqual(args.opt, "prodigy")
        self.assertEqual(args.prodigy_beta3, 0.999)
        self.assertIs(args.prodigy_decouple, True)
        self.assertIs(args.prodigy_use_bias_correction, True)
        self.assertIs(args.prodigy_safeguard_warmup, True)
        self.assertIs(args.prodigy_fsdp_in_use, True)

    def test_sophiag_specific_arguments_are_parseable(self):
        args = self.parse(
            "--opt",
            "sophiag",
            "--sophia_rho",
            "0.06",
            "--sophia_bs",
            "384",
        )

        self.assertEqual(args.opt, "sophiag")
        self.assertEqual(args.sophia_rho, 0.06)
        self.assertEqual(args.sophia_bs, 384)

    def test_muon_specific_arguments_are_parseable(self):
        args = self.parse(
            "--opt",
            "muon",
            "--muon_lr_factor",
            "0.015",
            "--muon_ns_steps",
            "7",
            "--momentum",
            "0.93",
            "--nesterov",
            "True",
        )

        self.assertEqual(args.opt, "muon")
        self.assertEqual(args.muon_lr_factor, 0.015)
        self.assertEqual(args.muon_ns_steps, 7)
        self.assertEqual(args.momentum, 0.93)
        self.assertIs(args.nesterov, True)

    def test_distributed_muon_specific_arguments_are_parseable(self):
        args = self.parse(
            "--opt",
            "d-muon",
            "--muon_ns_steps",
            "9",
            "--momentum",
            "0.88",
            "--nesterov",
            "True",
        )

        self.assertEqual(args.opt, "d-muon")
        self.assertEqual(args.muon_ns_steps, 9)
        self.assertEqual(args.momentum, 0.88)
        self.assertIs(args.nesterov, True)

    def test_softeq_k2000_muon_specific_arguments_are_parseable(self):
        args = self.parse(
            "--opt",
            "softeq-k2000-muon",
            "--muon_lr_factor",
            "0.014",
            "--momentum",
            "0.91",
        )

        self.assertEqual(args.opt, "softeq-k2000-muon")
        self.assertEqual(args.muon_lr_factor, 0.014)
        self.assertEqual(args.momentum, 0.91)

    def test_newton_muon_preconditioner_arguments_are_parseable(self):
        args = self.parse(
            "--opt",
            "newton-muon",
            "--muon_lr_factor",
            "0.012",
            "--muon_ns_steps",
            "6",
            "--newton_muon_precond_every",
            "48",
            "--newton_muon_precond_ewma",
            "0.93",
            "--newton_muon_precond_init_diag",
            "2e-3",
            "--newton_muon_precond_ridge_mult",
            "0.25",
            "--newton_muon_precond_eps",
            "2e-8",
        )

        self.assertEqual(args.opt, "newton-muon")
        self.assertEqual(args.muon_lr_factor, 0.012)
        self.assertEqual(args.muon_ns_steps, 6)
        self.assertEqual(args.newton_muon_precond_every, 48)
        self.assertEqual(args.newton_muon_precond_ewma, 0.93)
        self.assertEqual(args.newton_muon_precond_init_diag, 2e-3)
        self.assertEqual(args.newton_muon_precond_ridge_mult, 0.25)
        self.assertEqual(args.newton_muon_precond_eps, 2e-8)

    def test_magma_variant_arguments_are_parseable(self):
        for opt_name in ("adamw-magma", "muon-magma"):
            with self.subTest(opt=opt_name):
                args = self.parse(
                    "--opt",
                    opt_name,
                    "--magma_survival_p",
                    "0.65",
                    "--magma_tau",
                    "1.7",
                    "--magma_beta",
                    "0.84",
                    "--magma_scope",
                    "attn-mlp",
                )

                self.assertEqual(args.opt, opt_name)
                self.assertEqual(args.magma_survival_p, 0.65)
                self.assertEqual(args.magma_tau, 1.7)
                self.assertEqual(args.magma_beta, 0.84)
                self.assertEqual(args.magma_scope, "attn-mlp")

    def test_schedulefree_variant_arguments_are_parseable(self):
        for opt_name in ("sf-adamw", "sf-sgd"):
            with self.subTest(opt=opt_name):
                args = self.parse(
                    "--opt",
                    opt_name,
                    "--schedulefree_r",
                    "0.04",
                    "--weight_lr_power",
                    "1.5",
                )

                self.assertEqual(args.opt, opt_name)
                self.assertEqual(args.schedulefree_r, 0.04)
                self.assertEqual(args.weight_lr_power, 1.5)

    def test_sign_optimizer_arguments_are_parseable(self):
        for opt_name in ("signsgd", "signum"):
            with self.subTest(opt=opt_name):
                args = self.parse(
                    "--opt",
                    opt_name,
                    "--momentum",
                    "0.77",
                    "--dampening",
                    "0.03",
                    "--nesterov",
                    "True",
                )

                self.assertEqual(args.opt, opt_name)
                self.assertEqual(args.momentum, 0.77)
                self.assertEqual(args.dampening, 0.03)
                self.assertIs(args.nesterov, True)

    def test_adafactor_specific_argument_is_parseable(self):
        args = self.parse(
            "--opt",
            "adafactor",
            "--adafactor_decay_rate",
            "-0.72",
        )

        self.assertEqual(args.opt, "adafactor")
        self.assertEqual(args.adafactor_decay_rate, -0.72)

    def test_lamb_specific_argument_is_parseable(self):
        args = self.parse(
            "--opt",
            "lamb",
            "--lamb_use_bias_correction",
            "True",
        )

        self.assertEqual(args.opt, "lamb")
        self.assertIs(args.lamb_use_bias_correction, True)

    def test_scion_variant_arguments_are_parseable(self):
        for opt_name in ("scion", "scion-light"):
            with self.subTest(opt=opt_name):
                args = self.parse(
                    "--opt",
                    opt_name,
                    "--scion_lmh_scale",
                    "12.5",
                    "--scion_emb_scale",
                    "1.25",
                    "--scion_tr_scale",
                    "4.5",
                )

                self.assertEqual(args.opt, opt_name)
                self.assertEqual(args.scion_lmh_scale, 12.5)
                self.assertEqual(args.scion_emb_scale, 1.25)
                self.assertEqual(args.scion_tr_scale, 4.5)


if __name__ == "__main__":
    unittest.main()
