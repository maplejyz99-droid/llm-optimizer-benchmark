import unittest
from contextlib import redirect_stdout
from io import StringIO

from tests._helpers.behavior_harness import load_main_with_fakes, parse_base_args


class ExperimentNameBehaviorTest(unittest.TestCase):
    def setUp(self):
        self.capture = {}
        self.main_module = load_main_with_fakes(self.capture)
        self.backend = self.main_module.distributed.make_backend_from_args(
            parse_base_args([])[0]
        )

    def _name_for(self, argv):
        args, parser = parse_base_args(argv)
        return self.main_module.get_exp_name(args, parser, self.backend)

    def test_default_name_is_stable(self):
        self.assertEqual(
            self._name_for([]),
            "50x1_model-llama_dataset-slimpajama_opt-adamw",
        )

    def test_explicit_experiment_name_overrides_generated_name(self):
        self.assertEqual(
            self._name_for(["--experiment_name", "manual_name"]),
            "manual_name",
        )

    def test_ignored_runtime_arguments_do_not_change_name(self):
        baseline = self._name_for([])
        changed_runtime = self._name_for(
            [
                "--wandb",
                "--wandb_project",
                "other-project",
                "--wandb_entity",
                "entity",
                "--device",
                "cpu",
                "--seed",
                "123",
                "--results_base_folder",
                "/tmp/other",
                "--log_interval",
                "1",
                "--eval_batches",
                "7",
                "--eval_seq_prefix",
                "Hello",
                "--final_eval_batches",
                "7",
                "--distributed_control_timeout_seconds",
                "123",
            ]
        )

        self.assertEqual(changed_runtime, baseline)

        changed_token_budget = self._name_for(["--final_eval_tokens", "4096"])
        self.assertEqual(changed_token_budget, baseline)

    def test_notification_arguments_never_enter_generated_name(self):
        secret = "name-secret-value"
        baseline = self._name_for([])
        notified = self._name_for(
            [
                "--notify_interval",
                "10",
                "--notify_method",
                "email",
                "--notify_smtp_pass",
                secret,
            ]
        )

        self.assertEqual(notified, baseline)
        self.assertNotIn(secret, notified)

    def test_runtime_semantics_enrichment_does_not_warn_or_change_name(self):
        args, parser = parse_base_args([])
        baseline = self.main_module.get_exp_name(args, parser, self.backend)
        args.evaluation_protocol = {"identity": "evaluation-id"}
        args.training_semantics = {"schedule_free_parameter_point": "v2"}

        output = StringIO()
        with redirect_stdout(output):
            enriched = self.main_module.get_exp_name(args, parser, self.backend)

        self.assertEqual(enriched, baseline)
        self.assertEqual(output.getvalue(), "")

    def test_key_and_non_default_arguments_change_name(self):
        self.assertEqual(
            self._name_for(["--model", "mup_llama", "--opt", "muon", "--lr", "0.002"]),
            "50x1_model-mup_llama_dataset-slimpajama_opt-muon__lr-0.002",
        )

    def test_moe_model_name_prefix_is_stable(self):
        self.assertEqual(
            self._name_for(["--moe"]),
            "50x1_model-moe_llama_dataset-slimpajama_opt-adamw",
        )


if __name__ == "__main__":
    unittest.main()
