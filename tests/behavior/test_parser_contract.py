import argparse
import unittest

from tests._helpers.behavior_harness import parse_base_args


EXPECTED_PARSER_CONTRACTS = [
    (("-h", "--help"), "help", "_HelpAction", "==SUPPRESS==", None, None, 0, False),
    (("--config_format",), "config_format", "_StoreAction", "base", None, ("base",), None, False),
    (("--run_prefix",), "run_prefix", "_StoreAction", None, "str", None, None, False),
    (("--experiment_name",), "experiment_name", "_StoreAction", None, "str", None, None, False),
    (("--seed",), "seed", "_StoreAction", 0, "int", None, None, False),
    (("--data_seed",), "data_seed", "_StoreAction", 1337, "int", None, None, False),
    (("--eval_interval",), "eval_interval", "_StoreAction", 200, "int", None, None, False),
    (("--full_eval_at",), "full_eval_at", "_StoreAction", None, "int", None, "+", False),
    (("--eval_batches",), "eval_batches", "_StoreAction", 64, "int", None, None, False),
    (("--final_eval_batches",), "final_eval_batches", "_StoreAction", None, "positive_int", None, None, False),
    (("--device",), "device", "_StoreAction", "cuda:0", "str", None, None, False),
    (("--distributed_backend",), "distributed_backend", "_StoreAction", None, "str", (None, "nccl"), None, False),
    (("--log_interval",), "log_interval", "_StoreAction", 50, "int", None, None, False),
    (("--results_base_folder",), "results_base_folder", "_StoreAction", "./exps", "str", None, None, False),
    (("--permanent_ckpt_interval",), "permanent_ckpt_interval", "_StoreAction", 0, "int", None, None, False),
    (("--latest_ckpt_interval",), "latest_ckpt_interval", "_StoreAction", 0, "int", None, None, False),
    (("--resume_from",), "resume_from", "_StoreAction", None, "str", None, None, False),
    (("--resume_from_swa",), "resume_from_swa", "_StoreAction", None, "str", None, None, False),
    (("--auto_resume",), "auto_resume", "_StoreAction", True, "strict_bool", None, None, False),
    (("--allow_legacy_checkpoint_resume",), "allow_legacy_checkpoint_resume", "_StoreAction", False, "strict_bool", None, None, False),
    (("--wandb",), "wandb", "_StoreTrueAction", False, None, None, 0, False),
    (("--wandb_project",), "wandb_project", "_StoreAction", "my-project", "str", None, None, False),
    (("--wandb_run_prefix",), "wandb_run_prefix", "_StoreAction", "none", "str", None, None, False),
    (("--eval_seq_prefix",), "eval_seq_prefix", "_StoreAction", "none", "str", None, None, False),
    (("--log_dynamics",), "log_dynamics", "_StoreTrueAction", False, None, None, 0, False),
    (("--dynamics_logger_cfg",), "dynamics_logger_cfg", "_StoreAction", "./src/logger/rotational_logger.yaml", "str", None, None, False),
    (("--log_optimizer_groups",), "log_optimizer_groups", "_StoreTrueAction", False, None, None, 0, False),
    (("--wandb_entity",), "wandb_entity", "_StoreAction", None, "none_or_str", None, None, False),
    (("--log_parameter_norms",), "log_parameter_norms", "_StoreTrueAction", False, None, None, 0, False),
    (("--norm_order",), "norm_order", "_StoreAction", 2, None, None, None, False),
    (("--notify_interval",), "notify_interval", "_StoreAction", 0, "int", None, None, False),
    (("--notify_method",), "notify_method", "_StoreAction", "stdout", None, ("stdout", "email", "webhook", "pushplus"), None, False),
    (("--notify_email_to",), "notify_email_to", "_StoreAction", None, "none_or_str", None, None, False),
    (("--notify_email_from",), "notify_email_from", "_StoreAction", None, "none_or_str", None, None, False),
    (("--notify_smtp_host",), "notify_smtp_host", "_StoreAction", None, "none_or_str", None, None, False),
    (("--notify_smtp_port",), "notify_smtp_port", "_StoreAction", None, "none_or_str", None, None, False),
    (("--notify_smtp_user",), "notify_smtp_user", "_StoreAction", None, "none_or_str", None, None, False),
    (("--notify_smtp_pass",), "notify_smtp_pass", "_StoreAction", None, "none_or_str", None, None, False),
    (("--notify_webhook",), "notify_webhook", "_StoreAction", None, "none_or_str", None, None, False),
    (("--notify_pushplus_token",), "notify_pushplus_token", "_StoreAction", None, "none_or_str", None, None, False),
    (("--notify_pushplus_topic",), "notify_pushplus_topic", "_StoreAction", None, "none_or_str", None, None, False),
    (("--notify_pushplus_template",), "notify_pushplus_template", "_StoreAction", None, "none_or_str", None, None, False),
    (("--notify_pushplus_url",), "notify_pushplus_url", "_StoreAction", None, "none_or_str", None, None, False),
    (("--scheduler",), "scheduler", "_StoreAction", "cos", None, ("linear", "cos", "wsd", "none", "cos_inf"), None, False),
    (("--final_div_factor",), "final_div_factor", "_StoreAction", 1, "float", None, None, False),
    (("--cos_inf_steps",), "cos_inf_steps", "_StoreAction", 0, "int", None, None, False),
    (("--iterations",), "iterations", "_StoreAction", 15000, "int", None, None, False),
    (("--warmup_steps",), "warmup_steps", "_StoreAction", 3000, "int", None, None, False),
    (("--lr",), "lr", "_StoreAction", 0.001, "float", None, None, False),
    (("--wsd_final_lr_scale",), "wsd_final_lr_scale", "_StoreAction", 0.0, "float", None, None, False),
    (("--wsd_fract_decay",), "wsd_fract_decay", "_StoreAction", 0.1, "float", None, None, False),
    (("--decay_type",), "decay_type", "_StoreAction", "linear", None, ("linear", "cosine", "exp", "miror_cosine", "square", "sqrt"), None, False),
    (("--opt",), "opt", "_StoreAction", "adamw", None, ("adamw", "gn-prox", "gn-full", "cadamw", "adamw-magma", "sgd", "muon", "newton-muon", "muon-magma", "softeq-k2000-muon", "soap", "ademamix", "lion", "sf-adamw", "sf-sgd", "signsgd", "signum", "prodigy", "sophiag", "adopt", "mars", "adafactor", "lamb", "scion", "scion-light", "d-muon", "muon-pytorch"), None, False),
    (("--batch_size",), "batch_size", "_StoreAction", 50, "int", None, None, False),
    (("--acc_steps",), "acc_steps", "_StoreAction", 1, "int", None, None, False),
    (("--weight_decay",), "weight_decay", "_StoreAction", 0.1, "float", None, None, False),
    (("--beta1",), "beta1", "_StoreAction", 0.9, "float", None, None, False),
    (("--beta2",), "beta2", "_StoreAction", 0.95, "float", None, None, False),
    (("--grad_clip",), "grad_clip", "_StoreAction", 1.0, "float", None, None, False),
    (("--momentum",), "momentum", "_StoreAction", 0.9, "float", None, None, False),
    (("--shampoo_beta",), "shampoo_beta", "_StoreAction", -1.0, "float", None, None, False),
    (("--precondition_frequency",), "precondition_frequency", "_StoreAction", 10, "int", None, None, False),
    (("--max_precond_dim",), "max_precond_dim", "_StoreAction", 10000, "int", None, None, False),
    (("--merge_dims",), "merge_dims", "_StoreAction", False, "strict_bool", None, None, False),
    (("--precondition_1d",), "precondition_1d", "_StoreAction", False, "strict_bool", None, None, False),
    (("--normalize_grads",), "normalize_grads", "_StoreAction", False, "strict_bool", None, None, False),
    (("--soap_data_format",), "soap_data_format", "_StoreAction", "channels_first", "str", None, None, False),
    (("--correct_bias",), "correct_bias", "_StoreAction", True, "strict_bool", None, None, False),
    (("--nesterov",), "nesterov", "_StoreAction", False, "strict_bool", None, None, False),
    (("--muon_ns_steps",), "muon_ns_steps", "_StoreAction", 5, "int", None, None, False),
    (("--muon_lr_factor",), "muon_lr_factor", "_StoreAction", 1.0, "float", None, None, False),
    (("--newton_muon_precond_every",), "newton_muon_precond_every", "_StoreAction", 32, "int", None, None, False),
    (("--newton_muon_precond_ewma",), "newton_muon_precond_ewma", "_StoreAction", 0.95, "float", None, None, False),
    (("--newton_muon_precond_init_diag",), "newton_muon_precond_init_diag", "_StoreAction", 0.001, "float", None, None, False),
    (("--newton_muon_precond_ridge_mult",), "newton_muon_precond_ridge_mult", "_StoreAction", 0.2, "float", None, None, False),
    (("--newton_muon_precond_eps",), "newton_muon_precond_eps", "_StoreAction", 1e-08, "float", None, None, False),
    (("--cautious_xi",), "cautious_xi", "_StoreAction", 1.0, "float", None, None, False),
    (("--magma_survival_p",), "magma_survival_p", "_StoreAction", 0.5, "float", None, None, False),
    (("--magma_tau",), "magma_tau", "_StoreAction", 2.0, "float", None, None, False),
    (("--magma_beta",), "magma_beta", "_StoreAction", 0.9, "float", None, None, False),
    (("--magma_scope",), "magma_scope", "_StoreAction", "all", "str", ("all", "attn-mlp"), None, False),
    (("--adema_beta3",), "adema_beta3", "_StoreAction", 0.9, "float", None, None, False),
    (("--adema_alpha",), "adema_alpha", "_StoreAction", 2.0, "float", None, None, False),
    (("--adema_beta3_warmup",), "adema_beta3_warmup", "_StoreAction", None, "int", None, None, False),
    (("--adema_alpha_warmup",), "adema_alpha_warmup", "_StoreAction", None, "int", None, None, False),
    (("--schedulefree_r",), "schedulefree_r", "_StoreAction", 0.0, "float", None, None, False),
    (("--weight_lr_power",), "weight_lr_power", "_StoreAction", 2.0, "float", None, None, False),
    (("--dampening",), "dampening", "_StoreAction", 0.0, "float", None, None, False),
    (("--prodigy_beta3",), "prodigy_beta3", "_StoreAction", None, "float", None, None, False),
    (("--prodigy_decouple",), "prodigy_decouple", "_StoreAction", True, "strict_bool", None, None, False),
    (("--prodigy_use_bias_correction",), "prodigy_use_bias_correction", "_StoreAction", False, "strict_bool", None, None, False),
    (("--prodigy_safeguard_warmup",), "prodigy_safeguard_warmup", "_StoreAction", False, "strict_bool", None, None, False),
    (("--prodigy_fsdp_in_use",), "prodigy_fsdp_in_use", "_StoreAction", False, "strict_bool", None, None, False),
    (("--sophia_rho",), "sophia_rho", "_StoreAction", 0.04, "float", None, None, False),
    (("--sophia_bs",), "sophia_bs", "_StoreAction", 480, "int", None, None, False),
    (("--clipping_type",), "clipping_type", "_StoreAction", "no", None, ("no", "local", "elementwise"), None, False),
    (("--clip_eta",), "clip_eta", "_StoreAction", 1.0, "float", None, None, False),
    (("--mars_type",), "mars_type", "_StoreAction", "mars-adamw", None, ("mars-adamw", "mars-lion", "mars-shampoo"), None, False),
    (("--mars_vr_gamma",), "mars_vr_gamma", "_StoreAction", 0.025, "float", None, None, False),
    (("--mars_is_approx",), "mars_is_approx", "_StoreAction", True, "strict_bool", None, None, False),
    (("--mars_lr",), "mars_lr", "_StoreAction", 0.003, "float", None, None, False),
    (("--mars_beta1",), "mars_beta1", "_StoreAction", 0.95, "float", None, None, False),
    (("--mars_beta2",), "mars_beta2", "_StoreAction", 0.99, "float", None, None, False),
    (("--adafactor_decay_rate",), "adafactor_decay_rate", "_StoreAction", -0.8, "float", None, None, False),
    (("--lamb_use_bias_correction",), "lamb_use_bias_correction", "_StoreAction", False, "strict_bool", None, None, False),
    (("--adopt_decouple",), "adopt_decouple", "_StoreAction", True, "strict_bool", None, None, False),
    (("--adopt_eps",), "adopt_eps", "_StoreAction", 1e-06, "float", None, None, False),
    (("--scion_lmh_scale",), "scion_lmh_scale", "_StoreAction", 10.0, "float", None, None, False),
    (("--scion_emb_scale",), "scion_emb_scale", "_StoreAction", 1.0, "float", None, None, False),
    (("--scion_tr_scale",), "scion_tr_scale", "_StoreAction", 3.0, "float", None, None, False),
    (("--gn_inner_iters",), "gn_inner_iters", "_StoreAction", 8, "int", None, None, False),
    (("--gn_inner_lr",), "gn_inner_lr", "_StoreAction", 0.001, "float", None, None, False),
    (("--gn_inner_b1",), "gn_inner_b1", "_StoreAction", 0.9, "float", None, None, False),
    (("--gn_inner_b2",), "gn_inner_b2", "_StoreAction", 0.999, "float", None, None, False),
    (("--gn_inner_wd",), "gn_inner_wd", "_StoreAction", 0.0, "float", None, None, False),
    (("--gn_linesearch",), "gn_linesearch", "_StoreTrueAction", False, None, None, 0, False),
    (("--gn_ls_range",), "gn_ls_range", "_StoreAction", 5, "int", None, None, False),
    (("--gn_log_inner_steps",), "gn_log_inner_steps", "_StoreTrueAction", False, None, None, 0, False),
    (("--weight_average",), "weight_average", "_StoreTrueAction", False, None, None, 0, False),
    (("--wa_interval",), "wa_interval", "_StoreAction", 5, "int", None, None, False),
    (("--wa_horizon",), "wa_horizon", "_StoreAction", 500, "int", None, None, False),
    (("--wa_dtype",), "wa_dtype", "_StoreAction", "float32", "str", ("float32", "float64"), None, False),
    (("--wa_use_temp_dir",), "wa_use_temp_dir", "_StoreTrueAction", False, None, None, 0, False),
    (("--wa_sweep_horizon",), "wa_sweep_horizon", "_StoreTrueAction", False, None, None, 0, False),
    (("--max_num_wa_sweeps",), "max_num_wa_sweeps", "_StoreAction", 5, "int", None, None, False),
    (("--exponential_weight_average",), "exponential_weight_average", "_StoreTrueAction", False, None, None, 0, False),
    (("--ewa_interval",), "ewa_interval", "_StoreAction", 10, "int", None, None, False),
    (("--ewa_decay",), "ewa_decay", "_StoreAction", 0.95, "float", None, None, False),
    (("--ewa_after_warmup",), "ewa_after_warmup", "_StoreTrueAction", False, None, None, 0, False),
    (("--datasets_dir",), "datasets_dir", "_StoreAction", "./src/data/datasets/", "str", None, None, False),
    (("--dataset",), "dataset", "_StoreAction", "slimpajama", None, ("wikitext", "shakespeare-char", "arxiv", "arxiv2000", "arxiv+wiki", "openwebtext2", "redpajama", "slimpajama", "slimpajama_chunk1", "redpajamav2", "fineweb", "finewebedu", "c4", "arc_easy", "arc_challenge", "hellaswag", "logiqa", "piqa", "sciq", "humaneval", "gsm8k", "kodcode", "mathqa", "medqa"), None, False),
    (("--tokenizer",), "tokenizer", "_StoreAction", "gpt2", "str", ("gpt2",), None, False),
    (("--vocab_size",), "vocab_size", "_StoreAction", 50304, "int", None, None, False),
    (("--data_in_ram",), "data_in_ram", "_StoreTrueAction", False, None, None, 0, False),
    (("--model",), "model", "_StoreAction", "llama", None, ("base", "llama", "mup_gpt", "mup_llama"), None, False),
    (("--parallel_block",), "parallel_block", "_StoreTrueAction", False, None, None, 0, False),
    (("--use_pretrained",), "use_pretrained", "_StoreAction", "none", "str", None, None, False),
    (("--from_dense",), "from_dense", "_StoreTrueAction", False, None, None, 0, False),
    (("--init_std",), "init_std", "_StoreAction", 0.02, "float", None, None, False),
    (("--dropout",), "dropout", "_StoreAction", 0.0, "float", None, None, False),
    (("--n_head",), "n_head", "_StoreAction", 12, "int", None, None, False),
    (("--n_layer",), "n_layer", "_StoreAction", 24, "int", None, None, False),
    (("--sequence_length",), "sequence_length", "_StoreAction", 512, "int", None, None, False),
    (("--n_embd",), "n_embd", "_StoreAction", 768, "int", None, None, False),
    (("--multiple_of",), "multiple_of", "_StoreAction", 256, "int", None, None, False),
    (("--n_kv_head",), "n_kv_head", "_StoreAction", None, "int", None, None, False),
    (("--rmsnorm_eps",), "rmsnorm_eps", "_StoreAction", 1e-05, "float", None, None, False),
    (("--dtype",), "dtype", "_StoreAction", "bfloat16", "str", ("float32", "float16", "bfloat16"), None, False),
    (("--bias",), "bias", "_StoreAction", False, "strict_bool", None, None, False),
    (("--compile",), "compile", "_StoreTrueAction", False, None, None, 0, False),
    (("--untied_embeds",), "untied_embeds", "_StoreTrueAction", False, None, None, 0, False),
    (("--mlp_dim_exp_factor",), "mlp_dim_exp_factor", "_StoreAction", 1.0, "float", None, None, False),
    (("--moe",), "moe", "_StoreTrueAction", False, None, None, 0, False),
    (("--moe_routing",), "moe_routing", "_StoreAction", "standard_gating", "str", ("standard_gating", "expert_choice"), None, False),
    (("--moe_num_experts",), "moe_num_experts", "_StoreAction", 8, "int", None, None, False),
    (("--capacity_factor",), "capacity_factor", "_StoreAction", 2.0, "float", None, None, False),
    (("--moe_num_shared_experts",), "moe_num_shared_experts", "_StoreAction", 0, "int", None, None, False),
    (("--moe_router_loss",), "moe_router_loss", "_StoreAction", "load_balancing_z_loss", "str", ("entropy", "load_balancing_only", "load_balancing_z_loss"), None, False),
    (("--moe_num_experts_per_tok",), "moe_num_experts_per_tok", "_StoreAction", 2, "int", None, None, False),
    (("--moe_entropy_loss_factor",), "moe_entropy_loss_factor", "_StoreAction", 0.01, "float", None, None, False),
    (("--moe_aux_loss_factor",), "moe_aux_loss_factor", "_StoreAction", 0.1, "float", None, None, False),
    (("--moe_z_loss_factor",), "moe_z_loss_factor", "_StoreAction", 0.01, "float", None, None, False),
    (("--moe_softmax_order",), "moe_softmax_order", "_StoreAction", "topk_softmax", "str", ("softmax_topk", "topk_softmax"), None, False),
    (("--plot_router_logits",), "plot_router_logits", "_StoreTrueAction", False, None, None, 0, False),
    (("--scale_emb",), "scale_emb", "_StoreAction", 10, "int", None, None, False),
    (("--scale_base_model",), "scale_base_model", "_StoreAction", 256, "int", None, None, False),
    (("--scale_depth",), "scale_depth", "_StoreAction", 1.4, "float", None, None, False),
]


def _stable_value(value):
    if value is argparse.SUPPRESS:
        return "==SUPPRESS=="
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return repr(value)


def _stable_type(value):
    if value is None:
        return None
    return getattr(value, "__name__", repr(value))


def _parser_contract(parser):
    return [
        (
            tuple(action.option_strings),
            action.dest,
            type(action).__name__,
            _stable_value(action.default),
            _stable_type(getattr(action, "type", None)),
            _stable_value(getattr(action, "choices", None)),
            _stable_value(getattr(action, "nargs", None)),
            getattr(action, "required", False),
        )
        for action in parser._actions
        if action.option_strings
    ]


class ParserContractBehaviorTest(unittest.TestCase):
    maxDiff = None

    def test_parser_action_contract_is_stable(self):
        _, parser = parse_base_args([])

        self.assertEqual(_parser_contract(parser), EXPECTED_PARSER_CONTRACTS)


if __name__ == "__main__":
    unittest.main()
