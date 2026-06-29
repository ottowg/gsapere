"""Built-in pipeline presets — no local files required.

All models are loaded from HuggingFace Hub on first use.

Available presets
-----------------
``"gsap-ere"``
    GSAP-ERE: fine-grained entity and relation extraction on ML papers.
    Models: gesis/gsap-ere-pruner, gesis/gsap-ere-hgere,
            gesis/gsap-ere-prune-rules (rule-based pre-filter).
"""

from __future__ import annotations

from gsapere.pipeline.config import (
    FinalPruningConfig,
    HGEREConfig,
    PipelineConfig,
    PrunerConfig,
)
from gsapere.pre_filter.config import ThresholdPreFilterParams

PRESETS: dict[str, PipelineConfig] = {
    "gsap-ere": PipelineConfig(
        label_set="gsap",
        pruner=PrunerConfig(
            model_dir="gesis/gsap-ere-pruner",
            base_model_name_or_path="allenai/scibert_scivocab_uncased",
            do_lower_case=True,
            model_type="bertspanmarkerpruner",
            per_gpu_eval_batch_size=32,
            max_seq_length=256,
            max_pair_length=64,
            max_mention_ori_length=12,
            rulebased_pruner_file="hf://gesis/gsap-ere-prune-rules/rules.json",
            topk_ratio=8,
            min_mentions_num=1,
            max_mentions_num=60,
            final_pruning=FinalPruningConfig(method="threshold", threshold=0.0005),
        ),
        hgere=HGEREConfig(
            model_dir="gesis/gsap-ere-hgere",
            base_model_name_or_path="allenai/scibert_scivocab_uncased",
            model_type="hyper",
            do_lower_case=True,
            shuffle=True,
            no_sym=False,
            per_gpu_eval_batch_size=32,
            max_seq_length=512,
            factor_type="ternary",
            factor_encoder="biaf",
            ent_dim=400,
            rel_dim=400,
            mem_dim=400,
            ent_repr="mix",
            n_iter=3,
            layernorm=True,
            layernorm_1st=True,
            attn_self=True,
            pre_filter_params=ThresholdPreFilterParams(
                method="threshold", value=0.0125
            ),
        ),
    ),
}
