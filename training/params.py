"""Trimmed open-source registry.

This release ships a single training path: CKA-guided MoE upcycling (V5,
Differentiable NAS + Gumbel-Softmax). Competing continual-learning baselines
(EWC / LwF / GEM / OGD / O-LoRA / L2P / LoRA) and generative-replay variants
(LFPT5 / MbPA++), as well as the heavier MoE baselines (Drop-Upcycling,
Branch-Train-MiX), live in the full research repo and are intentionally
omitted here.
"""

from model.Dynamic_network.upcycling_refactored import Upcycle
from model.base_model import CL_Base_Model


Method2Class = {
    "base": CL_Base_Model,
    "upcycle": Upcycle,
    # Note: --cka-regularization with --cka-version v5 dispatches to
    # model/Dynamic_network/upcycling_cka_v5.create_cka_upcycle_v5() from
    # within training/main.py; see that file for the wiring.
}

AllDatasetName = [
    "C-STANCE", "FOMC", "MeetingBank", "Papyrus-f", "Py150",
    "ScienceQA", "ToolBench", "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
]
