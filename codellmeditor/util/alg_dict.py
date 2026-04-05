from ..models.rome import ROMEHyperParams, apply_rome_to_model
from ..models.ft import FTHyperParams, apply_ft_to_model
from ..models.grace import GraceHyperParams, apply_grace_to_model
from ..models.agrace import AGraceHyperParams, apply_agrace_to_model
from ..models.pmet import PMETHyperParams, apply_pmet_to_model
from ..models.non_edit import NON_EDITHyperParams, apply_non_edit_to_model
from ..models.malmen import MALMENHyperParams, MalmenRewriteExecutor
from ..models.memit import MEMITHyperParams, apply_memit_to_model
from ..models.alphaedit import AlphaEdit_hparams, apply_AlphaEdit_to_model
from ..models.lora import LoRAHyperParams, apply_lora_to_model
from ..models.replace_api import REPLACE_APIHyperParams, apply_replace_api_to_model
from ..models.routed_adalora import RoutedAdaLoRAHyperParams, apply_routed_adalora_to_model


ALG_DICT = {
    'ROME': apply_rome_to_model,
    "FT": apply_ft_to_model,
    'GRACE': apply_grace_to_model,
    'AGRACE': apply_agrace_to_model,
    'PMET': apply_pmet_to_model,
    'non-edit': apply_non_edit_to_model,
    'MALMEN': MalmenRewriteExecutor().apply_to_model,
    "MEMIT": apply_memit_to_model,
    'AlphaEdit': apply_AlphaEdit_to_model,
    'LoRA': apply_lora_to_model,
    'replace_api': apply_replace_api_to_model,
    'ROUTED_ADALORA': apply_routed_adalora_to_model
}
