"""Benchmark ASR engines on the Zulu and Sesotho COD clips.

Whisper itself ships no model for either language, so the engines are MMS
language adapters, Whisper checkpoints fine-tuned on the African Next Voices
corpus, and Omnilingual ASR. Run through run_benchmark with --language zu or st.
The Omnilingual engines pin old dependencies and need their own environment.
"""
import os

from engines import make_faster_whisper, make_hf_whisper, make_mms, \
    make_omniasr

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

MMS_REPO = "facebook/mms-1b-all"
# MMS has no Sesotho adapter, so Sesotho is decoded with Northern Sotho
MMS_SUBSTITUTES = {"sot": "nso"}

# Single-language fine-tunes, keyed by ISO 639-3
ANV_REPOS = {
    "zul": "dsfsi-anv/whisper-large-v3-turbo-anv-zul",
    "sot": "dsfsi-anv/whisper-large-v3-turbo-anv-sot",
}
ANV_MULTILINGUAL = "dsfsi-anv/za-anv-multilingual-whisper-v3-turbo"
# The multilingual checkpoint ships weights without any tokenizer files
ANV_BASE = "openai/whisper-large-v3-turbo"
# A CTranslate2 conversion of the multilingual checkpoint
SWIVURISO = "digiphyte/swivuriso-turbo"

OMNI_CARDS = {
    "omniasr-llm-300m": "omniASR_LLM_300M",
    "omniasr-llm-3b": "omniASR_LLM_3B",
}


def build_engines(args):
    language = args.language
    if language in MMS_SUBSTITUTES:
        mms_language = MMS_SUBSTITUTES[language]
        mms_name = f"mms-1b-all-{mms_language}"
    else:
        mms_language, mms_name = language, "mms-1b-all"
    engines = {mms_name: lambda: make_mms(MMS_REPO, mms_language, args.device)}
    # Whisper has no token for either language, so these decode on auto-detection
    engines[f"anv-{language}-turbo"] = \
        lambda: make_hf_whisper(ANV_REPOS[language], args.device)
    engines["anv-multilingual-turbo"] = \
        lambda: make_hf_whisper(ANV_MULTILINGUAL, args.device,
                                processor_repo=ANV_BASE)
    engines["swivuriso-turbo"] = \
        lambda: make_faster_whisper(SWIVURISO, None, args.fw_device,
                                    args.compute_type,
                                    condition_on_previous_text=False)
    # Omnilingual names a language by ISO 639-3 plus script
    for name, card in OMNI_CARDS.items():
        engines[name] = lambda card=card: \
            make_omniasr(card, f"{language}_Latn", args.device)
    return engines
