"""Sylheti dataset loading and scoring normalization."""
from dataset_io import match_recordings, read_references
from languages.bn import normalize

# No engine has a Sylheti code, and it is written in Bengali script
ASR_LANGUAGE = "bn"


def load_samples(directory):
    # Recordings are transcribed without a reference until one is supplied
    path = directory / "references" / "cases_syl.json"
    references = read_references(path, ("syl_narrative",)) \
        if path.exists() else None
    return match_recordings(directory / "audio", references,
                            lambda path: path.stem.removesuffix("_sn"))
