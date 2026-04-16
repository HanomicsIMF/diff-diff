"""Runtime accessor for bundled LLM guide files."""
from __future__ import annotations

from importlib.resources import files

_VARIANT_TO_FILE = {
    "concise": "llms.txt",
    "full": "llms-full.txt",
    "practitioner": "llms-practitioner.txt",
}


def get_llm_guide(variant: str = "concise") -> str:
    """Return the contents of a bundled LLM guide.

    Parameters
    ----------
    variant : str, default "concise"
        Which guide to load. Names are case-sensitive. One of:

        - ``"concise"`` -- compact API reference (llms.txt)
        - ``"full"`` -- complete API documentation (llms-full.txt)
        - ``"practitioner"`` -- 8-step practitioner workflow (llms-practitioner.txt)

    Returns
    -------
    str
        The full text of the requested guide.

    Raises
    ------
    ValueError
        If ``variant`` is not one of the known guide names.

    Examples
    --------
    >>> from diff_diff import get_llm_guide
    >>> concise = get_llm_guide()
    >>> workflow = get_llm_guide("practitioner")
    """
    try:
        filename = _VARIANT_TO_FILE[variant]
    except (KeyError, TypeError):
        valid = ", ".join(repr(k) for k in _VARIANT_TO_FILE)
        raise ValueError(
            f"Unknown guide variant {variant!r}. Valid options: {valid}."
        ) from None
    return files("diff_diff.guides").joinpath(filename).read_text(encoding="utf-8")
