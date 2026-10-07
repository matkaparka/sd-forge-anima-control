"""LoRA dropdown helpers and the `<lora:...>` auto-append."""
import re

from lib_anima_control import ControlError

NONE = "None"


def list_loras(refresh: bool = False) -> list[str]:
    try:
        import networks

        if refresh:
            networks.list_available_networks()
        names = sorted(networks.available_networks.keys(), key=str.lower)
    except Exception:
        names = []
    return [NONE] + names


def guess_lora(names: list[str], words: tuple[str, ...]) -> str:
    """First LoRA whose name contains ALL of `words`. No guess beats a wrong one: a folder that also
    holds e.g. Krea 2's canny LoRA must not get it pre-selected for Anima."""
    for n in names:
        low = n.lower()
        if all(w in low for w in words):
            return n
    return NONE


def lora_file(name: str) -> str:
    """Absolute path of a LoRA known to Forge; raises with a message that says what to do."""
    import networks

    net = networks.available_networks.get(name)
    if net is None:
        raise ControlError(f'Control LoRA "{name}" was not found -- press the refresh button, or check that the file is in your LoRA folder')
    return str(net.filename)


def has_tag(prompt: str, name: str) -> bool:
    """True if the prompt already contains `<lora:name:...>` or `<lora:name>`."""
    return re.search(r"<lora:" + re.escape(name) + r"[:>]", prompt, re.IGNORECASE) is not None


def append_tag(prompt: str, name: str, weight: float) -> str:
    """Add `<lora:name:weight>` unless the user already wrote their own tag for this LoRA (their weight wins)."""
    if has_tag(prompt, name):
        return prompt
    tag = f"<lora:{name}:{weight:g}>"
    return f"{prompt} {tag}" if prompt.strip() else tag


def inject_lora(p, name: str, weight: float) -> None:
    """Positive prompts and (if Hires. fix is on) Hires prompts. Negative prompts are left alone."""
    p.all_prompts = [append_tag(x, name, weight) for x in p.all_prompts]
    if getattr(p, "enable_hr", False) and getattr(p, "all_hr_prompts", None):
        p.all_hr_prompts = [append_tag(x, name, weight) for x in p.all_hr_prompts]
