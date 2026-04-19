import random

ADJECTIVES = [
    "swift", "bold", "keen", "calm", "bright", "sharp", "vivid", "steady",
    "fierce", "noble", "agile", "lucid", "rapid", "silent", "cosmic",
    "astral", "polar", "solar", "lunar", "crystal", "quantum", "neural",
    "primal", "sonic", "radiant", "golden", "silver", "iron", "amber",
    "crimson", "azure", "obsidian", "phantom", "blazing", "frozen",
]

NOUNS = [
    "falcon", "wolf", "hawk", "lynx", "otter", "raven", "viper", "fox",
    "crane", "tiger", "cobra", "eagle", "shark", "puma", "elk", "owl",
    "mantis", "phoenix", "hydra", "sphinx", "atlas", "nova", "pulse",
    "spark", "orbit", "flux", "prism", "forge", "nexus", "cipher",
    "vector", "vertex", "helix", "quasar", "photon", "beacon",
]

_used_names: set[str] = set()


def generate_agent_name() -> str:
    """Generate a unique two-word agent name."""
    for _ in range(100):
        name = f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"
        if name not in _used_names:
            _used_names.add(name)
            return name
    # Fallback with number
    name = f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}-{random.randint(10, 99)}"
    _used_names.add(name)
    return name


def load_used_names(names: set[str]) -> None:
    """Restore used names from DB on startup."""
    _used_names.update(names)
