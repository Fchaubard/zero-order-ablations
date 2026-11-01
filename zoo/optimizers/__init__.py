"""Registry: optimizer name -> class.  Names used everywhere (CLI, wandb, paper)."""
from .base import BaseOptimizer, per_sample_loss  # noqa: F401

OPTIMIZERS = {}


def register(name):
    def deco(cls):
        cls.name = name
        OPTIMIZERS[name] = cls
        return cls
    return deco


def get_optimizer(name, model, lr, task_kind, n_classes, **kw) -> BaseOptimizer:
    # import lazily so a broken module does not break the others
    import importlib
    for mod in ("backprop", "spsa", "random_search", "node_perturbation", "forward_forward"):
        try:
            importlib.import_module(f"zoo.optimizers.{mod}")
        except ImportError as e:  # pragma: no cover
            if name in _MODULE_OF.get(mod, ()):
                raise
    return OPTIMIZERS[name](model, lr, task_kind, n_classes, **kw)


_MODULE_OF = {
    "backprop": ("bp",),
    "spsa": ("spsa",),
    "random_search": ("rs",),
    "node_perturbation": ("np", "anp", "inp", "danp"),
    "forward_forward": ("ff",),
}
ALL_OPTIMIZERS = ["bp", "spsa", "rs", "np", "anp", "inp", "danp", "ff"]
