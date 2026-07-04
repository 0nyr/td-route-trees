"""td-route-trees: TD move-evaluation structures on an exact NDCPWLF engine.

Study repo for the Stream-4 protocol (P4.0-P4.5) of the KAYROS programme:
naive recomposition baseline, Visser & Spliet 2020 ready-time-function tree,
Blauth et al. 2024 LCA-BST, benchmarked on the canonical MAMUT TD families.
Every duration is anchored on the `mamut_routing_lib.td` Duration checker.
"""

from td_route_trees import _core
from td_route_trees.io import load_instance, to_core

__version__ = _core.__version__

__all__ = ["_core", "load_instance", "to_core", "__version__"]
