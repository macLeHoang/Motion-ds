"""Which attributes are meaningful per class and child slot.

AIO_t13_11_11.pt stores all-ones attr_mask / child_attr_mask, so the engine returns every attribute for every class and
slot. Child lists come from data_balanced_pose_erd_datafix_v2.yaml `children`; parent groups from the attribute names.
"""

from __future__ import annotations

import numpy as np

CHILD_ATTRS = {
    "head": ["has_helm", "has_cap_hat", "has_glasses", "has_mask"],
    "torso": ["has_jacket_coat", "has_tshirt_polo", "has_vest", "has_dress", "is_short_sleeve", "is_long_sleeve"],
    "legs": ["has_long_pants", "has_shorts", "has_skirt"],
    "foot_left": ["has_boots", "has_shoes", "has_sandals"],
    "foot_right": ["has_boots", "has_shoes", "has_sandals"],
}
PARENT_ATTRS = {
    "person": ("standing", "sitting", "laying", "bending", "view_", "is_occluded", "gender_", "age_", "hair_"),
    "two_wheeler": ("type_", "make_"),
    "light_vehicle": ("type_", "make_"),
    "heavy_machinery": ("type_", "make_"),
}


def display_masks(meta: dict):
    """(nc, na) parent and (nb, ka) child attribute masks; `meta["names"]` keyed by int or str class id."""
    attrs, cattrs, names = meta["attributes"], meta["child_attr_names"], meta["names"]
    nc = len(names)
    am = np.zeros((nc, len(attrs)), bool)
    for c in range(nc):
        groups = PARENT_ATTRS.get(names[c] if c in names else names[str(c)])
        if groups:
            am[c] = [a.startswith(groups) for a in attrs]
    cam = np.array([[a in CHILD_ATTRS.get(cn, cattrs) for a in cattrs] for cn in meta["child_names"]])
    return am & np.asarray(meta["attr_mask"], bool), cam & np.asarray(meta["child_attr_mask"], bool)
