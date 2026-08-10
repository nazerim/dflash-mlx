# Copyright 2026 jundot
# Licensed under the Apache License, Version 2.0 - see LICENSE file
"""Target model modules dflash-mlx supplies when mlx-lm has no implementation.

Each module registers itself into ``sys.modules["mlx_lm.models.<name>"]``
so ``mlx_lm.utils.load`` can resolve the checkpoint's model_type, and
yields automatically once upstream mlx-lm ships the family.
"""
