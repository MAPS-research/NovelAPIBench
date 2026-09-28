"""Stage 4: quality and novelty filtering (paper Section 3.2, Appendix B.2 "Stage 4").

``c1`` reference validity (shared by all backbones), ``c2`` empirical novelty (per backbone,
local vLLM), ``c3`` solvability with the Full bundle (GPT-5-mini). Each check writes one record
per task under ``outputs/construction/stage4/``; ``construction.instances`` intersects them.
"""
