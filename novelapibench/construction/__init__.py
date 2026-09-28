"""Benchmark construction (paper Section 3.2, Appendix B.2).

The four stages, each driven by ``scripts/build_benchmark.py``:

    Stage 1  stage1/    API discovery: introspect v_old and v_new of each library, diff the public
                        API surfaces, filter, and freeze one candidate pool (union over the
                        backbones' release boundaries; set-size-dependent caps applied once).
    Stage 2  stage2/    Knowledge extraction: S_name, S_param, E, M and C for each candidate.
    Stage 3  stage3/    Task generation: three tasks per bundle, each with an execute-then-assert
                        harness built from reference executions.
    Stage 4  stage4/    Filtering: C1 reference validity, C2 empirical novelty (per backbone),
                        C3 solvability with the Full bundle.

``call_records`` derives the reference call records of the call-record check and re-runs C3
under it; ``instances`` assembles the per-backbone instances, the exclusion list and the RQ3
split. Everything is written under ``outputs/construction/`` (``paths.construction_dir()``);
nothing here writes into ``data/``.
"""
