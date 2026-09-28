# Licenses and provenance

## Our materials

The code, task descriptions, context code, reference completions and test harnesses, the
generated usage examples and mechanism descriptions, call records, the Stage-4 filter records
(`benchmark/filters/`, including the sampled completions of the six backbones and of GPT-5-mini),
splits and manifests are released under Apache-2.0 (`../LICENSE`).

## Third-party content in the data

Every bundle, task and candidate-pool record names its source library in the `library` field; the
exact release it was taken from is `new_version` in `benchmark/manifest.json` (and in
`configs/libraries/<library>.yaml`). The following fields quote the library, its documentation,
or papers it cites:

| Where | What is quoted |
|---|---|
| `benchmark/bundles.jsonl` `implementation` (C) | the API's source code (and same-module helpers), with docstrings removed |
| `benchmark/bundles.jsonl` `s_name`, `s_param` | the API's name and signature; parameter descriptions taken or paraphrased from its documentation |
| `benchmark/bundles.jsonl` `examples` (E) | generated examples, some of which follow examples in the documentation |
| `benchmark/bundles.jsonl` `mechanism.references` | titles of papers cited by the documentation |
| `pool/candidates.jsonl` | the API's signature, docstring and source file path within the package |

This content remains under the license of the library it comes from. The license texts and
copyright notices shipped with each library at the release we used are in
`licenses/<library>/`:

| Domain | Library | Release | License | Files |
|---|---|---|---|---|
| agent tooling | fastmcp | 3.2.4 | Apache-2.0 | `licenses/fastmcp/` |
| agent tooling | langgraph | 1.1.10 | MIT | `licenses/langgraph/` |
| data science | numpy | 2.4.4 | BSD-3-Clause | `licenses/numpy/` |
| data science | pandas | 3.0.2 | BSD-3-Clause | `licenses/pandas/` |
| data science | scipy | 1.17.1 | BSD-3-Clause | `licenses/scipy/` |
| AI for science | ase | 3.28.0 | LGPL-2.1-or-later | `licenses/ase/` (upstream notice + LGPL-2.1 text) |
| AI for science | astropy | 7.2.0 | BSD-3-Clause | `licenses/astropy/` |
| AI for science | deepchem | 2.8.0 | MIT | `licenses/deepchem/` |
| AI for science | MDAnalysis | 2.10.0 | LGPL-3.0-or-later | `licenses/MDAnalysis/` |
| AI for science | pymatgen | 2026.3.23 | MIT | `licenses/pymatgen/` |
| AI for science | rdkit | 2026.3.1 | BSD-3-Clause | `licenses/rdkit/` (`license.txt`; `LICENSE.md` covers the PyPI packaging) |
| AI for science | scanpy | 1.11.5 | BSD-3-Clause | `licenses/scanpy/` |
| deep learning | diffusers | 0.38.0 | Apache-2.0 | `licenses/diffusers/` |
| deep learning | torch | 2.11.0 | BSD-3-Clause | `licenses/torch/` (with NOTICE) |
| deep learning | transformers | 5.7.0 | Apache-2.0 | `licenses/transformers/` |
| software engineering | django | 5.2.13 | BSD-3-Clause | `licenses/django/` |
| software engineering | fastapi | 0.136.1 | MIT | `licenses/fastapi/` |
| software engineering | flask | 3.1.3 | BSD-3-Clause | `licenses/flask/` |
| software engineering | httpx | 0.28.1 | BSD-3-Clause | `licenses/httpx/` |
| software engineering | pydantic | 2.13.3 | MIT | `licenses/pydantic/` |
| software engineering | sqlalchemy | 2.0.49 | MIT | `licenses/sqlalchemy/` |

Two libraries carry copyleft terms and are listed under `copyleft_libraries` in
`benchmark/manifest.json`: records whose `library` is `MDAnalysis` (LGPL-3.0-or-later) or `ase`
(LGPL-2.1-or-later) quote source from those projects under their licenses. The rest of the data
is unaffected.

## Third-party code

`../third_party/memit/` is upstream MEMIT (MIT; `LICENSE` and `NOTICE` there, with the changes we
made). Python dependencies are installed from PyPI (`../requirements.txt`, `../envs/locks/`) and
are not redistributed.
