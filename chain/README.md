# Final chain catalogue

Large chain and Nautilus checkpoint files are local scientific artefacts and
are intentionally excluded from Git.  The four final chains use the naming
scheme

```text
chain_<method>_<prior-set>_<sampler>.txt
```

| Method ID | Prior set | Final chain | Size (bytes) |
|---|---|---|---:|
| `reparameterized` | `with_planck` | `chain_reparameterized_with_planck_nautilus.txt` | 49,084,086 |
| `reparameterized` | `no_planck` | `chain_reparameterized_no_planck_nautilus.txt` | 45,070,075 |
| `explicit_zstar` | `with_planck` | `chain_explicit_zstar_with_planck_nautilus.txt` | 54,746,076 |
| `explicit_zstar` | `no_planck` | `chain_explicit_zstar_no_planck_nautilus.txt` | 48,904,681 |

Each chain has a same-basename `.sampler_status.hdf5` checkpoint beside it.
The text-chain hashes are recorded in `manifest.sha256` and can be verified
from this directory with:

```bash
sha256sum -c manifest.sha256
```

The paper-facing pair is the two `explicit_zstar` chains.  The two
`reparameterized` chains are retained as an independent implementation and
closure check.

Files containing `pilot`, `smoke`, `stalled`, or old non-canonical names are
diagnostic artefacts, not final chains.
