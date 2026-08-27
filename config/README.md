# Canonical Weyl pipeline configurations

The final analysis has two method IDs and two prior sets.  Configuration names
always use the form

```text
pipeline2x2pt_weyl_<method>_<prior-set>.ini
```

| Method | Prior set | Configuration | Completed Nautilus chain |
|---|---|---|---|
| amplitude reparameterization | With Planck/background priors | `pipeline2x2pt_weyl_reparameterized_with_planck.ini` | `chain_reparameterized_with_planck_nautilus.txt` |
| amplitude reparameterization | No Planck/background priors | `pipeline2x2pt_weyl_reparameterized_no_planck.ini` | `chain_reparameterized_no_planck_nautilus.txt` |
| explicit early--late split at $z_\star$ | With Planck/background priors | `pipeline2x2pt_weyl_explicit_zstar_with_planck.ini` | `chain_explicit_zstar_with_planck_nautilus.txt` |
| explicit early--late split at $z_\star$ | No Planck/background priors | `pipeline2x2pt_weyl_explicit_zstar_no_planck.ini` | `chain_explicit_zstar_no_planck_nautilus.txt` |

The completed chains live in the root repository under `chain/`.  Run these
entry-point files from the `kcap/` directory so that KCAP's data and module
paths resolve correctly.

Both methods share:

- `shared/values_weyl_late_time_amplitudes.ini`;
- `shared/priors_weyl_with_planck.ini`;
- `shared/priors_weyl_no_planck.ini`.

The explicit-$z_\star$ No-Planck configuration includes the validated 300 s
gRPT numerical guard.  It does not add a cosmological prior.

## Terminology and legacy identifiers

Use **amplitude reparameterization** or the method ID `reparameterized` in
filenames, scripts, documentation, and paper text.  Do not use `hatted` as a
method name.

New runs write derived quantities under `REPARAMETERIZED_DERIVED`.  The four
completed chains were generated before this naming cleanup and therefore
still contain the prefix `HATTED_DERIVED`; the analysis scripts retain support
for that immutable historical chain schema.  It does not define the public
method name.
