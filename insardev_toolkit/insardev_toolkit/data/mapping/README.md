# Coefficient tables

Coefficients of the troposphere model (utils_tropo), read once at import by `utils_tropo._mapping_tables()`: the
spherical harmonic coefficients of VMF3 and the coefficients of GMF. The decimal strings are the sources' own.

| File | Source | Layout | Bytes | SHA-256 |
|---|---|---|---|---|
| `vmf3_grid.txt` | `vmf3_grid.m`, https://vmf.geo.tuwien.ac.at/codes/vmf3_grid.m | 8 tables `[anm_bh]` ... `[bnm_cw]`, 91 rows x 5 values (A0 A1 B1 A2 B2) | 73678 | `a047d5e0a1255b03cf66d17a0f3af1fa8c9fff5036eb2aac24da5419acdaec5e` |
| `gmf.txt` | `gmf.f`, https://vmf.geo.tuwien.ac.at/codes/gmf.f | 8 tables `[ah_mean]` ... `[bw_amp]`, 11 rows x 5 values (55 terms, degree 0 to 9), `%e` | 6340 | `6b779b234be2fe79d6a9c58709deef17a99359168155f81c7cf5e2129d504d35` |

Source checksums: `vmf3_grid.m` (fetched 2026-10-22, 94058 bytes) has SHA-256 `8da9a4d296bd1160953e56fd0f2e9d20b9bb8c3f30b41ce81c9629d4a6eb2206`. `gmf.f` (fetched 2026-10-22, 10810 bytes) has SHA-256 `1d73070ea3bf4943c87252389abda64f8e59e0b0c4682517dbde01554cdc4f5a`.

License and terms: see `LICENSE`, which holds references and quotes of the sources only.

## Conversion

`mapping_tables.py` writes both files from the two sources:

    python3.13 mapping_tables.py vmf3_grid.m gmf.f

It parses the matrices of vmf3_grid.m and the DATA statements of gmf.f, and keeps the source's decimal strings (gmf values as %e). The fetched vmf3_grid.m (SHA-256 above) parses to the same 8 tables as the shipped vmf3_grid.txt (token lists equal, checked 2026-10-22). gmf.f was not parsed from its published bytes.
