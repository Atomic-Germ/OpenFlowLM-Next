# Binary Inventory

`inventory.py` creates a reproducible JSON inventory of precompiled artifacts.
By default it scans the shipping binary surfaces, `src/lib` and `src/xclbins`.

```bash
python utilities/binary-inventory/inventory.py
```

The default output is `docs/precompiled_artifacts.json`. It includes:

- path, size, SHA-256, format, and ownership classification;
  `ownership` distinguishes `closed_npu_kernel` (pending replacement) from
  `open_npu_kernel` (kernels we build and ship ourselves, under a family
  directory's `npu_matmul_f32/`);
- ELF SONAME, NEEDED entries, RUNPATH, and dynamic-symbol counts;
- XCLBIN UUID, format version, sections, and BUILD_METADATA availability;
- duplicate payload groups;
- XCLBIN model bundles and byte-identical bundle groups;
- model tags and family names from `src/model_list.json`.

Use `--all` to scan the whole repository, including archived examples. Build,
virtual-environment, dependency-cache, and Git metadata directories remain
excluded. A complete workspace inventory can be regenerated with:

```bash
python utilities/binary-inventory/inventory.py \
  --all --output docs/precompiled_artifacts_all.json
```

Required tools are detected at runtime. Missing optional tools reduce metadata
but do not prevent hashing and classification. XCLBIN inspection uses
`xclbinutil` from the installed XRT toolchain.
