# Vendored sources

- `native/motion_preprocess/`, `dls/core/`: from `vmsx-dls-vms-2026.10.09` (archive in `~/Downloads`). The first
  commit holds them unchanged; later commits are the changes for this test (`motion_preprocess.patch` collects them).
  `dls/core/` is only used by the library's `python_reference` test.
- `third_party/nvdsinfer_yolo_attr/`: see its `SOURCE.md`.
- `third_party/nlohmann_json/`: see its `SOURCE.md`.
