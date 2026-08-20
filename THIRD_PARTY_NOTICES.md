# Third-party source notices

- The feature-enabled Qwen3.5 modeling sibling and portable configuration are
  derived from Hugging Face Transformers v5.14.1 and retain their Apache-2.0
  headers. Only DFlash instrumentation and portable import paths are changed.
  Runtime source identities are listed in `SOURCE_LOCK.json`; the full license
  text is in `LICENSES/Apache-2.0.txt`.
- DFlash behavior follows the public z-lab DFlash software implementation,
  which is MIT licensed. The separate
  `z-lab/Qwen3.5-4B-DFlash` checkpoint model card declares Apache-2.0. Software
  and checkpoint licenses are tracked independently. The DFlash software
  license text is in `LICENSES/MIT-DFlash.txt`; the Apache-2.0 text also covers
  the declared checkpoint license.
- No third-party model weights are redistributed in this repository.
