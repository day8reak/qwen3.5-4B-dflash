# Reference contracts

The executable contracts are:

- `specs/mtp-contract.json` for official structure, shift, and acceptance;
- `targets/ascend310p/abi/runtime-v1.json` for backend tensor ABI;
- `docs/ACCURACY_CONTRACT.md` for numerical and promotion gates.

The CPU oracle uses the same interfaces as an internal target adapter.  The initial
implementation recomputes committed prefixes, so only committed tokens influence future
state.  Any incremental target implementation must additionally prove every acceptance
branch before replacing that oracle.
