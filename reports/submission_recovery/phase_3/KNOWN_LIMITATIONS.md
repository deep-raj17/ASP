# Phase 3 Known Limitations

- Detector calibration remains missing; production calibrated inference is
  blocked until an authorized calibration run completes.
- The simple neural baseline needs a dedicated frozen runner before its Phase 4
  pilot; changing the main model ad hoc is prohibited.
- Full reliability-aware CV has been tested synthetically but not yet on newly
  extracted validation features.
- The historical checkpoint remains provisional and underfit.
- The shared Python environment has seven dependency conflicts.
- Dataset archive lineage remains blocked despite manifest-level integrity.
- Three seeds provide limited resolution for seed-level significance tests.
- The Phase 4 full-model pilot is expected to consume one to two GPU-hours.
- No protected test evaluation has occurred.
