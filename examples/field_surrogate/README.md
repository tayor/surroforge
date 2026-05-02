# Field Surrogate Example

Field-output surrogates are planned for the next release track. The recommended shape is:

1. Run a simulator that writes field artifacts such as VTU files.
2. Extract fields onto a consistent mesh or grid.
3. Compress fields with an autoencoder or PCA-style basis.
4. Train a parameter-to-latent surrogate.
5. Decode latent predictions back to fields and report calibration error.

The scalar-output APIs in v0.1 are already structured so this workflow can reuse the same design-space, simulator, store, active-learning, and report layers.
