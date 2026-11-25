# Certified least-squares reference

The offline [Misra1a test](../tests/nist.rs) adapts the 14 observations from
[NIST StRD Misra1a](https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Misra1a.dat)
to the public `LeastSquares` residual/Jacobian interface. Tests do not download
data or need a NIST/MATLAB installation.

Model: `y=b1*(1-exp(-b2*x))`. The published reference parameters are
`b1=238.94212918`, `b2=0.00055015643181`; certified residual sum of squares is
`0.12455138894`. The solver objective is half that sum.
Both published starts `(500,0.0001)` and `(250,0.0005)` are tested.

This is a certified NIST case, not a claim that the constructed fits in the
benchmark catalog are NIST cases. Additional NIST datasets and exact COCO/BBOB
adapters remain separate expansion work. CEC/CUTEst imports require independent
provenance/licensing review.
