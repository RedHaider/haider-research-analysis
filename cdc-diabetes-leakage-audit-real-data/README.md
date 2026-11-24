# CDC Diabetes Representation Leakage Audit

Real-data companion to the professor-aligned medical feature-leakage audit for
Su-In Lee. A disease-trained neural representation is learned from CDC Diabetes
Health Indicators without directly using sex, then audited with linear and
nonlinear protected-attribute probes.

Three repeated splits compare the original representation, single-direction
projection and group-conditioned moment matching. Disease AUROC, strongest probe
AUROC and calibration are reported. The project is an audit benchmark, not a
clinical fairness claim or a reproduction of Professor Lee's method.

## Validated result

Across three 25,000-record samples, the original representation's strongest sex
probe AUROC was 0.6109. Linear projection reduced it only slightly to 0.6065
while preserving disease AUROC (0.81129 versus 0.81129). Moment matching made
the linear probe appear random (0.5030) but allowed a nonlinear probe to recover
sex perfectly (1.0000); quantile matching also increased nonlinear leakage to
0.6901. The defensible result is a multi-probe failure audit—not a claim that
these simple mitigations solved representation leakage.

