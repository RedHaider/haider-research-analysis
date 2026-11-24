# CDC Diabetes Federated Edge Benchmark

Real-data companion to the professor-aligned federated edge-healthcare project
for Radu Marculescu. It uses 253,680 public CDC survey records from the UCI
Diabetes Health Indicators dataset and forms eight naturally non-IID clients
from age and sex strata.

The benchmark compares float32 FedAvg with a dropout-tolerant method using
proximal local updates, clipped aggregation and int8 client deltas. Five seeds
measure global and worst-client AUROC, calibration, dropout resilience and actual
update bytes. Raw data stay on drive D.

This is a portfolio systems benchmark, not a clinical deployment or a
reproduction of Professor Marculescu's published system.

## Validated results

Across five seeds, int8 updates plus simulated 20% client dropout reduced update
traffic by **88.4%** (24,640 to 2,855 mean bytes). Global AUROC was effectively
retained at **0.8069 versus 0.8066**, and mean-client AUROC changed from 0.7966
to 0.7969. Brier score improved from **0.1899 to 0.1856** in all five seeds
(paired reduction 0.0043; 95% CI 0.0028 to 0.0057). The defensible claim is
communication and calibration efficiency, not higher discrimination.

