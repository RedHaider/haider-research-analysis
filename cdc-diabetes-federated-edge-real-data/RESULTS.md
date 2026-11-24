# Validated federated results

| Metric | Float32 FedAvg | Proposed int8/dropout method | Change |
| --- | ---: | ---: | ---: |
| Communication bytes | 24,640 | 2,855 | **-88.4%** |
| Global AUROC | 0.8069 | 0.8066 | -0.0004 |
| Mean-client AUROC | 0.7966 | 0.7969 | +0.0003 |
| Worst-client AUROC | 0.7112 | 0.7110 | -0.0001 |
| Brier score | 0.1899 | 0.1856 | **-0.0043** |
| ECE | 0.2804 | 0.2704 | **-0.0101** |

The Brier improvement was positive in all five seeds, with a paired 95% CI of
0.0028 to 0.0057. AUROC differences were negligible; the result supports a
communication-efficiency and calibration claim rather than improved discrimination.

Protocol: 253,680 CDC survey records, 21 features, eight age-sex clients, five
seeds, 18 rounds, float32 FedAvg baseline, int8 client deltas, proximal local
updates, clipped aggregation and simulated 20% dropout. Clients and network
conditions are simulated; this is not a clinical or professor-system claim.

