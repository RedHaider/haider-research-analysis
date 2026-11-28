# Learned admission for two-tier cache replay

Alex K. Jones co-authored [FusedCache](https://doi.org/10.1109/TMSCS.2016.2536020), which motivates studying the interaction of two cache levels. This independent cache simulator compares unconditional LRU demotion, admitting objects after repeated access and a learned admission policy. It uses an exclusive two-level cache and models bytes transferred. It does not reproduce the racetrack device or measure hardware energy.

The aligned study fits a classifier on one synthetic request stream and evaluates on a second stream with an independent key population. The companion uses the first 600,000 usable-prefix records from each of two [published anonymized Twitter trace samples](https://github.com/cacheMon/cache_dataset#twitter-twemcache-request-traces). The classifier trains on early evictions from cluster 17. Later requests from cluster 17 select the policy threshold, and cluster 52 supplies the held-out replay. Features use observed request size, count, recency and the age of the victim object. Future reuse generates labels within each training block. The archive's future-access field is unused. Object sizes are clipped to 16–65,536 bytes before replay.

    python training-logs/full-scale/cache-replay/train.py
    python training-logs/full-scale/cache-replay/train.py --real

Install NumPy, scikit-learn, joblib and threadpoolctl. The companion also requires zstandard and the saved 16 MiB compressed prefixes for clusters 17 and 52. Output includes the fitted classifier, its inference lookup, thresholds and byte-transfer metrics across cache sizes.
