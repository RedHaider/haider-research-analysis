# Multi-objective battery policy training

[Efficient Multi-objective Reinforcement Learning via Multiple-Gradient Descent with Iteratively Discovered Weight-Vector Sets](https://zhan-lab-ai.github.io/publications/), co-authored by Huixin Zhan, motivates reconciling gradients from separate objectives. This independent experiment learns preference-conditioned action values and compares equal averaging of the two training losses with a gradient mix calculated by minimizing the norm of the weighted objective gradients before clipping. It also measures hold and time-of-use heuristics. It evaluates charge spending and battery throughput at seven preferences with the same benchmark dynamics.

The aligned study uses simulated demand seasons. The companion uses measured 10-minute demand from the [UCI Appliances Energy Prediction dataset](https://archive.ics.uci.edu/dataset/374/appliances-energy-prediction). It only uses complete days and splits them in chronological order. The battery and tariff are simulated, while appliance and lighting inputs come from real measurements. The model learns from sampled simulated transitions on training days. Its test-day policy observes only current state and time. It performs no physical device control. The end-of-day charge balance adjusts the cost proxy using a fixed reset price.

    python training-logs/full-scale/multiobjective/train.py
    python training-logs/full-scale/multiobjective/train.py --real

Install pandas, PyTorch and NumPy. The companion expects datasets/appliances-energy.csv. Saved artifacts include chronological split IDs, fitted networks, validation histories and per-day held-out tradeoffs.
