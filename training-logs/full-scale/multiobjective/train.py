"""Learn preference-conditioned battery dispatch in a chronological load simulator."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
PER_DAY = 144
BATTERY = 1.20
FLOW = 0.16
EFFICIENCY = 0.92
PREFERENCES = (0.0, 0.15, 0.35, 0.55, 0.75, 0.9, 1.0)
TIME = np.arange(PER_DAY) / 6.0
PRICE = np.where(
    (TIME >= 17) & (TIME < 21), 0.44, np.where((TIME < 6) | (TIME >= 23), 0.10, 0.23)
)


def loads(real, seed):
    if real:
        table = pd.read_csv(
            ROOT / "datasets/appliances-energy.csv", parse_dates=["date"]
        ).sort_values("date")
        assert len(table) == 19735 and table.date.is_unique
        table["day"] = table.date.dt.date
        formatted = []
        for date, observed in table.groupby("day"):
            at = observed.date.dt.hour * 6 + observed.date.dt.minute // 10
            if len(at) == PER_DAY and np.array_equal(at.to_numpy(), np.arange(PER_DAY)):
                demand = (observed["Appliances"] + observed["lights"]).to_numpy(
                    dtype=np.float32
                ) / 1000.0
                formatted.append((str(date), demand))
        keys, days = zip(*formatted)
        return np.stack(days), list(keys)
    rng = np.random.default_rng(seed)
    periods = TIME[None]
    n = 520
    morning = np.exp(-0.5 * ((periods - rng.normal(7.0, 0.7, (n, 1))) / 1.4) ** 2)
    dinner = np.exp(-0.5 * ((periods - rng.normal(18.4, 0.9, (n, 1))) / 1.9) ** 2)
    season = np.sin(np.arange(n) * 2 * np.pi / 150)[:, None]
    demand = 0.04 + 0.12 * morning + 0.2 * dinner + season * 0.025
    demand = np.clip(demand + rng.normal(0, 0.03, demand.shape), 0.015, 0.7)
    return demand.astype(np.float32), list(range(n))


def step(load, state, action, currency):
    charge = np.minimum(FLOW, (BATTERY - state) / EFFICIENCY) * (action == 2)
    discharge = np.minimum(np.minimum(FLOW, state * EFFICIENCY), load) * (action == 0)
    next_charge = (state + EFFICIENCY * charge - discharge / EFFICIENCY).clip(
        0, BATTERY
    )
    grid = load + charge - discharge
    objective = np.stack([grid * currency, charge + discharge], axis=-1)
    return next_charge, objective


def state_vector(soc, at, load, preference):
    return np.column_stack(
        [
            soc / BATTERY,
            load * 4,
            PRICE[at] * 3,
            np.sin(TIME[at] * np.pi / 12),
            np.cos(TIME[at] * np.pi / 12),
            at / (PER_DAY - 1),
            preference,
        ]
    ).astype(np.float32)


class ActionValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(7, 128), nn.LayerNorm(128), nn.SiLU())
        self.res = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
        self.out = nn.Linear(128, 6)

    def forward(self, x):
        h = self.block(x)
        return self.out(F.silu(h + self.res(h))).view(-1, 3, 2)


def transitions(curves, rng, size):
    day = rng.integers(len(curves), size=size)
    at = rng.integers(0, PER_DAY, size)
    after = np.minimum(at + 1, PER_DAY - 1)
    terminal = at == PER_DAY - 1
    tariff = PRICE[at]
    load = curves[day, at]
    preference = rng.uniform(0, 1, size)
    soc = rng.uniform(0, BATTERY, size)
    action = rng.integers(0, 3, size)
    charge, objective = step(load, soc, action, tariff)
    objective[terminal, 0] += (0.6 - charge[terminal]) * PRICE[0] / EFFICIENCY
    inputs = state_vector(soc, at, load, preference)
    following = state_vector(charge, after, curves[day, after], preference)
    weights = np.stack([preference, 0.20 * (1 - preference)], axis=1)
    return (
        inputs,
        action,
        objective.astype(np.float32),
        following,
        weights.astype(np.float32),
        terminal,
    )


def reconcile(losses, params):
    cost = torch.autograd.grad(losses[0], params, retain_graph=True)
    throughput = torch.autograd.grad(losses[1], params)
    difference = [a - b for a, b in zip(cost, throughput)]
    denominator = sum(d.square().sum() for d in difference).clamp_min(1e-8)
    mix = (
        -sum((b * d).sum() for b, d in zip(throughput, difference)) / denominator
    ).clamp(0.15, 0.85)
    for parameter, a, b in zip(params, cost, throughput):
        parameter.grad = mix * a + (1 - mix) * b


def rollout(curves, model, preference, fixed=None):
    n = len(curves)
    soc = np.full(n, 0.6, dtype=np.float32)
    total = np.zeros((n, 2))
    decision = np.zeros(3, dtype=int)
    with torch.inference_mode():
        for at in range(PER_DAY):
            load = curves[:, at]
            if model is not None:
                x = state_vector(soc, np.full(n, at), load, np.full(n, preference))
                q = model(torch.from_numpy(x)).numpy()
                action = (
                    q[:, :, 0] * preference + 0.20 * q[:, :, 1] * (1 - preference)
                ).argmin(1)
            else:
                action = np.full(
                    n,
                    1
                    if fixed == "hold"
                    else (2 if PRICE[at] < 0.12 else (0 if PRICE[at] > 0.38 else 1)),
                )
            decision += np.bincount(action, minlength=3)
            soc, objective = step(load, soc, action, PRICE[at])
            total += objective
    total[:, 0] += (0.6 - soc) * PRICE[0] / EFFICIENCY
    return total, decision


def evaluate(curves, model):
    records, values = {}, []
    for preference in PREFERENCES:
        observed, decisions = rollout(curves, model, preference)
        values.append(observed)
        records[str(preference)] = {
            "demand_charge_proxy": float(observed[:, 0].mean()),
            "battery_throughput_kwh": float(observed[:, 1].mean()),
            "actions": decisions.tolist(),
        }
    return records, np.stack(values)


def fit(train, val, mode, args, output):
    rng = np.random.default_rng(args.seed + 1)
    torch.manual_seed(args.seed)
    q, target = ActionValue(), ActionValue()
    target.load_state_dict(q.state_dict())
    params = tuple(q.parameters())
    opt = torch.optim.AdamW(params, lr=0.0008, weight_decay=0.001)
    model_file = output / f"{mode}.pt"
    history, best = [], float("inf")
    for step_id in range(1, args.updates + 1):
        x, action, observed, following, weight, terminal = [
            torch.from_numpy(a) for a in transitions(train, rng, 320)
        ]
        index = torch.arange(len(x))
        with torch.no_grad():
            q_next = q(following)
            best_action = (q_next * weight[:, None]).sum(2).argmin(1)
            expected = (
                observed
                + 0.98 * target(following)[index, best_action] * (~terminal)[:, None]
            )
        estimate = q(x)[index, action]
        losses = F.smooth_l1_loss(estimate, expected, reduction="none", beta=0.08).mean(
            0
        )
        opt.zero_grad(set_to_none=True)
        if mode == "adaptive_gradient":
            reconcile(losses, params)
        else:
            losses.mean().backward()
        torch.nn.utils.clip_grad_norm_(params, 2.0)
        opt.step()
        if step_id % 40 == 0:
            target.load_state_dict(q.state_dict())
        if step_id % 250 == 0:
            _, outcomes = evaluate(val, q)
            demo = []
            for preference, samples in zip(PREFERENCES, outcomes):
                demo.append(
                    (
                        samples[:, 0] * preference
                        + samples[:, 1] * 0.2 * (1 - preference)
                    ).mean()
                )
            v = float(np.mean(demo))
            history.append(
                {
                    "update": step_id,
                    "validation_preference_cost": v,
                    "losses": losses.detach().tolist(),
                }
            )
            if v < best:
                best = v
                torch.save(copy.deepcopy(q.state_dict()), model_file)
            print(mode, history[-1], flush=True)
    q.load_state_dict(torch.load(model_file, weights_only=True))
    return q, history


def main(args):
    torch.set_num_threads(1)
    demand, ids = loads(args.real, args.seed)
    a, b = int(0.62 * len(ids)), int(0.80 * len(ids))
    train, val, test = demand[:a], demand[a:b], demand[b:]
    output = ROOT / (
        "real-data-projects/zhan-battery-measured-load-policy"
        if args.real
        else "professor-aligned/zhan-multiobjective-dispatch"
    )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "splits.npz", train=ids[:a], validation=ids[a:b], test=ids[b:]
    )
    results = {
        "configuration": vars(args),
        "total_days": len(ids),
        "steps_per_day": PER_DAY,
        "methods": {},
        "rules": {},
    }
    for rule in ("hold", "tariff"):
        result, decisions = rollout(test, None, 0.5, fixed=rule)
        results["rules"][rule] = {
            "held_out_mean_objectives": result.mean(0).tolist(),
            "actions": decisions.tolist(),
        }
    for name in ("equal_weight", "adaptive_gradient"):
        model, history = fit(train, val, name, args, output)
        scores, values = evaluate(test, model)
        results["methods"][name] = {"held_out": scores, "validation": history}
        np.savez_compressed(
            output / f"{name}-held-out.npz", preferences=PREFERENCES, per_day=values
        )
    (output / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--updates", type=int, default=3000)
    main(parser.parse_args())
