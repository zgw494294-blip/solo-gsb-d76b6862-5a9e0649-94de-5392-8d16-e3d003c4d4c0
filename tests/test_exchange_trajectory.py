"""Tests for the exchange-trajectory endpoint, including a brute-force cross-check."""

from __future__ import annotations

import random
from collections import Counter
from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def parse_fraction(text: str) -> Fraction:
    n, d = text.split("/")
    return Fraction(int(n), int(d))


def fmt(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def brute_force(payload: dict):
    """Enumerate every (molecule site, bath site) exchange sequence explicitly.

    Returns (probability, conditional): the exact joint probability of all
    observations and, per step 0..steps, the conditional molecule heavy-count
    distribution given all observations (None when the joint probability is
    zero).
    """
    n, m = payload["n"], payload["m"]
    steps = payload["steps"]
    bounds = {
        obs["step"]: (obs["lower"], obs["upper"])
        for obs in payload.get("observations", [])
    }
    molecule = (1,) * payload["h"] + (0,) * (n - payload["h"])
    bath = (1,) * payload["b"] + (0,) * (m - payload["b"])

    hits = 0
    counts = [Counter() for _ in range(steps + 1)]
    path = []

    def walk(t: int, molecule: tuple, bath: tuple) -> None:
        nonlocal hits
        heavy = sum(molecule)
        bound = bounds.get(t)
        if bound is not None and not bound[0] <= heavy <= bound[1]:
            return
        path.append(heavy)
        if t == steps:
            hits += 1
            for step, value in enumerate(path):
                counts[step][value] += 1
        else:
            for i in range(n):
                for j in range(m):
                    next_molecule = list(molecule)
                    next_bath = list(bath)
                    next_molecule[i], next_bath[j] = next_bath[j], next_molecule[i]
                    walk(t + 1, tuple(next_molecule), tuple(next_bath))
        path.pop()

    walk(0, molecule, bath)
    probability = Fraction(hits, (n * m) ** steps)
    if hits == 0:
        return probability, None
    conditional = [
        {heavy: Fraction(count, hits) for heavy, count in sorted(counter.items())}
        for counter in counts
    ]
    return probability, conditional


def assert_matches_brute_force(payload: dict) -> None:
    resp = client.post("/v1/exchange-trajectory", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    probability, conditional = brute_force(payload)
    assert body["probability"] == fmt(probability)
    if conditional is None:
        assert "distributions" not in body
        return
    assert len(body["distributions"]) == payload["steps"] + 1
    for t, entry in enumerate(body["distributions"]):
        assert entry["step"] == t
        got = {
            item["heavy_count"]: parse_fraction(item["probability"])
            for item in entry["counts"]
        }
        assert got == conditional[t]
        # Non-zero counts only, ascending, reduced, and normalized.
        heavy = [item["heavy_count"] for item in entry["counts"]]
        assert heavy == sorted(heavy)
        for item in entry["counts"]:
            value = parse_fraction(item["probability"])
            assert value > 0
            assert item["probability"] == fmt(value)
        assert sum(got.values(), Fraction(0)) == 1


def basic_payload():
    return {
        "n": 1,
        "m": 2,
        "h": 0,
        "b": 1,
        "steps": 2,
        "observations": [{"step": 2, "lower": 0, "upper": 0}],
    }


def test_conditioned_chain_hand_computed():
    # Molecule: 1 light site; bath: 1 heavy + 1 light. Step 1 swaps the heavy
    # in with probability 1/2; the observation pins the count after step 2 to
    # 0, which has probability 1/2*1 + 1/2*1/2 = 3/4. Conditioned on that,
    # the count after step 1 is 0 with (1/2*1/2)/(3/4) = 1/3 and 1 with 2/3.
    resp = client.post("/v1/exchange-trajectory", json=basic_payload())
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "3/4",
        "distributions": [
            {"step": 0, "counts": [{"heavy_count": 0, "probability": "1/1"}]},
            {
                "step": 1,
                "counts": [
                    {"heavy_count": 0, "probability": "1/3"},
                    {"heavy_count": 1, "probability": "2/3"},
                ],
            },
            {"step": 2, "counts": [{"heavy_count": 0, "probability": "1/1"}]},
        ],
    }


def test_no_observations_gives_unconditional_chain():
    # n = m = 1, the heavy isotope starts on the molecule: the only swap
    # moves it to the bath with probability 1.
    resp = client.post(
        "/v1/exchange-trajectory",
        json={"n": 1, "m": 1, "h": 1, "b": 0, "steps": 1},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "distributions": [
            {"step": 0, "counts": [{"heavy_count": 1, "probability": "1/1"}]},
            {"step": 1, "counts": [{"heavy_count": 0, "probability": "1/1"}]},
        ],
    }


def test_zero_steps_returns_initial_state():
    resp = client.post(
        "/v1/exchange-trajectory",
        json={"n": 3, "m": 2, "h": 2, "b": 1, "steps": 0, "observations": []},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "distributions": [
            {"step": 0, "counts": [{"heavy_count": 2, "probability": "1/1"}]}
        ],
    }


def test_observation_at_step_zero():
    payload = {"n": 2, "m": 2, "h": 1, "b": 0, "steps": 1}
    # The initial count is 1: observing [1, 1] at step 0 keeps everything.
    resp = client.post(
        "/v1/exchange-trajectory",
        json={**payload, "observations": [{"step": 0, "lower": 1, "upper": 1}]},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "distributions": [
            {"step": 0, "counts": [{"heavy_count": 1, "probability": "1/1"}]},
            {
                "step": 1,
                "counts": [
                    {"heavy_count": 0, "probability": "1/2"},
                    {"heavy_count": 1, "probability": "1/2"},
                ],
            },
        ],
    }
    # Observing [0, 0] at step 0 contradicts the initial state.
    resp = client.post(
        "/v1/exchange-trajectory",
        json={**payload, "observations": [{"step": 0, "lower": 0, "upper": 0}]},
    )
    assert resp.status_code == 200
    assert resp.json() == {"probability": "0/1"}
    assert "distributions" not in resp.json()


def test_impossible_observation_omits_distributions():
    # No heavy isotope anywhere: the count stays 0 forever.
    resp = client.post(
        "/v1/exchange-trajectory",
        json={
            "n": 1,
            "m": 1,
            "h": 0,
            "b": 0,
            "steps": 1,
            "observations": [{"step": 1, "lower": 1, "upper": 1}],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"probability": "0/1"}
    assert "distributions" not in resp.json()


def test_multiple_observations_match_brute_force():
    payload = {
        "n": 2,
        "m": 3,
        "h": 1,
        "b": 2,
        "steps": 4,
        "observations": [
            {"step": 1, "lower": 0, "upper": 1},
            {"step": 3, "lower": 1, "upper": 2},
            {"step": 4, "lower": 0, "upper": 0},
        ],
    }
    assert_matches_brute_force(payload)


def test_randomized_matches_brute_force():
    rng = random.Random(20260927)
    for _ in range(30):
        n = rng.randint(1, 3)
        m = rng.randint(1, 3)
        steps = rng.randint(0, 4)
        obs_count = rng.randint(0, 3)
        observations = [
            {
                "step": step,
                "lower": (lower := rng.randint(0, n)),
                "upper": rng.randint(lower, n),
            }
            for step in rng.sample(range(steps + 1), min(obs_count, steps + 1))
        ]
        payload = {
            "n": n,
            "m": m,
            "h": rng.randint(0, n),
            "b": rng.randint(0, m),
            "steps": steps,
            "observations": observations,
        }
        assert_matches_brute_force(payload)


def test_supports_max_scale():
    # Largest admissible input: n = m = 40, steps = 80, 8 observations.
    payload = {
        "n": 40,
        "m": 40,
        "h": 40,
        "b": 0,
        "steps": 80,
        "observations": [
            {"step": step, "lower": 0, "upper": 40} for step in range(0, 70, 10)
        ]
        + [{"step": 80, "lower": 0, "upper": 40}],
    }
    resp = client.post("/v1/exchange-trajectory", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert parse_fraction(body["probability"]) == 1  # the bounds exclude nothing
    assert len(body["distributions"]) == 81
    for entry in body["distributions"]:
        assert entry["counts"][0]["heavy_count"] >= 0
        total = sum(
            (parse_fraction(item["probability"]) for item in entry["counts"]),
            Fraction(0),
        )
        assert total == 1


@pytest.mark.parametrize(
    "mutation",
    [
        # n out of range / wrong type / missing
        lambda p: p.update(n=0),
        lambda p: p.update(n=41),
        lambda p: p.update(n=1.0),
        lambda p: p.update(n=1.5),
        lambda p: p.update(n="1"),
        lambda p: p.update(n=True),
        lambda p: p.pop("n"),
        # m out of range / wrong type / missing
        lambda p: p.update(m=0),
        lambda p: p.update(m=41),
        lambda p: p.update(m="2"),
        lambda p: p.update(m=False),
        lambda p: p.pop("m"),
        # h out of range / wrong type / missing
        lambda p: p.update(h=-1),
        lambda p: p.update(h=2),  # h > n
        lambda p: p.update(h=1.5),
        lambda p: p.update(h="0"),
        lambda p: p.pop("h"),
        # b out of range / wrong type / missing
        lambda p: p.update(b=-1),
        lambda p: p.update(b=3),  # b > m
        lambda p: p.update(b=0.5),
        lambda p: p.pop("b"),
        # steps out of range / wrong type / missing
        lambda p: p.update(steps=-1),
        lambda p: p.update(steps=81),
        lambda p: p.update(steps=2.0),
        lambda p: p.update(steps="2"),
        lambda p: p.pop("steps"),
        # observations: too many / wrong type
        lambda p: p.update(
            observations=[{"step": 0, "lower": 0, "upper": 0}] * 9
        ),
        lambda p: p.update(observations=1),
        lambda p: p.update(observations=[None]),
        # observation fields out of range / wrong type
        lambda p: p.update(observations=[{"step": -1, "lower": 0, "upper": 0}]),
        lambda p: p.update(observations=[{"step": 3, "lower": 0, "upper": 0}]),
        lambda p: p.update(observations=[{"step": 2.5, "lower": 0, "upper": 0}]),
        lambda p: p.update(observations=[{"step": 2, "lower": -1, "upper": 0}]),
        lambda p: p.update(observations=[{"step": 2, "lower": 0, "upper": 2}]),
        lambda p: p.update(observations=[{"step": 2, "lower": 1, "upper": 0}]),
        lambda p: p.update(observations=[{"step": 2, "lower": "0", "upper": 0}]),
        lambda p: p.update(observations=[{"step": 2, "lower": 0, "upper": True}]),
        lambda p: p.update(observations=[{"step": 2, "lower": 0}]),
        lambda p: p.update(
            observations=[{"step": 2, "lower": 0, "upper": 0, "extra": 1}]
        ),
        # duplicate observation steps
        lambda p: p.update(
            observations=[
                {"step": 2, "lower": 0, "upper": 0},
                {"step": 2, "lower": 0, "upper": 1},
            ]
        ),
        # unknown top-level field
        lambda p: p.update(extra=1),
    ],
)
def test_invalid_payloads_return_422(mutation):
    payload = basic_payload()
    mutation(payload)
    resp = client.post("/v1/exchange-trajectory", json=payload)
    assert resp.status_code == 422


def test_malformed_json_returns_422():
    resp = client.post(
        "/v1/exchange-trajectory",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
