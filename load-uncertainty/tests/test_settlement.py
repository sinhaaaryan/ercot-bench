import numpy as np
import pytest

from ercot_uncertainty.sim.settlement import Fleet, commitment_fraction, settle, simulate_day

rng = np.random.default_rng(0)
DA = rng.uniform(20, 80, 24)
RT = np.repeat(DA, 4) + rng.normal(0, 10, 96)


def test_do_nothing_is_zero():
    r = settle(np.zeros(24), np.zeros(96), DA, RT)
    assert r["total"] == 0 and r["da_rev"] == 0 and r["rt_rev"] == 0


def test_deliver_exactly_da_award_earns_da_revenue_only():
    award = np.zeros(24)
    award[[17, 18]] = 10.0
    r = settle(award, np.repeat(award, 4), DA, RT)
    assert r["rt_rev"] == pytest.approx(0)
    assert r["total"] == pytest.approx(10 * DA[17] + 10 * DA[18])


def test_under_delivery_buys_back_at_rt():
    award = np.zeros(24)
    award[17] = 10.0
    r = settle(award, np.zeros(96), DA, RT)
    assert r["rt_rev"] == pytest.approx(-10 * RT[68:72].mean())


def test_energy_never_exceeds_battery():
    spiky = RT.copy()
    spiky[60:80] = 5000
    f = Fleet()
    for frac in (0.0, 0.5, 1.0):
        r = simulate_day(frac, DA, spiky, f)
        assert r["discharged_mwh"] <= f.energy_mwh + 1e-6


def test_holding_back_pays_on_spike_days_and_costs_on_calm_days():
    calm = np.repeat(DA, 4)  # RT == DA: nothing to gain in RT
    spiky = calm.copy()
    spiky[70:74] = 3000
    assert simulate_day(1.0, DA, calm)["total"] >= simulate_day(0.3, DA, calm)["total"]
    assert simulate_day(0.3, DA, spiky)["total"] > simulate_day(1.0, DA, spiky)["total"]


def test_commitment_fraction_direction():
    assert commitment_fraction(2000, 1000) < commitment_fraction(1000, 1000) < commitment_fraction(500, 1000)
    assert 0.2 <= commitment_fraction(1e9, 1000) and commitment_fraction(1, 1000) <= 1.0


def test_rt_rule_is_causal():
    """A one-interval spike can't be captured: the rule only reacts to the previous interval's price."""
    flat = np.repeat(np.full(24, 30.0), 4)
    spike = flat.copy()
    spike[70] = 5000
    r = simulate_day(0.0, np.full(24, 30.0), spike)
    # with same-interval foresight it would earn ~20 MW * 0.25 h * $5000 = $25k; causally it reacts one
    # interval late and sells at the ordinary $30
    assert r["rt_rev"] < 1_000


def test_no_trading_below_marginal_cost():
    cheap_day = np.full(24, 5.0)  # peak price below charging cost / RTE + degradation
    r = simulate_day(1.0, cheap_day, np.repeat(cheap_day, 4))
    assert r["total"] == 0 and r["discharged_mwh"] == 0
