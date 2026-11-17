"""Lab 1 · Quantum coins & entangled pairs: auto-checks. Usage: from qff_utsa_grader.lab1 import check, scoreboard"""
from math import sqrt, sin
import numpy as np
from IPython import get_ipython
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

TASKS = {
    "A1": "Make a quantum coin", "A2": "Flip it 1,000 times", "A3": "Flip it twice",
    "B1": "Find the angle for P(1) = 1/4", "B2": "A coin with any bias", "B3": "Reveal the hidden phase",
    "C1": "Bit-order warm-up", "C2": "Entangle two coins", "C3": "Sample the pair",
    "C4": "Correlations with the Estimator", "C5": "All four Bell states", "C6": "Quantum vs classical",
    "D1": "Bonus: GHZ of any size", "D2": "Bonus: a noisy chip",
}
SCORE = {}
s2 = 1 / sqrt(2)


def _v(name):
    ns = get_ipython().user_ns
    if name not in ns:
        raise AssertionError(f"I can't find `{name}`. Run the cell above first.")
    return ns[name]


def _same(qc, target):
    assert isinstance(qc, QuantumCircuit), "Expected a QuantumCircuit."
    return Statevector(qc.remove_final_measurements(inplace=False)).equiv(Statevector(target))


def _close(x, y, tol=1e-6):
    return x is not None and abs(float(x) - y) < tol


def _A1():
    assert _same(_v("coin"), [s2, s2]), "The state isn't |+> yet. Which gate makes a superposition?"


def _A2():
    c = _v("coin_counts")
    assert isinstance(c, dict), "coin_counts should be a dict of counts."
    assert sum(c.values()) == 1000, f"Expected 1000 shots, got {sum(c.values())}."
    assert set(c) <= {"0", "1"}, "Only '0' and '1' should appear."
    assert all(400 <= c.get(k, 0) <= 600 for k in "01"), "That doesn't look like a fair coin. Is A1 correct?"


def _A3():
    d = _v("double")
    assert d.count_ops().get("h", 0) == 2, "Use exactly two H gates."
    assert _same(d, [1, 0]), "Two H gates should bring the coin back to |0>."


def _B1():
    th = _v("theta_B1")
    assert th is not None, "Set theta_B1 to a number."
    assert _close(sin(th / 2) ** 2, 0.25), f"With θ = {float(th):.4f}, P(1) = {sin(th / 2) ** 2:.4f}, not 0.25."


def _B2():
    make_coin = _v("make_coin")
    for p in [0, 0.1, 0.25, 0.5, 0.9, 1]:
        got = Statevector(make_coin(p)).probabilities()[1]
        assert _close(got, p), f"make_coin({p}) gives P(1) = {got:.4f}."


def _B3():
    reveal, plus, minus = _v("reveal"), _v("plus_coin"), _v("minus_coin")
    assert len(reveal(plus).data) == len(plus.data) + 1, "Add exactly one gate."
    assert _same(reveal(plus), [1, 0]), "reveal(plus_coin) should be |0>."
    assert _same(reveal(minus), [0, 1]), "reveal(minus_coin) should be |1>."


def _C1():
    assert _same(_v("always01"), [0, 1, 0, 0]), "Not '01' yet. Remember: q0 is the RIGHTMOST character."


def _C2():
    assert _same(_v("bell"), [s2, 0, 0, s2]), "Not (|00> + |11>)/√2 yet. Superposition first, then CNOT."


def _C3():
    c, p_q0, p_same = _v("bell_counts"), _v("p_q0"), _v("p_same")
    assert isinstance(c, dict) and sum(c.values()) == 1000, "bell_counts should hold 1000 shots."
    assert set(c) <= {"00", "11"}, "A Bell pair should only give '00' and '11'. Is C2 correct?"
    assert p_q0 is not None and 0.4 <= p_q0 <= 0.6, "p_q0 should be close to 0.5: each coin alone is fair."
    assert _close(p_same, 1.0), "p_same should be exactly 1: the coins always agree."


def _C4():
    evs = _v("bell_evs")
    assert evs is not None and len(evs) == 3, "bell_evs should have three numbers: <ZI>, <IZ>, <ZZ>."
    assert np.allclose(evs, [0, 0, 1], atol=1e-6), f"Expected three specific values, got {np.round(evs, 3)}. Check the order ZI, IZ, ZZ."


def _C5():
    assert _same(_v("phi_minus"), [s2, 0, 0, -s2]), "phi_minus isn't (|00> - |11>)/√2 yet."
    assert _same(_v("psi_plus"), [0, s2, s2, 0]), "psi_plus isn't (|01> + |10>)/√2 yet."
    assert _same(_v("psi_minus"), [0, s2, -s2, 0]), "psi_minus isn't (|01> - |10>)/√2 yet. Mind the order |q1 q0>."


def _C6():
    assert _close(_v("zz_classical"), 1.0), "zz_classical isn't right yet. Do the classical copies agree?"
    assert _close(_v("xx_classical"), 0.0), "xx_classical isn't right yet. Average over c00 and c11."
    assert _close(_v("xx_bell"), 1.0), "xx_bell isn't right yet. Is C2 correct?"


def _D1():
    make_ghz = _v("make_ghz")
    for n in (2, 3, 5):
        target = np.zeros(2 ** n)
        target[0] = target[-1] = s2
        assert _same(make_ghz(n), target), f"make_ghz({n}) isn't a GHZ state yet."


def _D2():
    noisy = _v("noisy_counts")
    expected = (noisy.get("01", 0) + noisy.get("10", 0)) / sum(noisy.values())
    assert _close(_v("wrong_fraction"), expected), "Count the '01' and '10' shots, divided by the total."
    assert set(_v("bell_counts") or {}) <= {"00", "11"}, "Finish C2 first, so the noisy run uses a real Bell pair."


_TESTS = {k: globals()["_" + k] for k in TASKS}


def check(task):
    try:
        _TESTS[task]()
        SCORE[task] = True
        print(f"{task} · {TASKS[task]}: correct! 🎉")
    except AssertionError as e:
        SCORE[task] = False
        print(f"{task} · {TASKS[task]}: not yet. {e}")
    except Exception as e:
        SCORE[task] = False
        print(f"{task} · {TASKS[task]}: not yet. Your code raised {type(e).__name__}: {e}")


def scoreboard():
    name = get_ipython().user_ns.get("YOUR_NAME", "Your Name")
    name = name if name != "Your Name" else "(put your name in the Setup cell!)"
    core = [t for t in TASKS if not t.startswith("D")]
    bonus = [t for t in TASKS if t.startswith("D")]
    print(f"Lab 1 · Quantum coins & entangled pairs · {name}\n")
    for t in TASKS:
        print(f"  {'✅' if SCORE.get(t) else '⬜'}  {t}  {TASKS[t]}")
    done = sum(bool(SCORE.get(t)) for t in core)
    print(f"\nCore: {done}/{len(core)}   Bonus: {sum(bool(SCORE.get(t)) for t in bonus)}/{len(bonus)}")
    print("🎉 Lab 1 complete!" if done == len(core) else "Keep going, or bring your questions to the walkthrough.")
