"""Lab 2 · Find the target with Grover's algorithm: auto-checks. Usage: from qff_utsa_grader.lab2 import check, scoreboard"""
from math import asin, sqrt, sin, pi
import numpy as np
from IPython import get_ipython
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

TASKS = {
    "A1": "Equal superposition of 2 qubits", "A2": "Oracle that marks |11>", "A3": "The 2-qubit diffuser",
    "A4": "One Grover iteration finds |11>",
    "B1": "An oracle for any target", "B2": "A diffuser for any size", "B3": "Grover for any target",
    "C1": "The best number of iterations", "C2": "Success vs iterations",
    "D1": "Transpile for a real chip", "D2": "Run it on the chip",
    "E1": "Bonus: two marked items", "E2": "Bonus: Qiskit's grover_operator",
}
SCORE = {}


def _v(name):
    ns = get_ipython().user_ns
    if name not in ns:
        raise AssertionError(f"I can't find `{name}`. Run the cell above first.")
    return ns[name]


def _circ(qc, n=None):
    assert isinstance(qc, QuantumCircuit), "Expected a QuantumCircuit."
    if n is not None:
        assert qc.num_qubits == n, f"Expected {n} qubits, got {qc.num_qubits}."
    return qc.remove_final_measurements(inplace=False)


def _uniform(n):
    return np.ones(2 ** n) / sqrt(2 ** n)


def _mark_matrix(ws, n):
    m = np.eye(2 ** n)
    for w in ws:
        m[int(w, 2), int(w, 2)] = -1
    return m


def _diffuser_matrix(n):
    s = _uniform(n)
    return 2 * np.outer(s, s) - np.eye(2 ** n)


def _p(qc, ws):
    probs = Statevector(_circ(qc)).probabilities_dict()
    return sum(probs.get(w, 0) for w in ws)


def _theory(n, k, m=1):
    th = asin(sqrt(m / 2 ** n))
    return sin((2 * k + 1) * th) ** 2


def _A1():
    assert Statevector(_circ(_v("start2"), 2)).equiv(Statevector(_uniform(2))), "Not an equal superposition yet. Which gate do you need on each qubit?"


def _A2():
    assert Operator(_circ(_v("oracle_11"), 2)).equiv(Operator(_mark_matrix(["11"], 2))), "This doesn't flip the sign of |11> only. Which 2-qubit gate is diag(1, 1, 1, -1)?"


def _A3():
    assert Operator(_circ(_v("diffuser2"), 2)).equiv(Operator(_diffuser_matrix(2))), "Not 2|s><s| - I yet (up to a global sign). Recipe: H, X, CZ, X, H."


def _A4():
    g = _circ(_v("grover2"), 2)
    p = _p(g, ["11"])
    assert abs(p - 1) < 1e-6, f"P(11) = {p:.3f}. Start with A1, then oracle, then diffuser: one iteration of each."


def _B1():
    make_oracle = _v("make_oracle")
    for w in ["01", "10", "101", "000", "110", "1011"]:
        O = make_oracle(w)
        assert Operator(_circ(O, len(w))).equiv(Operator(_mark_matrix([w], len(w)))), f"make_oracle('{w}') doesn't mark |{w}> only. Check which qubits need X (rightmost character = q0)."


def _B2():
    make_diffuser = _v("make_diffuser")
    for n in [2, 3, 4]:
        assert Operator(_circ(make_diffuser(n), n)).equiv(Operator(_diffuser_matrix(n))), f"make_diffuser({n}) isn't 2|s><s| - I yet (up to a global sign)."


def _B3():
    grover = _v("grover")
    for w, k in [("101", 2), ("11", 1), ("0110", 3)]:
        p, t = _p(grover(w, k), [w]), _theory(len(w), k)
        assert abs(p - t) < 1e-6, f"grover('{w}', {k}) gives P = {p:.3f}; theory says {t:.3f}."


def _C1():
    best_k = _v("best_k")
    for n in range(2, 11):
        th = asin(1 / sqrt(2 ** n))
        first_turn = int(pi / (2 * th))          # the first peak lies within the first half-turn
        ks = [_theory(n, k) for k in range(0, first_turn + 1)]
        k_best = int(np.argmax(ks))
        got = best_k(n)
        assert got == k_best, f"best_k({n}) returned {got}; the most likely k is a different number. Use θ = asin(1/√N) and k ≈ π/(4θ) − 1/2."


def _C2():
    ps = list(_v("p_success"))
    assert len(ps) == 7, "p_success should hold 7 numbers, for k = 0, 1, …, 6."
    for k, p in enumerate(ps):
        t = _theory(3, k)
        assert abs(float(p) - t) < 0.06, f"For k = {k} you have {float(p):.3f}; theory says {t:.3f}."


def _D1():
    isa, backend, cz = _v("isa"), _v("backend"), _v("cz_count")
    ops = set(isa.count_ops())
    allowed = set(backend.target.operation_names) | {"barrier", "measure"}
    assert ops <= allowed, f"These gates aren't native to {backend.name}: {sorted(ops - allowed)}. Did you transpile?"
    assert cz == isa.count_ops().get("cz", 0), "cz_count should be the number of CZ gates in your transpiled circuit."
    assert isa.count_ops().get("measure", 0) == 3, "Measure all 3 qubits before transpiling (measure_all)."


def _D2():
    counts, succ = _v("hw_counts"), _v("hw_success")
    assert isinstance(counts, dict) and sum(counts.values()) > 0, "hw_counts should be the counts dict from your job."
    shots = sum(counts.values())
    assert succ is not None and abs(float(succ) - counts.get("101", 0) / shots) < 1e-9, "hw_success = counts of '101' divided by the number of shots."
    assert max(counts, key=counts.get) == "101", "The most frequent result should be '101'. Is your circuit grover('101', 2)?"


def _E1():
    O = _circ(_v("oracle_two"), 3)
    assert Operator(O).equiv(Operator(_mark_matrix(["011", "110"], 3))), "oracle_two should flip the signs of |011> and |110> only. Hint: two oracles in a row."
    g = _v("grover_two")
    p = _p(g, ["011", "110"])
    assert abs(p - 1) < 1e-6, f"P(011 or 110) = {p:.3f}; with 2 marked out of 8, one iteration should give exactly 1."


def _E2():
    qc = _circ(_v("lib_circuit"), 3)
    p = _p(qc, ["101"])
    assert abs(p - _theory(3, 2)) < 1e-6, f"P(101) = {p:.3f}; expected {_theory(3, 2):.3f} for 2 iterations."


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
    core = [t for t in TASKS if not t.startswith("E")]
    bonus = [t for t in TASKS if t.startswith("E")]
    print(f"Lab 2 · Find the target with Grover's algorithm · {name}\n")
    for t in TASKS:
        print(f"  {'✅' if SCORE.get(t) else '⬜'}  {t}  {TASKS[t]}")
    done = sum(bool(SCORE.get(t)) for t in core)
    print(f"\nCore: {done}/{len(core)}   Bonus: {sum(bool(SCORE.get(t)) for t in bonus)}/{len(bonus)}")
    print("🎉 Lab 2 complete!" if done == len(core) else "Keep going, or bring your questions to the walkthrough.")
