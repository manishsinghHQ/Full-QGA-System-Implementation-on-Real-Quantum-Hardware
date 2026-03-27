import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService,Sampler


# =========================
# 🔐 Secure API Key
# =========================
API_KEY = st.secrets.get("IBM_API_KEY", None)

# =========================
# 🟢 SYSTEM 1: QGA KNAPSACK
# =========================
def init_population(pop_size, n_items):
    return np.ones((pop_size, n_items, 2)) / np.sqrt(2)

def measure(q_pop):
    probs = q_pop[:, :, 1]**2
    return (np.random.rand(*probs.shape) < probs).astype(int)

def fitness_vectorized(pop, weights, values, capacity):
    weights_sum = pop @ weights
    values_sum = pop @ values
    return np.where(weights_sum <= capacity, values_sum, 0)

def update(q_pop, pop, best):
    if best is None:
        return q_pop
    delta = 0.01
    for i in range(len(q_pop)):
        for j in range(len(q_pop[i])):
            if pop[i][j] != best[j]:
                theta = delta if best[j] == 1 else -delta
                a, b = q_pop[i][j]
                q_pop[i][j][0] = a*np.cos(theta) - b*np.sin(theta)
                q_pop[i][j][1] = a*np.sin(theta) + b*np.cos(theta)
    return q_pop

def run_qga(n_items=100, generations=50):
    weights = np.random.randint(1, 20, n_items)
    values = np.random.randint(10, 100, n_items)
    capacity = int(np.sum(weights) * 0.4)

    q_pop = init_population(20, n_items)
    best, best_fit = None, 0
    history = []

    for _ in range(generations):
        pop = measure(q_pop)
        fits = fitness_vectorized(pop, weights, values, capacity)

        idx = np.argmax(fits)

        if best is None or fits[idx] > best_fit:
            best_fit = fits[idx]
            best = pop[idx]

        if best is not None:
            q_pop = update(q_pop, pop, best)

        history.append(best_fit)

    return best_fit, history

# =========================
# 🔵 SYSTEM 2: MAX-CUT PQC
# =========================
edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)]

def create_circuit(params, n=6):
    qc = QuantumCircuit(n)

    for i in range(n):
        qc.h(i)

    for i in range(n):
        qc.ry(params[i], i)

    for i in range(n-1):
        qc.cx(i, i+1)

    qc.measure_all()
    return qc

def maxcut_cost(bitstring):
    return sum(1 for i, j in edges if bitstring[i] != bitstring[j])

def random_maxcut():
    best = 0
    for _ in range(500):
        sol = np.random.randint(0, 2, 6)
        score = sum(1 for i, j in edges if sol[i] != sol[j])
        best = max(best, score)
    return best

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

def run_maxcut(mode):
    params = np.random.rand(6)
    qc = create_circuit(params)

    if mode == "Ideal Simulator":
        backend = AerSimulator()
        tqc = transpile(qc, backend)
        job = backend.run(tqc, shots=1024)
        result = job.result()
        counts = result.get_counts()

    else:
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=API_KEY
        )

        if mode == "Noisy Simulator":
            backend = service.backend("ibm_brisbane")

        elif mode == "Real Quantum":
            backends = service.backends(simulator=False, operational=True)
            backend = min(backends, key=lambda b: b.status().pending_jobs)
            st.warning("⚠️ Real quantum jobs may take time (queue).")

        # ✅ Use Sampler instead of backend.run()
        sampler = Sampler(backend=backend)

        job = sampler.run([qc])
        result = job.result()

        quasi_dist = result.quasi_dists[0]

        # Convert quasi distribution → counts
        counts = {format(k, "06b"): int(v * 1024) for k, v in quasi_dist.items()}

    best_bitstring = max(counts, key=lambda x: maxcut_cost(x))
    best_score = maxcut_cost(best_bitstring)

    return counts, best_score, best_bitstring
# =========================
# 🎨 STREAMLIT UI
# =========================
st.set_page_config(page_title="QGA Quantum App", layout="wide")

st.title("🚀 Quantum Genetic Algorithm Dashboard")

option = st.sidebar.selectbox(
    "Select System",
    ["Knapsack QGA", "Max-Cut PQC"]
)

# =========================
# 🟢 KNAPSACK UI
# =========================
if option == "Knapsack QGA":
    st.header("🟢 Knapsack QGA (100 Items)")

    if st.button("Run QGA"):
        with st.spinner("Running QGA..."):
            best_fit, history = run_qga()

        st.success(f"Best Fitness: {best_fit}")

        fig, ax = plt.subplots()
        ax.plot(history)
        ax.set_xlabel("Generations")
        ax.set_ylabel("Best Fitness")
        ax.grid(True)
        st.pyplot(fig)

# =========================
# 🔵 MAX-CUT UI
# =========================
elif option == "Max-Cut PQC":
    st.header("🔵 Max-Cut Quantum Optimization")

    mode = st.selectbox(
        "Select Mode",
        ["Ideal Simulator", "Noisy Simulator", "Real Quantum"]
    )

    if st.button("Run Max-Cut"):
        if mode != "Ideal Simulator" and API_KEY is None:
            st.error("⚠️ Please add IBM API key in secrets.toml")
        else:
            with st.spinner("Running Quantum Circuit..."):
                counts, score, bitstring = run_maxcut(mode)

            st.success(f"Max-Cut Score: {score}")
            st.write("Best Bitstring:", bitstring)

            classical_score = random_maxcut()
            st.write("Classical Random Score:", classical_score)

            st.subheader("Counts")
            st.json(counts)

            fig, ax = plt.subplots()
            ax.bar(list(counts.keys())[:10], list(counts.values())[:10])
            plt.xticks(rotation=90)
            st.pyplot(fig)

            data = {
                "mode": mode,
                "score": score,
                "best_bitstring": bitstring,
                "counts": counts
            }

            st.download_button(
                "Download JSON",
                json.dumps(data, indent=4),
                file_name="results.json"
            )
