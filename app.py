import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService

# =========================
# 🔐 IBM API (Paste your key)
# =========================
API_KEY = "PASTE_YOUR_API_KEY_HERE"

# =========================
# 🟢 SYSTEM 1: QGA KNAPSACK
# =========================
def init_population(pop_size, n_items):
    return np.ones((pop_size, n_items, 2)) / np.sqrt(2)

def measure(q_pop):
    pop = []
    for individual in q_pop:
        bits = []
        for qbit in individual:
            prob = qbit[1]**2
            bits.append(1 if np.random.rand() < prob else 0)
        pop.append(bits)
    return np.array(pop)

def fitness(sol, weights, values, capacity):
    w = np.sum(sol * weights)
    v = np.sum(sol * values)
    return v if w <= capacity else 0

def update(q_pop, pop, best):
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
        fits = [fitness(p, weights, values, capacity) for p in pop]

        idx = np.argmax(fits)
        if fits[idx] > best_fit:
            best_fit = fits[idx]
            best = pop[idx]

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

def run_maxcut(mode):
    params = np.random.rand(6)

    qc = create_circuit(params)

    if mode == "Ideal Simulator":
        backend = AerSimulator()

    elif mode == "Noisy Simulator":
        service = QiskitRuntimeService(channel="ibm_quantum", token=API_KEY)
        real_backend = service.backend("ibm_brisbane")
        noise_model = NoiseModel.from_backend(real_backend)
        backend = AerSimulator(noise_model=noise_model)

    elif mode == "Real Quantum":
        service = QiskitRuntimeService(channel="ibm_quantum", token=API_KEY)
        backend = service.least_busy(simulator=False)

    tqc = transpile(qc, backend)
    job = backend.run(tqc, shots=1024)

    result = job.result()
    counts = result.get_counts()

    # Compute best score
    best_score = 0
    for bitstring in counts:
        score = maxcut_cost(bitstring)
        best_score = max(best_score, score)

    return counts, best_score

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
        ax.set_title("Convergence Graph")
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
        with st.spinner("Running Quantum Circuit..."):
            counts, score = run_maxcut(mode)

        st.success(f"Max-Cut Score: {score}")

        st.subheader("Counts")
        st.json(counts)

        # Plot
        fig, ax = plt.subplots()
        ax.bar(list(counts.keys())[:10], list(counts.values())[:10])
        plt.xticks(rotation=90)
        st.pyplot(fig)

        # Save JSON
        data = {
            "mode": mode,
            "score": score,
            "counts": counts
        }

        st.download_button(
            "Download JSON",
            json.dumps(data, indent=4),
            file_name="results.json"
        )
