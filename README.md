# 🎯 Multi-Fidelity Bayesian Optimization (Cost-Aware)

**MultiFidelityBO** is a simulation framework for **adaptive black-box optimization** under **heterogeneous noise and cost constraints**. It compares multiple sampling strategies—**Greedy**, **Gaussian Process-based Acquisition**, and **Thompson Sampling**—to identify high-value points on the unit hypersphere within a fixed query budget.

---

## 🧠 Key Features

* 🧪 **Black-box Function Evaluation**: Unknown function defined on a high-dimensional sphere with multiple Gaussian bumps and global structure.
* 🎲 **Deterministic Noise Injection**: Reproducible noise added per approximation method.
* 💸 **Per-Evaluator Cost Model**: Each approximation method has a distinct cost and noise level.
* 📈 **Offline Evaluation Benchmark**: Compares total best performance under equal budget constraints.
* 🧠 **Gaussian Process Regression**: Used for acquisition-guided exploration (UCB & Thompson Sampling).

---

## 📊 Methods Compared

| Method        | Description                                                                            |
| ------------- | -------------------------------------------------------------------------------------- |
| **GreedySeq** | Uses the cheapest approximator to greedily select points with highest estimated value. |
| **GP**        | Bayesian Optimization via Upper Confidence Bound (UCB) acquisition.                    |
| **Thompson**  | Samples from GP posterior to perform Thompson Sampling over approximators.             |

---

## 🛠️ How It Works

1. **Synthetic Function**: Defined on the unit hypersphere with multiple Gaussian-like peaks.
2. **Approximators**: Each approximator `j` has a `(cost_j, noise_j)` pair, simulating real-world tradeoffs.
3. **Sampling Strategies**: Use each method to query the function while staying within a fixed total cost budget.
4. **Metrics Logged**: For each trial and budget, logs the best observed function value (`BestG`).

---

## 🧪 How to Run

```bash
# Clone and install dependencies
pip install numpy pandas matplotlib scikit-learn tqdm

# Run the full benchmark
python MultiFidelityBO.py
```

Results will be printed to the console with summaries of:

* Mean and standard deviation of best value found (`BestG`)
* Comparison across methods (`GreedySeq`, `GP`, `Thompson`)

---

## 📁 Structure

All logic is implemented in a **single file**: `MultiFidelityBO.py`

### Key Functions / Classes

* `true_g(x)` – Ground-truth black-box function
* `Approximation` – Evaluator class modeling cost and noise
* `greedy_sequence(...)` – Greedy budgeted selection
* `multi_shot_gp(...)` – GP-UCB acquisition under cost
* `multi_shot_thompson(...)` – GP Thompson Sampling under cost

---

## 📌 Use Cases

* Benchmarking **adaptive query selection** under constraints
* Simulating **multi-fidelity optimization** or **multi-armed bandit** settings
* Demonstrating **Bayesian optimization** with per-evaluation cost tradeoffs
---

## 📬 Author

**Kalyan Cherukuri**
For questions or collaboration: \[[kcherukuri@imsa.edu](mailto:kcherukuri@imsa.edu)]
