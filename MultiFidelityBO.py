import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.exceptions import ConvergenceWarning
import warnings
from tqdm import tqdm

def gp_predict(gp, X):
    mu, std = gp.predict(X, return_std=True)
    return mu, np.maximum(std, 1e-8)

def deterministic_noise(x, j, noise_level):
    x_rounded = np.round(x, 6).flatten()
    seed_input = np.concatenate([x_rounded, [j]])
    key = int(abs(hash(tuple(seed_input))) % (2**32))
    rng = np.random.RandomState(key)
    return rng.randn() * noise_level

def sample_on_sphere(n, num_samples):
    X = np.random.normal(size=(num_samples, n))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def true_g(x):
    n = x.shape[1]
    num_points = x.shape[0]

    local_rng = np.random.RandomState(123)
    num_centers = max(5, min(10, n // 10))
    centers = local_rng.normal(size=(num_centers, n))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    heights = np.linspace(50.0, 10.0, num_centers)
    widths = np.linspace(0.6, 0.2, num_centers)

    result = np.zeros(num_points)
    for i in range(num_centers):
        dots = np.dot(x, centers[i])
        distances = np.arccos(np.clip(dots, -1, 1))
        result += heights[i] * np.exp(-distances**2 / widths[i]**2)

    global_component = 2.0 * np.sin(np.sum(x[:, :min(5, n)], axis=1))
    result += global_component
    return result

class Approximation:
    def __init__(self, cost, noise_level):
        self.cost = cost
        self.noise_level = noise_level

    def evaluate(self, x, j):
        y_true = true_g(x)
        noise = np.array([deterministic_noise(x[i:i+1], j, self.noise_level) for i in range(len(x))])
        return y_true + noise

def greedy_sequence(apps, pool, budgets, trials, lam=1.0):
    costs = np.array([ap.cost for ap in apps])
    rec = []

    for B in budgets:
        best_overall = -np.inf
        remaining_budget = B
        used_indices = set()

        cheapest_j = np.argmin(costs)
        subset_size = min(8, len(pool))
        eval_pool = pool[:subset_size]

        while remaining_budget >= costs[cheapest_j] and len(used_indices) < len(eval_pool):
            available_indices = [i for i in range(len(eval_pool)) if i not in used_indices]
            if not available_indices:
                break

            remaining_budget -= costs[cheapest_j]
            available_points = eval_pool[available_indices]
            approx_values = apps[cheapest_j].evaluate(available_points, cheapest_j)

            max_idx = np.argmax(approx_values)
            best_idx = available_indices[max_idx]
            observed_val = approx_values[max_idx]
            best_overall = max(best_overall, observed_val)
            used_indices.add(best_idx)

        rec.append((len(apps), B, 'GreedySeq', best_overall))
    return rec

def create_gp():
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2))
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        alpha=1e-6
    )

def multi_shot_gp(apps, pool, budgets, trials, lam=1):
    m, costs = len(apps), np.array([ap.cost for ap in apps])
    rec = []

    for B in tqdm(budgets, desc='GP budgets'):
        for trial in tqdm(range(trials), desc='GP trials', leave=False):
            np.random.seed(trial)
            R = B
            data = {'X': [], 'y': [], 'j': []}
            best = -np.inf
            available = np.ones(len(pool), bool)

            if R >= costs.min():
                cheapest_idx = np.argmin(costs)
                n_warmup = min(5, len(pool), R // costs[cheapest_idx])
                warmup_indices = np.random.choice(len(pool), n_warmup, replace=False)

                for idx in warmup_indices:
                    if R >= costs[cheapest_idx] and available[idx]:
                        R -= costs[cheapest_idx]
                        x = pool[idx:idx+1]
                        y_obs = apps[cheapest_idx].evaluate(x, cheapest_idx)
                        data['X'].append(x.flatten())
                        data['y'].append(y_obs[0])
                        data['j'].append(cheapest_idx)
                        best = max(best, y_obs[0])
                        available[idx] = False

            iteration = 0
            while R >= costs.min() and iteration < 200:
                iteration += 1

                if len(data['X']) >= 3:

                    gp = create_gp()
                    X_train = np.array(data['X'])
                    y_train = np.array(data['y'])

                    noise_levels = np.array([apps[j].noise_level for j in data['j']])

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', ConvergenceWarning)
                        try:

                            gp.alpha = noise_levels**2
                            gp.fit(X_train, y_train)
                            gp_success = True
                        except:
                            gp_success = False

                    if gp_success:

                        best_acq = -np.inf
                        best_j, best_idx = None, None

                        available_indices = np.where(available)[0]
                        if len(available_indices) == 0:
                            break

                        X_pred = pool[available_indices]

                        mu, std = gp_predict(gp, X_pred)

                        for j in range(m):
                            if costs[j] <= R:

                                sigma_j = apps[j].noise_level
                                total_std = np.sqrt(std**2 + sigma_j**2)

                                acq = mu + lam * total_std

                                if len(acq) > 0:
                                    max_idx = np.argmax(acq)
                                    if acq[max_idx] > best_acq:
                                        best_acq = acq[max_idx]
                                        best_j = j
                                        best_idx = available_indices[max_idx]

                        if best_j is not None:

                            R -= apps[best_j].cost
                            x = pool[best_idx:best_idx+1]
                            y_obs = apps[best_j].evaluate(x, best_j)
                            data['X'].append(x.flatten())
                            data['y'].append(y_obs[0])
                            data['j'].append(best_j)
                            best = max(best, y_obs[0])
                            available[best_idx] = False
                            continue

                affordable = [j for j in range(m) if costs[j] <= R]
                if not affordable:
                    break

                available_indices = np.where(available)[0]
                if len(available_indices) == 0:
                    break

                j_star = np.random.choice(affordable)
                i_star = np.random.choice(available_indices)

                R -= apps[j_star].cost
                x = pool[i_star:i_star+1]
                y_obs = apps[j_star].evaluate(x, j_star)

                data['X'].append(x.flatten())
                data['y'].append(y_obs[0])
                data['j'].append(j_star)
                best = max(best, y_obs[0])
                available[i_star] = False

            rec.append((m, B, 'GP', best))
    return rec

def multi_shot_thompson(apps, pool, budgets, trials, lam=1):
    m, costs = len(apps), np.array([ap.cost for ap in apps])
    rec = []

    for B in tqdm(budgets, desc='TS budgets'):
        for trial in tqdm(range(trials), desc='TS trials', leave=False):
            np.random.seed(trial + 1000)
            R = B
            data = {'X': [], 'y': [], 'j': []}
            best = -np.inf
            available = np.ones(len(pool), bool)

            if R >= costs.min():
                cheapest_idx = np.argmin(costs)
                n_warmup = min(5, len(pool), R // costs[cheapest_idx])
                warmup_indices = np.random.choice(len(pool), n_warmup, replace=False)

                for idx in warmup_indices:
                    if R >= costs[cheapest_idx] and available[idx]:
                        R -= costs[cheapest_idx]
                        x = pool[idx:idx+1]
                        y_obs = apps[cheapest_idx].evaluate(x, cheapest_idx)
                        data['X'].append(x.flatten())
                        data['y'].append(y_obs[0])
                        data['j'].append(cheapest_idx)
                        best = max(best, y_obs[0])
                        available[idx] = False

            iteration = 0
            while R >= costs.min() and iteration < 200:
                iteration += 1

                if len(data['X']) >= 3:
                    gp = create_gp()
                    X_train = np.array(data['X'])
                    y_train = np.array(data['y'])

                    noise_levels = np.array([apps[j].noise_level for j in data['j']])

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', ConvergenceWarning)
                        try:
                            gp.alpha = noise_levels**2
                            gp.fit(X_train, y_train)
                            gp_success = True
                        except:
                            gp_success = False

                    if gp_success:
                        best_sample = -np.inf
                        best_j, best_idx = None, None

                        available_indices = np.where(available)[0]
                        if len(available_indices) == 0:
                            break

                        X_pred = pool[available_indices]
                        mu, std = gp_predict(gp, X_pred)

                        for j in range(m):
                            if costs[j] <= R:

                                sigma_j = apps[j].noise_level
                                total_std = np.sqrt(std**2 + sigma_j**2)

                                samples = mu + total_std * np.random.randn(len(mu))

                                if len(samples) > 0:
                                    max_idx = np.argmax(samples)
                                    if samples[max_idx] > best_sample:
                                        best_sample = samples[max_idx]
                                        best_j = j
                                        best_idx = available_indices[max_idx]

                        if best_j is not None:
                            R -= apps[best_j].cost
                            x = pool[best_idx:best_idx+1]
                            y_obs = apps[best_j].evaluate(x, best_j)
                            data['X'].append(x.flatten())
                            data['y'].append(y_obs[0])
                            data['j'].append(best_j)
                            best = max(best, y_obs[0])
                            available[best_idx] = False
                            continue

                affordable = [j for j in range(m) if costs[j] <= R]
                if not affordable:
                    break

                available_indices = np.where(available)[0]
                if len(available_indices) == 0:
                    break

                j_star = np.random.choice(affordable)
                i_star = np.random.choice(available_indices)

                R -= apps[j_star].cost
                x = pool[i_star:i_star+1]
                y_obs = apps[j_star].evaluate(x, j_star)

                data['X'].append(x.flatten())
                data['y'].append(y_obs[0])
                data['j'].append(j_star)
                best = max(best, y_obs[0])
                available[i_star] = False

            rec.append((m, B, 'Thompson', best))
    return rec

if __name__ == '__main__':

    dim = 128
    pool_size, trials = 1000, 50
    budgets = [100]
    ms = [100]

    records = []
    pool = sample_on_sphere(dim, pool_size)

    print(f"Function value range on pool: [{np.min(true_g(pool)):.3f}, {np.max(true_g(pool)):.3f}]")
    print(f"Function mean: {np.mean(true_g(pool)):.3f}")

    for m in ms:

        apps = [Approximation(c, n)
                for c, n in zip(np.linspace(1, 5, m),
                               np.linspace(0.2, 0.05, m))]

        records += [(dim,)+r for r in greedy_sequence(apps, pool, budgets, trials)]
        records += [(dim,)+r for r in multi_shot_gp(apps, pool, budgets, trials)]
        records += [(dim,)+r for r in multi_shot_thompson(apps, pool, budgets, trials)]

    df = pd.DataFrame(records, columns=['dim','m','Budget','Method','BestG'])

    grouped = df.groupby(['dim','m','Budget','Method'])['BestG']
    mean_results = grouped.mean().unstack('Method')
    std_results = grouped.std().unstack('Method')

    print("\nMean BestG by method:")
    print(mean_results)

    print("\nOverall method comparison:")
    print(df.groupby('Method')['BestG'].agg(['mean', 'std', 'min', 'max']))
