"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
# YOUR ANSWER HERE
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    A = np.array([[3, 4], [2, 1]], dtype=float) if A is None else np.asarray(A, dtype=float)
    B = np.array([[1, 2], [0, -1]], dtype=float) if B is None else np.asarray(B, dtype=float)
    AB = A @ B
    BA = B @ A
    return AB, BA, (A + B).T, A @ B.T, np.trace(A), np.trace(AB)


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    M = np.array([[2, -1], [-1, 2]], dtype=float) if M is None else np.asarray(M, dtype=float)
    values, vectors = np.linalg.eigh(M) # eigh is for symmetric matrices: real eigenvalues, orthonormal eigenvectors
    return values, vectors


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    C = np.array([[4, 7], [2, 6]], dtype=float) if C is None else np.asarray(C, dtype=float)
    return np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# YOUR ANSWER HERE
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    rate = float(rate)
    return np.exp(-rate), rate, rate


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    p_ab = p_a_given_b * p_b
    return p_ab, p_ab / p_a, p_a + p_b - p_ab


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    mle = heads / tosses
    map_est = (heads + 1) / (tosses + 1)
    return mle, map_est


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    x = np.array([1, 3, 5, 7], dtype=float) if data is None else np.asarray(data, dtype=float)
    mu = x.mean()
    var = np.mean((x - mu) ** 2)
    return mu, var


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    w_obs = 1.0 / observation_variance
    w_prior = 1.0 / prior_variance
    return (w_obs * x + w_prior * prior_mean) / (w_obs + w_prior)


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# YOUR ANSWER HERE

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# YOUR ANSWER HERE

# P3 Q2: Describe the Gaussian generator and observed correlation.
# YOUR ANSWER HERE
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    mean = np.array([2.0, -1.0]) if mean is None else np.array(mean, dtype=float)
    covariance = (np.array([[3.0, 1.2], [1.2, 1.0]]) if covariance is None
                  else np.array(covariance, dtype=float))
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size=n_samples)


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# YOUR ANSWER HERE
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    mean = X.mean(axis=0)
    centered = X - mean
    covariance = centered.T @ centered / (n - 1)
    covariance = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order]

    # flip each eigenvectorr,so that its largest entry is +ve,
    flip = np.sign(components[np.argmax(np.abs(components), axis=0), range(components.shape[1])])
    flip[flip == 0] = 1
    components = components * flip

    eigenvalues = np.clip(eigenvalues, 0.0, None) # rounding off can make a true zero eigenvalue slightly negative.

    scores = centered @ components
    return {
        "mean": mean,
        "centered": centered,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "components": components,
        "scores": scores,
    }


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# YOUR ANSWER HERE
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return eigenvalues / eigenvalues.sum()


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    import matplotlib.pyplot as plt

    X = np.asarray(X, dtype=float)
    mean = result["mean"]
    eigenvalues = result["eigenvalues"]
    components = result["components"]
    fractions = explained_variance(eigenvalues)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.5)
    r = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
    ax.set_title(f"Generated data (sample correlation = {r:.2f})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.3, label="samples")
    colors = ["tab:red", "tab:green"]
    for i in range(2):
        vec = components[:, i] * 2 * np.sqrt(eigenvalues[i])
        ax.annotate("", xy=mean + vec, xytext=mean,
                    arrowprops=dict(arrowstyle="->", color=colors[i], lw=2.5))
        ax.plot([], [], color=colors[i], lw=2.5,
                label=f"PC{i + 1} (eigenvalue {eigenvalues[i]:.2f})")
    ax.scatter(*mean, color="black", marker="x", s=60, label="mean")
    ax.set_title("Principal directions (length = 2 std)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    labels = [f"PC{i + 1}" for i in range(len(fractions))]
    bars = ax.bar(labels, fractions, color=colors[:len(fractions)])
    ax.plot(labels, np.cumsum(fractions), "ko-", label="cumulative")
    for b, f in zip(bars, fractions):
        ax.text(b.get_x() + b.get_width() / 2, f + 0.02, f"{f:.1%}", ha="center")
    ax.set_ylim(0, 1.1)
    ax.set_title("Explained variance")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Fraction of total variance")
    ax.legend(loc="center right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE

if __name__ == '__main__':
    import matplotlib
    matplotlib.use("Agg")

    print("Part 1")
    AB, BA, ApBT, ABT, trA, trAB = matrix_operations()
    print("AB =\n", AB, "\nBA =\n", BA, "\n(A+B)^T =\n", ApBT, "\nAB^T =\n", ABT)
    print("trace(A) =", trA, " trace(AB) =", trAB)
    vals, vecs = eigensystem()
    print("eigenvalues:", vals, "\neigenvectors (columns):\n", vecs)
    M = np.array([[2, -1], [-1, 2]], dtype=float)
    for i in range(2):
        print(f"  M v{i} = {M @ vecs[:, i]},  lambda v{i} = {vals[i] * vecs[:, i]}")
    print("C^-1 =\n", matrix_inverse())

    print("\nPart 2")
    print("c, E[X], Var(X):", distribution_moments())
    print("P(AnB), P(B|A), P(AuB):", event_probabilities())
    print("coin MLE, MAP:", coin_estimates())
    print("Gaussian mu, sigma^2 MLE:", gaussian_mle())
    print("Gaussian MAP mu:", gaussian_map())

    print("\nPart 3")
    X = generate_data()
    result = pca(X)
    print("sample covariance:\n", result["covariance"])
    print("eigenvalues:", result["eigenvalues"])
    print("components (columns):\n", result["components"])
    print("explained variance:", explained_variance(result["eigenvalues"]))

    fig = make_plots(X, result)
    fig.savefig("pca_plots.png", dpi=150)
    print("saved pca_plots.png")
