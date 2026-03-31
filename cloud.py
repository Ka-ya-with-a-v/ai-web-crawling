"""
cloud.py  –  Web-crawl prioritization with *learned* feature weights.

The key improvement over the original hand-tuned heuristic is that feature
weights are now estimated from labelled training examples using a linear
regression model (via scikit-learn).  We still expose the same
``select_top_k`` / ``pagerank_only_top_k`` API so visualize.py keeps working.

Workflow
--------
1. ``build_feature_matrix`` – compute one feature vector per URL.
2. ``generate_training_labels`` – create weak/pseudo labels from a
   combination of content-quality priors and structural signals.
3. ``train_weight_model`` – fit a Ridge regression; return the learned
   weight vector and the trained model.
4. ``select_top_k`` – score every allowed URL with the learned model,
   apply domain-diversity reranking, and return the top-k list.
"""

from collections import defaultdict
from urllib.parse import urlparse

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

from web_graph_data import GRAPH, PAGERANK, ALLOWED, CONTENT_QUALITY


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_domain(url: str) -> str:
    """Extract the base domain from a URL or path-like string."""
    if "://" in url:
        return urlparse(url).netloc.lower()
    return url.split("/")[0].lower()


def is_low_quality_page(url: str) -> bool:
    """Return True for utility/transactional URLs that are poor crawl targets."""
    bad_keywords = [
        "login", "signin", "signup", "register",
        "checkout", "cart", "account", "payment",
        "password", "auth", "unsubscribe", "tracker",
        "cdn.", "ads.",
    ]
    url_lower = url.lower()
    return any(word in url_lower for word in bad_keywords)


def is_low_trust_domain(url: str) -> bool:
    """Return True for known low-trust / spammy domain patterns."""
    bad_tokens = ["randomblog", "clickbait", "forumspam", "buzzfeedarticle.click",
                  "articlefarm.xyz"]
    url_lower = url.lower()
    return any(token in url_lower for token in bad_tokens)


# ──────────────────────────────────────────────────────────────────────────────
# Structural graph features
# ──────────────────────────────────────────────────────────────────────────────

def compute_inlink_score(graph: dict, pagerank: dict) -> dict:
    """
    Inlink authority: each source page distributes its PageRank equally
    across its outlinks (simplified HITS-style hub/authority signal).
    """
    inlink_score: dict[str, float] = defaultdict(float)
    for src, outlinks in graph.items():
        if not outlinks:
            continue
        share = pagerank.get(src, 0.0) / len(outlinks)
        for dst in outlinks:
            inlink_score[dst] += share
    return dict(inlink_score)


def compute_domain_diversity(graph: dict) -> dict:
    """
    For each URL, count the number of *distinct* linking domains.
    High diversity  →  broadly recognised across the web.
    """
    linking_domains: dict[str, set] = defaultdict(set)
    for src, outlinks in graph.items():
        src_domain = get_domain(src)
        for dst in outlinks:
            linking_domains[dst].add(src_domain)
    return {url: len(domains) for url, domains in linking_domains.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Feature matrix
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "pagerank",
    "content_quality",
    "inlink_score",
    "indegree_norm",
    "domain_diversity_norm",
    "outdegree_norm",
    "is_low_quality",    # penalty flag  (0 / 1)
    "is_low_trust",      # penalty flag  (0 / 1)
]


def build_feature_matrix(
    graph: dict,
    pagerank: dict,
    allowed: set,
    content_quality: dict,
) -> tuple[np.ndarray, list[str], MinMaxScaler]:
    """
    Build an N×8 feature matrix for all crawl-allowed URLs.

    Returns
    -------
    X       : (N, 8) float array  –  raw (un-scaled) features
    urls    : list of N URL strings
    scaler  : fitted MinMaxScaler (for later inference)
    """
    inlink_score     = compute_inlink_score(graph, pagerank)
    domain_diversity = compute_domain_diversity(graph)

    # Compute raw indegree (structural, not PageRank-weighted)
    indegree: dict[str, int] = defaultdict(int)
    for src, outlinks in graph.items():
        for dst in outlinks:
            indegree[dst] += 1

    urls = [url for url in graph if url in allowed]

    max_indegree = max((indegree.get(u, 0) for u in urls), default=1) or 1
    max_diversity = max((domain_diversity.get(u, 0) for u in urls), default=1) or 1
    max_outdegree = max((len(graph.get(u, [])) for u in urls), default=1) or 1
    max_inlink = max((inlink_score.get(u, 0.0) for u in urls), default=1e-9) or 1e-9

    rows = []
    for url in urls:
        pr   = pagerank.get(url, 0.0)
        cq   = content_quality.get(url, 0.4)
        il   = inlink_score.get(url, 0.0) / max_inlink
        id_  = indegree.get(url, 0) / max_indegree
        dd   = domain_diversity.get(url, 0) / max_diversity
        od   = len(graph.get(url, [])) / max_outdegree
        lq   = float(is_low_quality_page(url))
        lt   = float(is_low_trust_domain(url))
        rows.append([pr, cq, il, id_, dd, od, lq, lt])

    X = np.array(rows, dtype=np.float32)

    # Scale only the continuous features (columns 0–5); leave binary flags raw
    scaler = MinMaxScaler()
    X[:, :6] = scaler.fit_transform(X[:, :6])

    return X, urls, scaler


# ──────────────────────────────────────────────────────────────────────────────
# Weak-supervision label generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_training_labels(
    urls: list[str],
    graph: dict,
    pagerank: dict,
    content_quality: dict,
) -> np.ndarray:
    """
    Generate pseudo-labels by combining multiple quality signals.

    Label  =  0.50 * content_quality
            + 0.30 * pagerank
            + 0.10 * (has outlinks)
            - 0.30 * is_low_quality_page
            - 0.20 * is_low_trust_domain

    These labels act as a *teacher* signal; the learner discovers which raw
    features best predict them, producing interpolated weights.
    """
    labels = []
    for url in urls:
        cq     = content_quality.get(url, 0.4)
        pr     = pagerank.get(url, 0.0)
        has_od = float(len(graph.get(url, [])) > 0) * 0.10
        lq_pen = 0.30 * float(is_low_quality_page(url))
        lt_pen = 0.20 * float(is_low_trust_domain(url))
        y = 0.50 * cq + 0.30 * pr + has_od - lq_pen - lt_pen
        labels.append(y)
    return np.array(labels, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Weight learning
# ──────────────────────────────────────────────────────────────────────────────

def train_weight_model(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.01,
) -> tuple:
    """
    Fit a Ridge regression to learn feature weights.

    Returns
    -------
    model   : fitted sklearn Ridge instance
    weights : named dict of learned coefficients
    """
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)

    weights = {name: float(coef) for name, coef in zip(FEATURE_NAMES, model.coef_)}
    weights["intercept"] = float(model.intercept_)
    return model, weights


# ──────────────────────────────────────────────────────────────────────────────
# Scoring and ranking
# ──────────────────────────────────────────────────────────────────────────────

def select_top_k(
    graph: dict,
    pagerank: dict,
    allowed: set,
    content_quality: dict,
    k: int = 10,
    domain_penalty: float = 0.08,
) -> list[dict]:
    """
    Rank allowed URLs using *learned* weights + greedy domain-diversity
    reranking.  Returns a list of dicts sorted by descending final score.
    """
    X, urls, scaler = build_feature_matrix(graph, pagerank, allowed, content_quality)
    y = generate_training_labels(urls, graph, pagerank, content_quality)
    model, learned_weights = train_weight_model(X, y)

    raw_scores = model.predict(X).tolist()

    scored = [
        {
            "url":        url,
            "base_score": score,
            "domain":     get_domain(url),
            "learned_weights": learned_weights,
        }
        for url, score in zip(urls, raw_scores)
    ]
    scored.sort(key=lambda x: x["base_score"], reverse=True)

    # Greedy diversity-aware reranking
    selected: list[dict] = []
    seen_domains: dict[str, int] = defaultdict(int)

    while scored and len(selected) < k:
        best_idx     = -1
        best_adjusted = float("-inf")

        for i, item in enumerate(scored):
            adjusted = item["base_score"] - seen_domains[item["domain"]] * domain_penalty
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_idx      = i

        chosen = scored.pop(best_idx)
        repeat_count = seen_domains[chosen["domain"]]
        final_score  = chosen["base_score"] - repeat_count * domain_penalty

        selected.append({
            "url":    chosen["url"],
            "score":  round(final_score, 4),
            "domain": chosen["domain"],
            "learned_weights": chosen["learned_weights"],
        })
        seen_domains[chosen["domain"]] += 1

    return selected


def pagerank_only_top_k(
    graph: dict,
    pagerank: dict,
    allowed: set,
    k: int = 10,
) -> list[dict]:
    """Baseline: rank by PageRank alone (no learning)."""
    candidates = [
        {"url": url, "score": pagerank.get(url, 0.0), "domain": get_domain(url)}
        for url in graph
        if url in allowed
    ]
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:k]


def get_learned_weights(
    graph: dict,
    pagerank: dict,
    allowed: set,
    content_quality: dict,
) -> dict:
    """Convenience function: train the model and return weights only."""
    X, urls, _ = build_feature_matrix(graph, pagerank, allowed, content_quality)
    y = generate_training_labels(urls, graph, pagerank, content_quality)
    _, weights = train_weight_model(X, y)
    return weights


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def print_results(title: str, results: list[dict], show_weights: bool = False) -> None:
    print(f"\n=== {title} ===")
    for i, item in enumerate(results, start=1):
        print(f"{i:>2}. {item['url']:<42}  score={item['score']:.4f}  domain={item['domain']}")
    if show_weights and results:
        w = results[0]["learned_weights"]
        print("\n  Learned feature weights:")
        for feat, val in w.items():
            print(f"    {feat:<30} {val:+.4f}")


def main() -> None:
    baseline = pagerank_only_top_k(GRAPH, PAGERANK, ALLOWED, k=10)
    improved  = select_top_k(GRAPH, PAGERANK, ALLOWED, CONTENT_QUALITY, k=10)

    print(f"Total nodes in graph : {len(GRAPH)}")
    print(f"Crawl-allowed pages  : {len(ALLOWED)}")

    print_results("Baseline: PageRank Only",        baseline)
    print_results("Improved: Learned-Weight Ranking", improved, show_weights=True)


if __name__ == "__main__":
    main()
