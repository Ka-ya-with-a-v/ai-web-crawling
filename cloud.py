from urllib.parse import urlparse
from collections import defaultdict
from web_graph_data import GRAPH, PAGERANK, ALLOWED, CONTENT_QUALITY


def get_domain(url: str) -> str:
    """Extract the base domain from a URL or path-like string."""
    if "://" in url:
        return urlparse(url).netloc.lower()
    return url.split("/")[0].lower()


def is_low_quality_page(url: str) -> bool:
    """Detect utility pages that are usually poor crawl targets."""
    bad_keywords = [
        "login", "signin", "signup", "register",
        "checkout", "cart", "account", "payment",
        "password", "auth"
    ]
    url_lower = url.lower()
    return any(word in url_lower for word in bad_keywords)


def compute_inlink_score(graph, pagerank):
    """
    Compute a simple inlink authority score.
    Each source page distributes its PageRank equally across its outlinks.
    """
    inlink_score = defaultdict(float)

    for src, outlinks in graph.items():
        if not outlinks:
            continue

        share = pagerank.get(src, 0.0) / len(outlinks)
        for dst in outlinks:
            inlink_score[dst] += share

    return dict(inlink_score)


def score_url(url, graph, pagerank, allowed, content_quality, inlink_score):
    """
    Score a URL using a weighted combination of:
    - PageRank
    - content quality prior
    - inlink support
    - outdegree bonus
    - low-quality penalties
    """
    if url not in allowed:
        return None

    reasons = []

    # 1. PageRank contribution
    pr = pagerank.get(url, 0.0)
    score = 0.55 * pr
    reasons.append(f"PageRank contribution = 0.55*{pr:.2f}")

    # 2. Content quality contribution
    cq = content_quality.get(url, 0.4)
    score += 0.30 * cq
    reasons.append(f"content-quality contribution = 0.30*{cq:.2f}")

    # 3. Inlink authority bonus
    il = inlink_score.get(url, 0.0)
    score += 0.20 * il
    reasons.append(f"inlink bonus = 0.20*{il:.2f}")

    # 4. Outdegree bonus
    outdegree = len(graph.get(url, []))
    out_bonus = min(outdegree * 0.012, 0.06)
    score += out_bonus
    reasons.append(f"outdegree bonus = +{out_bonus:.3f}")

    # 5. Utility-page penalty
    if is_low_quality_page(url):
        score -= 0.50
        reasons.append("junk-page penalty = -0.50")

    # 6. Low-trust domain penalty
    if any(token in url.lower() for token in ["randomblog", "clickbait", "forumspam"]):
        score -= 0.18
        reasons.append("low-trust-site penalty = -0.18")

    return score, reasons


def select_top_k(graph, pagerank, allowed, content_quality, k=10, domain_penalty=0.10):
    """
    Select top-k pages using greedy diversity-aware reranking.
    """
    inlink_score = compute_inlink_score(graph, pagerank)
    scored = []

    for url in graph:
        result = score_url(url, graph, pagerank, allowed, content_quality, inlink_score)
        if result is None:
            continue

        score, reasons = result
        scored.append({
            "url": url,
            "base_score": score,
            "reasons": reasons
        })

    scored.sort(key=lambda x: x["base_score"], reverse=True)

    selected = []
    seen_domains = defaultdict(int)

    while scored and len(selected) < k:
        best_idx = None
        best_adjusted = float("-inf")

        for i, item in enumerate(scored):
            domain = get_domain(item["url"])
            adjusted = item["base_score"] - seen_domains[domain] * domain_penalty

            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_idx = i

        chosen = scored.pop(best_idx)
        domain = get_domain(chosen["url"])
        repeat_count = seen_domains[domain]
        final_score = chosen["base_score"] - repeat_count * domain_penalty

        reasons = list(chosen["reasons"])
        if repeat_count > 0:
            reasons.append(f"domain diversity penalty = -{repeat_count * domain_penalty:.2f}")

        selected.append({
            "url": chosen["url"],
            "score": round(final_score, 3),
            "domain": domain,
            "reasons": reasons,
        })

        seen_domains[domain] += 1

    return selected


def pagerank_only_top_k(graph, pagerank, allowed, k=10):
    """Baseline ranking using PageRank only on crawl-allowed pages."""
    candidates = [
        {
            "url": url,
            "score": pagerank.get(url, 0.0),
            "domain": get_domain(url)
        }
        for url in graph
        if url in allowed
    ]

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:k]


def print_results(title, results, show_reasons=False):
    """Pretty-print ranking results."""
    print(f"\n=== {title} ===")
    for i, item in enumerate(results, start=1):
        print(f"{i}. {item['url']:<30} score={item['score']:.3f}   domain={item['domain']}")
        if show_reasons:
            for reason in item["reasons"]:
                print(f"   - {reason}")


def main():
    baseline = pagerank_only_top_k(GRAPH, PAGERANK, ALLOWED, k=10)
    improved = select_top_k(GRAPH, PAGERANK, ALLOWED, CONTENT_QUALITY, k=10)

    print(f"Total nodes in graph: {len(GRAPH)}")
    print(f"Total crawl-allowed pages: {len(ALLOWED)}")

    print_results("Baseline: PageRank Only", baseline)
    print_results("Improved Crawl Prioritization", improved, show_reasons=True)


if __name__ == "__main__":
    main()