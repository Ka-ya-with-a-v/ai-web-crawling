from urllib.parse import urlparse
from collections import defaultdict


def get_domain(url: str) -> str:
    """
    Extract domain family.
    Example:
        'bbc.com/news' -> 'bbc.com'
        'https://cnn.com/world' -> 'cnn.com'
    """
    if "://" in url:
        parsed = urlparse(url)
        domain = parsed.netloc
    else:
        domain = url.split("/")[0]
    return domain.lower()


def is_low_quality_page(url: str) -> bool:
    """
    Heuristic for utility/noise pages.
    """
    bad_keywords = [
        "login", "signin", "signup", "register",
        "checkout", "cart", "account", "payment",
        "password", "auth"
    ]
    url_lower = url.lower()
    return any(word in url_lower for word in bad_keywords)


def compute_inlink_score(graph, pagerank):
    """
    Bonus for being linked to by strong pages.
    This makes the graph structure matter more.
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
    Produce a richer score and explanation for one URL.
    """
    if url not in allowed:
        return None

    reasons = []

    # Base authority
    pr = pagerank.get(url, 0.0)
    score = 0.60 * pr
    reasons.append(f"PageRank contribution = 0.60*{pr:.2f}")

    # Content quality prior
    cq = content_quality.get(url, 0.4)
    score += 0.30 * cq
    reasons.append(f"content-quality contribution = 0.30*{cq:.2f}")

    # Inlink authority bonus
    il = inlink_score.get(url, 0.0)
    score += 0.25 * il
    reasons.append(f"inlink bonus = 0.25*{il:.2f}")

    # Outdegree bonus: content-rich pages tend to link out
    outdegree = len(graph.get(url, []))
    out_bonus = min(outdegree * 0.015, 0.06)
    score += out_bonus
    reasons.append(f"outdegree bonus = +{out_bonus:.3f}")

    # Junk penalty
    if is_low_quality_page(url):
        score -= 0.50
        reasons.append("junk-page penalty = -0.50")

    return score, reasons


def select_top_k(graph, pagerank, allowed, content_quality, k=5, domain_penalty=0.12):
    """
    Greedy selection with diversity-aware reranking.
    """
    inlink_score = compute_inlink_score(graph, pagerank)

    # Score all allowed candidates
    scored = []
    for url in graph:
        result = score_url(url, graph, pagerank, allowed, content_quality, inlink_score)
        if result is not None:
            score, reasons = result
            scored.append({
                "url": url,
                "base_score": score,
                "reasons": reasons
            })

    # Sort by score first
    scored.sort(key=lambda x: x["base_score"], reverse=True)

    selected = []
    seen_domains = defaultdict(int)

    while scored and len(selected) < k:
        best_idx = None
        best_adjusted = float("-inf")

        for i, item in enumerate(scored):
            domain = get_domain(item["url"])
            repeat_count = seen_domains[domain]

            adjusted = item["base_score"] - (repeat_count * domain_penalty)

            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_idx = i

        chosen = scored.pop(best_idx)
        domain = get_domain(chosen["url"])
        repeat_count = seen_domains[domain]

        final_score = chosen["base_score"] - (repeat_count * domain_penalty)

        reasons = list(chosen["reasons"])
        if repeat_count > 0:
            reasons.append(f"domain diversity penalty = -{repeat_count * domain_penalty:.2f}")

        seen_domains[domain] += 1

        selected.append({
            "url": chosen["url"],
            "score": round(final_score, 3),
            "domain": domain,
            "reasons": reasons
        })

    return selected


def pagerank_only_top_k(graph, pagerank, allowed, k=5):
    candidates = [
        {"url": url, "score": pagerank.get(url, 0.0), "domain": get_domain(url)}
        for url in graph
        if url in allowed
    ]
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:k]


def domain_diversity(results):
    return len({item["domain"] for item in results})


def average_score(results):
    if not results:
        return 0.0
    return round(sum(item["score"] for item in results) / len(results), 3)


def print_results(title, results, show_reasons=False):
    print(f"\n=== {title} ===")
    for i, item in enumerate(results, start=1):
        print(f"{i}. {item['url']:<20} score={item['score']:.3f}   domain={item['domain']}")
        if show_reasons:
            for reason in item["reasons"]:
                print(f"   - {reason}")


def print_comparison_table(baseline, improved):
    print("\n=== Comparison Table ===")
    print(f"{'Rank':<6}{'PageRank Only':<22}{'Improved Method':<22}")
    print("-" * 50)
    for i in range(max(len(baseline), len(improved))):
        left = baseline[i]["url"] if i < len(baseline) else "-"
        right = improved[i]["url"] if i < len(improved) else "-"
        print(f"{i+1:<6}{left:<22}{right:<22}")


def main():
    # Better graph: includes multiple pages from same domains and some junk pages.
    graph = {
        "bbc.com": ["cnn.com", "nytimes.com", "bbc.com/news", "reuters.com"],
        "bbc.com/news": ["bbc.com", "reuters.com", "theguardian.com"],
        "cnn.com": ["bbc.com", "reuters.com", "nytimes.com", "cnn.com/world"],
        "cnn.com/world": ["bbc.com", "nytimes.com"],
        "nytimes.com": ["cnn.com", "theguardian.com", "reuters.com"],
        "reuters.com": ["bbc.com", "bloomberg.com", "nytimes.com"],
        "theguardian.com": ["bbc.com", "cnn.com", "medium.com"],
        "bloomberg.com": ["reuters.com", "wsj.com"],
        "wsj.com": ["bloomberg.com", "reuters.com"],
        "medium.com": ["bbc.com", "cnn.com", "randomblog123.com"],
        "randomblog123.com": ["login.site.com", "medium.com"],
        "login.site.com": [],
        "shoppingcart.com": ["checkout.com"],
        "checkout.com": [],
        "forumspam.net": ["login.site.com", "shoppingcart.com"]
    }

    # Precomputed PageRank
    pagerank = {
        "bbc.com": 0.95,
        "bbc.com/news": 0.78,
        "cnn.com": 0.90,
        "cnn.com/world": 0.73,
        "nytimes.com": 0.84,
        "reuters.com": 0.88,
        "theguardian.com": 0.76,
        "bloomberg.com": 0.72,
        "wsj.com": 0.70,
        "medium.com": 0.58,
        "randomblog123.com": 0.22,
        "login.site.com": 0.05,
        "shoppingcart.com": 0.08,
        "checkout.com": 0.04,
        "forumspam.net": 0.10
    }

    # Simulated crawl permission
    allowed = {
        "bbc.com", "bbc.com/news",
        "cnn.com", "cnn.com/world",
        "nytimes.com", "reuters.com",
        "theguardian.com", "bloomberg.com",
        "wsj.com", "medium.com",
        "randomblog123.com"
    }

    # Heuristic prior for likely training-data quality
    content_quality = {
        "bbc.com": 0.95,
        "bbc.com/news": 0.92,
        "cnn.com": 0.88,
        "cnn.com/world": 0.86,
        "nytimes.com": 0.91,
        "reuters.com": 0.94,
        "theguardian.com": 0.84,
        "bloomberg.com": 0.87,
        "wsj.com": 0.86,
        "medium.com": 0.62,
        "randomblog123.com": 0.25,
        "login.site.com": 0.05,
        "shoppingcart.com": 0.05,
        "checkout.com": 0.05,
        "forumspam.net": 0.02
    }

    k = 5

    baseline = pagerank_only_top_k(graph, pagerank, allowed, k)
    improved = select_top_k(graph, pagerank, allowed, content_quality, k, domain_penalty=0.12)

    print_results("Baseline: PageRank Only", baseline)
    print_results("Improved Crawl Prioritization", improved, show_reasons=True)
    print_comparison_table(baseline, improved)

    print("\n=== Simple Metrics ===")
    print(f"Domain diversity (PageRank only): {domain_diversity(baseline)}")
    print(f"Domain diversity (Improved):      {domain_diversity(improved)}")
    print(f"Average score (PageRank only):    {average_score(baseline)}")
    print(f"Average score (Improved):         {average_score(improved)}")

    print("\n=== Excluded Pages ===")
    for url in ["login.site.com", "shoppingcart.com", "checkout.com", "forumspam.net"]:
        if url not in allowed:
            print(f"- {url}: excluded because crawling is not allowed")
        elif is_low_quality_page(url):
            print(f"- {url}: allowed but penalized as a low-quality utility page")
        else:
            print(f"- {url}: allowed")

    print("\n=== Interpretation ===")
    print("The improved method differs from PageRank-only because it combines authority,")
    print("estimated content quality, graph support from inlinks, and a diversity-aware")
    print("selection rule. This makes the crawl frontier more useful than ranking by")
    print("authority alone.")


if __name__ == "__main__":
    main()