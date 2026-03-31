"""
visualize.py – Plots for the web-crawl prioritization experiments.
Now works with the 100-node graph and learned-weight ranking.
"""

from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from web_graph_data import GRAPH, PAGERANK, ALLOWED, CONTENT_QUALITY
from cloud import (
    pagerank_only_top_k,
    select_top_k,
    get_learned_weights,
    FEATURE_NAMES,
)

sns.set_theme(style="whitegrid", context="talk")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def classify_page(url: str) -> str:
    url = url.lower()
    if any(x in url for x in ["login", "checkout", "cart", "account", "spam",
                               "forumspam", "unsubscribe", "tracker", "cdn."]):
        return "low-quality/utility"
    if any(x in url for x in ["nih.gov", "who.int", "cdc.gov", "nature.com", "arxiv.org",
                               "science.org", "pubmed", "thelancet", "nejm", "medlineplus"]):
        return "research/health"
    if any(x in url for x in ["bbc.com", "cnn.com", "reuters.com", "nytimes.com",
                               "theguardian.com", "wsj.com", "bloomberg.com", "aljazeera",
                               "vox.com", "theatlantic", "politico", "foreignpolicy",
                               "techcrunch", "wired.com", "theverge"]):
        return "news/media"
    if any(x in url for x in ["github.com", "stackoverflow.com", "developer.mozilla.org",
                               "kaggle.com", "docs.docker", "kubernetes.io",
                               "tensorflow.org", "pytorch.org", "huggingface.co"]):
        return "technical/AI"
    if any(x in url for x in ["wikipedia.org", "stanford.edu", "mit.edu", "harvard.edu",
                               "cambridge.org", "ox.ac.uk", "khanacademy", "coursera",
                               "edx.org"]):
        return "reference/education"
    if any(x in url for x in ["un.org", "worldbank", "imf.org", "oecd.org",
                               "ecb.europa.eu", "europa.eu", "gov.uk", "nhs.uk",
                               "data.gov"]):
        return "government/policy"
    return "general/blog"


def build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for src, dsts in GRAPH.items():
        g.add_node(src)
        for dst in dsts:
            g.add_edge(src, dst)
    return g


def get_results(k: int = 10):
    baseline = pagerank_only_top_k(GRAPH, PAGERANK, ALLOWED, k)
    improved  = select_top_k(GRAPH, PAGERANK, ALLOWED, CONTENT_QUALITY, k)
    return baseline, improved


def shorten_label(label: str, max_len: int = 32) -> str:
    return label if len(label) <= max_len else label[: max_len - 3] + "..."


# ──────────────────────────────────────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────────────────────────────────────

CATEGORY_COLORS = {
    "news/media":           "#4C78A8",
    "research/health":      "#59A14F",
    "technical/AI":         "#F28E2B",
    "reference/education":  "#B07AA1",
    "government/policy":    "#76B7B2",
    "general/blog":         "#9C755F",
    "low-quality/utility":  "#E15759",
    "blocked/not allowed":  "#BAB0AC",
}


# ──────────────────────────────────────────────────────────────────────────────
# 1) Full graph – allowed pages highlighted
# ──────────────────────────────────────────────────────────────────────────────

def draw_allowed_highlight_graph(output_file="allowed_highlight_graph.png"):
    g = build_graph()
    pos = nx.spring_layout(g, seed=42, k=2.2, iterations=300)
    plt.figure(figsize=(26, 18))

    nx.draw_networkx_edges(g, pos, edge_color="gray", alpha=0.15,
                           arrows=True, arrowsize=6, width=0.6,
                           connectionstyle="arc3,rad=0.05")

    blocked = [n for n in g.nodes if n not in ALLOWED]
    nx.draw_networkx_nodes(g, pos, nodelist=blocked,
                           node_color=CATEGORY_COLORS["blocked/not allowed"],
                           node_size=[100 + PAGERANK.get(n, 0.05) * 400 for n in blocked],
                           alpha=0.5, linewidths=0.6, edgecolors="black")

    for cat, color in CATEGORY_COLORS.items():
        if cat in ("blocked/not allowed",):
            continue
        nodes = [n for n in ALLOWED if classify_page(n) == cat and n in g.nodes]
        if not nodes:
            continue
        nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=color,
                               node_size=[160 + PAGERANK.get(n, 0.05) * 700 for n in nodes],
                               alpha=0.9, linewidths=0.8, edgecolors="black", label=cat)

    label_pos = {n: (x, y + 0.03) for n, (x, y) in pos.items()}
    texts = nx.draw_networkx_labels(g, label_pos,
                                    labels={n: n for n in g.nodes}, font_size=6)
    for t in texts.values():
        t.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.12))

    legend_handles = [mpatches.Patch(color=c, label=l) for l, c in CATEGORY_COLORS.items()]
    plt.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1),
               fontsize=9, frameon=True, title="Page type")
    plt.title("100-Node Simulated Web Graph  –  Crawl-Allowed Pages Highlighted", fontsize=18, pad=18)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_file}")


# ──────────────────────────────────────────────────────────────────────────────
# 2) Allowed-only subgraph
# ──────────────────────────────────────────────────────────────────────────────

def draw_allowed_only_graph(output_file="allowed_only_graph.png"):
    g = build_graph()
    sub = g.subgraph(ALLOWED).copy()
    pos = nx.spring_layout(sub, seed=42, k=2.0, iterations=300)
    plt.figure(figsize=(24, 17))

    nx.draw_networkx_edges(sub, pos, edge_color="gray", alpha=0.18,
                           arrows=True, arrowsize=7, width=0.65,
                           connectionstyle="arc3,rad=0.05")

    for cat, color in CATEGORY_COLORS.items():
        if cat in ("blocked/not allowed",):
            continue
        nodes = [n for n in sub.nodes if classify_page(n) == cat]
        if not nodes:
            continue
        nx.draw_networkx_nodes(sub, pos, nodelist=nodes, node_color=color,
                               node_size=[160 + PAGERANK.get(n, 0.05) * 700 for n in nodes],
                               alpha=0.9, linewidths=0.8, edgecolors="black", label=cat)

    label_pos = {n: (x, y + 0.03) for n, (x, y) in pos.items()}
    texts = nx.draw_networkx_labels(sub, label_pos,
                                    labels={n: n for n in sub.nodes}, font_size=6)
    for t in texts.values():
        t.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.12))

    legend_handles = [mpatches.Patch(color=CATEGORY_COLORS[l], label=l)
                      for l in list(CATEGORY_COLORS)[:-1]]
    plt.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1),
               fontsize=9, frameon=True, title="Page type")
    plt.title("Crawl-Allowed Subgraph (100-Node Graph)", fontsize=18, pad=18)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_file}")


# ──────────────────────────────────────────────────────────────────────────────
# 3) Rank-movement chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_rank_movement(k=10, output_file="rank_movement_chart.png"):
    baseline, improved = get_results(k)
    b_ranks = {item["url"]: rank for rank, item in enumerate(baseline, 1)}
    i_ranks = {item["url"]: rank for rank, item in enumerate(improved, 1)}
    all_urls = list(dict.fromkeys([item["url"] for item in baseline] +
                                  [item["url"] for item in improved]))

    plt.figure(figsize=(13, 8))
    for url in all_urls:
        color = CATEGORY_COLORS.get(classify_page(url), "#333333")
        xs, ys = [], []
        if url in b_ranks:
            xs.append(0); ys.append(b_ranks[url])
        if url in i_ranks:
            xs.append(1); ys.append(i_ranks[url])
        if len(xs) == 2:
            plt.plot(xs, ys, marker="o", linewidth=2, alpha=0.85, color=color)
        else:
            plt.scatter(xs, ys, s=80, alpha=0.85, color=color)

        label_x = 1.03 if url in i_ranks else 0.03
        label_y = i_ranks.get(url, b_ranks.get(url))
        plt.text(label_x, label_y, shorten_label(url, 28), fontsize=8.5, va="center")

    plt.xticks([0, 1], ["PageRank Only", "Learned-Weight"])
    plt.yticks(range(1, k + 1))
    plt.gca().invert_yaxis()
    plt.ylabel("Rank Position")
    plt.title("Rank Movement: PageRank-Only vs Learned-Weight Ranking", fontsize=15)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_file}")


# ──────────────────────────────────────────────────────────────────────────────
# 4) Top-k comparison bars
# ──────────────────────────────────────────────────────────────────────────────

def plot_topk_comparison(k=10, output_file="topk_comparison.png"):
    baseline, improved = get_results(k)
    b_df = pd.DataFrame(baseline)[["url", "score"]].copy()
    i_df = pd.DataFrame(improved)[["url", "score"]].copy()
    b_df["label"] = b_df["url"].apply(lambda x: shorten_label(x, 32))
    i_df["label"] = i_df["url"].apply(lambda x: shorten_label(x, 32))

    fig, axes = plt.subplots(2, 1, figsize=(14, 11))
    sns.barplot(data=b_df, x="score", y="label", ax=axes[0], color="#7DA9D8")
    axes[0].set_title("Top-k Pages: PageRank Only")
    axes[0].set_xlabel("Score"); axes[0].set_ylabel("")

    sns.barplot(data=i_df, x="score", y="label", ax=axes[1], color="#86BC86")
    axes[1].set_title("Top-k Pages: Learned-Weight Ranking")
    axes[1].set_xlabel("Score"); axes[1].set_ylabel("")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_file}")


# ──────────────────────────────────────────────────────────────────────────────
# 5) Category breakdown
# ──────────────────────────────────────────────────────────────────────────────

def plot_category_breakdown(k=10, output_file="category_breakdown.png"):
    baseline, improved = get_results(k)
    b_cats = [classify_page(item["url"]) for item in baseline]
    i_cats = [classify_page(item["url"]) for item in improved]

    all_cats = sorted(set(b_cats) | set(i_cats))
    b_counts = Counter(b_cats)
    i_counts = Counter(i_cats)

    plot_df = pd.DataFrame({
        "category": all_cats * 2,
        "count":    [b_counts.get(c, 0) for c in all_cats] +
                    [i_counts.get(c, 0) for c in all_cats],
        "method":   ["PageRank Only"] * len(all_cats) + ["Learned-Weight"] * len(all_cats),
    })

    plt.figure(figsize=(13, 6))
    sns.barplot(data=plot_df, x="category", y="count", hue="method")
    plt.xticks(rotation=20)
    plt.ylabel("Count in Top-k")
    plt.xlabel("")
    plt.title("Category Distribution of Selected Top-k Pages", fontsize=15)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_file}")


# ──────────────────────────────────────────────────────────────────────────────
# 6) NEW – Learned weight bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_learned_weights(output_file="learned_weights.png"):
    weights = get_learned_weights(GRAPH, PAGERANK, ALLOWED, CONTENT_QUALITY)
    # exclude intercept for the bar chart
    feat_weights = {k: v for k, v in weights.items() if k != "intercept"}
    features = list(feat_weights.keys())
    values   = list(feat_weights.values())
    colors   = ["#59A14F" if v >= 0 else "#E15759" for v in values]

    plt.figure(figsize=(11, 5))
    bars = plt.barh(features, values, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, values):
        plt.text(val + (0.003 if val >= 0 else -0.003),
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:+.4f}", va="center",
                 ha="left" if val >= 0 else "right", fontsize=10)
    plt.xlabel("Learned Coefficient")
    plt.title("Feature Weights Learned by Ridge Regression", fontsize=15)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_file}")


# ──────────────────────────────────────────────────────────────────────────────
# 7) Ranking table image
# ──────────────────────────────────────────────────────────────────────────────

def save_pretty_table_image(k=10, output_file="ranking_table.png"):
    baseline, improved = get_results(k)
    b_map = {r: item["url"] for r, item in enumerate(baseline, 1)}
    i_map = {r: item["url"] for r, item in enumerate(improved, 1)}

    rows = [{"Rank": r,
             "PageRank Only":   b_map.get(r, "-"),
             "Learned-Weight":  i_map.get(r, "-"),
             "Changed?":        "Yes" if b_map.get(r) != i_map.get(r) else "No"}
            for r in range(1, k + 1)]

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(17, 1.2 + 0.55 * len(df)))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    changed_col = list(df.columns).index("Changed?")
    for col in range(len(df.columns)):
        table[(0, col)].set_text_props(weight="bold", color="black")
        table[(0, col)].set_facecolor("#D9EAF7")
    for row in range(1, len(df) + 1):
        fill = "#FCE8E6" if df.iloc[row - 1]["Changed?"] == "Yes" else "#EEF5EA"
        for col in range(len(df.columns)):
            table[(row, col)].set_facecolor(fill)

    plt.title("Top-k Ranking Comparison Table", fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_file}")


# ──────────────────────────────────────────────────────────────────────────────
# 8) CSV export
# ──────────────────────────────────────────────────────────────────────────────

def save_results_csv(k=10, output_file="ranking_results.csv"):
    baseline, improved = get_results(k)
    rows = []
    for rank, item in enumerate(baseline, 1):
        rows.append({"method": "PageRank Only", "rank": rank, "url": item["url"],
                     "domain": item["domain"], "score": item["score"],
                     "category": classify_page(item["url"])})
    for rank, item in enumerate(improved, 1):
        rows.append({"method": "Learned-Weight", "rank": rank, "url": item["url"],
                     "domain": item["domain"], "score": item["score"],
                     "category": classify_page(item["url"])})
    pd.DataFrame(rows).to_csv(output_file, index=False)
    print(f"Saved {output_file}")


# ──────────────────────────────────────────────────────────────────────────────
# Run everything
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    k = 10
    draw_allowed_highlight_graph()
    draw_allowed_only_graph()
    plot_topk_comparison(k=k)
    plot_rank_movement(k=k)
    plot_category_breakdown(k=k)
    plot_learned_weights()
    save_pretty_table_image(k=k)
    save_results_csv(k=k)

    print("\nAll files generated successfully.")
