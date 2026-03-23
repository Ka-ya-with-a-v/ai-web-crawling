from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import seaborn as sns

from web_graph_data import GRAPH, PAGERANK, ALLOWED, CONTENT_QUALITY
from cloud import pagerank_only_top_k, select_top_k


# -----------------------------
# Styling
# -----------------------------
sns.set_theme(style="whitegrid", context="talk")


# -----------------------------
# Helpers
# -----------------------------
def classify_page(url: str) -> str:
    url = url.lower()

    if any(x in url for x in ["login", "checkout", "cart", "account", "spam", "forumspam"]):
        return "low-quality/utility"
    if any(x in url for x in ["nih.gov", "who.int", "cdc.gov", "nature.com", "arxiv.org", "sciencedirect.com", "science.org"]):
        return "research/health"
    if any(x in url for x in ["bbc.com", "cnn.com", "reuters.com", "nytimes.com", "theguardian.com", "aljazeera.com", "wsj.com", "bloomberg.com"]):
        return "news"
    if any(x in url for x in ["github.com", "stackoverflow.com", "developer.mozilla.org", "kaggle.com"]):
        return "technical"
    if any(x in url for x in ["wikipedia.org", "stanford.edu", "mit.edu", "harvard.edu"]):
        return "reference/education"

    return "general"


def build_graph():
    g = nx.DiGraph()
    for src, dsts in GRAPH.items():
        if src not in g:
            g.add_node(src)
        for dst in dsts:
            g.add_edge(src, dst)
    return g


def get_results(k=10):
    baseline = pagerank_only_top_k(GRAPH, PAGERANK, ALLOWED, k)
    improved = select_top_k(GRAPH, PAGERANK, ALLOWED, CONTENT_QUALITY, k)
    return baseline, improved


def shorten_label(label: str, max_len: int = 30) -> str:
    if len(label) <= max_len:
        return label
    return label[: max_len - 3] + "..."


# -----------------------------
# 1) Full graph with allowed pages highlighted
# -----------------------------
def draw_allowed_highlight_graph(output_file="allowed_highlight_graph.png"):
    g = build_graph()

    pos = nx.spring_layout(g, seed=42, k=2.2, iterations=300)

    plt.figure(figsize=(22, 16))

    category_colors = {
        "news": "#4C78A8",
        "research/health": "#59A14F",
        "technical": "#F28E2B",
        "reference/education": "#B07AA1",
        "general": "#9C755F",
        "low-quality/utility": "#E15759",
        "blocked/not allowed": "#BAB0AC",
    }

    allowed_nodes = [n for n in g.nodes if n in ALLOWED]
    blocked_nodes = [n for n in g.nodes if n not in ALLOWED]

    # Lighter edges
    nx.draw_networkx_edges(
        g,
        pos,
        edge_color="gray",
        alpha=0.18,
        arrows=True,
        arrowsize=8,
        width=0.7,
        connectionstyle="arc3,rad=0.05"
    )

    # Blocked nodes first
    blocked_sizes = [120 + PAGERANK.get(n, 0.05) * 500 for n in blocked_nodes]
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=blocked_nodes,
        node_color=category_colors["blocked/not allowed"],
        node_size=blocked_sizes,
        alpha=0.55,
        linewidths=0.8,
        edgecolors="black",
        label="blocked/not allowed"
    )

    # Allowed nodes grouped by category
    category_to_nodes = {
        "news": [],
        "research/health": [],
        "technical": [],
        "reference/education": [],
        "general": [],
        "low-quality/utility": [],
    }

    for node in allowed_nodes:
        category_to_nodes[classify_page(node)].append(node)

    for category, nodes in category_to_nodes.items():
        if not nodes:
            continue

        sizes = [180 + PAGERANK.get(n, 0.05) * 900 for n in nodes]
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=nodes,
            node_color=category_colors[category],
            node_size=sizes,
            alpha=0.9,
            linewidths=1.0,
            edgecolors="black",
            label=category
        )

    # Label every node, slightly offset upward
    label_pos = {node: (x, y + 0.03) for node, (x, y) in pos.items()}
    text_items = nx.draw_networkx_labels(
        g,
        label_pos,
        labels={node: node for node in g.nodes},
        font_size=7,
        font_color="black"
    )

    for _, text in text_items.items():
        text.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.15))

    plt.title("Simulated Web Graph with Crawl-Allowed Pages Highlighted", fontsize=20, pad=20)
    plt.axis("off")

    legend_order = [
        "blocked/not allowed",
        "news",
        "research/health",
        "technical",
        "reference/education",
        "general",
        "low-quality/utility",
    ]

    legend_handles = [
        mpatches.Patch(color=category_colors[name], label=name)
        for name in legend_order
    ]

    plt.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        fontsize=10,
        frameon=True,
        title="Page type"
    )


    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()


# -----------------------------
# 2) Allowed-only graph, fully labeled
# -----------------------------
def draw_allowed_only_graph(output_file="allowed_only_graph.png"):
    g = build_graph()
    allowed_subgraph = g.subgraph(ALLOWED).copy()

    pos = nx.spring_layout(allowed_subgraph, seed=42, k=2.0, iterations=300)

    plt.figure(figsize=(22, 16))

    category_colors = {
        "news": "#4C78A8",
        "research/health": "#59A14F",
        "technical": "#F28E2B",
        "reference/education": "#B07AA1",
        "general": "#9C755F",
        "low-quality/utility": "#E15759",
    }

    nx.draw_networkx_edges(
        allowed_subgraph,
        pos,
        edge_color="gray",
        alpha=0.20,
        arrows=True,
        arrowsize=8,
        width=0.7,
        connectionstyle="arc3,rad=0.05"
    )

    for category, color in category_colors.items():
        nodes = [n for n in allowed_subgraph.nodes if classify_page(n) == category]
        if not nodes:
            continue

        sizes = [180 + PAGERANK.get(n, 0.05) * 900 for n in nodes]

        nx.draw_networkx_nodes(
            allowed_subgraph,
            pos,
            nodelist=nodes,
            node_color=color,
            node_size=sizes,
            alpha=0.9,
            linewidths=1.0,
            edgecolors="black",
            label=category
        )

    label_pos = {node: (x, y + 0.03) for node, (x, y) in pos.items()}
    text_items = nx.draw_networkx_labels(
        allowed_subgraph,
        label_pos,
        labels={node: node for node in allowed_subgraph.nodes},
        font_size=7,
        font_color="black"
    )

    for _, text in text_items.items():
        text.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.15))

    plt.title("Crawl-Allowed Subgraph (Fully Labeled)", fontsize=20, pad=20)
    plt.axis("off")
    import matplotlib.patches as mpatches

    category_colors = {
        "news": "#4C78A8",
        "research/health": "#59A14F",
        "technical": "#F28E2B",
        "reference/education": "#B07AA1",
        "general": "#9C755F",
    }

    legend_order = [
        "news",
        "research/health",
        "technical",
        "reference/education",
        "general",
    ]

    legend_handles = [
        mpatches.Patch(color=category_colors[name], label=name)
        for name in legend_order
    ]

    plt.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        frameon=True,
        title="Page type",
        title_fontsize=11,
        labelspacing=0.6,
        borderpad=0.8
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()


# -----------------------------
# 3) Before/after rank movement chart
# -----------------------------
def plot_rank_movement(k=10, output_file="rank_movement_chart.png"):
    baseline, improved = get_results(k)

    baseline_ranks = {item["url"]: rank for rank, item in enumerate(baseline, start=1)}
    improved_ranks = {item["url"]: rank for rank, item in enumerate(improved, start=1)}

    union_urls = list(dict.fromkeys([item["url"] for item in baseline] + [item["url"] for item in improved]))
    rows = []

    for url in union_urls:
        rows.append({
            "url": url,
            "baseline_rank": baseline_ranks.get(url),
            "improved_rank": improved_ranks.get(url),
            "category": classify_page(url)
        })

    df = pd.DataFrame(rows)

    plt.figure(figsize=(13, 8))

    palette = {
        "news": "#4C78A8",
        "research/health": "#59A14F",
        "technical": "#F28E2B",
        "reference/education": "#B07AA1",
        "general": "#9C755F",
        "low-quality/utility": "#E15759",
    }

    for _, row in df.iterrows():
        x_vals = []
        y_vals = []

        if pd.notna(row["baseline_rank"]):
            x_vals.append(0)
            y_vals.append(row["baseline_rank"])

        if pd.notna(row["improved_rank"]):
            x_vals.append(1)
            y_vals.append(row["improved_rank"])

        color = palette.get(row["category"], "#333333")

        if len(x_vals) == 2:
            plt.plot(x_vals, y_vals, marker="o", linewidth=2, alpha=0.9, color=color)
        else:
            plt.scatter(x_vals, y_vals, s=90, alpha=0.9, color=color)

        if pd.notna(row["improved_rank"]):
            plt.text(1.03, row["improved_rank"], shorten_label(row["url"], 28), fontsize=9, va="center")
        elif pd.notna(row["baseline_rank"]):
            plt.text(0.03, row["baseline_rank"], shorten_label(row["url"], 28), fontsize=9, va="center")

    plt.xticks([0, 1], ["PageRank Only", "Improved"])
    plt.yticks(range(1, k + 1))
    plt.gca().invert_yaxis()
    plt.ylabel("Rank Position")
    plt.title("Before/After Rank Movement for Top-k Pages", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()


# -----------------------------
# 4) Pretty comparison table image
# -----------------------------
def save_pretty_table_image(k=10, output_file="ranking_table.png"):
    baseline, improved = get_results(k)

    baseline_map = {rank: item["url"] for rank, item in enumerate(baseline, start=1)}
    improved_map = {rank: item["url"] for rank, item in enumerate(improved, start=1)}

    rows = []
    for rank in range(1, k + 1):
        b_url = baseline_map.get(rank, "-")
        i_url = improved_map.get(rank, "-")
        rows.append({
            "Rank": rank,
            "PageRank Only": b_url,
            "Improved": i_url,
            "Changed?": "Yes" if b_url != i_url else "No"
        })

    df = pd.DataFrame(rows)

    fig_height = 1.2 + 0.55 * len(df)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="left",
        colLoc="left"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    for col_idx in range(len(df.columns)):
        cell = table[(0, col_idx)]
        cell.set_text_props(weight="bold", color="black")
        cell.set_facecolor("#D9EAF7")

    changed_col_idx = list(df.columns).index("Changed?")
    for row_idx in range(1, len(df) + 1):
        changed = df.iloc[row_idx - 1]["Changed?"]
        for col_idx in range(len(df.columns)):
            cell = table[(row_idx, col_idx)]
            if changed == "Yes":
                cell.set_facecolor("#FCE8E6")
            else:
                cell.set_facecolor("#EEF5EA")

    plt.title("Top-k Ranking Comparison Table", fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()


# -----------------------------
# 5) Top-k score comparison bars
# -----------------------------
def plot_topk_comparison(k=10, output_file="topk_comparison.png"):
    baseline, improved = get_results(k)

    baseline_df = pd.DataFrame(baseline)[["url", "score"]].copy()
    improved_df = pd.DataFrame(improved)[["url", "score"]].copy()

    baseline_df["label"] = baseline_df["url"].apply(lambda x: shorten_label(x, 30))
    improved_df["label"] = improved_df["url"].apply(lambda x: shorten_label(x, 30))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    sns.barplot(data=baseline_df, x="score", y="label", ax=axes[0], color="#7DA9D8")
    axes[0].set_title("Top-k Pages: PageRank Only")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("")

    sns.barplot(data=improved_df, x="score", y="label", ax=axes[1], color="#86BC86")
    axes[1].set_title("Top-k Pages: Improved Ranking")
    axes[1].set_xlabel("Score")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()


# -----------------------------
# 6) Category breakdown
# -----------------------------
def plot_category_breakdown(k=10, output_file="category_breakdown.png"):
    baseline, improved = get_results(k)

    baseline_categories = [classify_page(item["url"]) for item in baseline]
    improved_categories = [classify_page(item["url"]) for item in improved]

    baseline_counts = Counter(baseline_categories)
    improved_counts = Counter(improved_categories)

    all_categories = sorted(set(baseline_counts.keys()) | set(improved_counts.keys()))
    baseline_values = [baseline_counts.get(cat, 0) for cat in all_categories]
    improved_values = [improved_counts.get(cat, 0) for cat in all_categories]

    plot_df = pd.DataFrame({
        "category": all_categories * 2,
        "count": baseline_values + improved_values,
        "method": ["PageRank Only"] * len(all_categories) + ["Improved"] * len(all_categories)
    })

    plt.figure(figsize=(13, 6))
    sns.barplot(data=plot_df, x="category", y="count", hue="method")
    plt.xticks(rotation=20)
    plt.ylabel("Count in Top-k")
    plt.xlabel("")
    plt.title("Category Breakdown of Selected Pages", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()


# -----------------------------
# 7) CSV export
# -----------------------------
def save_results_csv(k=10, output_file="ranking_results.csv"):
    baseline, improved = get_results(k)

    rows = []
    for rank, item in enumerate(baseline, start=1):
        rows.append({
            "method": "PageRank Only",
            "rank": rank,
            "url": item["url"],
            "domain": item["domain"],
            "score": item["score"],
            "category": classify_page(item["url"])
        })

    for rank, item in enumerate(improved, start=1):
        rows.append({
            "method": "Improved",
            "rank": rank,
            "url": item["url"],
            "domain": item["domain"],
            "score": item["score"],
            "category": classify_page(item["url"])
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Saved CSV: {output_file}")


# -----------------------------
# Run everything
# -----------------------------
if __name__ == "__main__":
    k = 10

    draw_allowed_highlight_graph(output_file="allowed_highlight_graph.png")
    draw_allowed_only_graph(output_file="allowed_only_graph.png")
    plot_topk_comparison(k=k, output_file="topk_comparison.png")
    plot_rank_movement(k=k, output_file="rank_movement_chart.png")
    plot_category_breakdown(k=k, output_file="category_breakdown.png")
    save_pretty_table_image(k=k, output_file="ranking_table.png")
    save_results_csv(k=k, output_file="ranking_results.csv")

    print("\nGenerated files:")
    print("- allowed_highlight_graph.png")
    print("- allowed_only_graph.png")
    print("- topk_comparison.png")
    print("- rank_movement_chart.png")
    print("- category_breakdown.png")
    print("- ranking_table.png")
    print("- ranking_results.csv")