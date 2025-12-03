from pathlib import Path
from .evaluate import evaluate
from .data_loader import load_posts

BASE_DIR = Path(__file__).resolve().parents[1]

def generate_markdown_report(results_path: str | Path, labels_path: str | Path) -> Path:
    results_path = Path(results_path)
    labels_path = Path(labels_path)
    
    out_path = BASE_DIR / "results" / f"report_{results_path.stem}.md"
    
    metrics_df = evaluate(results_path, labels_path)
    res_df = load_posts(results_path)

    lines = []
    lines.append(f"# Sentiment Evaluation Report - {results_path.stem}")
    lines.append("\n## Overall Metrics\n")
    lines.append(metrics_df.to_markdown(index=False))
    lines.append("\n")

    # Example: count disagreements where models differ
    sentiment_cols = [c for c in res_df.columns if c.endswith("_sentiment")]
    disagree = res_df[sentiment_cols].nunique(axis=1) > 1
    lines.append(f"Total rows: {len(res_df)}")
    lines.append(f"Rows where models disagree: {disagree.sum()}")
    lines.append("\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    
    print(f"Report saved to {out_path}")
    return out_path

if __name__ == "__main__":
    results_path = BASE_DIR / "results" / "results_posts_sample.csv"
    labels_path = BASE_DIR / "data" / "samples" / "labels_sample.csv"
    
    generate_markdown_report(results_path, labels_path)
