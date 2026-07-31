"""Build the Assignment 3 high-level Markdown and PDF report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "submission.yaml"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def shorten(value: Any, length: int = 38) -> str:
    value = str(value)
    return value if len(value) <= length else value[: length - 3] + "..."


def footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#5F6B76"))
    canvas.drawString(0.65 * inch, 0.35 * inch, "ADSP 31021 - Assignment 3 AutoML")
    canvas.drawRightString(7.85 * inch, 0.35 * inch, f"Page {document.page}")
    canvas.restoreState()


def styled_table(rows: list[list[Any]], widths: list[float], font_size: float = 7.5) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C4CE")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def leaderboard_rows(
    dataframe: pd.DataFrame,
    model_column: str,
    score_column: str,
    speed_column: str,
) -> list[list[Any]]:
    rows = [["Rank", "Model", "Validation RMSE", "Speed"]]
    for index, row in dataframe.head(3).iterrows():
        rank = row.get("validation_rank", row.get("speed_rank", index + 1))
        rows.append(
            [
                str(int(rank)),
                shorten(row[model_column], 42),
                fmt(row[score_column]),
                fmt(row[speed_column]),
            ]
        )
    return rows


def main() -> None:
    config = load_yaml(CONFIG)
    assignment = config["assignment"]
    paths = {key: ROOT / value for key, value in config["paths"].items()}
    outputs = config["outputs"]

    required = [
        "feature_list",
        "target_quality",
        "autogluon_all_summary",
        "autogluon_all_score",
        "autogluon_all_speed",
        "autogluon_top_summary",
        "h2o_all_summary",
        "h2o_all_score",
        "h2o_all_speed",
        "h2o_top_summary",
        "final_comparison",
        "recommendation",
        "feature_overlap",
        "assignment1_baseline",
    ]
    missing = [
        str(paths[key].relative_to(ROOT))
        for key in required
        if not paths[key].exists()
    ]
    if missing:
        raise FileNotFoundError("Missing report inputs: " + ", ".join(missing))

    features = load_json(paths["feature_list"])
    quality = load_json(paths["target_quality"])
    ag_all = load_json(paths["autogluon_all_summary"])
    ag_top = load_json(paths["autogluon_top_summary"])
    ag_score = pd.read_csv(paths["autogluon_all_score"])
    ag_speed = pd.read_csv(paths["autogluon_all_speed"])
    h2o_all = load_json(paths["h2o_all_summary"])
    h2o_top = load_json(paths["h2o_top_summary"])
    h2o_score = pd.read_csv(paths["h2o_all_score"])
    h2o_speed = pd.read_csv(paths["h2o_all_speed"])
    comparison = pd.read_csv(paths["final_comparison"])
    recommendation = load_json(paths["recommendation"])
    overlap = load_json(paths["feature_overlap"])
    assignment1 = load_yaml(paths["assignment1_baseline"])["assignment1_baseline"]

    best = recommendation["best_predictive_run"]
    rows_after = quality.get("rows_after", quality.get("clean_rows", 53505))
    rows_removed = quality.get("rows_removed", 29)

    md_path = ROOT / outputs["high_level_report_markdown"]
    pdf_path = ROOT / outputs["high_level_report_pdf"]
    md_path.parent.mkdir(parents=True, exist_ok=True)

    md_path.write_text(
        f"""# Assignment 3 - AutoML High-Level Report

**Course:** {assignment['course']}  
**Author:** {assignment['author']}  
**Primary platform:** {assignment['primary_platform']}  
**Repository:** {assignment.get('repository_url', 'N/A')}

## Executive Summary

The strongest run by validation RMSE was **{best['platform']} -
{best['feature_set']}**, using model `{best['best_model']}`.

- Validation RMSE: {fmt(best['validation_rmse'])}
- Test RMSE: {fmt(best['test_rmse'])}
- Test MAE: {fmt(best['test_mae'])}
- Test R-squared: {fmt(best['test_r2'])}

## Dataset

- Raw rows: 423,006
- Final modeling rows: {rows_after}
- Target-quality rows removed: {rows_removed}
- Approved features: {features['feature_count']}
- Split: 64% train / 16% validation / 20% test
- Random seed: 42

## Comparison

{comparison.to_markdown(index=False)}

## Top Features

- AutoGluon: {', '.join(ag_all.get('top_five_features', []))}
- H2O: {', '.join(h2o_all.get('top_five_features', []))}

## Assignment 1 Baseline

Assignment 1 Dataset v2 Random Forest reported test RMSE
{fmt(assignment1.get('test_rmse'))}, test MAE
{fmt(assignment1.get('test_mae'))}, and test R-squared
{fmt(assignment1.get('test_r2'))}. It used a different processed dataset and
80/20 split, and did not retain directly comparable validation or runtime evidence.

## Recommendation

Select the model primarily by validation RMSE and confirm performance on the
untouched test set. Prefer a reduced-feature run only when its validation
performance remains within the documented tolerance and operational simplicity is
valuable.
""",
        encoding="utf-8",
    )

    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "TitleX",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=28,
            textColor=colors.HexColor("#17365D"),
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleX",
            parent=base["Normal"],
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#465A6C"),
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "H1X",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#1F4E78"),
            spaceBefore=8,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2X",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#2F75B5"),
            spaceBefore=5,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BodyX",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.3,
            leading=13,
            textColor=colors.HexColor("#263238"),
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "BulletX",
            parent=base["BodyText"],
            fontSize=9,
            leading=12,
            leftIndent=13,
            firstLineIndent=-8,
            spaceAfter=3,
        ),
    }

    doc = BaseDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.7 * inch,
        title="Assignment 3 AutoML High-Level Report",
        author=assignment["author"],
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=footer))

    story: list[Any] = [
        Spacer(1, 0.2 * inch),
        Paragraph("Assignment 3 - AutoML", styles["title"]),
        Paragraph(
            "Athlete Total-Lift Prediction with AutoGluon, MLflow, and H2O AutoML",
            styles["subtitle"],
        ),
    ]

    metadata = [
        ["Course", assignment["course"]],
        ["Author", assignment["author"]],
        ["Primary platform", assignment["primary_platform"]],
        ["Platform mode", assignment["primary_platform_mode"]],
        ["Repository", assignment.get("repository_url", "UPDATE_WITH_GITHUB_URL")],
    ]
    meta_table = Table(metadata, colWidths=[1.45 * inch, 5.35 * inch])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#DCE6F1")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAB7C4")),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story += [
        meta_table,
        Spacer(1, 0.18 * inch),
        Paragraph("Executive Summary", styles["h1"]),
        Paragraph(
            "This project evaluates AutoGluon with MLflow and H2O AutoML using "
            "identical cleaned train, validation, and test partitions. Each platform "
            "is run with all approved features and its top three features.",
            styles["body"],
        ),
    ]

    metric_rows = [
        ["Recommended run", f"{best['platform']} - {best['feature_set']}"],
        ["Best model", shorten(best["best_model"], 68)],
        ["Validation RMSE", fmt(best["validation_rmse"])],
        ["Test RMSE", fmt(best["test_rmse"])],
        ["Test MAE", fmt(best["test_mae"])],
        ["Test R-squared", fmt(best["test_r2"])],
    ]
    story.append(styled_table([["Metric", "Result"], *metric_rows], [2.0 * inch, 4.8 * inch], 8.5))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("Key conclusion", styles["h2"]))
    story.append(
        Paragraph(
            "The final recommendation is selected primarily by validation RMSE and "
            "confirmed using the untouched test set. Speed, feature count, "
            "interpretability, and deployment complexity are secondary selection factors.",
            styles["body"],
        )
    )
    story.append(PageBreak())

    story += [
        Paragraph("1. Dataset and Reproducible Design", styles["h1"]),
        styled_table(
            [
                ["Item", "Value", "Item", "Value"],
                ["Raw rows", "423,006", "Final rows", f"{int(rows_after):,}"],
                ["Rows removed", f"{int(rows_removed):,}", "Approved features", str(features["feature_count"])],
                ["Train rows", "34,243", "Validation rows", "8,561"],
                ["Test rows", "10,701", "Target range", "8 to 2,330"],
                ["Random seed", "42", "Primary metric", "Validation RMSE"],
            ],
            [1.2 * inch, 1.5 * inch, 1.3 * inch, 2.6 * inch],
            8.2,
        ),
        Spacer(1, 0.12 * inch),
        Paragraph("Quality and leakage controls", styles["h2"]),
    ]
    for item in [
        "The target is the sum of deadlift, clean and jerk, snatch, and back squat.",
        "Target components, total_lift, and athlete identifiers are excluded from predictors.",
        "Sentinel lift value 1 is treated as missing before target construction.",
        "Twenty-nine corrupted target rows are audited and removed rather than capped.",
        "Features above 70% missingness are excluded before modeling.",
        "The same fixed 64/16/20 partitions are used across all current experiments.",
    ]:
        story.append(Paragraph(item, styles["bullet"], bulletText="-"))

    story += [
        Paragraph("Workflow overview", styles["h2"]),
        styled_table(
            [
                ["Stage", "Purpose"],
                ["1", "Profile and validate the athlete dataset"],
                ["2", "Build leakage-safe features and fixed data partitions"],
                ["3", "Run AutoGluon with all features and top three features"],
                ["4", "Repeat the workflow with H2O AutoML"],
                ["5", "Train conventional baselines and compare platforms"],
                ["6", "Generate reports, validate evidence, and package submission"],
            ],
            [0.55 * inch, 6.25 * inch],
            8.5,
        ),
        PageBreak(),
        Paragraph("2. AutoGluon Primary Workflow", styles["h1"]),
        Paragraph(
            "AutoGluon is classified as full-code AutoML. It automates model "
            "training, preprocessing, algorithm selection, hyperparameter exploration, "
            "leaderboard ranking, and ensembling where supported. Python and YAML "
            "still control data quality, feature eligibility, validation design, "
            "runtime budgets, tracking, and final approval.",
            styles["body"],
        ),
    ]

    ag_summary_rows = [
        ["Feature set", "Count", "Best model", "Val. RMSE", "Test RMSE", "Test R2"],
        [
            "All",
            str(ag_all["feature_count"]),
            shorten(ag_all["best_model"], 30),
            fmt(ag_all["validation_rmse"]),
            fmt(ag_all["test_rmse"]),
            fmt(ag_all["test_r2"]),
        ],
        [
            "Top 3",
            str(ag_top["feature_count"]),
            shorten(ag_top["best_model"], 30),
            fmt(ag_top["validation_rmse"]),
            fmt(ag_top["test_rmse"]),
            fmt(ag_top["test_r2"]),
        ],
    ]
    story.append(styled_table(ag_summary_rows, [0.65*inch,0.5*inch,2.3*inch,0.9*inch,0.9*inch,0.8*inch]))
    story += [
        Paragraph("Top three by validation RMSE - all features", styles["h2"]),
        styled_table(
            leaderboard_rows(
                ag_score,
                "model",
                "validation_rmse",
                "fit_time_marginal" if "fit_time_marginal" in ag_score.columns else "fit_time",
            ),
            [0.45 * inch, 3.65 * inch, 1.25 * inch, 1.2 * inch],
        ),
        Paragraph("Top three by speed - all features", styles["h2"]),
        styled_table(
            leaderboard_rows(
                ag_speed,
                "model",
                "validation_rmse",
                ag_all.get("speed_measure", "fit_time_marginal"),
            ),
            [0.45 * inch, 3.65 * inch, 1.25 * inch, 1.2 * inch],
        ),
        Paragraph(
            "Top five AutoGluon features: <b>"
            + ", ".join(ag_all.get("top_five_features", []))
            + "</b>.",
            styles["body"],
        ),
        Paragraph(
            "Top-three-feature run: <b>"
            + ", ".join(ag_top.get("model_features", []))
            + "</b>.",
            styles["body"],
        ),
        PageBreak(),
        Paragraph("3. H2O AutoML and Baseline Context", styles["h1"]),
    ]

    h2o_summary_rows = [
        ["Feature set", "Count", "Best model", "Val. RMSE", "Test RMSE", "Test R2"],
        [
            "All",
            str(h2o_all["feature_count"]),
            shorten(h2o_all["best_model"], 30),
            fmt(h2o_all["validation_rmse"]),
            fmt(h2o_all["test_rmse"]),
            fmt(h2o_all["test_r2"]),
        ],
        [
            "Top 3",
            str(h2o_top["feature_count"]),
            shorten(h2o_top["best_model"], 30),
            fmt(h2o_top["validation_rmse"]),
            fmt(h2o_top["test_rmse"]),
            fmt(h2o_top["test_r2"]),
        ],
    ]
    story.append(styled_table(h2o_summary_rows, [0.65*inch,0.5*inch,2.3*inch,0.9*inch,0.9*inch,0.8*inch]))
    story += [
        Paragraph("Top three by validation RMSE - all features", styles["h2"]),
        styled_table(
            leaderboard_rows(h2o_score, "model_id", "rmse", "training_time_ms"),
            [0.45 * inch, 3.65 * inch, 1.25 * inch, 1.2 * inch],
        ),
        Paragraph("Top three by training speed - all features", styles["h2"]),
        styled_table(
            leaderboard_rows(h2o_speed, "model_id", "rmse", "training_time_ms"),
            [0.45 * inch, 3.65 * inch, 1.25 * inch, 1.2 * inch],
        ),
        Paragraph(
            "Top five H2O features: <b>"
            + ", ".join(h2o_all.get("top_five_features", []))
            + "</b>.",
            styles["body"],
        ),
        Paragraph("Assignment 1 historical baseline", styles["h2"]),
        styled_table(
            [
                ["Model", "Dataset", "Split", "Test RMSE", "Test MAE", "Test R2"],
                [
                    assignment1["model_name"],
                    assignment1["feature_or_dataset_version"],
                    assignment1["split_strategy"],
                    fmt(assignment1["test_rmse"]),
                    fmt(assignment1["test_mae"]),
                    fmt(assignment1["test_r2"]),
                ],
            ],
            [1.2*inch,1.2*inch,1.4*inch,0.9*inch,0.85*inch,0.8*inch],
            7.2,
        ),
        Spacer(1, 0.08 * inch),
        Paragraph(
            "Assignment 1 used a different processed dataset and 80/20 split, and "
            "did not retain separate validation or reliable runtime evidence. Its "
            "metrics are historical context. The Phase 4 same-split Random Forest "
            "is the controlled current-workflow baseline.",
            styles["body"],
        ),
        PageBreak(),
        Paragraph("4. Cross-Platform Comparison and Recommendation", styles["h1"]),
    ]

    comp_rows = [["Platform", "Features", "Count", "Model", "Val. RMSE", "Test RMSE", "Test R2", "Train sec."]]
    for _, row in comparison.iterrows():
        comp_rows.append(
            [
                shorten(row["platform"], 16),
                shorten(row["feature_set"], 15),
                str(int(row["feature_count"])),
                shorten(row["best_model"], 25),
                fmt(row["validation_rmse"]),
                fmt(row["test_rmse"]),
                fmt(row["test_r2"]),
                fmt(row["training_seconds"]),
            ]
        )
    story.append(
        styled_table(
            comp_rows,
            [0.82*inch,0.72*inch,0.4*inch,1.7*inch,0.8*inch,0.8*inch,0.65*inch,0.75*inch],
            6.8,
        )
    )

    for key, title in [
        ("test_rmse_plot", "Test RMSE comparison"),
        ("training_time_plot", "Training-time comparison"),
    ]:
        path = paths.get(key)
        if path and path.exists():
            story.append(Paragraph(title, styles["h2"]))
            image = Image(str(path))
            image._restrictSize(6.6 * inch, 2.25 * inch)
            story.append(image)

    shared = overlap.get("autogluon_vs_h2o_top_five", {})
    story += [
        Paragraph("Recommendation", styles["h2"]),
        Paragraph(
            f"The strongest run by validation RMSE is <b>{best['platform']} - "
            f"{best['feature_set']}</b>, model <b>{shorten(best['best_model'], 72)}</b>. "
            f"It achieved test RMSE <b>{fmt(best['test_rmse'])}</b>, test MAE "
            f"<b>{fmt(best['test_mae'])}</b>, and test R-squared "
            f"<b>{fmt(best['test_r2'])}</b>.",
            styles["body"],
        ),
        Paragraph(
            f"AutoGluon and H2O shared {shared.get('shared_feature_count', 0)} "
            "of their top five features. Feature importance is model-dependent "
            "and does not imply causality.",
            styles["body"],
        ),
        Paragraph("Operational implications", styles["h2"]),
    ]
    for item in [
        "AutoML reduces manual model selection and tuning effort.",
        "Ensembles can improve accuracy while increasing deployment complexity.",
        "Reduced-feature models simplify scoring and monitoring when validation quality is maintained.",
        "Runtime comparisons depend on hardware and platform measurement definitions.",
        "Production use still requires schema, quality, drift, latency, and cost monitoring.",
    ]:
        story.append(Paragraph(item, styles["bullet"], bulletText="-"))

    story += [
        Paragraph("Reproducibility", styles["h2"]),
        Paragraph(
            "The repository includes Python 3.11 dependency files, YAML "
            "configurations, fixed random seeds, deterministic partitions, automated "
            "experiment scripts, MLflow tracking, saved leaderboards, tests, a "
            "submission validator, and a clean ZIP builder. Run "
            "<b>python scripts/finalize_submission.py</b> after all experiments pass.",
            styles["body"],
        ),
    ]

    doc.build(story)
    print(f"Markdown report: {md_path}")
    print(f"PDF report: {pdf_path}")
    print("HIGH-LEVEL REPORT STATUS: PASS")


if __name__ == "__main__":
    main()
