#!/usr/bin/env python3

import re
import math
import argparse
from pathlib import Path

import pysam
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


def setup_matplotlib():
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    })


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality stacked coverage tracks around focal bacterial mutations."
    )
    parser.add_argument(
        "--bam",
        nargs="+",
        required=True,
        help="Sorted BAM files with indexes."
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional sample names in the same order as BAM files."
    )
    parser.add_argument(
        "--mutation-table",
        required=True,
        help="Mutation table TSV/CSV with columns including POS, REF, ALT, Gene, AA_CHANGE, etc."
    )
    parser.add_argument(
        "--vcf",
        default=None,
        help="Optional SnpEff-annotated VCF for fallback annotation parsing."
    )
    parser.add_argument(
        "--contig",
        default="NC_003197.2",
        help="Reference contig name."
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        type=int,
        default=[357924, 357925],
        help="1-based focal mutation positions."
    )
    parser.add_argument(
        "--window",
        type=int,
        default=700,
        help="Half-window size in bp around focal mutations."
    )
    parser.add_argument(
        "--output-prefix",
        default="mutation_evidence_NC_003197_2_357924_357925",
        help="Output file prefix."
    )
    return parser.parse_args()


def detect_delimiter(path):
    ext = Path(path).suffix.lower()
    if ext in [".tsv", ".txt"]:
        return "\t"
    if ext == ".csv":
        return ","
    with open(path, "r", encoding="utf-8") as fh:
        first = fh.readline()
    if first.count("\t") >= first.count(","):
        return "\t"
    return ","


def load_mutation_table(path):
    sep = detect_delimiter(path)
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    return df


def normalize_aa_change(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    s = s.replace("p.", "")
    s = s.replace(" ", "")
    return s


def format_aa_label(val):
    if val is None:
        return ""
    s = str(val).strip()
    if not s:
        return ""
    if s.startswith("p."):
        return s
    if re.match(r"^[A-Z][a-z]{2}\d+[A-Z][a-z]{2}$", s):
        return f"p.{s}"
    if re.match(r"^[A-Z]\d+[A-Z*]$", s):
        return s
    return s


def parse_ann_field(info_str, alt=None):
    if info_str is None or pd.isna(info_str):
        return None
    info_str = str(info_str)
    ann_match = re.search(r"(?:^|;)ANN=([^;]+)", info_str)
    if not ann_match:
        return None

    ann_entries = ann_match.group(1).split(",")
    parsed = []
    for ann in ann_entries:
        fields = ann.split("|")
        if len(fields) < 11:
            continue
        allele = fields[0]
        effect = fields[1]
        gene = fields[3]
        hgvs_c = fields[9] if len(fields) > 9 else ""
        hgvs_p = fields[10] if len(fields) > 10 else ""
        parsed.append({
            "allele": allele,
            "effect": effect,
            "gene": gene,
            "hgvs_c": hgvs_c,
            "hgvs_p": hgvs_p
        })

    if alt is not None:
        alt = str(alt)
        alt_hits = [x for x in parsed if x["allele"] == alt]
        if alt_hits:
            parsed = alt_hits

    for p in parsed:
        if "missense_variant" in p["effect"]:
            return p
    return parsed[0] if parsed else None


def load_vcf_annotations(vcf_path, contig, positions):
    ann = {}
    if vcf_path is None:
        return ann

    vf = pysam.VariantFile(vcf_path)
    for rec in vf.fetch(contig, min(positions) - 1, max(positions)):
        if rec.pos not in positions:
            continue

        info_dict = dict(rec.info)
        ann_value = info_dict.get("ANN", None)

        if ann_value:
            entries = ann_value if isinstance(ann_value, (list, tuple)) else [ann_value]
            for alt in rec.alts:
                chosen = None
                for item in entries:
                    fields = str(item).split("|")
                    if len(fields) < 11:
                        continue
                    if fields[0] != alt:
                        continue
                    candidate = {
                        "allele": fields[0],
                        "effect": fields[1],
                        "gene": fields[3],
                        "hgvs_c": fields[9] if len(fields) > 9 else "",
                        "hgvs_p": fields[10] if len(fields) > 10 else ""
                    }
                    if chosen is None or "missense_variant" in candidate["effect"]:
                        chosen = candidate
                        if "missense_variant" in candidate["effect"]:
                            break
                if chosen:
                    ann[(rec.pos, alt)] = chosen
    vf.close()
    return ann


def choose_annotation_rows(df, contig, positions):
    df2 = df.copy()

    if "CHROM" in df2.columns:
        df2 = df2[df2["CHROM"].astype(str) == str(contig)]
    elif "Contig" in df2.columns:
        df2 = df2[df2["Contig"].astype(str) == str(contig)]
    elif "Reference" in df2.columns:
        df2 = df2[df2["Reference"].astype(str) == str(contig)]

    df2 = df2[df2["POS"].isin(positions)].copy()

    if df2.empty:
        raise ValueError("No rows found in mutation table for requested contig/positions.")

    return df2

    selected = []
    for pos in positions:
        sub = df2[df2["POS"] == pos].copy()
        if sub.empty:
            continue

        if "AF_decimal" in sub.columns:
            sub["AF_decimal"] = pd.to_numeric(sub["AF_decimal"], errors="coerce")
            sub = sub.sort_values("AF_decimal", ascending=False)
        elif "ALT_count" in sub.columns:
            sub["ALT_count"] = pd.to_numeric(sub["ALT_count"], errors="coerce")
            sub = sub.sort_values("ALT_count", ascending=False)

        selected.append(sub.iloc[0].to_dict())

    out = pd.DataFrame(selected)
    if out.empty:
        raise ValueError("Could not select mutation annotations for focal positions.")
    return out


def unique_preserve_order(values):
    seen = set()
    out = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
    
def build_mutation_annotations(mut_df, vcf_ann):
    muts = []

    for pos, sub in mut_df.groupby("POS", sort=True):
        sub = sub.copy()

        ref_vals = unique_preserve_order(sub["REF"]) if "REF" in sub.columns else []
        gene_vals = unique_preserve_order(sub["Gene"]) if "Gene" in sub.columns else []

        ref = ref_vals[0] if ref_vals else "N"

        alt_aa_pairs = []
        for _, row in sub.iterrows():
            alt = str(row["ALT"]).strip() if "ALT" in row and not pd.isna(row["ALT"]) else ""
            aa = None

            if "AA_CHANGE" in row:
                aa = normalize_aa_change(row["AA_CHANGE"])
                aa = format_aa_label(aa) if aa else ""

            if not aa and (int(pos), alt) in vcf_ann:
                aa = vcf_ann[(int(pos), alt)].get("hgvs_p", "")
                aa = format_aa_label(aa) if aa else ""

            if alt:
                alt_aa_pairs.append((alt, aa))

        seen = set()
        unique_pairs = []
        for alt, aa in alt_aa_pairs:
            key = (alt, aa)
            if key not in seen:
                seen.add(key)
                unique_pairs.append((alt, aa))

        if not gene_vals:
            vcf_genes = []
            for alt, _ in unique_pairs:
                if (int(pos), alt) in vcf_ann:
                    g = vcf_ann[(int(pos), alt)].get("gene", "")
                    if g and g not in vcf_genes:
                        vcf_genes.append(g)
            gene_vals = vcf_genes

        muts.append({
            "pos": int(pos),
            "ref": ref,
            "genes": gene_vals,
            "alt_aa_pairs": unique_pairs
        })

    muts = sorted(muts, key=lambda x: x["pos"])
    return muts



def sample_name_from_bam(path):
    base = Path(path).name
    for ext in [".sorted.bam", ".bam"]:
        if base.endswith(ext):
            return base[:-len(ext)]
    return Path(path).stem


def get_sample_names(bams, supplied=None):
    if supplied:
        if len(supplied) != len(bams):
            raise ValueError("If --samples is provided, it must match the number of BAMs.")
        return supplied
    return [sample_name_from_bam(b) for b in bams]


def extract_coverage(bam_path, contig, start0, end0, quality_threshold=0):
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        a, c, g, t = bam.count_coverage(
            contig=contig,
            start=start0,
            stop=end0,
            quality_threshold=quality_threshold,
            read_callback="all"
        )
    cov = np.array(a) + np.array(c) + np.array(g) + np.array(t)
    return cov


def add_mutation_lines(ax, mutations):
    for mut in mutations:
        ax.axvline(
            mut["pos"],
            color="#b2182b",
            linestyle=(0, (3, 2)),
            linewidth=1.0,
            alpha=0.95,
            zorder=4
        )


def make_label(mut):
    gene_text = ", ".join(mut["genes"]) if mut.get("genes") else "NA"

    line1 = f"{mut['pos']}"
    line2 = gene_text

    pair_lines = []
    for alt, aa in mut.get("alt_aa_pairs", []):
        if aa:
            pair_lines.append(f"{mut['ref']}→{alt}   {aa}")
        else:
            pair_lines.append(f"{mut['ref']}→{alt}")

    if pair_lines:
        return line1 + "\n" + line2 + "\n" + "\n".join(pair_lines)
    else:
        return line1 + "\n" + line2


def annotate_top_track(ax, mutations, y_top):
    if len(mutations) == 1:
        offsets = [0]
    elif len(mutations) == 2:
        offsets = [-55, 55]
    else:
        offsets = np.linspace(-80, 80, len(mutations))

    for mut, dx in zip(mutations, offsets):
        label = make_label(mut)
        ax.annotate(
            label,
            xy=(mut["pos"], y_top * 0.98),
            xycoords="data",
            xytext=(mut["pos"] + dx, y_top * 1.14),
            textcoords="data",
            ha="center",
            va="bottom",
            fontsize=8,
            linespacing=1.1,
            bbox=dict(
                boxstyle="round,pad=0.25,rounding_size=0.15",
                facecolor="white",
                edgecolor="#7f7f7f",
                linewidth=0.6
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="#7f7f7f",
                linewidth=0.7,
                shrinkA=0,
                shrinkB=4
            ),
            zorder=6,
            clip_on=False
        )


def plot_coverage_figure(
    coverages,
    sample_names,
    positions,
    contig,
    mutations,
    out_prefix
):
    genomic_x = np.arange(positions["start1"], positions["end1"] + 1)
    n = len(sample_names)

    fig_h = 1.5 + n * 1.55
    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=(9.2, fig_h),
        sharex=True,
        constrained_layout=False
    )
    if n == 1:
        axes = [axes]

    track_color = "#4C78A8"
    line_color = "#2F5D8A"

    top_track_ymax = None

    for idx, (ax, sample) in enumerate(zip(axes, sample_names)):
        cov = coverages[sample]
        local_max = max(cov) if len(cov) else 0
        local_ymax = max(10, math.ceil(local_max * 1.20))

        ax.fill_between(genomic_x, cov, color=track_color, alpha=0.85, linewidth=0)
        ax.plot(genomic_x, cov, color=line_color, linewidth=0.8, zorder=3)

        add_mutation_lines(ax, mutations)

        ax.set_ylim(0, local_ymax)
        ax.set_ylabel(sample, rotation=0, ha="right", va="center", labelpad=28)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.grid(False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#666666")
        ax.spines["bottom"].set_color("#666666")
        ax.set_facecolor("white")

        if idx == 0:
            top_track_ymax = local_ymax
            annotate_top_track(ax, mutations, top_track_ymax)

    axes[-1].set_xlabel(f"{contig} genomic position (bp)")
    axes[-1].set_xlim(positions["start1"], positions["end1"])

    title = (
        f"Sequencing evidence for adjacent missense mutations on {contig}\n"
        f"Positions {mutations[0]['pos']} and {mutations[1]['pos']}"
        if len(mutations) == 2
        else f"Sequencing evidence for focal mutations on {contig}"
    )
    fig.suptitle(title, y=0.995, fontsize=11, fontweight="bold")

    legend_handles = [
        Line2D([0], [0], color=line_color, lw=1.2, label="Coverage"),
        Line2D([0], [0], color="#b2182b", lw=1.0, linestyle=(0, (3, 2)), label="Mutation site")
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        bbox_to_anchor=(0.0, 1.34),
        ncol=2,
        handlelength=2.5,
        columnspacing=1.4
    )

    plt.subplots_adjust(left=0.12, right=0.985, top=0.82, bottom=0.12, hspace=0.16)

    pdf_path = f"{out_prefix}.pdf"
    svg_path = f"{out_prefix}.svg"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return pdf_path, svg_path


def main():
    args = parse_args()
    setup_matplotlib()

    sample_names = get_sample_names(args.bam, args.samples)
    mut_table = load_mutation_table(args.mutation_table)
    vcf_ann = load_vcf_annotations(args.vcf, args.contig, args.positions) if args.vcf else {}

    mut_rows = choose_annotation_rows(mut_table, args.contig, args.positions)
    mutations = build_mutation_annotations(mut_rows, vcf_ann)

    if not mutations:
        raise ValueError("No focal mutations were found after parsing the mutation table.")

    focal_min = min(args.positions)
    focal_max = max(args.positions)
    start1 = max(1, focal_min - args.window)
    end1 = focal_max + args.window
    start0 = start1 - 1
    end0 = end1

    coverages = {}
    for bam, sample in zip(args.bam, sample_names):
        coverages[sample] = extract_coverage(
            bam,
            args.contig,
            start0,
            end0,
            quality_threshold=0
        )

    positions = {"start1": start1, "end1": end1}
    pdf_path, svg_path = plot_coverage_figure(
        coverages=coverages,
        sample_names=sample_names,
        positions=positions,
        contig=args.contig,
        mutations=mutations,
        out_prefix=args.output_prefix
    )

    print("Generated:")
    print(pdf_path)
    print(svg_path)


if __name__ == "__main__":
    main()