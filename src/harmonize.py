"""
AI-Augmented Transdiagnostic Pleiotropy & Discordance Explorer for PGC Psychiatric GWAS.

Data attribution notice:
This module analyzes CC-BY 4.0 data from the OpenMed_AI PGC collection on Hugging Face,
derived from Psychiatric Genomics Consortium (PGC) summary statistics. Users must cite
OpenMed_AI, the PGC, and the original PGC publication/config used for each trait.
"""

from __future__ import annotations

import heapq
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import polars as pl
import requests
from datasets import get_dataset_config_names, load_dataset
from scipy.stats import combine_pvalues
from tqdm.auto import tqdm

ATTRIBUTION_NOTICE = (
    "Data source: OpenMed_AI PGC collection on Hugging Face (CC-BY 4.0), "
    "derived from Psychiatric Genomics Consortium (PGC) summary statistics. "
    "Cite OpenMed_AI, PGC, and the original PGC publications/configs."
)

DATASETS_SERVER_URL = "https://datasets-server.huggingface.co"
GENOME_WIDE_SIGNIFICANCE = 5e-8

_CANONICAL_COLUMNS = {
    "snp": {"snp", "rsid", "rs_id", "id", "markername", "variant_id"},
    "chr": {"chr", "chromosome", "#chrom", "chrom"},
    "pos": {"bp", "pos", "position", "base_pair_location", "bp_hg19"},
    "a1": {"a1", "alt", "effect_allele", "allele1", "ea"},
    "a2": {"a2", "ref", "other_allele", "allele2", "nea"},
    "beta": {"beta", "effect", "log_odds", "estimate"},
    "or": {"or", "odds_ratio"},
    "se": {"se", "stderr", "standard_error", "sebeta"},
    "p": {"p", "p_value", "p-value", "pval"},
    "info": {"info", "imputation_info"},
    "eaf": {"eaf", "maf", "freq", "frq", "effect_allele_frequency"},
    "n": {"n", "neff", "n_total"},
    "ncases": {"nca", "ncases", "n_cases"},
    "ncontrols": {"nco", "ncontrols", "n_controls"},
    "z": {"z", "zscore"},
    "source_file": {"_source_file", "source_file"},
}

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


@dataclass(frozen=True)
class TraitSpec:
    disorder: str
    label: str
    dataset_id: str
    preferred_config: str
    fallbacks: tuple[str, ...] = ()
    note: str = ""
    exclude_parquet_suffixes: tuple[str, ...] = ()


def default_trait_specs() -> list[TraitSpec]:
    """Return the six core traits used in the notebook."""
    return [
        TraitSpec(
            disorder="SCZ",
            label="Schizophrenia",
            dataset_id="OpenMed/pgc-schizophrenia",
            preferred_config="scz2022",
            fallbacks=("scz2018clozuk", "scz2014"),
            note="Latest flagship schizophrenia config in the OpenMed collection.",
        ),
        TraitSpec(
            disorder="BIP",
            label="Bipolar disorder",
            dataset_id="OpenMed/pgc-bipolar",
            preferred_config="bip2024",
            fallbacks=("bip2021", "bip2019"),
            note="Latest bipolar config in the OpenMed collection.",
        ),
        TraitSpec(
            disorder="MDD",
            label="Major depressive disorder",
            dataset_id="OpenMed/pgc-mdd",
            preferred_config="mdd2025",
            fallbacks=("mdd2023diverse", "mdd2018"),
            note=(
                "The `mdd2025` parquet listing contains one metadata shard named "
                "`0000.parquet`; this module excludes that shard automatically."
            ),
            exclude_parquet_suffixes=("0000.parquet",),
        ),
        TraitSpec(
            disorder="ADHD",
            label="ADHD",
            dataset_id="OpenMed/pgc-adhd",
            preferred_config="adhd2022",
            fallbacks=("adhd2019",),
            note="Latest ADHD config in the OpenMed collection.",
        ),
        TraitSpec(
            disorder="ASD",
            label="Autism spectrum disorder",
            dataset_id="OpenMed/pgc-autism",
            preferred_config="asd2019",
            fallbacks=("asd2017",),
            note=(
                "As of 2026-04-07, `asd2019` is the latest Autism config exposed in "
                "the OpenMed collection."
            ),
        ),
        TraitSpec(
            disorder="PTSD",
            label="PTSD",
            dataset_id="OpenMed/pgc-ptsd",
            preferred_config="ptsd2024",
            fallbacks=("ptsd2019", "ptsd2018"),
            note=(
                "As of 2026-04-07, `ptsd2024` exists in the collection but currently "
                "previews as VCF-header metadata rather than analyzable GWAS rows. "
                "This module therefore falls back to `ptsd2019`."
            ),
        ),
    ]


def trait_table(traits: Iterable[TraitSpec]) -> pd.DataFrame:
    """Inspect available configs and return the resolved runtime selection."""
    rows: list[dict[str, Any]] = []
    for trait in traits:
        resolved = resolve_trait_config(trait)
        rows.append(
            {
                "trait": trait.disorder,
                "label": trait.label,
                "dataset_id": trait.dataset_id,
                "preferred_config": trait.preferred_config,
                "selected_config": resolved["selected_config"],
                "available_configs": ", ".join(resolved["available_configs"]),
                "preview_columns": ", ".join(resolved["preview_columns"]),
                "selection_reason": resolved["selection_reason"],
                "note": trait.note,
            }
        )
    return pd.DataFrame(rows)


def resolve_trait_config(trait: TraitSpec) -> dict[str, Any]:
    """Choose a usable config by previewing real rows through `datasets`."""
    config_lookup_note = ""
    try:
        available_configs = get_dataset_config_names(trait.dataset_id)
    except Exception as exc:  # pragma: no cover - notebook safety path
        available_configs = [trait.preferred_config, *trait.fallbacks]
        config_lookup_note = f"config lookup fell back to curated defaults because dataset listing failed ({exc})"
    seen: set[str] = set()
    curated_candidates = [trait.preferred_config, *trait.fallbacks]
    candidates = [candidate for candidate in curated_candidates if candidate in available_configs]
    if not candidates:
        candidates = curated_candidates
    inspected: list[str] = []

    for config in candidates:
        if config in seen:
            continue
        seen.add(config)

        preview_columns: list[str] = []
        try:
            preview_rows = fetch_preview_rows(trait.dataset_id, config)
            preview_columns = list(preview_rows[0].keys()) if preview_rows else []
            if is_summary_stat_schema(preview_columns):
                reason = (
                    "preferred config is usable"
                    if config == trait.preferred_config
                    else f"fell back after previewing malformed or unusable schemas: {'; '.join(inspected)}"
                )
                if config_lookup_note:
                    reason = f"{reason}; {config_lookup_note}"
                return {
                    "available_configs": available_configs,
                    "selected_config": config,
                    "preview_columns": preview_columns,
                    "selection_reason": reason,
                }
            inspected.append(f"{config}: first streamed rows were not GWAS summary rows")
        except Exception as exc:  # pragma: no cover - notebook safety path
            inspected.append(f"{config}: preview error ({exc})")

        try:
            parquet_columns = fetch_preview_columns_from_parquet(
                dataset_id=trait.dataset_id,
                config=config,
                exclude_suffixes=trait.exclude_parquet_suffixes,
            )
            if is_summary_stat_schema(parquet_columns):
                reason = (
                    "preferred config is usable after parquet schema inspection"
                    if config == trait.preferred_config
                    else f"fell back after preview/parquet inspection: {'; '.join(inspected)}"
                )
                if config_lookup_note:
                    reason = f"{reason}; {config_lookup_note}"
                return {
                    "available_configs": available_configs,
                    "selected_config": config,
                    "preview_columns": parquet_columns,
                    "selection_reason": reason,
                }
            inspected.append(f"{config}: parquet schema missing GWAS columns")
        except Exception as exc:  # pragma: no cover - notebook safety path
            inspected.append(f"{config}: parquet schema error ({exc})")

    raise RuntimeError(
        f"Could not find a usable config for {trait.dataset_id}. "
        f"Preview attempts: {'; '.join(inspected)}"
    )


def fetch_preview_rows(dataset_id: str, config: str, n_rows: int = 2) -> list[dict[str, Any]]:
    """Preview rows via `datasets` streaming without materializing the full table."""
    dataset = load_dataset(dataset_id, config, split="train", streaming=True)
    preview: list[dict[str, Any]] = []
    iterator = iter(dataset)
    for _ in range(n_rows):
        try:
            preview.append(dict(next(iterator)))
        except StopIteration:
            break
    return preview


def is_summary_stat_schema(columns: Iterable[str]) -> bool:
    """Check whether the preview looks like a usable GWAS summary-stat table."""
    mapping = infer_column_map(columns)
    has_variant = bool(mapping.get("snp") or (mapping.get("chr") and mapping.get("pos")))
    has_effect = bool(mapping.get("beta") or mapping.get("or") or mapping.get("z"))
    has_p = bool(mapping.get("p"))
    return has_variant and has_effect and has_p


def infer_column_map(columns: Iterable[str]) -> dict[str, str]:
    """Infer canonical GWAS column roles from heterogeneous source schemas."""
    resolved: dict[str, str] = {}
    columns = list(columns)
    lowered = {column.lower(): column for column in columns}

    for canonical, aliases in _CANONICAL_COLUMNS.items():
        for alias in aliases:
            if alias in lowered:
                resolved[canonical] = lowered[alias]
                break

    for column in columns:
        lower = column.lower()
        if "frq_a_" in lower or lower.startswith("freq1") or lower.endswith("_frequency"):
            resolved.setdefault("eaf", column)
        if lower.startswith("frq_u_"):
            resolved.setdefault("control_freq", column)
        if lower.startswith("heterogeneity") and lower.endswith("_p_value"):
            resolved.setdefault("heterogeneity_p", column)
        if lower in {"weight"} and "n" not in resolved:
            resolved["n"] = column
        if lower.startswith("##fileformat"):
            resolved.setdefault("header_only", column)

    return resolved


def get_parquet_urls(
    dataset_id: str,
    config: str,
    split: str = "train",
    exclude_suffixes: Iterable[str] = (),
) -> list[str]:
    """Return converted parquet shard URLs for a dataset/config pair."""
    response = requests.get(
        f"{DATASETS_SERVER_URL}/parquet",
        params={"dataset": dataset_id, "config": config, "split": split},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    exclude_suffixes = tuple(exclude_suffixes)

    urls = []
    for item in payload.get("parquet_files", []):
        url = item["url"]
        if exclude_suffixes and url.endswith(exclude_suffixes):
            continue
        urls.append(url)
    return urls


def fetch_preview_columns_from_parquet(
    dataset_id: str,
    config: str,
    exclude_suffixes: Iterable[str] = (),
    max_files: int = 3,
) -> list[str]:
    """
    Inspect parquet schemas directly.

    This is more robust than streaming the first row because some OpenMed configs
    begin with a metadata shard before the real summary-stat shards.
    """
    urls = get_parquet_urls(
        dataset_id=dataset_id,
        config=config,
        exclude_suffixes=exclude_suffixes,
    )
    if not urls:
        return []

    candidate_columns: list[str] = []
    for url in urls[:max_files]:
        schema = pl.scan_parquet(url, low_memory=True, retries=3).collect_schema()
        candidate_columns = list(schema.names())
        if is_summary_stat_schema(candidate_columns):
            return candidate_columns
    return candidate_columns


def extract_trait_hits(
    trait: TraitSpec,
    p_threshold: float = GENOME_WIDE_SIGNIFICANCE,
    top_n: int = 15_000,
    backend: str = "polars",
    row_cap: int | None = None,
) -> pd.DataFrame:
    """
    Extract a reduced hit table for one trait.

    The preferred route uses Polars lazy parquet scanning against the Hugging Face
    parquet conversion. If that fails, the function falls back to `datasets`
    streaming with a heap-based top-hit tracker.
    """
    resolved = resolve_trait_config(trait)
    config = resolved["selected_config"]
    columns = resolved["preview_columns"]
    column_map = infer_column_map(columns)

    if backend.lower() == "polars":
        try:
            return extract_trait_hits_polars(
                trait=trait,
                config=config,
                column_map=column_map,
                p_threshold=p_threshold,
                top_n=top_n,
            )
        except BaseException as exc:  # pragma: no cover - operational fallback
            print(
                f"[warn] Polars extraction failed for {trait.disorder} {config}: {exc}. "
                "Falling back to pandas/pyarrow parquet scanning."
            )

    try:
        return extract_trait_hits_parquet(
            trait=trait,
            config=config,
            column_map=column_map,
            p_threshold=p_threshold,
            top_n=top_n,
        )
    except Exception as exc:  # pragma: no cover - operational fallback
        print(
            f"[warn] Pandas parquet extraction failed for {trait.disorder} {config}: {exc}. "
            "Falling back to datasets streaming."
        )

    return extract_trait_hits_streaming(
        trait=trait,
        config=config,
        column_map=column_map,
        p_threshold=p_threshold,
        top_n=top_n,
        row_cap=row_cap,
    )


def extract_all_traits(
    traits: Iterable[TraitSpec],
    p_threshold: float = GENOME_WIDE_SIGNIFICANCE,
    top_n: int = 15_000,
    backend: str = "polars",
    row_cap: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Extract reduced hit tables for all configured traits."""
    extracted: dict[str, pd.DataFrame] = {}
    for trait in tqdm(list(traits), desc="Extracting trait hit panels"):
        extracted[trait.disorder] = extract_trait_hits(
            trait=trait,
            p_threshold=p_threshold,
            top_n=top_n,
            backend=backend,
            row_cap=row_cap,
        )
    return extracted


def extract_trait_hits_polars(
    trait: TraitSpec,
    config: str,
    column_map: dict[str, str],
    p_threshold: float,
    top_n: int,
) -> pd.DataFrame:
    """Use Polars lazy parquet scanning for the exact trait hit panel."""
    urls = get_parquet_urls(
        dataset_id=trait.dataset_id,
        config=config,
        exclude_suffixes=trait.exclude_parquet_suffixes,
    )
    if not urls:
        raise RuntimeError(f"No parquet URLs were found for {trait.dataset_id}::{config}.")

    lazy = pl.scan_parquet(urls, low_memory=True, retries=3, allow_missing_columns=True)

    p_expr = _optional_expr(column_map.get("p"), dtype=pl.Float64)
    beta_expr = _optional_expr(column_map.get("beta"), dtype=pl.Float64)
    effect_expr = _optional_expr(column_map.get("beta"), dtype=pl.Float64)
    or_expr = _optional_expr(column_map.get("or"), dtype=pl.Float64)
    se_expr = _optional_expr(column_map.get("se"), dtype=pl.Float64)
    z_expr = _optional_expr(column_map.get("z"), dtype=pl.Float64)

    curated = (
        lazy.select(
            [
                _optional_expr(column_map.get("snp"), dtype=pl.Utf8).alias("snp"),
                _optional_expr(column_map.get("chr"), dtype=pl.Utf8).alias("chr_raw"),
                _optional_expr(column_map.get("pos"), dtype=pl.Int64).alias("pos"),
                _optional_expr(column_map.get("a1"), dtype=pl.Utf8).alias("a1"),
                _optional_expr(column_map.get("a2"), dtype=pl.Utf8).alias("a2"),
                p_expr.alias("p"),
                se_expr.alias("se"),
                _optional_expr(column_map.get("info"), dtype=pl.Float64).alias("info"),
                _optional_expr(column_map.get("eaf"), dtype=pl.Float64).alias("eaf"),
                _optional_expr(column_map.get("n"), dtype=pl.Float64).alias("n"),
                _optional_expr(column_map.get("ncases"), dtype=pl.Float64).alias("ncases"),
                _optional_expr(column_map.get("ncontrols"), dtype=pl.Float64).alias("ncontrols"),
                _optional_expr(column_map.get("source_file"), dtype=pl.Utf8).alias("source_file"),
                pl.coalesce(
                    [
                        beta_expr,
                        effect_expr,
                        pl.when(or_expr > 0).then(or_expr.log()).otherwise(None),
                        pl.when(se_expr.is_not_null() & z_expr.is_not_null()).then(se_expr * z_expr).otherwise(None),
                    ]
                ).alias("beta"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("snp").is_null() | (pl.col("snp").str.len_chars() == 0))
                .then(
                    pl.when(pl.col("chr_raw").is_not_null() & pl.col("pos").is_not_null())
                    .then(pl.format("{}:{}", pl.col("chr_raw"), pl.col("pos")))
                    .otherwise(None)
                )
                .otherwise(pl.col("snp"))
                .alias("snp"),
            ]
        )
        .filter(pl.col("p").is_not_null() & (pl.col("p") > 0))
        .filter(pl.col("beta").is_not_null())
    )

    gws = curated.filter(pl.col("p") < p_threshold).collect(streaming=True).to_pandas()
    top = curated.sort("p").limit(top_n).collect(streaming=True).to_pandas()
    combined = pd.concat([gws, top], ignore_index=True)
    return finalize_trait_hits(combined, trait=trait, config=config)


def extract_trait_hits_streaming(
    trait: TraitSpec,
    config: str,
    column_map: dict[str, str],
    p_threshold: float,
    top_n: int,
    row_cap: int | None = None,
) -> pd.DataFrame:
    """Fallback route using `datasets` streaming and an in-memory top-hit heap."""
    dataset = load_dataset(trait.dataset_id, config, split="train", streaming=True)
    significant_records: list[dict[str, Any]] = []
    top_heap: list[tuple[float, int, dict[str, Any]]] = []

    for row_index, row in enumerate(tqdm(dataset, desc=f"Streaming {trait.disorder}:{config}")):
        record = normalize_row(row=row, trait=trait.disorder, config=config, column_map=column_map)
        if record is None:
            continue

        if record["p"] < p_threshold:
            significant_records.append(record)

        heap_item = (-record["p"], row_index, record)
        if len(top_heap) < top_n:
            heapq.heappush(top_heap, heap_item)
        elif heap_item[0] > top_heap[0][0]:
            heapq.heapreplace(top_heap, heap_item)

        if row_cap is not None and row_index + 1 >= row_cap:
            print(
                f"[warn] Row cap of {row_cap:,} reached for {trait.disorder}:{config}. "
                "This streaming fallback becomes approximate when row_cap is used."
            )
            break

    top_records = [item[2] for item in sorted(top_heap, key=lambda x: (-x[0], x[1]))]
    combined = pd.DataFrame(significant_records + top_records)
    return finalize_trait_hits(combined, trait=trait, config=config)


def extract_trait_hits_parquet(
    trait: TraitSpec,
    config: str,
    column_map: dict[str, str],
    p_threshold: float,
    top_n: int,
) -> pd.DataFrame:
    """Robust fallback that scans parquet shards one-by-one with pandas/pyarrow."""
    urls = get_parquet_urls(
        dataset_id=trait.dataset_id,
        config=config,
        exclude_suffixes=trait.exclude_parquet_suffixes,
    )
    if not urls:
        raise RuntimeError(f"No parquet URLs were found for {trait.dataset_id}::{config}.")

    significant_chunks: list[pd.DataFrame] = []
    running_top = pd.DataFrame()

    for url in tqdm(urls, desc=f"Scanning parquet shards for {trait.disorder}:{config}"):
        raw = pd.read_parquet(url, engine="pyarrow")
        chunk = normalize_chunk_frame(raw=raw, trait=trait.disorder, config=config, column_map=column_map)
        if chunk.empty:
            continue

        significant_chunks.append(chunk[chunk["p"] < p_threshold].copy())
        top_chunk = chunk.nsmallest(min(top_n, len(chunk)), "p")
        running_top = (
            pd.concat([running_top, top_chunk], ignore_index=True)
            .nsmallest(top_n, "p")
            .reset_index(drop=True)
        )

    combined_parts = [frame for frame in significant_chunks if not frame.empty]
    if not running_top.empty:
        combined_parts.append(running_top)
    combined = pd.concat(combined_parts, ignore_index=True) if combined_parts else pd.DataFrame()
    return finalize_trait_hits(combined, trait=trait, config=config)


def normalize_chunk_frame(
    raw: pd.DataFrame,
    trait: str,
    config: str,
    column_map: dict[str, str],
) -> pd.DataFrame:
    """Normalize one parquet shard into the canonical schema."""
    frame = pd.DataFrame(index=raw.index)

    frame["trait"] = trait
    frame["config"] = config
    frame["snp"] = _series_or_none(raw, column_map.get("snp"))
    frame["chr"] = _series_or_none(raw, column_map.get("chr"))
    frame["pos"] = _series_or_none(raw, column_map.get("pos"))
    frame["a1"] = _series_or_none(raw, column_map.get("a1"))
    frame["a2"] = _series_or_none(raw, column_map.get("a2"))
    frame["se"] = _numeric_series_or_nan(raw, column_map.get("se"))
    frame["p"] = _numeric_series_or_nan(raw, column_map.get("p"))
    frame["info"] = _numeric_series_or_nan(raw, column_map.get("info"))
    frame["eaf"] = _numeric_series_or_nan(raw, column_map.get("eaf"))
    frame["n"] = _numeric_series_or_nan(raw, column_map.get("n"))
    frame["ncases"] = _numeric_series_or_nan(raw, column_map.get("ncases"))
    frame["ncontrols"] = _numeric_series_or_nan(raw, column_map.get("ncontrols"))
    frame["source_file"] = _series_or_none(raw, column_map.get("source_file"))

    beta = _numeric_series_or_nan(raw, column_map.get("beta"))
    or_series = _numeric_series_or_nan(raw, column_map.get("or"))
    z_series = _numeric_series_or_nan(raw, column_map.get("z"))
    beta = beta.where(beta.notna(), np.log(or_series.where(or_series > 0)))
    beta = beta.where(beta.notna(), z_series * frame["se"])
    frame["beta"] = beta

    frame["chr"] = frame["chr"].apply(_normalize_chromosome)
    frame["pos"] = pd.to_numeric(frame["pos"], errors="coerce").astype("Int64")
    frame["a1"] = frame["a1"].apply(_normalize_allele)
    frame["a2"] = frame["a2"].apply(_normalize_allele)
    frame["snp"] = frame["snp"].apply(_safe_text)
    frame["source_file"] = frame["source_file"].apply(_safe_text)

    missing_snp = frame["snp"].isna() | (frame["snp"].astype("string").str.len().fillna(0) == 0)
    frame.loc[missing_snp & frame["chr"].notna() & frame["pos"].notna(), "snp"] = (
        frame.loc[missing_snp & frame["chr"].notna() & frame["pos"].notna(), "chr"].astype(int).astype(str)
        + ":"
        + frame.loc[missing_snp & frame["chr"].notna() & frame["pos"].notna(), "pos"].astype(int).astype(str)
    )

    frame.loc[frame["n"].isna() & frame["ncases"].notna() & frame["ncontrols"].notna(), "n"] = (
        frame["ncases"] + frame["ncontrols"]
    )
    frame["z"] = frame["beta"] / frame["se"]

    frame = frame[frame["p"].notna() & (frame["p"] > 0) & frame["beta"].notna() & frame["snp"].notna()].copy()
    return frame


def normalize_row(
    row: dict[str, Any],
    trait: str,
    config: str,
    column_map: dict[str, str],
) -> dict[str, Any] | None:
    """Normalize one streaming row into the canonical hit schema."""
    snp = _safe_text(row.get(column_map.get("snp", "")))
    chr_value = _normalize_chromosome(row.get(column_map.get("chr", "")))
    pos = _safe_int(row.get(column_map.get("pos", "")))
    a1 = _normalize_allele(row.get(column_map.get("a1", "")))
    a2 = _normalize_allele(row.get(column_map.get("a2", "")))
    p_value = _safe_float(row.get(column_map.get("p", "")))
    se = _safe_float(row.get(column_map.get("se", "")))
    info = _safe_float(row.get(column_map.get("info", "")))
    eaf = _safe_float(row.get(column_map.get("eaf", "")))
    n_total = _safe_float(row.get(column_map.get("n", "")))
    ncases = _safe_float(row.get(column_map.get("ncases", "")))
    ncontrols = _safe_float(row.get(column_map.get("ncontrols", "")))

    beta = _safe_float(row.get(column_map.get("beta", "")))
    if beta is None and column_map.get("or") in row:
        odds_ratio = _safe_float(row.get(column_map["or"]))
        if odds_ratio is not None and odds_ratio > 0:
            beta = math.log(odds_ratio)
    if beta is None and column_map.get("z") in row and se is not None:
        z_value = _safe_float(row.get(column_map["z"]))
        if z_value is not None:
            beta = z_value * se

    if p_value is None or p_value <= 0 or beta is None:
        return None

    if not snp:
        if chr_value is not None and pos is not None:
            snp = f"{chr_value}:{pos}"
        else:
            return None

    if n_total is None and ncases is not None and ncontrols is not None:
        n_total = ncases + ncontrols

    z_score = beta / se if se not in (None, 0) else None
    return {
        "trait": trait,
        "config": config,
        "snp": snp,
        "chr": chr_value,
        "pos": pos,
        "a1": a1,
        "a2": a2,
        "beta": beta,
        "se": se,
        "z": z_score,
        "p": p_value,
        "info": info,
        "eaf": eaf,
        "n": n_total,
        "ncases": ncases,
        "ncontrols": ncontrols,
        "source_file": _safe_text(row.get(column_map.get("source_file", ""))),
    }


def finalize_trait_hits(df: pd.DataFrame, trait: TraitSpec, config: str) -> pd.DataFrame:
    """Finalize per-trait hits after either extraction backend."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "trait",
                "config",
                "snp",
                "chr",
                "pos",
                "a1",
                "a2",
                "beta",
                "se",
                "z",
                "p",
                "info",
                "eaf",
                "n",
                "ncases",
                "ncontrols",
                "source_file",
                "locus_id",
                "match_key",
                "is_gws",
                "rank_within_trait",
            ]
        )

    standardized = df.copy()
    standardized["trait"] = trait.disorder
    standardized["config"] = config
    standardized["snp"] = standardized["snp"].astype(str)
    standardized["chr"] = standardized["chr"].apply(_normalize_chromosome)
    standardized["pos"] = pd.to_numeric(standardized["pos"], errors="coerce").astype("Int64")
    standardized["a1"] = standardized["a1"].apply(_normalize_allele)
    standardized["a2"] = standardized["a2"].apply(_normalize_allele)
    standardized["beta"] = pd.to_numeric(standardized["beta"], errors="coerce")
    standardized["se"] = pd.to_numeric(standardized["se"], errors="coerce")
    standardized["p"] = pd.to_numeric(standardized["p"], errors="coerce")
    standardized["info"] = _numeric_column_or_nan(standardized, "info")
    standardized["eaf"] = _numeric_column_or_nan(standardized, "eaf")
    standardized["n"] = _numeric_column_or_nan(standardized, "n")
    standardized["ncases"] = _numeric_column_or_nan(standardized, "ncases")
    standardized["ncontrols"] = _numeric_column_or_nan(standardized, "ncontrols")
    standardized["z"] = standardized["beta"] / standardized["se"]
    standardized["locus_id"] = standardized.apply(
        lambda row: f"{row['chr']}:{int(row['pos'])}" if pd.notna(row["chr"]) and pd.notna(row["pos"]) else None,
        axis=1,
    )
    standardized["match_key"] = standardized.apply(_make_match_key, axis=1)
    standardized["is_gws"] = standardized["p"] < GENOME_WIDE_SIGNIFICANCE

    standardized = standardized.dropna(subset=["p", "beta", "match_key"])
    standardized = standardized.sort_values("p", ascending=True)
    standardized = standardized.drop_duplicates(subset=["match_key"], keep="first")
    standardized["rank_within_trait"] = np.arange(1, len(standardized) + 1)
    return standardized.reset_index(drop=True)


def build_cross_trait_tables(
    trait_hits: dict[str, pd.DataFrame],
    p_threshold: float = GENOME_WIDE_SIGNIFICANCE,
) -> dict[str, pd.DataFrame]:
    """Build cross-trait summary tables, overlap matrices, and a Z-score matrix."""
    non_empty_frames = [frame.copy() for frame in trait_hits.values() if not frame.empty]
    if not non_empty_frames:
        raise ValueError("No trait hits were extracted.")

    long_df = pd.concat(non_empty_frames, ignore_index=True)
    long_df = long_df.sort_values(["trait", "p"]).drop_duplicates(["trait", "match_key"], keep="first")

    reference = (
        long_df.sort_values("p")
        .drop_duplicates("match_key", keep="first")[["match_key", "a1", "a2", "trait", "p"]]
        .rename(
            columns={
                "a1": "ref_a1",
                "a2": "ref_a2",
                "trait": "lead_trait",
                "p": "lead_trait_p",
            }
        )
    )

    long_df = long_df.merge(reference, on="match_key", how="left")
    alignment = long_df.apply(
        lambda row: _align_to_reference(row["beta"], row["a1"], row["a2"], row["ref_a1"], row["ref_a2"]),
        axis=1,
        result_type="expand",
    )
    alignment.columns = ["aligned_beta", "alignment_status"]
    long_df = pd.concat([long_df, alignment], axis=1)
    long_df["aligned_z"] = long_df["aligned_beta"] / long_df["se"]

    membership = (
        long_df.assign(significant=long_df["p"] < p_threshold)
        .pivot_table(index="match_key", columns="trait", values="significant", aggfunc="max", fill_value=False)
        .astype(bool)
    )

    z_matrix = long_df.pivot_table(index="match_key", columns="trait", values="aligned_z", aggfunc="first")
    p_matrix = long_df.pivot_table(index="match_key", columns="trait", values="p", aggfunc="first")
    overlap_matrix = membership.astype(int).T.dot(membership.astype(int))

    summary_rows: list[dict[str, Any]] = []
    for match_key, group in long_df.groupby("match_key", sort=False):
        group = group.sort_values("p", ascending=True)
        significant = group[group["p"] < p_threshold].copy()
        available_traits = group["trait"].nunique()
        significant_traits = significant["trait"].tolist()
        discordant_pairs = _discordant_pairs(significant)
        fisher_p = np.nan
        valid_p = group["p"].dropna().clip(lower=1e-300)
        if valid_p.shape[0] >= 2:
            fisher_p = combine_pvalues(valid_p.values.tolist(), method="fisher")[1]

        lead = group.iloc[0]
        summary_rows.append(
            {
                "match_key": match_key,
                "snp": lead["snp"],
                "chr": lead["chr"],
                "pos": lead["pos"],
                "ref_a1": lead["ref_a1"],
                "ref_a2": lead["ref_a2"],
                "lead_trait": lead["lead_trait"],
                "lead_trait_p": lead["lead_trait_p"],
                "pleiotropy_score": len(significant_traits),
                "available_traits": available_traits,
                "significant_traits": "|".join(significant_traits),
                "all_traits_seen": "|".join(group["trait"].tolist()),
                "best_p": group["p"].min(),
                "best_beta": lead["aligned_beta"],
                "best_z": lead["aligned_z"],
                "fisher_combined_p": fisher_p,
                "discordant_flag": bool(discordant_pairs),
                "discordant_pairs": "|".join(discordant_pairs),
                "supporting_traits": group["trait"].nunique(),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["pleiotropy_score", "discordant_flag", "fisher_combined_p", "best_p"],
        ascending=[False, False, True, True],
    )
    discordant_df = summary_df[summary_df["discordant_flag"]].copy()

    return {
        "long_df": long_df.reset_index(drop=True),
        "reference_df": reference.reset_index(drop=True),
        "summary_df": summary_df.reset_index(drop=True),
        "discordant_df": discordant_df.reset_index(drop=True),
        "membership_df": membership.reset_index(),
        "zscore_matrix": z_matrix.reset_index(),
        "pvalue_matrix": p_matrix.reset_index(),
        "overlap_matrix": overlap_matrix.reset_index(),
    }


def download_refgene(cache_dir: str | Path) -> pd.DataFrame:
    """
    Download a compact gene-coordinate reference for nearest-gene annotation.

    UCSC `refGene.txt.gz` is small enough for a notebook workflow and avoids
    shipping a large static annotation file in the repository.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / "refGene_hg19.tsv.gz"
    if not destination.exists():
        response = requests.get(
            "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz",
            timeout=120,
        )
        response.raise_for_status()
        destination.write_bytes(response.content)

    columns = [
        "bin",
        "name",
        "chrom",
        "strand",
        "txStart",
        "txEnd",
        "cdsStart",
        "cdsEnd",
        "exonCount",
        "exonStarts",
        "exonEnds",
        "score",
        "name2",
        "cdsStartStat",
        "cdsEndStat",
        "exonFrames",
    ]
    genes = pd.read_csv(destination, sep="\t", header=None, names=columns, compression="gzip")
    genes = genes[["chrom", "txStart", "txEnd", "name2"]].rename(
        columns={"chrom": "chr", "txStart": "start", "txEnd": "end", "name2": "gene"}
    )
    genes["chr"] = genes["chr"].str.replace("^chr", "", regex=True)
    genes["chr"] = genes["chr"].apply(_normalize_chromosome)
    genes = genes.dropna(subset=["chr", "start", "end", "gene"]).copy()
    genes["start"] = pd.to_numeric(genes["start"], errors="coerce")
    genes["end"] = pd.to_numeric(genes["end"], errors="coerce")
    genes = genes.groupby(["gene", "chr"], as_index=False).agg(start=("start", "min"), end=("end", "max"))
    return genes.sort_values(["chr", "start", "end"]).reset_index(drop=True)


def annotate_with_nearest_gene(variants: pd.DataFrame, genes: pd.DataFrame) -> pd.DataFrame:
    """Annotate each variant with the nearest gene on the same chromosome."""
    annotated = variants.copy()
    genes_by_chr = {
        chrom: subset.reset_index(drop=True)
        for chrom, subset in genes.groupby("chr", sort=False)
    }

    nearest_gene: list[str | None] = []
    nearest_distance: list[float | None] = []
    for row in tqdm(
        annotated[["chr", "pos"]].itertuples(index=False),
        total=len(annotated),
        desc="Annotating nearest genes",
    ):
        chrom_genes = genes_by_chr.get(row.chr)
        if chrom_genes is None or pd.isna(row.pos):
            nearest_gene.append(None)
            nearest_distance.append(None)
            continue

        starts = chrom_genes["start"].to_numpy(dtype=float)
        ends = chrom_genes["end"].to_numpy(dtype=float)
        positions = np.full_like(starts, float(row.pos))
        distances = np.where(
            (positions >= starts) & (positions <= ends),
            0.0,
            np.minimum(np.abs(positions - starts), np.abs(positions - ends)),
        )
        best_index = int(np.argmin(distances))
        nearest_gene.append(chrom_genes.iloc[best_index]["gene"])
        nearest_distance.append(float(distances[best_index]))

    annotated["nearest_gene"] = nearest_gene
    annotated["nearest_gene_distance_bp"] = nearest_distance
    return annotated


def summarize_trait_hits(trait_hits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a simple per-trait extraction summary."""
    rows: list[dict[str, Any]] = []
    for trait, frame in trait_hits.items():
        rows.append(
            {
                "trait": trait,
                "n_rows_retained": int(frame.shape[0]),
                "n_gws": int(frame["is_gws"].sum()) if not frame.empty else 0,
                "best_p": float(frame["p"].min()) if not frame.empty else np.nan,
                "best_locus": frame.iloc[0]["match_key"] if not frame.empty else None,
            }
        )
    return pd.DataFrame(rows).sort_values("trait").reset_index(drop=True)


def _optional_expr(column_name: str | None, dtype: pl.DataType) -> pl.Expr:
    if column_name is None:
        return pl.lit(None, dtype=dtype)
    return pl.col(column_name).cast(dtype, strict=False)


def _series_or_none(frame: pd.DataFrame, column_name: str | None) -> pd.Series:
    if column_name is None or column_name not in frame.columns:
        return pd.Series([None] * len(frame), index=frame.index, dtype="object")
    return frame[column_name]


def _numeric_series_or_nan(frame: pd.DataFrame, column_name: str | None) -> pd.Series:
    if column_name is None or column_name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column_name], errors="coerce")


def _numeric_column_or_nan(frame: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column_name], errors="coerce")


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _safe_int(value: Any) -> int | None:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_allele(value: Any) -> str | None:
    text = _safe_text(value)
    if text is None:
        return None
    text = text.upper()
    return text if re.fullmatch(r"[ACGT]+", text) else text


def _normalize_chromosome(value: Any) -> int | None:
    text = _safe_text(value)
    if text is None:
        return None
    text = text.upper().replace("CHR", "")
    mapping = {"X": 23, "Y": 24, "XY": 25, "MT": 26, "M": 26}
    if text in mapping:
        return mapping[text]
    try:
        return int(float(text))
    except ValueError:
        return None


def _make_match_key(row: pd.Series) -> str | None:
    snp = row.get("snp")
    if isinstance(snp, str) and snp.lower().startswith("rs"):
        return snp.lower()
    chrom = row.get("chr")
    pos = row.get("pos")
    a1 = row.get("a1")
    a2 = row.get("a2")
    if pd.notna(chrom) and pd.notna(pos):
        if isinstance(a1, str) and isinstance(a2, str):
            ordered = sorted([a1, a2])
            return f"{chrom}:{int(pos)}:{ordered[0]}:{ordered[1]}"
        return f"{chrom}:{int(pos)}"
    return None


def _is_palindromic(a1: str | None, a2: str | None) -> bool:
    if not a1 or not a2:
        return False
    return {a1, a2} in ({"A", "T"}, {"C", "G"})


def _complement(allele: str | None) -> str | None:
    if allele is None:
        return None
    return allele.translate(_COMPLEMENT)


def _align_to_reference(
    beta: float,
    a1: str | None,
    a2: str | None,
    ref_a1: str | None,
    ref_a2: str | None,
) -> tuple[float, str]:
    if beta is None:
        return np.nan, "missing_beta"
    if a1 is None or a2 is None or ref_a1 is None or ref_a2 is None:
        return beta, "unresolved"
    if _is_palindromic(a1, a2):
        return beta, "palindromic"
    if a1 == ref_a1 and a2 == ref_a2:
        return beta, "aligned"
    if a1 == ref_a2 and a2 == ref_a1:
        return -beta, "flipped"
    if _complement(a1) == ref_a1 and _complement(a2) == ref_a2:
        return beta, "complement_aligned"
    if _complement(a1) == ref_a2 and _complement(a2) == ref_a1:
        return -beta, "complement_flipped"
    return beta, "unresolved"


def _discordant_pairs(group: pd.DataFrame) -> list[str]:
    discordant: list[str] = []
    if group.shape[0] < 2:
        return discordant
    rows = list(group[["trait", "aligned_beta"]].itertuples(index=False, name=None))
    for index, (trait_i, beta_i) in enumerate(rows):
        if pd.isna(beta_i):
            continue
        for trait_j, beta_j in rows[index + 1 :]:
            if pd.isna(beta_j):
                continue
            if np.sign(beta_i) != 0 and np.sign(beta_j) != 0 and np.sign(beta_i) != np.sign(beta_j):
                discordant.append(f"{trait_i}~{trait_j}")
    return discordant
