"""
CSV upload handling for Streamlit app.

Provides:
- Smart column mapping with fuzzy matching
- User-friendly validation error reporting
- Integration with existing schema and validation
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import pandas as pd

from src.data.schema import SCHEMA, _LEGACY_REMAP


# ── Schema metadata ──

# Required columns per table (cannot be NULL / must exist)
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "underlying_bars": ["ticker", "timestamp", "open", "high", "low", "close", "volume"],
    "options_chain": [
        "ticker", "timestamp", "expiration", "strike", "option_type", "bid", "ask",
    ],
}

# Optional columns (will be set to NULL if missing)
OPTIONAL_COLUMNS: dict[str, list[str]] = {
    "underlying_bars": ["dividend"],
    "options_chain": [
        "last", "volume", "open_interest", "implied_vol",
        "delta", "gamma", "theta", "vega",
    ],
}

# Common alternative names → canonical name
COLUMN_ALIASES: dict[str, str] = {
    "date": "timestamp",
    "time": "timestamp",
    "datetime": "timestamp",
    "dt": "timestamp",
    "symbol": "ticker",
    "sym": "ticker",
    "underlying": "ticker",
    "close_price": "close",
    "open_price": "open",
    "high_price": "high",
    "low_price": "low",
    "adj_close": "close",
    "adjusted_close": "close",
    "vol": "volume",
    "oi": "open_interest",
    "openinterest": "open_interest",
    "open_int": "open_interest",
    "iv": "implied_vol",
    "impl_vol": "implied_vol",
    "implied_volatility": "implied_vol",
    "type": "option_type",
    "opt_type": "option_type",
    "call_put": "option_type",
    "cp": "option_type",
    "exp": "expiration",
    "expiry": "expiration",
    "expiration_date": "expiration",
    "exp_date": "expiration",
    "strike_price": "strike",
    "bid_price": "bid",
    "ask_price": "ask",
    "last_price": "last",
}


@dataclass
class ColumnMapping:
    """Result of column mapping detection."""
    source_col: str
    target_col: str
    method: str  # "exact", "alias", "fuzzy", "legacy"
    confidence: float  # 0.0 - 1.0


@dataclass
class MappingResult:
    """Full mapping result for one file."""
    mappings: list[ColumnMapping] = field(default_factory=list)
    unmapped_required: list[str] = field(default_factory=list)
    unmapped_optional: list[str] = field(default_factory=list)
    extra_source_cols: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.unmapped_required) == 0

    def get_rename_dict(self) -> dict[str, str]:
        """Return {source_col: target_col} for pandas rename."""
        return {m.source_col: m.target_col for m in self.mappings}


def detect_column_mapping(
    source_columns: list[str],
    table_name: str,
) -> MappingResult:
    """
    Detect how source CSV columns map to the target schema.

    Matching priority:
    1. Exact match (case-insensitive)
    2. Known alias match
    3. Legacy remap (from schema.py)
    4. Fuzzy match (>= 0.8 similarity)
    """
    required = REQUIRED_COLUMNS.get(table_name, [])
    optional = OPTIONAL_COLUMNS.get(table_name, [])
    all_target = required + optional

    legacy = _LEGACY_REMAP.get(table_name, {})

    # Normalize source columns
    source_lower = {col.strip().lower().replace(" ", "_"): col for col in source_columns}

    mappings: list[ColumnMapping] = []
    matched_targets: set[str] = set()
    matched_sources: set[str] = set()

    # Pass 1: Exact match (case-insensitive)
    for target in all_target:
        if target.lower() in source_lower and target not in matched_targets:
            src_original = source_lower[target.lower()]
            mappings.append(ColumnMapping(src_original, target, "exact", 1.0))
            matched_targets.add(target)
            matched_sources.add(src_original)

    # Pass 2: Known aliases
    for src_norm, src_original in source_lower.items():
        if src_original in matched_sources:
            continue
        if src_norm in COLUMN_ALIASES:
            target = COLUMN_ALIASES[src_norm]
            if target in all_target and target not in matched_targets:
                mappings.append(ColumnMapping(src_original, target, "alias", 0.95))
                matched_targets.add(target)
                matched_sources.add(src_original)

    # Pass 3: Legacy remap (e.g., "date" → "timestamp")
    for old_name, new_name in legacy.items():
        if new_name not in matched_targets:
            if old_name.lower() in source_lower:
                src_original = source_lower[old_name.lower()]
                if src_original not in matched_sources:
                    mappings.append(ColumnMapping(src_original, new_name, "legacy", 0.9))
                    matched_targets.add(new_name)
                    matched_sources.add(src_original)

    # Pass 4: Fuzzy match for remaining
    unmatched_targets = [t for t in all_target if t not in matched_targets]
    unmatched_sources = [
        (norm, orig) for norm, orig in source_lower.items()
        if orig not in matched_sources
    ]

    for target in unmatched_targets:
        best_score = 0.0
        best_src = None
        for src_norm, src_original in unmatched_sources:
            score = SequenceMatcher(None, src_norm, target.lower()).ratio()
            if score > best_score and score >= 0.8:
                best_score = score
                best_src = (src_norm, src_original)

        if best_src:
            mappings.append(ColumnMapping(best_src[1], target, "fuzzy", round(best_score, 2)))
            matched_targets.add(target)
            matched_sources.add(best_src[1])
            unmatched_sources = [
                (n, o) for n, o in unmatched_sources if o != best_src[1]
            ]

    return MappingResult(
        mappings=mappings,
        unmapped_required=[c for c in required if c not in matched_targets],
        unmapped_optional=[c for c in optional if c not in matched_targets],
        extra_source_cols=[
            col for col in source_columns if col not in matched_sources
        ],
    )


def parse_csv_file(
    uploaded_file,
    encoding: str = "utf-8",
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Parse an uploaded CSV file with error handling.

    Returns (dataframe, error_message).
    """
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        content = uploaded_file.read()

        if len(content) == 0:
            return None, "File is empty."

        # Try specified encoding, fallback to latin-1
        for enc in [encoding, "latin-1"]:
            try:
                df = pd.read_csv(
                    io.BytesIO(content),
                    encoding=enc,
                    sep=None,    # auto-detect delimiter
                    engine="python",
                )
                break
            except UnicodeDecodeError:
                continue
        else:
            return None, "Could not decode file. Please use UTF-8 or Latin-1 encoding."

        if df.empty:
            return None, "File contains no data rows."

        if len(df.columns) < 2:
            return None, (
                "Only 1 column detected. The file might use an unsupported delimiter. "
                "Please use comma-separated values."
            )

        return df, None

    except pd.errors.ParserError as e:
        return None, f"CSV parsing error: {e}"
    except Exception as e:
        return None, f"Error reading file: {e}"


def apply_mapping_and_convert(
    df: pd.DataFrame,
    mapping: MappingResult,
    table_name: str,
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Rename columns according to mapping and convert types.

    Returns (processed_df, error_message).
    """
    try:
        # Keep only mapped columns
        rename_dict = mapping.get_rename_dict()
        mapped_cols = list(rename_dict.keys())
        df_mapped = df[mapped_cols].rename(columns=rename_dict).copy()

        # Add missing optional columns as NULL
        all_cols = REQUIRED_COLUMNS.get(table_name, []) + OPTIONAL_COLUMNS.get(table_name, [])
        for col in all_cols:
            if col not in df_mapped.columns:
                df_mapped[col] = None

        # Type conversions
        if table_name == "underlying_bars":
            df_mapped["timestamp"] = pd.to_datetime(df_mapped["timestamp"])
            for col in ["open", "high", "low", "close"]:
                df_mapped[col] = pd.to_numeric(df_mapped[col], errors="coerce")
            df_mapped["volume"] = pd.to_numeric(df_mapped["volume"], errors="coerce").astype("Int64")
            if "dividend" in df_mapped.columns:
                df_mapped["dividend"] = pd.to_numeric(
                    df_mapped["dividend"], errors="coerce"
                ).fillna(0.0)

        elif table_name == "options_chain":
            df_mapped["timestamp"] = pd.to_datetime(df_mapped["timestamp"])
            df_mapped["expiration"] = pd.to_datetime(df_mapped["expiration"]).dt.date
            df_mapped["strike"] = pd.to_numeric(df_mapped["strike"], errors="coerce")
            df_mapped["option_type"] = df_mapped["option_type"].str.strip().str.upper()

            for col in ["bid", "ask"]:
                df_mapped[col] = pd.to_numeric(df_mapped[col], errors="coerce")

            for col in ["last", "implied_vol", "delta", "gamma", "theta", "vega"]:
                if col in df_mapped.columns and df_mapped[col] is not None:
                    df_mapped[col] = pd.to_numeric(df_mapped[col], errors="coerce")

            for col in ["volume", "open_interest"]:
                if col in df_mapped.columns and df_mapped[col] is not None:
                    df_mapped[col] = pd.to_numeric(
                        df_mapped[col], errors="coerce"
                    ).astype("Int64")

        # Reorder columns to match schema
        df_mapped = df_mapped[[c for c in all_cols if c in df_mapped.columns]]

        return df_mapped, None

    except Exception as e:
        return None, f"Error converting data: {e}"


def generate_template_csv(table_name: str) -> str:
    """Generate a template CSV string with headers and a few example rows."""
    if table_name == "underlying_bars":
        return (
            "ticker,timestamp,open,high,low,close,volume,dividend\n"
            "AAPL,2023-01-03,130.28,130.90,124.17,125.07,112117471,0\n"
            "AAPL,2023-01-04,126.89,128.66,125.08,126.36,89113633,0\n"
            "AAPL,2023-01-05,127.13,127.77,124.76,125.02,80962708,0\n"
        )
    elif table_name == "options_chain":
        return (
            "ticker,timestamp,expiration,strike,option_type,bid,ask,last,"
            "volume,open_interest,implied_vol,delta,gamma,theta,vega\n"
            "AAPL,2023-01-03,2023-02-17,130.00,C,5.10,5.30,5.20,"
            "1250,15000,0.2800,0.4500,0.0320,-0.1200,0.2500\n"
            "AAPL,2023-01-03,2023-02-17,130.00,P,4.80,5.00,4.90,"
            "980,12000,0.2900,-0.5500,0.0320,-0.1100,0.2500\n"
            "AAPL,2023-01-03,2023-02-17,135.00,C,3.20,3.40,3.30,"
            "2100,18000,0.2700,0.3200,0.0280,-0.1000,0.2200\n"
        )
    return ""
