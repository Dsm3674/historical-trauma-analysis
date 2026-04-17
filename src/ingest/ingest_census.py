
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests


@dataclass
class SourceMeta:
    dataset_name: str
    source_org: str
    source_url: str
    accessed_on: str
    citation: str
    notes: str


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_dataset_with_manifest(df: pd.DataFrame, output_csv: str | Path, meta: SourceMeta) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    manifest = {"rows": int(len(df)), "columns": list(df.columns), "source_meta": asdict(meta)}
    output_csv.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _request_json(url: str, params: dict) -> list[list[str]]:
    response = requests.get(url, params=params, timeout=90)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError("Unexpected Census API response format.")
    return payload


def fetch_acs_custom_variables(year: int, variables: Iterable[str], api_key: Optional[str] = None) -> pd.DataFrame:
    variables = list(dict.fromkeys(variables))
    if not variables:
        raise ValueError("At least one ACS variable must be provided.")
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {"get": ",".join(["NAME", *variables]), "for": "state:*"}
    if api_key:
        params["key"] = api_key
    rows = _request_json(url, params)
    df = pd.DataFrame(rows[1:], columns=rows[0]).rename(columns={"NAME": "State", "state": "StateFIPS"})
    df["Year"] = year
    for column in variables:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df[["Year", "State", "StateFIPS", *variables]]


def build_state_ai_an_dataset(
    year: int,
    ai_an_total_var: str,
    total_population_var: str,
    disability_var: Optional[str] = None,
    poverty_var: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    if ai_an_total_var == "B01003_001E":
        raise ValueError("ai_an_total_var cannot be B01003_001E; use an AI/AN-specific ACS field.")
    variables = [ai_an_total_var, total_population_var]
    if disability_var:
        variables.append(disability_var)
    if poverty_var:
        variables.append(poverty_var)
    df = fetch_acs_custom_variables(year=year, variables=variables, api_key=api_key)
    rename_map = {
        ai_an_total_var: "AI_AN_Population",
        total_population_var: "Total_Population",
    }
    if disability_var:
        rename_map[disability_var] = "AI_AN_With_Disability"
    if poverty_var:
        rename_map[poverty_var] = "AI_AN_In_Poverty"
    df = df.rename(columns=rename_map)
    if "AI_AN_With_Disability" in df.columns:
        df["Disability_Rate"] = df["AI_AN_With_Disability"] / df["AI_AN_Population"]
    if "AI_AN_In_Poverty" in df.columns:
        df["Poverty_Rate"] = df["AI_AN_In_Poverty"] / df["AI_AN_Population"]
    return df


if __name__ == "__main__":
    output_dir = ensure_dir("data/processed")
    raise SystemExit(
        "Set explicit ACS variable IDs in build_state_ai_an_dataset(...). "
        "This script intentionally requires real AI/AN-specific variable IDs."
    )
