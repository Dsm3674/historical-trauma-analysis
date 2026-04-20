    def load_missing_persons(self) -> DatasetBundle:
        df = self._read_csv(
            "missing_persons/missing_persons.csv",
            # Total_Missing is no longer required — NamUs May 2020 report
            # provides AI/AN counts only at state level
            ["Year", "State", "AI_AN_Missing"],
            "missing_persons",
        )
        df = to_numeric_columns(
            df,
            ["Year", "AI_AN_Missing", "AI_AN_Population_Percent"],
        )
        self._validate_nonnegative(
            df,
            ["Year", "AI_AN_Missing"],
            "missing_persons",
        )
        if "AI_AN_Population_Percent" in df.columns:
            self._validate_percentage(df, ["AI_AN_Population_Percent"], "missing_persons")

        # AI_AN_Percent_Missing and Overrepresentation_Ratio require Total_Missing.
        # Since the NamUs May 2020 report does not publish total missing counts
        # at the state level, these derived columns are omitted. The primary
        # outcome variable is AI_AN_Missing (absolute count).
        if "Total_Missing" in df.columns:
            df = to_numeric_columns(df, ["Total_Missing"])
            self._validate_nonnegative(df, ["Total_Missing"], "missing_persons")
            if (df["AI_AN_Missing"] > df["Total_Missing"]).any():
                raise ValueError("missing_persons.AI_AN_Missing exceeds Total_Missing.")
            df["AI_AN_Percent_Missing"] = np.where(
                df["Total_Missing"] > 0,
                100.0 * df["AI_AN_Missing"] / df["Total_Missing"],
                np.nan,
            )
            if "AI_AN_Population_Percent" in df.columns:
                df["Overrepresentation_Ratio"] = np.where(
                    df["AI_AN_Population_Percent"] > 0,
                    df["AI_AN_Percent_Missing"] / df["AI_AN_Population_Percent"],
                    np.nan,
                )
            self._validate_percentage(
                df, ["AI_AN_Percent_Missing", "AI_AN_Population_Percent"], "missing_persons"
            )
            self._validate_nonnegative(df, ["Overrepresentation_Ratio"], "missing_persons")

        meta = SourceMeta(
            dataset_name="missing_persons",
            source_org="National Institute of Justice / NamUs",
            source_url="https://www.namus.gov",
            accessed_on=utc_now(),
            file_path=str(self.raw_dir / "missing_persons/missing_persons.csv"),
            notes=(
                "AI/AN missing persons counts sourced from NamUs Unresolved Cases report, "
                "May 1, 2020, National Institute of Justice. State-level total missing counts "
                "are not published in this report; AI_AN_Percent_Missing and "
                "Overrepresentation_Ratio are therefore not computed."
            ),
            geographic_unit="state",
        )
        return DatasetBundle("missing_persons", df, meta)
