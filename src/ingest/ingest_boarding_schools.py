import pandas as pd


def load_boarding_school_listing(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


def build_state_boarding_school_features(df):
    grouped = (
        df.groupby("State")["School_Name"]
        .nunique()
        .reset_index(name="BoardingSchool_Count")
    )
    return grouped


def to_indicator_table(df):
    return df.melt(
        id_vars=["State"],
        var_name="Indicator",
        value_name="Value"
    ).assign(
        Definition=lambda x: x["Indicator"],
        Source_Label="Boarding School Data"
    )
