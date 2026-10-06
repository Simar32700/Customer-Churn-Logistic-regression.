"""Ask for customer details in the terminal, then predict customer churn."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model_train import predict_churn


INPUT_FIELDS = [
    "Gender", "Age", "Married", "Number of Dependents", "Zip Code",
    "Latitude", "Longitude", "Number of Referrals", "Tenure in Months",
    "Offer", "Phone Service", "Avg Monthly Long Distance Charges",
    "Multiple Lines", "Internet Service", "Internet Type",
    "Avg Monthly GB Download", "Online Security", "Online Backup",
    "Device Protection Plan", "Premium Tech Support", "Streaming TV",
    "Streaming Movies", "Streaming Music", "Unlimited Data", "Contract",
    "Paperless Billing", "Payment Method", "Monthly Charge", "Total Charges",
    "Total Refunds", "Total Extra Data Charges", "Total Long Distance Charges",
    "Total Revenue",
]

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "cleaned"


def ask_customer():
    with (DATA_DIR / "preprocessing.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    encoder = joblib.load(DATA_DIR / metadata["encoder_file"])
    choices = dict(zip(metadata["categorical_columns"], encoder.categories_))
    choices["Gender"] = ["Female", "Male"]
    for column in metadata["binary_columns"]:
        choices[column] = ["Yes", "No"]
    population = pd.read_csv(
        PROJECT_DIR / "data" / "telecom_zipcode_population.csv", dtype={"Zip Code": "string"}
    )
    known_zips = set(population["Zip Code"])
    count_fields = {"Age", "Number of Dependents", "Number of Referrals", "Tenure in Months"}
    phone_fields = {"Multiple Lines": "Not applicable", "Avg Monthly Long Distance Charges": 0}
    internet_fields = {
        "Internet Type": "Not applicable", "Avg Monthly GB Download": 0,
        **{column: "Not applicable" for column in [
            "Online Security", "Online Backup", "Device Protection Plan",
            "Premium Tech Support", "Streaming TV", "Streaming Movies",
            "Streaming Music", "Unlimited Data",
        ]},
    }
    values = {}
    print("Enter the customer's details:")
    for column in INPUT_FIELDS:
        if values.get("Phone Service") == "No" and column in phone_fields:
            values[column] = phone_fields[column]
            continue
        if values.get("Internet Service") == "No" and column in internet_fields:
            values[column] = internet_fields[column]
            continue
        options = choices.get(column)
        prompt = f"{column} ({' / '.join(options)}): " if options is not None else f"{column}: "
        while True:
            answer = input(prompt).strip()
            if options is not None:
                matches = [option for option in options if option.casefold() == answer.casefold()]
                if matches:
                    values[column] = matches[0]
                    break
                print("Choose one of the listed options.")
            elif column == "Zip Code":
                if len(answer) == 5 and answer.isascii() and answer.isdigit() and answer in known_zips:
                    values[column] = answer
                    break
                print("Enter a five-digit ZIP code present in our population dataset.")
            else:
                try:
                    number = float(answer)
                except ValueError:
                    print("Enter a number.")
                    continue
                valid = np.isfinite(number)
                if column == "Latitude":
                    valid = valid and -90 <= number <= 90
                elif column == "Longitude":
                    valid = valid and -180 <= number <= 180
                else:
                    valid = valid and number >= 0
                if column in count_fields:
                    valid = valid and number.is_integer()
                if column == "Age":
                    valid = valid and 1 <= number <= 120
                if valid:
                    values[column] = int(number) if column in count_fields else number
                    break
                print("Enter a valid value (counts must be whole numbers; charges cannot be negative).")
    return values


def prepare_customer(values):
    with (DATA_DIR / "preprocessing.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    values = {key: value.strip() if isinstance(value, str) else value for key, value in values.items()}
    zipcode = str(values.get("Zip Code", ""))
    if len(zipcode) != 5 or not zipcode.isascii() or not zipcode.isdigit():
        raise ValueError("Enter Zip Code as a five-digit string, such as '93225'.")
    population = pd.read_csv(
        PROJECT_DIR / "data" / "telecom_zipcode_population.csv", dtype={"Zip Code": "string"}
    )
    match = population.loc[population["Zip Code"].eq(zipcode), "Population"]
    if len(match) != 1:
        raise ValueError("ZIP code must have exactly one population match in our dataset.")
    values["Population"] = match.iloc[0]

    required = metadata["scaled_columns"] + metadata["binary_columns"] + ["Gender"] + metadata["categorical_columns"]
    missing = sorted(set(required) - set(values))
    if missing:
        raise ValueError(f"Missing customer fields: {missing}")
    row = pd.DataFrame([values])[required]
    for column in metadata["scaled_columns"]:
        value = pd.to_numeric(row[column], errors="raise")
        value = value.fillna(metadata["training_medians"].get(column, np.nan))
        if not np.isfinite(value.to_numpy()).all():
            raise ValueError(f"Enter a valid number for {column}.")
        if column not in ["Latitude", "Longitude"] and value.lt(0).any():
            raise ValueError(f"{column} must not be negative.")
        row[column] = value

    for column in metadata["binary_columns"]:
        value = row.at[0, column]
        if pd.isna(value) or value == "":
            value = metadata["training_binary_modes"][column]
        if isinstance(value, str):
            value = {"yes": "Yes", "no": "No", "not applicable": "Not applicable"}.get(value.casefold(), value)
        if value not in metadata["binary_mapping"]:
            raise ValueError(f"{column}: enter Yes or No.")
        row[column] = metadata["binary_mapping"][value]
    gender = str(row.at[0, "Gender"]).capitalize()
    if gender not in metadata["gender_mapping"]:
        raise ValueError("Gender: enter Female or Male, matching this training dataset.")
    row["Gender"] = metadata["gender_mapping"][gender]

    scaler = joblib.load(DATA_DIR / metadata["scaler_file"])
    row[metadata["scaled_columns"]] = scaler.transform(row[metadata["scaled_columns"]])
    encoder = joblib.load(DATA_DIR / metadata["encoder_file"])
    categories = metadata["categorical_columns"]
    for column, allowed in zip(categories, encoder.categories_):
        if row.at[0, column] not in allowed:
            raise ValueError(f"{column}: choose one of {allowed.tolist()}")
    encoded = pd.DataFrame(
        encoder.transform(row[categories]), columns=encoder.get_feature_names_out(categories), index=row.index,
    )
    inputs = pd.concat([row.drop(columns=categories), encoded], axis=1)
    return inputs[metadata["feature_columns"]]


def main():
    inputs = prepare_customer(ask_customer())
    model = joblib.load(PROJECT_DIR / "models" / "customer_churn_model.joblib")
    with (PROJECT_DIR / "models" / "training_summary.json").open(encoding="utf-8") as file:
        threshold = json.load(file)["prediction_threshold"]
    answer = predict_churn(model, inputs, threshold=threshold)[0]
    print("Will this customer churn?", "Yes" if answer == 1 else "No")


if __name__ == "__main__":
    main()
