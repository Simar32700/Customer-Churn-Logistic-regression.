"""Data cleaning for the customer churn project."""

from pathlib import Path
import json

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path(__file__).resolve().parent / "data"


def main():
    # ZIP codes are identifiers; reading them as text preserves leading zeros.
    customers = pd.read_csv(
        DATA_DIR / "telecom_customer_churn.csv",
        dtype={"Customer ID": "string", "Zip Code": "string"},
        keep_default_na=False,
        na_values=[""],
    )
    population = pd.read_csv(
        DATA_DIR / "telecom_zipcode_population.csv",
        dtype={"Zip Code": "string"},
        keep_default_na=False,
        na_values=[""],
    )
    dictionary = pd.read_csv(
        DATA_DIR / "telecom_data_dictionary.csv",
        encoding="cp1252",
        keep_default_na=False,
        na_values=[""],
    )

    for name, dataset in [
        ("Customer records", customers),
        ("ZIP-code population", population),
        ("Data dictionary", dictionary),
    ]:
        print(f"\n{name}: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
        print("Columns:", dataset.columns.tolist())
        print("\nMissing values in each column:")
        print(dataset.isna().sum().to_string())
        print("Extra identical rows:", dataset.duplicated().sum())
        print("\nColumn data types:")
        print(dataset.dtypes.to_string())

        print("\nText whitespace checks:")
        for column in dataset.select_dtypes(include=["object", "string"]).columns:
            text = dataset[column].astype("string")
            trimmed = text.str.strip()
            changed = text.ne(trimmed).fillna(False).sum()
            blank = trimmed.eq("").fillna(False).sum()
            print(f"{column}: extra spaces={changed}, whitespace-only blanks={blank}")
            dataset[column] = trimmed.replace("", pd.NA)

    print("\nZIP-code format checks (five ASCII digits):")
    for name, dataset in [("Customers", customers), ("Population", population)]:
        invalid_zip = ~dataset["Zip Code"].str.fullmatch(r"[0-9]{5}", na=False)
        print(f"{name}: {invalid_zip.sum()} invalid or missing ZIP codes")
        if invalid_zip.any():
            print(dataset.loc[invalid_zip, ["Zip Code"]].head(10).to_string(index=False))
            raise ValueError(f"Review {name.lower()} ZIP codes before merging.")

    # The dictionary spells this one field differently from the customer CSV.
    field_aliases = {"CustomerID": "Customer ID"}
    print("\nDictionary alignment checks:")
    print("Recognized field alias: CustomerID -> Customer ID")
    for table, dataset in [("Customer Churn", customers), ("Zip Code Population", population)]:
        definitions = dictionary.loc[dictionary["Table"].eq(table)]
        fields = definitions["Field"].replace(field_aliases)
        undocumented = sorted(set(dataset.columns) - set(fields))
        unused = sorted(set(fields) - set(dataset.columns))
        print(f"{table}: undocumented columns={undocumented}, definitions without columns={unused}")
        if fields.duplicated().any() or definitions["Description"].isna().any():
            raise ValueError(f"Duplicate definitions or missing descriptions in {table}.")
        if undocumented or unused:
            raise ValueError(f"Resolve dictionary mismatches for {table} before preprocessing.")
    unknown_tables = sorted(set(dictionary["Table"].dropna()) - {"Customer Churn", "Zip Code Population"})
    print(f"Dictionary tables without a loaded dataset: {unknown_tables}")

    # Descriptions can flag timing concerns, but cannot prove pre-churn availability.
    customer_definitions = dictionary.loc[dictionary["Table"].eq("Customer Churn")].copy()
    timing_flags = customer_definitions["Description"].str.contains(
        r"end of the quarter|quarter ended|by the end|at the end|when they leave",
        case=False, na=False,
    )
    timing_review = customer_definitions.loc[timing_flags, ["Field", "Description"]]
    print("\nTiming review: quarter-end or post-churn definitions:")
    print(timing_review.to_string(index=False))
    print("Availability before churn is NOT verified: the files lack dated customer snapshots and churn dates.")

    binary_columns = [
        "Married", "Phone Service", "Multiple Lines", "Internet Service",
        "Online Security", "Online Backup", "Device Protection Plan",
        "Premium Tech Support", "Streaming TV", "Streaming Movies",
        "Streaming Music", "Unlimited Data", "Paperless Billing",
    ]
    for column in binary_columns:
        labels = customers[column].str.casefold()
        unexpected = customers[column].notna() & ~labels.isin(["yes", "no"])
        if unexpected.any():
            print(f"{column}: marking {unexpected.sum()} invalid labels as missing: {customers.loc[unexpected, column].unique().tolist()}")
        customers[column] = labels.map({"yes": "Yes", "no": "No"}).astype("string")

    customers["Customer Status"] = customers["Customer Status"].str.casefold().map(
        {"churned": "Churned", "stayed": "Stayed", "joined": "Joined"}
    )
    if customers["Customer Status"].isna().any():
        raise ValueError("Missing or unrecognized Customer Status labels.")

    allowed_offers = ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"]
    invalid_offers = customers["Offer"].notna() & ~customers["Offer"].isin(allowed_offers)
    print(f"Offer: marking {invalid_offers.sum()} invalid labels as missing")
    customers.loc[invalid_offers, "Offer"] = pd.NA

    count_columns = [
        "Number of Dependents", "Number of Referrals", "Tenure in Months",
    ]
    nonnegative_columns = [
        "Avg Monthly Long Distance Charges", "Avg Monthly GB Download",
        "Monthly Charge", "Total Charges", "Total Refunds",
        "Total Extra Data Charges", "Total Long Distance Charges", "Total Revenue",
    ]
    print("\nNumeric checks (flagged values need review):")
    for name, dataset, key, columns in [
        ("Customers", customers, "Customer ID",
         ["Age", "Latitude", "Longitude"] + count_columns + nonnegative_columns),
        ("Population", population, "Zip Code", ["Population"]),
    ]:
        for column in columns:
            original = dataset[column]
            numeric = pd.to_numeric(original, errors="coerce")
            nonnumeric = original.notna() & numeric.isna()
            nonfinite = numeric.isin([float("inf"), float("-inf")])
            flagged = nonnumeric | nonfinite
            print(f"{name} / {column}: nonnumeric={nonnumeric.sum()}, nonfinite={nonfinite.sum()}")

            if column == "Age":
                invalid = numeric.notna() & (~numeric.between(1, 120) | numeric.mod(1).ne(0))
                rule = "outside 1-120 or not a whole number (review)"
            elif column in ["Latitude", "Longitude"]:
                limit = 90 if column == "Latitude" else 180
                invalid = numeric.notna() & ~numeric.between(-limit, limit)
                rule = "outside valid coordinate range"
            else:
                invalid = numeric.lt(0)
                rule = "negative"
                if column in count_columns + ["Population"]:
                    invalid = invalid | (numeric.notna() & numeric.mod(1).ne(0))
                    rule = "negative or not a whole number"

            flagged = flagged | invalid
            print(f"  {rule}: {invalid.sum()}")
            if flagged.any():
                print(dataset.loc[flagged, [key, column]].head(10).to_string(index=False))

    # Count all rows involved in a repeated key, including the first occurrence.
    for name, dataset, key in [
        ("Customer IDs", customers, ["Customer ID"]),
        ("Population ZIP codes", population, ["Zip Code"]),
        ("Dictionary table/field definitions", dictionary, ["Table", "Field"]),
    ]:
        repeated = dataset.duplicated(subset=key, keep=False)
        print(f"\n{name}: {repeated.sum()} rows with repeated keys")
        if repeated.any():
            print("First 10 rows to review:")
            print(dataset.loc[repeated].head(10).to_string(index=False))

    no_phone = customers["Phone Service"].eq("No")
    no_internet = customers["Internet Service"].eq("No")

    phone_options = ["Multiple Lines"]
    internet_options = [
        "Internet Type", "Online Security", "Online Backup",
        "Device Protection Plan", "Premium Tech Support", "Streaming TV",
        "Streaming Movies", "Streaming Music", "Unlimited Data",
    ]

    print("\nService consistency checks:")
    for service, absent, options, usage in [
        ("Phone", no_phone, phone_options, "Avg Monthly Long Distance Charges"),
        ("Internet", no_internet, internet_options, "Avg Monthly GB Download"),
    ]:
        contradictions = absent & customers[usage].gt(0)
        for column in options:
            if column == "Internet Type":
                contradictions = contradictions | (absent & customers[column].notna())
            else:
                contradictions = contradictions | (absent & customers[column].eq("Yes").fillna(False))
        print(f"{service}: {contradictions.sum()} customers with no service but positive usage/options")
        if contradictions.any():
            print(customers.loc[contradictions, ["Customer ID", usage] + options].head(10).to_string(index=False))
            raise ValueError(f"Review contradictory {service.lower()} records before preprocessing.")

    # Fill only missing fields whose service is explicitly absent.
    for column in phone_options:
        customers.loc[no_phone, column] = customers.loc[no_phone, column].fillna(
            "Not applicable"
        )
    customers.loc[no_phone, "Avg Monthly Long Distance Charges"] = customers.loc[
        no_phone, "Avg Monthly Long Distance Charges"
    ].fillna(0)

    for column in internet_options:
        customers.loc[no_internet, column] = customers.loc[no_internet, column].fillna(
            "Not applicable"
        )
    customers.loc[no_internet, "Avg Monthly GB Download"] = customers.loc[
        no_internet, "Avg Monthly GB Download"
    ].fillna(0)

    print("\nCustomer missing values after service-based filling:")
    print(customers.isna().sum().to_string())

    # Many customers can share one ZIP code; keep every customer in the merge.
    if population["Zip Code"].isna().any():
        raise ValueError("Population ZIP codes must not be missing before merging.")
    customer_count = len(customers)
    customers = customers.merge(
        population,
        on="Zip Code",
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=False,
    )
    unmatched = customers["_merge"].eq("left_only")
    print(f"\nCustomers without a population match: {unmatched.sum()}")
    if unmatched.any():
        print(customers.loc[unmatched, ["Customer ID", "Zip Code"]].head(10).to_string(index=False))
        raise ValueError("Review unmatched ZIP codes before saving the merged data.")
    if len(customers) != customer_count:
        raise ValueError("The population merge changed the customer row count.")
    if customers["Population"].isna().any():
        raise ValueError("Matched ZIP codes have missing population values.")
    customers = customers.drop(columns="_merge")
    print(f"Merged customer data: {len(customers)} rows, {len(customers.columns)} columns")

    # Negative counts are invalid; treat negative service charges as invalid too.
    imputation_columns = ["Number of Referrals", "Monthly Charge"]
    for column in imputation_columns:
        negative = customers[column].lt(0)
        print(f"\n{column}: marking {negative.sum()} negative values as missing")
        customers.loc[negative, column] = float("nan")

    # Unknown referrals need training-only imputation; churn reasons stay unchanged.
    cleaned_dir = DATA_DIR / "cleaned"
    cleaned_dir.mkdir(exist_ok=True)
    output_path = cleaned_dir / "telecom_customer_churn.csv"
    customers.to_csv(output_path, index=False)
    print(f"\nSaved customer data to: {output_path}")

    if customers["Customer ID"].isna().any() or customers["Customer ID"].duplicated().any():
        raise ValueError("Customer IDs must be present and unique before splitting.")

    eligible = customers["Customer Status"].isin(["Churned", "Stayed"])
    model_data = customers.loc[eligible].drop(columns=["Churn Category", "Churn Reason"])
    print(f"\nCustomers excluded without a Churned/Stayed outcome: {(~eligible).sum()}")

    # Stratification keeps similar churn proportions; the seed repeats this split.
    train_data, test_data = train_test_split(
        model_data,
        test_size=0.20,
        random_state=42,
        stratify=model_data["Customer Status"],
    )
    train_data = train_data.copy()
    test_data = test_data.copy()

    training_medians = train_data[imputation_columns].median()
    if training_medians.isna().any():
        raise ValueError("No valid training values to calculate an imputation median.")
    print("\nTraining-only medians:")
    print(training_medians.to_string())

    training_modes = {}
    for column in binary_columns + ["Offer"]:
        mode = train_data[column].mode()
        if mode.empty:
            raise ValueError(f"No known training labels to fill {column}.")
        training_modes[column] = mode.iloc[0]

    binary_mapping = {"Yes": 1, "No": 0, "Not applicable": 0}
    gender_mapping = {"Female": 0, "Male": 1}
    target_mapping = {"Stayed": 0, "Churned": 1}
    metadata = {
        "training_medians": training_medians.to_dict(),
        "training_binary_modes": training_modes,
        "binary_columns": binary_columns,
        "binary_mapping": binary_mapping,
        "gender_mapping": gender_mapping,
        "target_column": "Customer Status",
        "target_mapping": target_mapping,
        "test_size": 0.20,
        "random_state": 42,
        "field_aliases": field_aliases,
        "pre_churn_availability_verified": False,
        "timing_review": timing_review.to_dict(orient="records"),
    }
    for name, dataset in [("train", train_data), ("test", test_data)]:
        for column in binary_columns + ["Offer"]:
            missing_count = dataset[column].isna().sum()
            dataset[column] = dataset[column].fillna(training_modes[column])
            if missing_count:
                print(f"{name.title()} / {column}: filled {missing_count} unknown labels with training mode {training_modes[column]}")
        for column in imputation_columns:
            missing_count = dataset[column].isna().sum()
            dataset[column] = dataset[column].fillna(training_medians[column])
            print(f"{name.title()} / {column}: filled {missing_count} missing values")
        if dataset[imputation_columns].isna().any().any() or dataset[imputation_columns].lt(0).any().any():
            raise ValueError("Missing or negative values remain in the imputed columns.")
        for column, mapping in [
            *[(column, binary_mapping) for column in binary_columns],
            ("Gender", gender_mapping),
            ("Customer Status", target_mapping),
        ]:
            encoded = dataset[column].map(mapping)
            if encoded.isna().any():
                raise ValueError(f"Missing or unexpected labels in {column} during encoding.")
            dataset[column] = encoded.astype("int64")
        print(f"{name.title()}: encoded {len(binary_columns) + 2} columns as numeric 0/1")
        split_path = cleaned_dir / f"customers_{name}.csv"
        dataset.to_csv(split_path, index=False)
        print(f"\n{name.title()} set: {len(dataset)} customers")
        print("Customer Status counts:")
        print(dataset["Customer Status"].value_counts().to_string())
        print("Remaining missing values:", dataset.isna().sum().sum())
        print(f"Saved to: {split_path}")

        # Keep identical row order in X and y so each input matches its outcome.
        X = dataset.drop(columns=["Customer Status", "Customer ID"])
        y = dataset[["Customer Status"]]
        X_path = cleaned_dir / f"X_{name}.csv"
        y_path = cleaned_dir / f"y_{name}.csv"
        X.to_csv(X_path, index=False)
        y.to_csv(y_path, index=False)
        print(f"Inputs: {X.shape[0]} rows, {X.shape[1]} features -> {X_path}")
        print(f"Outcome: {y.shape[0]} rows, 1 target -> {y_path}")

    # Exclude binary indicators and the outcome; ZIP codes remain text labels.
    measurement_columns = [
        column for column in train_data.select_dtypes(include="number").columns
        if column not in binary_columns + ["Gender", "Customer Status"]
    ]
    scaler = StandardScaler()
    scaler.fit(train_data[measurement_columns])

    excluded_inputs = ["Customer Status", "Customer ID", "City", "Zip Code"]
    model_inputs = train_data.drop(columns=excluded_inputs)
    categorical_columns = model_inputs.select_dtypes(include=["object", "string"]).columns.tolist()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int)
    encoder.fit(model_inputs[categorical_columns])
    encoded_columns = encoder.get_feature_names_out(categorical_columns).tolist()
    print("\nOne-hot encoding categories (learned from training):")
    for column, categories in zip(categorical_columns, encoder.categories_):
        print(f"{column}: {categories.tolist()}")

    for name, dataset in [("train", train_data), ("test", test_data)]:
        X_scaled = dataset.drop(columns=["Customer Status", "Customer ID"]).copy()
        X_scaled[measurement_columns] = scaler.transform(dataset[measurement_columns])
        scaled_path = cleaned_dir / f"X_{name}_scaled.csv"
        X_scaled.to_csv(scaled_path, index=False)
        print(f"Saved scaled {name} inputs to: {scaled_path}")

        inputs = X_scaled.drop(columns=["City", "Zip Code"])
        for column, categories in zip(categorical_columns, encoder.categories_):
            unknown = ~inputs[column].isin(categories)
            print(f"{name.title()} / {column}: {unknown.sum()} unseen categories (encoded as all zeros)")
        encoded = pd.DataFrame(
            encoder.transform(inputs[categorical_columns]),
            columns=encoded_columns,
            index=inputs.index,
        )
        X_ready = pd.concat([inputs.drop(columns=categorical_columns), encoded], axis=1)
        if X_ready.columns.duplicated().any():
            raise ValueError("One-hot encoding created duplicate feature names.")
        if len(X_ready.select_dtypes(include="number").columns) != len(X_ready.columns):
            raise ValueError("Text columns remain in the encoded inputs.")
        if X_ready.isna().any().any() or X_ready.isin([float("inf"), float("-inf")]).any().any():
            raise ValueError("Missing or infinite values in the encoded inputs.")
        ready_path = cleaned_dir / f"X_{name}_encoded.csv"
        X_ready.to_csv(ready_path, index=False)
        print(f"Saved numeric {name} inputs: {X_ready.shape[0]} rows, {X_ready.shape[1]} features -> {ready_path}")
        if name == "train":
            metadata["feature_columns"] = X_ready.columns.tolist()
        elif X_ready.columns.tolist() != metadata["feature_columns"]:
            raise ValueError("Training and test feature columns do not match.")

    joblib.dump(scaler, cleaned_dir / "numeric_scaler.joblib")
    joblib.dump(encoder, cleaned_dir / "categorical_encoder.joblib")
    metadata["categorical_columns"] = categorical_columns
    metadata["excluded_input_columns"] = excluded_inputs
    metadata["encoder_file"] = "categorical_encoder.joblib"
    metadata["scaled_columns"] = measurement_columns
    metadata["scaler_file"] = "numeric_scaler.joblib"
    with (cleaned_dir / "preprocessing.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Scaled {len(measurement_columns)} measurement columns using training-only statistics.")


if __name__ == "__main__":
    main()
