import pandas as pd
import glob
import os


def parse_signals(file_path, columns_to_keep):
    df = pd.read_csv(file_path)

    columns_to_keep = [" II", " RESP", " PLETH"]
    df = df[columns_to_keep]

    return df


def parse_fix(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        age = lines[5].replace("Age: ", "").rstrip().rstrip("+")
        gender = lines[6].replace("Gender: ", "").rstrip()
        id = (
            os.path.basename(file_path)
            .replace("bidmc_", "")
            .replace("_Fix.txt", "")
        )

        # print(filename, age, gender)

        data = {"ID": [id], "Age": [age], "Gender": [gender]}

        df = pd.DataFrame(data)

        return df


# dataset_dir = "datasets"
# if not os.path.exists(dataset_dir):
#     os.mkdir(dataset_dir)

signal_file_paths = glob.glob(os.path.join("bidmc_csv", "bidmc_??_Signals.csv"))
fix_file_paths = glob.glob(os.path.join("bidmc_csv", "bidmc_??_Fix.txt"))
signal_file_paths.sort()
fix_file_paths.sort()

output_df = pd.DataFrame()
columns = [" II", " RESP", " PLETH"]

for signal_file, fix_file in zip(signal_file_paths, fix_file_paths):
    if "_18_" in signal_file:
        continue

    df = parse_signals(signal_file, columns)

    fix_df = parse_fix(fix_file)

    # Create a new DataFrame with a single row for each column
    new_df = pd.DataFrame()

    # Loop through each column and join its values into a single row
    for col in columns:
        joined_string = ";".join(df[col].astype(str))
        new_df[col] = [joined_string]

    new_df = pd.concat([fix_df, new_df], axis=1)

    output_df = pd.concat([output_df, new_df], axis=0)


output_df.rename(columns={" II": "ECG"}, inplace=True)
output_df.rename(columns={" RESP": "RESP"}, inplace=True)
output_df.rename(columns={" PLETH": "PPG"}, inplace=True)

output_df.to_csv(os.path.join("bidmc.csv"), index=False)

print(output_df)
# for file_path in fix_file_paths:
#     parse_fix(file_path)
