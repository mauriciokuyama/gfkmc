import pandas as pd
import glob
import os
import wfdb

# dataset_dir = "datasets"
# if not os.path.exists(dataset_dir):
#     os.mkdir(dataset_dir)

csv_file = os.path.join("physionet.org", "files" ,"wearable-exercise-frailty", "1.0.0", "subject-info.csv")

df = pd.read_csv(csv_file, header=1)

# print(df.columns)
print(df.shape)

"""
Index(['Patient ID', 'Age, years', 'Gender    (0 - male, 1 - female)',
       'Height, cm', 'Weight, kg', 'EFS score', 'Days after surgery',
       'Surgery type (0 - coronary artery bypass graft, 1 - isolated valve, 2 - combined)',
       'Heart failure based on NYHA class',
       'Atrial fibrillation (0 - no AF, 1 - permanent, 2 - persistent, 3 - paroxysmal)',
       'Chronic obstructive pulmonary disease', 'Depression',
       'Multisceletal system diseases', 'Oncological diseases',
       'ACE inhibitors', 'Beta blockers ', 'Calcium channel blockers',
       'Distance, m', 'Time, s', 'Duration, mm:ss', 'Max load, watt',
       'Max HR, bpm', 'Step length (left), cm', 'Step length (right), cm',
       'Stride length, cm', 'Step width, cm', 'Stance phase (left), %',
       'Stance phase (right), %', 'Swing phase (left), %',
       'Swing phase (right), %', 'Double stance phase, %',
       'Step time (left), s', 'Step time (right), s', 'Stride time, s',
       'Cadence, steps/min', 'Velocity, km/h',
       'Length of gait line (left), mm', 'Length of gait line (right), mm',
       'Single limb support line (left), mm',
       'Single limb support line (right), mm', 'Ant/post position, mm',
       'Lateral symmetry, mm', 'Max gait line velocity, cm/s'],
      dtype='object')
"""
columns_to_keep = [
    "Patient ID",
    "Age, years",
    "Gender    (0 - male, 1 - female)",
    "Height, cm",
    "Weight, kg",
]

df = df[columns_to_keep]
# print(df.head(5))

df.rename(
    columns={
        "Patient ID": "ID",
        "Gender    (0 - male, 1 - female)": "Gender",
        "Age, years": "Age",
        "Height, cm": "Height(cm)",
        "Weight, kg": "Weight(kg)"
    },
    inplace=True,
)

df["Gender"] = df["Gender"].replace(0, "M")
df["Gender"] = df["Gender"].replace(1, "F")
# print(df.head())

df["ID"] = df["ID"].map(lambda id: f'{id:03}')
df.loc[df['ID']=='203', 'Weight(kg)'] = 85

df.to_csv(os.path.join("wearable-exercise-frailty.csv"), index=False)
