import pandas

df = pandas.read_csv("RHMD_Data_Uncleaned.csv")

# Symptom keywords unrelated to influenza which were included in the dataset
keywords = ["addiction", "allergic", "alzheimer", "asthma", "cancer", "depression", "diabetes", "heart attack", "migraine", "OCD", "PTSD", "stroke"]

for word in keywords:
    df = df[~df["Text"].str.contains(word, case=False)]


df_HM = df[df["Label"] == 2]

# Remove health mentions before sampling
df = df[df["Label"] != 2]
df_FMNHM = df.sample(frac=0.1, replace=True, random_state=1)

# Relabel data
df_HM['Label'] = df_HM["Label"].replace(2,1)
df_FMNHM['Label'] = df_FMNHM["Label"].replace(1,0)

# Concat frames
frames = [df_HM, df_FMNHM]
df_final = pandas.concat(frames)

df.to_csv('RHMD_Cleaned.csv', index=False)