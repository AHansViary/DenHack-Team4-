import pandas as pd

df = pd.read_excel('src\results_cityu_mcgill3.xlsx')
threshold = 0.4
filtered_df = df[df['Similarity Index Cosine'] > threshold]
print(filtered_df)