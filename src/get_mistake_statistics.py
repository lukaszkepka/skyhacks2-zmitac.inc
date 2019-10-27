import pandas as pd

answers_path = 'D:\\answers(2).csv'
annotations_path = "D:\\Programowanie\\zmitac.inc\\Models\\labels.csv"

annotations = pd.read_csv(annotations_path)
answers = pd.read_csv(annotations_path)
mistakes = {}
columns_to_wyjabania = ['filename', 'tech_cond', 'standard', 'task2_class']

for index, row in answers.iterrows():
    filename = row['filename']
    matched = annotations.loc[annotations['filename'] == filename].head(1).iloc[0]
    if not matched.empty:
        row = row.drop(columns_to_wyjabania)
        matched = matched.drop(columns_to_wyjabania)

        matched.as_numpty()

        for i, column in enumerate(answers.columns.to_list()):
            if column not in columns_to_wyjabania and str(row[column]) != str(matched[column]):
                if column not in mistakes:
                    mistakes[column] = 0

                mistakes[column] += 1

print(mistakes)
