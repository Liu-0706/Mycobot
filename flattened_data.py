import pandas as pd
import ast

input_file = "data.csv"
output_file = "flattened_data.csv"

df = pd.read_csv(input_file)

columns = ["x_target", "y_target", "z_target",
           "x_error", "y_error", "z_error",
           "j1", "j2", "j3", "j4", "j5", "j6"]

def parse_list(cell):
    try:
        return ast.literal_eval(cell) if isinstance(cell, str) else cell
    except:
        return None

flattened_rows = []

for i, row in df.iterrows():
    for cell in row:
        values = parse_list(cell)
        if values and len(values) == 12:
            flattened_rows.append(values)
        else:
            print("####################",len(values))
            print("i",i)
            print("cell",cell)

    print("i",i)

flattened_df = pd.DataFrame(flattened_rows, columns=columns)

flattened_df = flattened_df.apply(pd.to_numeric, errors='coerce')

flattened_df.to_csv(output_file, index=False)

print(f"save as {output_file}")
