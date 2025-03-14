# Loading libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from scipy.interpolate import PchipInterpolator, interp1d
import re
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import io

# Global variables
input_file_path = ''  # inicialize as empty

# Start Window 1 ----------------------------------------------------------------------------------------------------------------------

root = tk.Tk()
root.geometry("500x278")
root.title("2D MAP INTER-EXTRAPOLATOR, by DGPB, NefMoto Community")

# Initial tag
label = tk.Label(root, text="This app will help you design a new axis for your map.\n"
                 "\n"
                 "1) In WinOLS window, open the map you'd like to modify,\n"
                 "2) Go to Edit -> Copy advanced -> Copy map.\n"
                 "3) Paste it as plain text in a blank csv and save it.\n"
                 "4) Load it by clicking the button below.",
                 font=("Calibri", 12),
                 justify="left",
                 )
label.pack(pady=20, anchor="w", fill="x")

# Load CSV Button
button = tk.Button(root, text="Load CSV Map", font=("Calibri", 12),
                  command=lambda: [
                      # Select file
                      globals().__setitem__('input_file_path', filedialog.askopenfilename(
                          title="Select your input CSV",
                          filetypes=[("CSV Files", "*.csv")]  # only CSV
                      )),
                      # comprobation
                      tag.config(text=f"{input_file_path}" if input_file_path != '' else "No file selected"),
                      # comprobation
                      (tk.Button(root, text="Confirm", command=root.destroy, font=("Calibri", 12)).pack(padx=10, pady=0, anchor="w", fill="x") if input_file_path != '' else None)
                  ])
button.pack(padx=10, pady=0, anchor="w", fill="x")

# tag for loaded file path
tag = tk.Label(root, text="No file selected", font=("Calibri", 12))
tag.pack(pady=10)

root.mainloop()

# End Window 1 ----------------------------------------------------------------------------------------------------------------------

#print(input_file_path)

with open(input_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
lines

# Extract 'length' from 1st line
first_line = lines[0].strip()
match = re.search(r'x(\d+)\(', first_line[::-1])  # from right to left
length = int(match.group(1)[::-1]) if match else None

# Extract X_name and y_name from 3rd line
third_line = lines[2].strip()
x_part, y_part = third_line.split('/')
X_name = x_part.replace('(', '').replace(')', '').strip()
y_name = y_part.replace('(', '').replace(')', '').strip()

# Extract X_data from 6th line
sixth_line = lines[5].strip()
X_data = list(map(float, sixth_line.split('\t')[0:]))  # from 1st value onward

# Extract y_data from 7th line
seventh_line = lines[6].strip()
y_data = list(map(float, seventh_line.split('\t')[2:]))  # skip '-', '0'

# Create DataFrame
df = pd.DataFrame({X_name: X_data, y_name: y_data})
df['weight'] = 0.0

#Expand original DF
df_expanded_ori = pd.DataFrame({X_name: np.linspace(min(df[df.columns[0]]), max(df[df.columns[0]]), num=10000), y_name: np.full(10000, np.nan)})
df_expanded_ori = pd.concat([df, df_expanded_ori])
df_expanded_ori = df_expanded_ori.drop_duplicates(subset=df_expanded_ori.columns[0], keep='first')
df_expanded_ori = df_expanded_ori.sort_values(df_expanded_ori.columns[0])
df_expanded_ori.reset_index(inplace=True, drop=True)
df_expanded_ori.drop(df_expanded_ori.columns[2], axis=1, inplace=True)

# Linear interpolation between original setpoints
df_expanded_lin = df_expanded_ori.copy()
df_expanded_lin[df_expanded_lin.columns[1]] = np.interp(df_expanded_lin[df_expanded_lin.columns[0]], df_expanded_lin.dropna()[df_expanded_lin.columns[0]], df_expanded_lin.dropna()[df_expanded_lin.columns[1]])
df_expanded_lin

# Adding columns for "df_aux_values" and "abs_error"
df_expanded_1 = df_expanded_lin.copy()
df_expanded_1[y_name+'_n-1_setpoints'] = np.nan
df_expanded_1['abs_error'] = 0.0
df_expanded_1

# Create lists with each one of the middle setpoints deleted
# To store graphs
figures = []
plt.ioff()
# index w/o first and last
for i in X_data[1:-1]:
    lista_sin_i = [x for x in X_data if x != i]
    df_aux = df[df[df.columns[0]].isin(lista_sin_i)]
    df_aux = df_aux.drop(columns = df.columns[2])
    # Asign '%' from df_aux to df_expanded_1 only where Motordrehzahl exists in df
    df_expanded_1.iloc[:, 2] = df_expanded_1.iloc[:, 0].map(df_aux.set_index(df_aux.columns[0])[df_aux.columns[1]])
    # Order by 'X' just in case
    df_expanded_1 = df_expanded_1.sort_values(df_expanded_1.columns[0])
    df_expanded_1.reset_index(inplace=True, drop=True)
    # Interpolate only NaN in '%'
    df_expanded_1[df_expanded_1.columns[2]] = np.interp(df_expanded_1[df_expanded_1.columns[0]], df_expanded_1.dropna()[df_expanded_1.columns[0]], df_expanded_1.dropna()[df_expanded_1.columns[2]])
    # Calculate error
    df_expanded_1.iloc[:, 3] = abs(df_expanded_1.iloc[:, 1] - df_expanded_1.iloc[:, 2])
    sumatorio_error = df_expanded_1['abs_error'].sum()
    df.loc[df.iloc[:, 0] == i, df.columns[2]] = sumatorio_error
    #Graph
    fig = plt.subplots(figsize=(8, 3))
    sns.lineplot(data=df_expanded_1, x=df_expanded_1.columns[0], y=df_expanded_1.columns[1], linewidth=4, label="All Setpoints")
    sns.lineplot(data=df_expanded_1, x=df_expanded_1.columns[0], y=df_expanded_1.columns[2], label="n-1 SetPoints")
    sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], s=75)
    plt.grid()
    plt.legend()
    plt.title(f"No SetPoint on {i}")
    figures.append(fig)
plt.close()

# Re establish df_expanded_1
# Linear interpolation between original setpoints
df_expanded_1[df_expanded_1.columns[1]] = np.interp(df_expanded_1[df_expanded_1.columns[0]], df_expanded_1.dropna()[df_expanded_1.columns[0]], df_expanded_1.dropna()[df_expanded_1.columns[1]])

# Multiple subplots
num_plots = len(figures)
fig, axes = plt.subplots(num_plots, 1, figsize=(8, 3 * num_plots))

# in case only 1 plot -> array
if num_plots == 1:
    axes = [axes]

# add figures to subplots
for idx, (fig_i, ax) in enumerate(zip(figures, axes)):
    original_ax = fig_i[1]
    for line in original_ax.get_lines():
        ax.plot(line.get_xdata(), line.get_ydata(), 
                label=line.get_label(), 
                linewidth=line.get_linewidth(),
                color=line.get_color())
    sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], s=75, ax=ax)
    ax.set_title(original_ax.get_title())
    ax.legend()
    ax.grid()
    plt.close(fig_i[0])
plt.tight_layout()

# save the figure in a buffer

buf_1 = io.BytesIO()
plt.savefig(buf_1, format='png')
buf_1.seek(0)

# Start Window 2 ----------------------------------------------------------------------------------------------------------------------

root = tk.Tk()
root.geometry("1450x800")
root.title("2D MAP INTER-EXTRAPOLATOR, by DGPB, NefMoto Community")

# grid conf
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

frame_image = ttk.Frame(root)
frame_image.grid(row=1, column=0, sticky="nsew")

frame_right = ttk.Frame(root)
frame_right.grid(row=1, column=1, sticky="nsew")

frame_table = ttk.Frame(frame_right)
frame_table.grid(row=0, column=0, sticky="nsew")

frame_inputs = ttk.Frame(frame_right)
frame_inputs.grid(row=1, column=0, sticky="nsew", padx=10)

# section tags
label_image = tk.Label(root, text="Map with n-1 SetPoints", font=("Calibri", 12, "bold"))
label_table = tk.Label(root, text="Less significant SetPoints", font=("Calibri", 12, "bold"))
label_inputs = tk.Label(frame_inputs, text="Editable X_data Values", font=("Calibri", 12, "bold"))

label_image.grid(row=0, column=0, padx=10, pady=5)
label_table.grid(row=0, column=1, padx=10, pady=5, sticky="w")
label_inputs.pack(pady=5)

# load and show img
image = Image.open(buf_1)
photo = ImageTk.PhotoImage(image)
canvas = tk.Canvas(frame_image, width=image.width, height=image.height)
canvas.pack(side="left", fill="both", expand=True)
scrollbar = tk.Scrollbar(frame_image, orient="vertical", command=canvas.yview)
scrollbar.pack(side="right", fill="y")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.create_image(0, 0, anchor="nw", image=photo)
canvas.config(scrollregion=canvas.bbox("all"))

# show df as table
tree = ttk.Treeview(frame_table, columns=list(df.iloc[1:-1].sort_values(by=df.columns[2], ascending=True)), show="headings")

# scrollbar
scrollbar_tree = ttk.Scrollbar(frame_table, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scrollbar_tree.set)
scrollbar_tree.pack(side="right", fill="y")

for col in df.iloc[1:-1].sort_values(by=df.columns[2], ascending=True):
    tree.heading(col, text=col)

for index, row in df.iloc[1:-1].sort_values(by=df.columns[2], ascending=True).iterrows():
    tree.insert("", "end", values=list(row))

tree.pack(fill="both", expand=True, padx=5, pady=5)

entries = []

for i, val in enumerate(X_data):
    frame_row = tk.Frame(frame_inputs)
    frame_row.pack(fill="x", pady=2)
    label = tk.Label(frame_row, text=f"X{i+1}:", width=5, anchor="w")
    label.pack(side="left")
    entry = tk.Entry(frame_row, width=10)
    entry.insert(0, str(val))
    entry.pack(side="left", padx=5)
    entries.append(entry)

# compute function
def compute_new_axis():
    global new_axis
    new_axis = np.array([float(entry.get()) for entry in entries]).astype(float)
    print(new_axis)
    root.quit()  # Termina el mainloop
    root.destroy()  # Cierra la ventana

# compute button
btn_compute = tk.Button(frame_inputs, text="Compute", command=compute_new_axis, bg="green", fg="white", font=("Calibri", 12, "bold"))
btn_compute.pack(pady=10)

# close
buf_1.close()
root.mainloop()

# End Window 2 ----------------------------------------------------------------------------------------------------------------------

# New Map creation
df_new_linear = df.copy()
df_new_linear.drop(df_new_linear.columns[2], axis=1, inplace=True)

# To know what values are original
# placing new setpoints onto the pchip curve,
# avoiding that way the piecewise quantization error
for i in new_axis:
    if i not in df[df_new_linear.columns[0]].values:
        df_new_linear = pd.concat([df_new_linear, pd.DataFrame({df_new_linear.columns[0]: [i]})], ignore_index=True)
df_new_linear = df_new_linear.sort_values(df_new_linear.columns[0], ascending = True).reset_index(drop=True)
df_new_linear[y_name+'_linear'] = df_new_linear[df_new_linear.columns[1]]
df_new_linear = df_new_linear.drop(df_new_linear.columns[1], axis=1)

# function for linear interpolation/extrapolation
interp_func = interp1d(
    df_new_linear.dropna(subset=[df_new_linear.columns[1]]).iloc[:, 0],  
    df_new_linear.dropna(subset=[df_new_linear.columns[1]]).iloc[:, 1],
    kind='linear',
    fill_value='extrapolate'
)

# Apply
df_new_linear.iloc[:, 1] = interp_func(df_new_linear.iloc[:, 0])

# New Map creation
df_new_pchip = df.copy()
df_new_pchip.drop(df_new_pchip.columns[2], axis=1, inplace=True)

# To know what values are original
# placing new setpoints onto the pchip curve,
# avoiding that way the piecewise quantization error
for i in new_axis:
    if i not in df[df_new_pchip.columns[0]].values:
        df_new_pchip = pd.concat([df_new_pchip, pd.DataFrame({df_new_pchip.columns[0]: [i]})], ignore_index=True)
df_new_pchip = df_new_pchip.sort_values(df_new_pchip.columns[0], ascending = True).reset_index(drop=True)
df_new_pchip[y_name+'_pchip'] = df_new_pchip[df_new_pchip.columns[1]]
df_new_pchip = df_new_pchip.drop(df_new_pchip.columns[1], axis=1)
# fill values
interp_pchip = PchipInterpolator(
    df_new_pchip.dropna(subset=[df_new_pchip.columns[0], df_new_pchip.columns[1]])[df_new_pchip.columns[0]].values,
    df_new_pchip.dropna(subset=[df_new_pchip.columns[0], df_new_pchip.columns[1]])[df_new_pchip.columns[1]].values,
    axis=0,
    extrapolate=True
)

# apply interpolation/extrapolation
df_new_pchip.iloc[:, 1] = interp_pchip(df_new_linear.iloc[:, 0])
df_new_combined = pd.concat([df_new_linear, df_new_pchip[df_new_pchip.columns[1]]], axis=1)
df_new_combined = df_new_combined[df_new_combined.iloc[:, 0].isin(new_axis)]
df_new_combined[y_name+'_chosen'] = 0.0
df_interp = df_new_combined[(df_new_combined.iloc[:, 0] >= min(X_data)) & (df_new_combined.iloc[:, 0] <= max(X_data))]
df_extrap = df_new_combined[(df_new_combined.iloc[:, 0] < min(X_data)) | (df_new_combined.iloc[:, 0] > max(X_data))]

# Start Window 3 ----------------------------------------------------------------------------------------------------------------------

root = tk.Tk()
root.geometry("500x278")
root.title("2D MAP INTER-EXTRAPOLATOR, by DGPB, NefMoto Community")

# Initialize global variables
method_interp = ''
method_extrap = ''

# assignment functions
def set_interpolation_method(method):
    global method_interp
    method_interp = method
    print(f"Interpolation method set to: {method_interp}")

def set_extrapolation_method(method):
    global method_extrap
    method_extrap = method
    print(f"Extrapolation method set to: {method_extrap}")

# close function
def close_window():
    print(f"Final interpolation method: {method_interp}")
    print(f"Final extrapolation method: {method_extrap}")
    root.quit()  # Termina el mainloop
    root.destroy()  # Cierra la ventana

# raw 1 - title and buttons for interpolation
label_interp = tk.Label(root, text="Interpolation method", font=("Calibri", 12))
label_interp.grid(row=0, column=0, columnspan=2, pady=10)

button_linear = tk.Button(root, text="Linear", command=lambda: set_interpolation_method('Linear'))
button_linear.grid(row=1, column=0, padx=10)

button_pchip = tk.Button(root, text="Pchip", command=lambda: set_interpolation_method('Pchip'))
button_pchip.grid(row=1, column=1, padx=10)

# raw 2 - title and buttons for extrapolation
label_extrap = tk.Label(root, text="Extrapolation method", font=("Calibri", 12))
label_extrap.grid(row=2, column=0, columnspan=2, pady=10)

button_linear_extrap = tk.Button(root, text="Linear", command=lambda: set_extrapolation_method('Linear'))
button_linear_extrap.grid(row=3, column=0, padx=10)

button_pchip_extrap = tk.Button(root, text="Pchip", command=lambda: set_extrapolation_method('Pchip'))
button_pchip_extrap.grid(row=3, column=1, padx=10)

# raw - confirmation
button_confirm = tk.Button(root, text="Confirm", command=close_window)
button_confirm.grid(row=4, column=0, columnspan=2, pady=20)

root.mainloop()

# End Window 3 ----------------------------------------------------------------------------------------------------------------------

if method_interp == 'Linear':
    df_interp.iloc[:, 3] = df_interp.iloc[:, 1]
if method_interp == 'Pchip':
    df_interp.iloc[:, 3] = df_interp.iloc[:, 2]

if method_extrap == 'Linear':
    df_extrap.iloc[:, 3] = df_extrap.iloc[:, 1]
if method_extrap == 'Pchip':
    df_extrap.iloc[:, 3] = df_extrap.iloc[:, 2]

df_final = pd.concat([df_interp, df_extrap])
df_final = df_final.sort_values(df_final.columns[0], ascending = True).reset_index(drop=True)

df_theoretical_curve = df_expanded_ori.copy()

#fill values
interp_pchip = PchipInterpolator(
    df_theoretical_curve.dropna(subset=[df_theoretical_curve.columns[0], df_theoretical_curve.columns[1]])[df_theoretical_curve.columns[0]].values,
    df_theoretical_curve.dropna(subset=[df_theoretical_curve.columns[0], df_theoretical_curve.columns[1]])[df_theoretical_curve.columns[1]].values,
    axis=0,
    extrapolate=True
)

# Apply interpolation/extrapolation
df_theoretical_curve.iloc[:, 1] = interp_pchip(df_theoretical_curve.iloc[:, 0])

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=df.columns[0], y=df.columns[1], label="Original", color = 'black')
sns.lineplot(data=df_final, x=df_final.columns[0], y=df_final.columns[3], label="New", color = 'green')
sns.lineplot(data=df_theoretical_curve, x=df_theoretical_curve.columns[0], y=df_theoretical_curve.columns[1], label="Theoretical curve", color = 'red', linestyle='--')
plt.grid()
plt.title('Overlayed maps')

# buffer for img
buf_2 = io.BytesIO()
plt.savefig(buf_2, format='png')
plt.close()
buf_2.seek(0)

# Start Window 4 ----------------------------------------------------------------------------------------------------------------------

root = tk.Tk()
root.geometry("1200x600")
root.title("2D MAP INTER-EXTRAPOLATOR, by DGPB, NefMoto Community")

# show img
image2 = Image.open(buf_2)
photo = ImageTk.PhotoImage(image2)
label = tk.Label(root, image=photo)
label.image = photo
label.pack()

# close
buf_2.close()
root.mainloop()