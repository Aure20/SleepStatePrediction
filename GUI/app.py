from tkinter import *
import pandas as pd
from datetime import timedelta
import lightgbm as lgb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

 
bst = lgb.Booster(model_file='GUI/25M.txt')

def open_file():
    df = pd.read_parquet('GUI/test_app.parquet')
    # Do something with the DataFrame, e.g., store it in a global variable for later use
    select_user(df)

def select_user(df):
    users = df['series_id'].unique()
    users_menu['menu'].delete(0, 'end')  # Clear existing menu items
    for user in users:
        users_menu['menu'].add_command(label=user, command=lambda user=user: generate_time_windows(df[df['series_id'] == user], user))


def generate_time_windows(df,user):
    var_user.set(f'Selected user: {user}')
    if 'timestamp' not in df.columns:
        print("Please select a file first.")
        return

    time_windows_menu['menu'].delete(0, 'end')  # Clear existing menu items

    start_timestamp = df['timestamp'].min()
    end_timestamp = df['timestamp'].max()

    current_start = start_timestamp
    index = 1
    
    fig, ax = plt.subplots(figsize=(10, 6))
    canvas = FigureCanvasTkAgg(fig, master=root)

    while current_start + timedelta(days=1) <= end_timestamp:
        current_end = current_start + timedelta(days=1)
        option_text = f"Day {index} - {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}"
        time_windows_menu['menu'].add_command(label=option_text, command=lambda start=current_start, end=current_end: predict_time_window(canvas, ax,
                                                                                                                                          df[(df['timestamp'] >= start) & (df['timestamp'] <= end)], start, end))
        current_start += timedelta(days=1)
        index += 1

def predict_time_window(canvas, ax, df, start, end): 
    var_day.set(f"Selected day: {start.strftime('%m-%d')}")
    preds_proba = bst.predict(df.drop(columns=['state', 'series_id', 'step', 'timestamp'])) 
    preds = np.argmax(preds_proba, axis = 1)
    wake_up_time, onset_time = get_predictions(preds,preds_proba) 
    # Create a stacked area plot
    y1 = preds_proba[:, 0]
    y2 = preds_proba[:, 1]
    y3 = preds_proba[:, 2]

    ax.clear()
    
    # Calculate x values based on the number of rows
    x = np.arange(start, end+timedelta(seconds=5), timedelta(seconds=5))

    # Plot vertical lines for sleep onset and wake up
    if wake_up_time is not None:
        ax.axvline(x[wake_up_time], color='red', linestyle='--', label='Wake up')

    if onset_time is not None:
        ax.axvline(x[onset_time], color='blue', linestyle='--', label='Sleep Onset')
    
    
    day.config(text=f"Predicting for the day: {start.strftime('%m-%d')}")
    wakeup.config(text=f"Predicted wake up time: {np.datetime_as_string(x[wake_up_time], unit='m')}")
    onset.config(text=f"Predicet onset time: {np.datetime_as_string(x[onset_time], unit='m')}")

    # Use fill_between to create a stacked area plot
    ax.fill_between(x, 0, y1, label='Awake')
    ax.fill_between(x, y1, y1 + y2, label='Sleeping')
    ax.fill_between(x, y1 + y2, y1 + y2 + y3, label='Not wearing')

    ax.set_xlim(x[0], x[-1])  # Adjusted xlim to be consistent with zero-based indexing
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time of the day')
    ax.set_ylabel('Probabilities')
    ax.legend(loc = 'lower left')
    
    canvas.draw() 
    canvas.get_tk_widget().grid(row=3,column=0,columnspan=2)

def get_predictions(preds, preds_proba):
    for event in ['onset', 'wakeup']:
        step = 50
        indices = list(range(step, len(preds)-step))
        column_difference = []
        for index in indices:
            if event == 'onset':
                column_difference.append(np.mean(preds_proba[index:index+step, 0] + preds_proba[index-step:index, 1]))
            else:
                column_difference.append(np.mean(preds_proba[index-step:index, 0] + preds_proba[index:index+step, 1]))
        if event == 'onset': 
            onset_event = indices[np.argmax(column_difference)] 
        else:
            wake_event = indices[np.argmax(column_difference)]

    return onset_event, wake_event



# GUI setup   
root = Tk()
root.title("File Loader")
root.geometry("720x1280")
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)

var_user = StringVar(root)
var_user.set('Select user:')
var_day = StringVar(root)
var_day.set('Select day:')

pady = 10
# Buttons and labels 
button = Button(root, text="Upload Parquet File", command=open_file)
button.grid(row=0,column=0,padx=10,pady=pady)

# Dropdown menu for time windows
users_menu = OptionMenu(root, var_user, "No File Selected")
users_menu.grid(row=1,column=0,padx=10,pady=pady)

# Dropdown menu for time windows
time_windows_menu = OptionMenu(root, var_day, "No File Selected")
time_windows_menu.grid(row=2,column=0,padx=10,pady=pady)

# Elements on the top right corner
day = Label(root, text="Predicting for the day:") 
day.grid(row=0,column=1,padx=10,pady=pady)

wakeup = Label(root, text="Wakeup event recorder at:") 
wakeup.grid(row=2,column=1,padx=10,pady=pady)

onset = Label(root, text="Onset event recorder at:") 
onset.grid(row=1,column=1,padx=10,pady=pady)

root.mainloop()

