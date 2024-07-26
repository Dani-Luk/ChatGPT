"""
This script implements a neural network model for solving the XOR problem using a custom feature.
The XOR problem is a classic problem in machine learning where the task is to predict the output based on two binary inputs.
The neural network model consists of two layers: the first layer takes the input values, and the second layer combines the inputs with a custom feature.
The custom feature is calculated based on the selected function from the `custom_features` module.
The model is trained using the XOR dataset and the weights, loss, and accuracy are saved for each epoch.
The script also includes a graphical user interface (GUI) implemented using the tkinter library to visualize the results and interact with the model.
:))
"""

from typing import List, Literal, Callable, Dict, Callable
import time

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Lambda
from keras.optimizers import Adam
# from keras import initializers
from keras import callbacks 
import tensorflow as tf

# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import initializers, callbacks
# from tensorflow.keras.callbacks import EarlyStopping

# Ensure we use the CPU to avoid non-deterministic behavior on the GPU
tf.config.set_visible_devices([], 'GPU')
tf.config.experimental.enable_op_determinism()

# region Classes
class SignalTraining:
    def __init__(self):
        self.callbacks: List[Callable[[int]]] = []
        self.onTraining = True

    def connect(self, callback: Callable[[int], None]):
        self.callbacks.append(callback)
    
    def signal(self, value: int):
        for callback in self.callbacks:
            callback(value)


class WeightsHistory(callbacks.Callback):
    def __init__(self, sig:SignalTraining):
        super().__init__()
        self.signal = sig

    def on_train_begin(self, logs={}):
        self.weights = []
        self.predictions = []
        self.loss = []
        self.accuracy = []
        self.signal.onTraining = True

        if bAddBias:
            int_weights = self.model.layers[-1].get_weights()[0] # get the weights of the second layer
            # int_bias = - np.inner(int_weights.ravel(), np.array([0.5, 0.5, 0])) # lets take the 'obvious' mean, starting however from 0 z-elevation
            int_bias = - np.inner(int_weights.ravel(), np.full(int_weights.shape[0], 0.5)) 
            self.model.layers[-1].set_weights([int_weights, np.array([int_bias])])  
        super().on_train_begin(logs)

    def on_epoch_begin(self, epoch, logs={}):
        self.weights.append(self.model.layers[-1].get_weights())
        self.predictions.append(self.model.predict(x_data))
        super().on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.signal.signal(epoch)
        super().on_epoch_end(epoch, logs)
    
    def on_train_end(self, logs=None):
        self.signal.onTraining = False
        self.signal.signal(len(self.weights) - 1) # signal the last epoch at the end of the training
        return super().on_train_end(logs)
# endregion Classes
# --------------------------------------------------------------

# region initializations of global variables
SHOW_PROGRESS_IN_REAL_TIME = False # set to True to see the training(current epoch number) in real time
DIFF_LOSS_FOR_REAL_TIME = 0.05 # the minimum difference in loss to update the plot in real time
last_loss = 10e10 # a big number to force the first update
bAddBias = True # set to True to manually adjust the biases in the on_train_begin -> put it towards the center
layer2_init_seed = 403 # initial seed for the second layer training for reproducibility
function_dict: Dict[str, Callable] = {} # dictionary to store the custom feature functions

# Datasets with the 4 possible values for x0 and x1
x0_data = np.array([0, 0, 1, 1])
x1_data = np.array([0, 1, 0, 1])
x_data = np.stack((x0_data, x1_data), axis=-1)
y_true_XOR_data = x0_data ^ x1_data

layer2_signal = SignalTraining()

layer2_weights_history = WeightsHistory(layer2_signal)

x_input = Input(shape=(2,), name='x') # Input layer of x0 and x1 

# Initialize the fixed output data for the custom feature
z1_fixed_output_data = np.zeros(4) # when we change de custom feature in the combobox, we will update this variable

# endregion initializations of global variables
# --------------------------------------------------------------

# region Second layer model

# Define the custom feature calculation
def my_custom_feature(inputs):
    x0, x1 = tf.split(inputs, num_or_size_splits=2, axis=-1)  # Split the input tensor into x0 and x1
    selected_function_name = function_combobox.get()
    selected_function = function_dict[selected_function_name]
    return tf.cast(selected_function(tf.cast(x0, tf.int32), tf.cast(x1, tf.int32)), tf.float32)

def train_second_layer_model(seed: int):
    """ 
    Create, compile and fit the second layer model 
    In the training process the weights, loss and accuracy are saved in the layer2_weights_history object for each epoch
    """
    # clear the ax plot
    layer2_ax.cla()
    # force plot repaint
    fig.canvas.draw_idle()
    layer2_ax.set_title('XOR Model Decision Boundary', pad=-50) # save some space 

    tf.keras.utils.set_random_seed(seed) # for reproducibility

    # Define the custom feature output
    custom_feature_output = Lambda(my_custom_feature)(x_input)

    # Concatenate the inputs with the custom feature
    combined_input = Concatenate()([x_input, custom_feature_output]) 

    # Define the second layer with a single neuron and sigmoid activation of the combined input
    output = Dense(1, activation='sigmoid', name='z2')(combined_input)

    # Create the full model with the second layer
    full_model = Model(inputs=x_input, outputs=output)

    # Compile the full model
    # full_model.compile(optimizer=Adam(learning_rate=0.3), loss='binary_crossentropy', metrics=['accuracy'])
    full_model.compile(optimizer=Adam(learning_rate=0.2), loss='binary_crossentropy', metrics=['accuracy'])

    # Initialize the callback for recording the weights history
    global layer2_weights_history
    layer2_weights_history = WeightsHistory(layer2_signal) 

    # Train the second layer model
    full_model.fit(x_data, y_true_XOR_data, epochs=300, batch_size=4, verbose=1, 
                # callbacks=[callbacks.EarlyStopping(monitor='loss', patience=30, min_delta=0.1), layer2_weights_history], # for debug
                callbacks=[callbacks.EarlyStopping(monitor='loss', patience=50, min_delta=0.005), layer2_weights_history],
                shuffle=False, # for preserving the reproducibility
                )
    print(f"second layer trained with {seed=}")
    
    # Update the slider with the number of epochs and put it at the 0 position (force the repaint if already at 0)
    slider.config(to=len(layer2_weights_history.weights) - 1)
    if slider.get() == 0:
        slider_on_change(0)
    else:
        slider.set(0)
    # put the focus on the slider
    slider.focus_set()

# endregion Second layer model
# --------------------------------------------------------------

# region UI

# region UI-tkinter
import tkinter as tk
from tkinter import ttk, messagebox
import custom_features
import inspect
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to get all functions starting with "CustomFeature_" from a module
prefix_custom_function = "CustomFeature_"

def get_functions(module):
    return {
        name[len(prefix_custom_function):]: func
        for name, func in inspect.getmembers(module, inspect.isfunction)
        if name.startswith(prefix_custom_function)
    }

# Get all functions from custom_feature.py
function_dict = get_functions(custom_features)

# Function to center the window on the screen
def center_window(window):
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    x -= 8 # Account for window border width
    y -= 16 # Account for window title bar height
    window.geometry(f'{width}x{height}+{x}+{y}')

# Create the main window
root = tk.Tk()
root.title("NN XOR with Custom Feature")

# Set the initial size of the window
root.geometry("1000x1000")
root.minsize(750, 850)
center_window(root)
root.update_idletasks()

# Function to update the grid with custom function results
def update_grid():
    selected_function_name = function_combobox.get()
    selected_function = function_dict[selected_function_name]
    global z1_fixed_output_data
    for i, (x0, x1) in enumerate(coordinates):
        result = selected_function(x0, x1)
        lst_result_labels[i].config(text=f"{result:.2f}")
        z1_fixed_output_data[i] = result

# Function to update the grid when the combobox selection changes and set the seed for RND
def combobox_on_change(event):
    custom_features.set_global_Rnd_Seed(int(rng_seed_entry.get()))
    update_grid()

# Function to allow only digits in the entry box
def validate_digit_input(new_value):
    ret = new_value.isdigit() or new_value == ""
    ret = ret and len(new_value) <= 6
    print(f"{new_value=} {ret=}")
    return ret

def on_key_release(event):
    custom_features.set_global_Rnd_Seed(int(rng_seed_entry.get().strip() or 0))
    update_grid()

# Function to insert a random integer into rng_seed_entry
def insert_random_seed():
    np.random.seed(int(time.time()))
    rnd_int = np.random.randint(0, 1000)
    rng_seed_entry.delete(0, tk.END)
    rng_seed_entry.insert(0, str(rnd_int))
    combobox_on_change(None)

# Function to display the value of rng_seed_entry
def train_model():
    if rng_seed_entry.get().strip() == "":
        rng_seed_entry.delete(0, tk.END)
        rng_seed_entry.insert(0, "0")
    train_second_layer_model(int(rng_seed_entry.get()))

# Create the upper zone frame
upper_frame = tk.Frame(root)
upper_frame.pack(side=tk.TOP, fill=tk.X, anchor="n")

# Create a nested frame to hold the label and combobox
nested_frame = tk.Frame(upper_frame)
nested_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

# Add label "Custom Feature" above the combobox in the nested frame
custom_feature_label = tk.Label(nested_frame, text="Custom Feature", font=('Helvetica', 10, 'bold'))
custom_feature_label.grid(row=0, column=0, padx=4, pady=(0, 1), sticky="w")

# Create the combobox in the same nested frame
function_combobox = ttk.Combobox(nested_frame, values=list(function_dict.keys()), state='readonly')
function_combobox.current(0)
function_combobox.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

# Create the right part of the upper frame with a 4x2 grid
grid_frame = tk.Frame(upper_frame)
grid_frame.grid(row=0, column=1, padx=1, pady=1, sticky="nsew")

# Define coordinates for the grid
coordinates = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Fill the grid with static text and create labels for results
lst_result_labels = []
for row, (x0, x1) in enumerate(coordinates):
    label = tk.Label(grid_frame, text=f"({x0}, {x1}) =")
    label.grid(row=row, column=0, padx=1, pady=1, sticky="e")
    result_label = tk.Label(grid_frame, text="", anchor="e", bg="white", width=5, bd=1, relief="solid")  # Right-align the text
    result_label.grid(row=row, column=1, padx=1, pady=1, sticky="e")
    lst_result_labels.append(result_label)

# Bind the combobox selection to the update_grid function
function_combobox.bind("<<ComboboxSelected>>", combobox_on_change)

# Call update_grid initially to populate the grid with default values
update_grid()

# Create the new frame to the right of grid_frame
control_frame = tk.Frame(upper_frame, bd=1, relief="solid")
control_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

# First row: Label "Rng Seed", Entry box, and Button "Rnd"
rng_seed_label = tk.Label(control_frame, text="Rng Seed")
rng_seed_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

vcmd = (control_frame.register(validate_digit_input), '%P')
rng_seed_entry = tk.Entry(control_frame, validate='key', validatecommand=vcmd)

# set max 6 characters for the entry
rng_seed_entry.config(width=6)

# Bind the <<Modified>> event to the entry widget
rng_seed_entry.bind('<KeyRelease>', on_key_release)

# Insert the initial seed value
rng_seed_entry.insert(0, str(layer2_init_seed))

# Place the entry box in the control frame
rng_seed_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

# on lost focus, if the entry is empty, insert 0, then call the combobox_on_change
def on_focus_out(event):
    if rng_seed_entry.get().strip() == "":
        rng_seed_entry.delete(0, tk.END)
        rng_seed_entry.insert(0, "0")
    combobox_on_change(None)
rng_seed_entry.bind("<FocusOut>", on_focus_out)

# add a "Rnd" button to insert a random seed in the entry box
rnd_button = tk.Button(control_frame, text="Rnd", command=insert_random_seed)
rnd_button.grid(row=0, column=2, padx=5, pady=5, sticky="w")

# Second row: Button "Train model"
train_button = tk.Button(control_frame, text="Train model", command=train_model)
train_button.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

# Create the frame to the right of control_frame for "Loss" and "Accuracy"
stats_frame = tk.Frame(upper_frame)
stats_frame.grid(row=0, column=3, padx=10, pady=10, sticky="nsew")

# First row: Label "Loss: " and a label with white background and thin border
loss_label = tk.Label(stats_frame, text="Loss: ")
loss_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

loss_value = tk.Label(stats_frame, text="0.12345", bg="white", bd=1, relief="solid", width=10, anchor='w')
loss_value.grid(row=0, column=1, padx=5, pady=5, sticky="w")

# Second row: Label "Accuracy: " and a label with white background and thin border
accuracy_label = tk.Label(stats_frame, text="Accuracy: ")
accuracy_label.grid(row=1, column=0, padx=5, pady=0, sticky="w")

accuracy_value = tk.Label(stats_frame, text="0.75   ", bg="white", bd=1, relief="solid", width=10, anchor='w')
accuracy_value.grid(row=1, column=1, padx=5, pady=0, sticky="w")

# in the 2nd row of stats_frame add a checkbox "Show Progress in real-time" span col0 + col1
show_progress_var = tk.BooleanVar()
show_progress_var.set(SHOW_PROGRESS_IN_REAL_TIME)
show_progress_checkbox = tk.Checkbutton(stats_frame, 
                                        text=f"Show Progress in real-time (min diff loss = {DIFF_LOSS_FOR_REAL_TIME:.3})", 
                                        variable=show_progress_var)
show_progress_checkbox.grid(row=2, column=0, columnspan=2, padx=5, pady=(3, 0), sticky="w")


# Create the middle zone frame for the matplotlib plot
middle_frame = tk.Frame(root,  bd=1, relief="solid")
middle_frame.pack(fill=tk.BOTH, expand=True)

# Create the bottom zone frame
bottom_frame = tk.Frame(root, height=50)
bottom_frame.pack_propagate(False)  # Prevent bottom_frame from resizing
bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, anchor="s")

# plot the model at the epoch indicated by the slider
def slider_on_change(str_val):
    plot_model(int(str_val))

# Create a horizontal slider in the bottom frame
slider = tk.Scale(bottom_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=slider_on_change)
slider.pack(fill=tk.X, padx=10, pady=0)

# Function to focus the slider on left/right click
def focus_slider(event):
    slider.focus_set()

# Bind mouse click to the slider to regain focus
slider.bind("<Button-1>", focus_slider)
slider.bind("<Button-3>", focus_slider)

# Set the focus on the slider on startup
slider.focus_set()
# endregion UI-tkinter
# --------------------------------------------------------------

# region UI-plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
# Filter out the specific VisibleDeprecationWarning from mpl_toolkits.mplot3d.proj3d and art3d
# Filter out the specific VisibleDeprecationWarning from mpl_toolkits.mplot3d.proj3d
warnings.filterwarnings("ignore", message="Creating an ndarray from ragged nested sequences.*", category=np.VisibleDeprecationWarning, module="mpl_toolkits.mplot3d.proj3d")

# Filter out the same VisibleDeprecationWarning from mpl_toolkits.mplot3d.art3d
warnings.filterwarnings("ignore", message="Creating an ndarray from ragged nested sequences.*", category=np.VisibleDeprecationWarning, module="mpl_toolkits.mplot3d.art3d")

# Ignore MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Alternatively, to catch a broader range of warnings, including MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=UserWarning)

# Create a placeholder for the matplotlib plot in the middle zone
fig = plt.figure(num="NN XOR by Custom Feature") 
layer2_ax: Axes3D = fig.add_subplot(111, projection='3d')

# Remove padding/margins
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
layer2_ax.set_position([0, 0, 1, 0.97])  # [left, bottom, width, height] let some space for the title

canvas = FigureCanvasTkAgg(fig, master=middle_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def plot_model(val:int):
    print(f" current epoch = {val}")
    # always update the texts
    loss_value.config(text=f"{layer2_weights_history.loss[int(val)]:.5f}")
    accuracy_value.config(text=f"{layer2_weights_history.accuracy[int(val)]:.2f}")
    global last_loss
    # see if we need to update the plot
    if (not layer2_signal.onTraining or 
        (show_progress_var.get() and 
         (abs(layer2_weights_history.loss[int(val)] - last_loss) > DIFF_LOSS_FOR_REAL_TIME))
        ):
        last_loss = layer2_weights_history.loss[int(val)]
        
        # Get the weights and bias of the full model at the slider value
        weights = layer2_weights_history.weights[val][0]
        a, b, c = weights[0][0], weights[1][0], weights[2][0]
        d = layer2_weights_history.weights[val][1][0] # bias
        print(f"Second layer decision boundary plain: {a}•𝓧 + {b}•𝓨 + {c}•𝓩 + {d} = 0")
        
        # Define the x, y plane
        x_meshgrid = np.linspace(-0.5, 1.5, 10)
        y_meshgrid = np.linspace(-0.5, 1.5, 10)
        x_meshgrid, y_meshgrid = np.meshgrid(x_meshgrid, y_meshgrid)

        # Calculate z values (z = -(ax + by + d) / c)
        z = -(a*x_meshgrid + b*y_meshgrid + d) / c

        # Clear the current plot
        layer2_ax.cla()

        # Get the predicted z2_output values at the slider value
        z2_output_pred = layer2_weights_history.predictions[val]

        # Calculate z values for the points
        z_points = -(a*x0_data + b*x1_data + d) / c

        # Determine whether each point is above or below the plane
        above_plane = z2_output_pred.ravel() >= z_points

        # Add labels to the points with the predicted z2_output values
        for i in range(len(x0_data)):
            layer2_ax.text(
                x0_data[i], x1_data[i], z1_fixed_output_data[i],
                f'  ({x0_data[i]}, {x1_data[i]}, {z1_fixed_output_data.ravel()[i]:.2f})\n= {z2_output_pred[i][0]:.2f}',
                va='center_baseline', ha='left', ma='center',
                )

        layer2_ax.set_title('XOR Model Decision Boundary', pad=-50)
        layer2_ax.set_xlabel('x0_data')
        layer2_ax.set_ylabel('x1_data')
        layer2_ax.set_zlabel('custom_feature_data')

        # Input data
        # construct X from x0_data, x1_data and z1_fixed_output_data
        X = np.column_stack((x0_data, x1_data, z1_fixed_output_data))

        Y = np.array([0, 1, 1, 0])  # Labels for the XOR

        # gets the positions of input data points: above the plane = TRUE, below = FALSE
        above_plane = z1_fixed_output_data.ravel() >= z_points
        
        above_plane_class0 = above_plane[Y==0] # get the indexes of the input data points for class 0
        facecolors_class0 = ['r' if ap else 'b' for ap in above_plane_class0]

        above_plane_class1 = above_plane[Y==1] # get the indexes of the input data points for class 1
        facecolors_class1 = ['r' if ap else 'b' for ap in above_plane_class1]
        
        # plot the 1-st class points, with facecolors according to the position relative to the plane
        layer2_ax.scatter(X[Y==0][:, 0], X[Y==0][:, 1], X[Y==0][:, 2], 
                          color='red', 
                          facecolors=facecolors_class0, 
                          label='Class 0', 
                          s=50, linewidths=1.7 , alpha=1.0) 
        
        # plot the 2-nd class points, with facecolors according to the position relative to the plane
        layer2_ax.scatter(X[Y==1][:, 0], X[Y==1][:, 1], X[Y==1][:, 2], 
                          color='blue', 
                          facecolors=facecolors_class1, 
                          label='Class 1', 
                          s=50, linewidths=1.7 , alpha=1.0)

        # Plot the decision boundary plane
        layer2_ax.plot_surface(x_meshgrid, y_meshgrid, z, alpha=0.4, rstride=1, cstride=1, color='yellow')
        
        # plot the horizontal plane at z=0
        layer2_ax.plot_surface(x_meshgrid, y_meshgrid, np.zeros_like(x_meshgrid), alpha=0.2, rstride=1, cstride=1, color='cyan')

        # Prompt: add a legend with this 2 colors of the 2 planes
        # Copilot: Unfortunately, the plot_surface function in Matplotlib does not support adding a legend directly. 
        #   However, you can create a workaround by plotting invisible scatter plots with the same colors and adding a legend for those. 
        #   Here's how you can do it:
        
        # Create invisible scatter plots for the legend
        scatter1 = layer2_ax.scatter([], [], [], color='yellow', alpha=0.7)
        scatter2 = layer2_ax.scatter([], [], [], color='cyan', alpha=0.4)
        scatter3 = layer2_ax.scatter([], [], [], color='#80bf3b', alpha=1)
        
        # Add the legend
        layer2_ax.legend([scatter1, scatter2, scatter3], 
                         ['Decision boundary', 'z1=0 plane', 'Intersection line'], 
                         bbox_to_anchor=(0.8, 0.98), loc='upper left', facecolor='#f2f2f2'
                         )

        # the x, y domains of the plot of the intersection line
        xx = np.linspace(-0.5, 1.5, 100)
        # Solve the equation of the decision boundary plane for z = 0
        yy = -(a*xx + d) / b

        yyclip = yy[(yy >= -0.5) & (yy <= 1.5)]
        xxclip = xx[(yy >= -0.5) & (yy <= 1.5)]
        
        # Plot the intersection line between the two planes (z=0 and the decision boundary)
        zz = np.zeros_like(xxclip)
        layer2_ax.plot(xxclip, yyclip, zz, color='forestgreen', alpha=0.8)

        # Set the limits of the z-axis
        layer2_ax.set_xlim(-0.5, 1.5)
        layer2_ax.set_ylim(-0.5, 1.5)
        layer2_ax.set_zlim(-1, 2)

        # Mark the origin by drawing the 3 axes
        #  x-axis
        layer2_ax.quiver(0, 0, 0, 1.5, 0, 0, color='black', alpha=0.5, arrow_length_ratio=0.1) 
        #  y-axis
        layer2_ax.quiver(0, 0, 0, 0, 1.5, 0, color='black', alpha=0.5, arrow_length_ratio=0.1)
        #  z-axis
        layer2_ax.quiver(0, 0, 0, 0, 0, 1.5, color='black', alpha=0.5, arrow_length_ratio=0.1)

        # Generate the ticks
        ticks = np.arange(-0.5, 1.5, 0.5)

        # Set the ticks
        layer2_ax.set_xticks(ticks)
        layer2_ax.set_yticks(ticks) 
        # layer2_ax.set_zticks(ticks) 

        fig.canvas.draw_idle()
    # end if need to update the plot
# end plot_model function
# --------------------------------------------------------------

# see progress in 'real time'
def setSliderMaxFromTrainning (val: int):
    slider.config(to=(val))
    slider.set(val)
    # force repaint the slider
    slider.update()

# Connect the signal to the plot_model function
layer2_signal.connect(setSliderMaxFromTrainning)

# endregion UI-plot
# --------------------------------------------------------------

# endregion UI
# --------------------------------------------------------------

# Train the second layer model with the initial seed
train_second_layer_model(layer2_init_seed)

# Start the Tkinter event loop
root.mainloop()
