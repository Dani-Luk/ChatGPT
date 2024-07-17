# ChatGPT_Cross_NN.py

"""
This script defines a neural network model for performing bitwise AND and XOR operations.
The model consists of two layers: the first layer performs the AND operation, and the second layer performs the XOR operation.
The weights and predictions of each layer are saved at the end of each epoch using custom callbacks.
The script also includes a user interface for visualizing the model's performance.
"""
# Rest of the code...

from typing import List, Literal, Callable
import time

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.optimizers import Adam
# from keras import initializers
from keras import callbacks 
import tensorflow as tf

# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, Concatenate
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import initializers, callbacks
# from tensorflow.keras.callbacks import EarlyStopping


# Ensure we use the CPU to avoid non-deterministic behavior on the GPU
tf.config.set_visible_devices([], 'GPU')
tf.config.experimental.enable_op_determinism()

SHOW_PROGRESS_IN_REAL_TIME = False # set to True to see the training(current epoch number) in real time
    # NOTE: performance will be affected

bTrainFirstLayer = False # set to True to train the first layer model according with the goals_ax
bAddBias = True # set to True to manually adjust the biases in the on_train_begin -> go to center

# init seeds for the first and second layer models
layer1_init_seed = 1
# layer1_init_seed = np.random.randint(0, 1000)
# layer2_init_seed = np.random.randint(0, 1000)
# layer2_init_seed = 502
layer2_init_seed = 403

# Datasets with 4 possible values for x0 and x1
x0_data = np.array([0, 0, 1, 1])
x1_data = np.array([0, 1, 0, 1])
x_data = np.stack((x0_data, x1_data), axis=-1)
x_data_mean = np.mean(x_data, axis=0) # x_data_mean = np.array([np.mean(x0_data), np.mean(x1_data)])
y_true_XOR_data = x0_data ^ x1_data


class SignalTraining:
    def __init__(self):
        self.callbacks: List[Callable[[int]]] = []
        self.onTraining = True

    def connect(self, callback: Callable[[int], None]):
        self.callbacks.append(callback)
    
    def signal(self, value: int):
        for callback in self.callbacks:
            callback(value)

layer1_signal = SignalTraining()
layer2_signal = SignalTraining()

class WeightsHistory(callbacks.Callback):
    def __init__(self, sig:SignalTraining, previousLayerCallback:"WeightsHistory | None" = None):
        super().__init__()
        self.signal = sig
        self.previousLayerCallback = previousLayerCallback

    def on_train_begin(self, logs={}):
        self.weights = []
        self.predictions = []
        self.loss = []
        self.accuracy = []
        self.signal.onTraining = True
        if self.previousLayerCallback:
            self.previousLayerCallback.signal.onTraining = True            
            self.previousLayerCallback.weights = []
            self.previousLayerCallback.predictions = []
            self.previousLayerCallback.loss= []
            self.previousLayerCallback.accuracy= []
            self.previousLayerCallback.model = Model(inputs=self.model.input, outputs=self.model.layers[-3].output)
            if bAddBias:
                # finding that the probability that the decision boundary line separates the (1,1) point over the rest is very small
                # I thought it might be because of the 0 bias initializers  
                # which therefore forces the line from the beginning to go through the point (0,0). 
                # So I think to myself: let's put the starting point in the middle and see if we balance the results
                init_weights = self.model.layers[-3].get_weights()[0] # get the weights of the first layer
                init_bias = - np.inner(init_weights.ravel(), x_data_mean) # np.inner([w1, w2], [x0_mean, x1_mean])
                self.model.layers[-3].set_weights([init_weights, np.array([init_bias])])
                # doesn't seem to be enough, as long as the second starts at (0,0,0) anyway ... so
        # lets do something for the second layer model too
        if bAddBias:
            int_weights = self.model.layers[-1].get_weights()[0] # get the weights of the second layer
            # int_bias = - np.inner(int_weights.ravel(), np.array([0.5, 0.5, 0])) # lets take the 'obvious' mean, starting however from 0 z-elevation
            int_bias = - np.inner(int_weights.ravel(), np.full(int_weights.shape[0], 0.5)) 
            self.model.layers[-1].set_weights([int_weights, np.array([int_bias])])  
        super().on_train_begin(logs)

    def on_epoch_begin(self, epoch, logs={}):
        self.weights.append(self.model.layers[-1].get_weights())
        self.predictions.append(self.model.predict(x_data))
        if self.previousLayerCallback:
            self.previousLayerCallback.weights.append(self.model.layers[-3].get_weights())
            self.previousLayerCallback.predictions.append(self.previousLayerCallback.model.predict(x_data))
        super().on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.signal.signal(epoch)
        super().on_epoch_end(epoch, logs)
    
    def on_train_end(self, logs=None):
        self.signal.onTraining = False
        if self.previousLayerCallback:
            self.previousLayerCallback.signal.onTraining = False
        self.signal.signal(len(self.weights) - 1) # signal the last epoch at the end of the training
        return super().on_train_end(logs)


layer1_weights_history = WeightsHistory(layer1_signal) # initialized only once it's enough
if not bTrainFirstLayer: # this should be done also in the Go of layer 2
    layer2_weights_history = WeightsHistory(layer2_signal, layer1_weights_history) 
    # the first layer weights history will be recorded also by the second layer training process
else:
    layer2_weights_history = WeightsHistory(layer2_signal)


x_input = Input(shape=(2,), name='x') # Input layer for the first AND the second layer 

# region First layer model

# Target for the first layer: AND function
# y_true_layer1 = np.array([0, 0, 0, 1])  # AND
y_true_layer1 = x0_data & x1_data  # AND
    # then you can try this one
    # y_true_layer1 = np.array([0, 1, 0, 0]) 
    # y_true_layer1 = np.array([1, 0, 0, 0])  # NOT OR
z1_fixed_output_data = np.array([]) # when we move the slider on the first layer model, we will update this variable


def create_first_layer_model() -> Model:
    # Define the first layer model 
    z1_output = Dense(1, activation='sigmoid', name='z1')(x_input)
    # Create the first layer model
    model = Model(inputs=x_input, outputs=z1_output)
    return model

first_layer_model = create_first_layer_model() # we always need it, even if we don't train it, for plotting the decision boundary

def train_first_layer_model(seed: int, y_true_param=y_true_layer1) -> np.ndarray:
    tf.keras.utils.set_random_seed(seed)
    global first_layer_model
    first_layer_model = create_first_layer_model()
    
    # Compile the first layer model
    first_layer_model.compile(optimizer=Adam(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the first layer model
    first_layer_model.fit(x_data, y_true_param, epochs=200, batch_size=4, 
                        verbose=1, 
                        callbacks=[callbacks.EarlyStopping(monitor='loss', patience=20, min_delta=0.1), layer1_weights_history]
                        )
    
    # Freeze the first layer, so its weights are not updated during the training of the second layer
    first_layer_model.trainable = False
    print(f"First layer trained with {seed=}")
    # return the final output of the first layer model for the dataset
    return layer1_weights_history.predictions[-1]

if bTrainFirstLayer:
    z1_fixed_output_data = train_first_layer_model(layer1_init_seed)
    print(f"{z1_fixed_output_data=}")

# endregion First layer model

# --------------------------------------------------------------
# region Second layer model

def train_second_layer_model(seed: int):
    """ 
    Create, compile and fit the second layer model 
    In the training process the weights, loss and accuracy are saved in the layer2_weights_history object for each epoch
    """

    tf.keras.utils.set_random_seed(seed) # for reproducibility

    # Define the second layer model using the output of the first layer (trained or not) 
    global layer2_weights_history
    if bTrainFirstLayer:
        z1_output_fixed = first_layer_model(x_input) # type: ignore # if bTrainFirstLayer the first layer model exist AND is trained
        layer2_weights_history = WeightsHistory(layer2_signal) # standalone, not connected to the first layer model
    else:
        z1_output_fixed = Dense(1, activation='sigmoid', name='z1')(x_input)
        # z1_output_fixed = Dense(1, activation='relu', name='z1')(x_input)
        layer2_weights_history = WeightsHistory(layer2_signal, layer1_weights_history) # connected to the first layer model

    # Second layer for further processing (example XOR operation)
    second_layer_input = Concatenate()([x_input, z1_output_fixed]) # Concatenate the inputs with the output of the first layer

    z2_output = Dense(1, activation='sigmoid', name='z2')(second_layer_input)

    # Create the full model with the second layer
    full_model = Model(inputs=x_input, outputs=z2_output)

    # Compile the full model
    full_model.compile(optimizer=Adam(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the second layer model
    full_model.fit(x_data, y_true_XOR_data, epochs=300, batch_size=4, verbose=1, 
                # callbacks=[callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=0.1), layer2_weights_history],
                callbacks=[callbacks.EarlyStopping(monitor='loss', patience=50, min_delta=0.01), layer2_weights_history],
                shuffle=False, # for preserving the reproducibility
                )
    print(f"second layer trained with {seed=}")

# endregion Second layer model

train_second_layer_model(layer2_init_seed)

# --------------------------------------------------------------
# region UI
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.widgets import Slider, TextBox, Button
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backend_tools import Cursors

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

# --------------------------------------------------------------
# region UI classes
    # region LabelAs_0_1_ToggleButton
class LabelAs_0_1_ToggleButton:
    instances: List['LabelAs_0_1_ToggleButton'] = []
    sigClick: Callable | None = None

    def __init__(self, ax, left, top, text, val:Literal[0, 1] = 0):

        # if val not in (0, 1):
        #     raise ValueError("value must be 0 or 1")
        
        self.ax = ax
        self.top = top
        
        self._bottom = 1 - self.top
        
        # Create the result and operation texts
        self.text_result = self.ax.text(left, self._bottom, '' + str(val), ha='left', va='center', fontsize=10,
                                        bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='square, pad=0.5', lw=0.5))
        self.text_result.set_backgroundcolor('none')
        
        self.text_operation = self.ax.text(left, self._bottom, text, ha='left', va='center', fontsize=10,
                                           bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='square, pad=0.5', lw=0.5))
        self.text_operation.set_backgroundcolor('white')
        
        # Add this instance to the list of instances
        LabelAs_0_1_ToggleButton.instances.append(self)
        
        # Connect event handlers only once
        if len(LabelAs_0_1_ToggleButton.instances) == 1:
            self.fig = ax.figure
            self.fig.canvas.mpl_connect('motion_notify_event', LabelAs_0_1_ToggleButton.on_hover)
            self.fig.canvas.mpl_connect('button_press_event', LabelAs_0_1_ToggleButton.on_click)

        self.value = val # set the value and update the text

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val
        self.text_result.set_text('                ' + str(self._value))
        self.ax.figure.canvas.draw_idle()
        
    @staticmethod
    def on_hover(event):
        for instance in LabelAs_0_1_ToggleButton.instances:
            if instance.text_result.contains(event)[0]:
                # used to check if the mouse event occurred within the area of the text_result object for the given instance of LabelLikeToggleButton.
                # The contains method returns a tuple. 
                # The first element of the tuple is a boolean indicating whether the event occurred within the bounding box of the text_result object. 
                # The second element of the tuple provides additional information (which we are not using in this case).
                instance.text_result.set_color('blue')  # Change text color
                instance.text_result.set_backgroundcolor('0.85')
                instance.ax.figure.canvas.draw_idle()
            else:
                instance.text_result.set_color('black')  # Reset text color
                instance.text_result.set_backgroundcolor('none')  # Reset background color
                instance.ax.figure.canvas.draw_idle()
        

    @staticmethod
    def on_click(event):
        for instance in LabelAs_0_1_ToggleButton.instances:
            if instance.text_result.contains(event)[0]:
                instance.value = 1 - instance.value
                if LabelAs_0_1_ToggleButton.sigClick:
                     LabelAs_0_1_ToggleButton.sigClick()
    # endregion LabelAs_0_1_ToggleButton

class IntTextBox(TextBox):
    def __init__(self, ax, label, initial='0', max_length=5):
        super().__init__(ax, label, initial=initial)
        self.old_value = initial
        self.max_length = max_length
        self.on_text_change(self.on_change)
        self.on_submit(self.on_submit)

    def on_change(self, text):
        if (text.isdigit() or text == "") and (0 <= len(text) <= self.max_length):
            # If the input is a digit and within the length limit, update the old value
            self.old_value = text
        else:
            # If the input is not a valid digit, revert to the old value
            self.set_val(self.old_value)
            self.cursor_index = max(len(text), 1)  # Adjust cursor position if needed
            self._rendercursor()
    
    def on_submit(self, text):
        # print(f"on_submit({text})")
        # hmmm does not trigger ?...
        if text == "":
            self.set_val("0")

    def get_val(self):
        if self.text == "":
            self.set_val("0")
        return int(self.text)
    
    def set_val(self, val):
        TextBox.set_val(self, val)
        self.cursor_index = max(len(val), 1)
        self._rendercursor()
        # Make the TextBox active
        self.begin_typing(None)        
# endregion UI classes

# --------------------------------------------------------------
# Create a plot
fig = plt.figure(num="NN XOR by crossing 2 layers", figsize=(18, 9))

# Add ax for parameters of layer1
goals_ax = fig.add_subplot(131, frameon=True)
goals_ax.set_title(' 1st layer goals\n  (y_true)', loc='left')
goals_ax.set_xticks([])
goals_ax.set_yticks([])
checkTrain1st = goals_ax.text(0.615, 1.013, '☐', ha='center', va='center', fontsize=17,) # fontweight='bold')
checkTrain1st.set_backgroundcolor('none')
checkTrain1st.set_text('☑' if bTrainFirstLayer else '☐')

def checkTrain1st_click(event):
    if checkTrain1st.contains(event)[0]:
        global bTrainFirstLayer
        bTrainFirstLayer = not bTrainFirstLayer
        checkTrain1st.set_text('☑' if bTrainFirstLayer else '☐')
        set_UI_visibility()
        resize_figure(None)

fig.canvas.mpl_connect('button_press_event', checkTrain1st_click)


# Create the 4 instances of LabelAs_0_1_ToggleButton for zip(x0_data, x1_data)
lst_label_x00_x11: List[LabelAs_0_1_ToggleButton] = [
    LabelAs_0_1_ToggleButton(goals_ax, 0.1, 0.05 * (i + 1), 
                             str(x0_data[i]) + " ∘ " + str(x1_data[i]) + " = ", y_true_layer1[i]) 
    for i in range(4)
    ]

# region layer 1
# --------------------------------------------------------------

# Add ax for the first layer model
layer1_ax: Axes = fig.add_subplot(132)  
# layer1_ax.set_aspect('equal')

# texts for the loss and accuracy of the first layer model
layer1_loss_txt = fig.text(0.45, 0.97, f'Loss: ', color='black')
layer1_accuracy_txt = fig.text(0.45, 0.945, f'Accuracy: ', color='black')

# Prompt: add a matplotlib.widgets.TextBox ("Rng Seed") mapping an integer and feed keras with this value on text update
layer1_rng_seed_ax_left = 0.125
layer1_rng_seed_ax_width = 0.04
layer1_rng_seed_ax = plt.axes([layer1_rng_seed_ax_left, 0.95, layer1_rng_seed_ax_width, 0.03])
# initial value is the random seed for the first layer model
rng_seed1_txt = IntTextBox(layer1_rng_seed_ax, 'Rng Seed: ', initial=str(layer1_init_seed)  )

# Prompt: add a button to randomly(0 to 9999) reset the seed when pressed. After the press event put the focus into the TextBox
layer1_btn_randomize_ax = plt.axes([layer1_rng_seed_ax_left + layer1_rng_seed_ax_width + 0.01, 0.95, 0.04, 0.03])
layer1_btn_randomize = Button(layer1_btn_randomize_ax, 'Rnd', color='lightgoldenrodyellow', hovercolor='0.975')
def layer1_randomize_seed(event):
    np.random.seed(int(time.time()))
    rnd_int = np.random.randint(0, 1000)
    rng_seed1_txt.set_val(str(rnd_int))  # Set the value to the corresponding rng seed IntTextBox

layer1_btn_randomize.on_clicked(layer1_randomize_seed)

# add a "GO" button after the Rnd button to start the training of the second layer
layer1_btn_go_ax = plt.axes([layer1_btn_randomize_ax.get_position().x1 + 0.02, 0.95, 0.05, 0.03])
layer1_btn_go = Button(layer1_btn_go_ax, 'Train layer 1', color='lightgreen', hovercolor='0.975')
def on_click1(event):
    layer1_ax.cla()
    layer1_ax.draw(fig.canvas.get_renderer())
    fig.canvas.flush_events()
    global z1_fixed_output_data 
    z1_fixed_output_data = train_first_layer_model(int(rng_seed1_txt.get_val()), np.array(list(map( lambda x: x.value, lst_label_x00_x11))))
    print(f"{z1_fixed_output_data=}")
    if not SHOW_PROGRESS_IN_REAL_TIME: #if REAL_TIME already updated
        layer1_slider.valmax = len(layer1_weights_history.weights) - 1
        layer1_slider.ax.set_xlim(0, layer1_slider.valmax)
        layer1_slider.ax.draw(fig.canvas.get_renderer())  
    layer1_slider.set_val(len(layer1_weights_history.weights) - 1)

layer1_btn_go.on_clicked(on_click1)

# Create a slider for the epoch of the first layer model
layer1_slider_ax = plt.axes([layer1_rng_seed_ax_left - 0.048, 0.03, 0.45, 0.02])
if bTrainFirstLayer:
    layer1_slider = Slider(layer1_slider_ax, 'Epoch: ', 0, len(layer1_weights_history.weights) - 1, valinit=0, valstep=1)
else:
    layer1_slider = Slider(layer1_slider_ax, 'Epoch: ', 0, 1, valinit=0, valstep=1)

# z1_fixed_output_data = np.array([1, 0, 0, 0]) # just to avoid the error in the next line
# Define a function to update the decision boundary
def layer1_slider_on_changed(val:int):
    # always update the texts , if ...
    # z1_fixed_output_data = np.array([1, 0, 0, 0]) # just to avoid the error in the next line
    if bTrainFirstLayer:
        print(f"Layer 1 current epoch = {val}")
        layer1_loss_txt.set_text(f'Loss: {layer1_weights_history.loss[val]:.5f}')
        layer1_accuracy_txt.set_text(f'Accuracy: {layer1_weights_history.accuracy[val]:.2f}')
    if layer1_signal.onTraining:
        if SHOW_PROGRESS_IN_REAL_TIME:
            layer1_loss_txt.draw(fig.canvas.get_renderer())
            layer1_accuracy_txt.draw(fig.canvas.get_renderer())
            fig.canvas.flush_events()
    else:
        # Extract y_true from lst_label_x00_x11
        y_true = np.array(list(map( lambda x: x.value, lst_label_x00_x11)))

        # Extract weights and bias from the first layer of the model
        weights, bias = layer1_weights_history.weights[val]
        w1, w2 = weights[0][0], weights[1][0]
        b = bias[0]
        print(f"First layer decision boundary line: {w1}•𝓧 + {w2}•𝓨 + {b} = 0")

        line_equation = lambda x, y : w1 * x + w2 * y + b

        # Clear the current plot
        layer1_ax.cla()
        layer1_ax.set_xlabel('x0_data')
        layer1_ax.set_ylabel('x1_data')
        layer1_ax.set_title('First Layer Model Decision Boundary')

        # Plot data points, coloring by class and position relative to the side of the decision boundary
        colors = ['r' if point_class else 'b' for point_class in y_true]
        facecolors = ['r' if line_equation(x, y) >= 0 else 'b' for (x, y) in zip(x0_data, x1_data)]
        _ = layer1_ax.scatter(x0_data, x1_data, color=colors, facecolors=facecolors, s=50, linewidths=1.7 , alpha=1.0)

        _point_to_center = {
            (0,0): (0.07, 0.1),
            (0,1): (0.07, 0.9),
            (1,0): (0.78, 0.1),
            (1,1): (0.78, 0.9)
            }

        z1_output_pred = layer1_weights_history.predictions[val] # saved predictions of the first layer model

        # label each point with its prediction (might clutter the plot)
        for i, txt in enumerate(z1_output_pred):
            layer1_ax.annotate(f'{x0_data[i]} ∘ {x1_data[i]} → {txt[0]:.2f} ≈ {y_true[i]}', 
                            xy=(x0_data[i], x1_data[i]),
                            xytext=(_point_to_center[x0_data[i], x1_data[i]]),
                            arrowprops=dict(facecolor='black', shrink=0.07, linewidth=0.5, width=0.1, headwidth=5)
                            )
        epsilon=1e-4
        # Adjust w1 and w2 if they are very close to zero
        if abs(w1) < epsilon:
            w1 = np.sign(w1) * epsilon if w1 != 0 else epsilon
        if abs(w2) < epsilon:
            w2 = np.sign(w2) * epsilon if w2 != 0 else epsilon

        # Calculate the intercepts
        if w2 != 0:
            y0 = -b / w2  # y when x = 0
            y1 = -(w1 + b) / w2  # y when x = 1
        else:
            y0 = y1 = float('inf')  # Vertical line
        
        if w1 != 0:
            x0 = -b / w1  # x when y = 0
            x1 = -(w2 + b) / w1  # x when y = 1
        else:
            x0 = x1 = float('inf')  # Horizontal line

        # Determine min and max values for x and y
        min_x = min(0, x0, x1)
        max_x = max(1, x0, x1)
        min_y = min(0, y0, y1)
        max_y = max(1, y0, y1)

        # Calculate square dimensions
        range_x = max_x - min_x
        range_y = max_y - min_y
        square_dim = max(range_x, range_y)

        plot_min_x = min_x
        plot_max_x = min_x + square_dim
        plot_min_y = min_y
        plot_max_y = min_y + square_dim

        # construct the line
        x_vals = np.linspace(plot_min_x, plot_max_x, 100)
        y_vals = -(w1 * x_vals + b) / w2

        # Plot the decision boundary
        layer1_ax.plot(x_vals, y_vals, 'y--')

        layer1_ax.grid(True, color='grey', linestyle='--', linewidth=0.25, alpha=0.8)

        x_min, x_max = layer1_ax.get_xlim()
        y_min, y_max = layer1_ax.get_ylim()

        # Find the common range
        common_min = min(x_min, y_min)
        common_max = max(x_max, y_max)

        # Set the common range to both axes
        layer1_ax.set_xlim(common_min, common_max)
        layer1_ax.set_ylim(common_min, common_max)

        # NOTE: and then ... ;)
        first_layer_model.set_weights(layer1_weights_history.weights[val]) 
        global z1_fixed_output_data
        z1_fixed_output_data = layer1_weights_history.predictions[val]
        # NOTE: ... we can train the second layer from here!
        layer1_ax.draw(fig.canvas.get_renderer())

    fig.canvas.draw_idle()
    # end if layer1 _signal.onTraining
        
# Connect the function with the slider value changes event
layer1_slider.on_changed(layer1_slider_on_changed)

# redraw first layer on goals changed, for better visualization of points color
def _redraw1stLayer():
    if bTrainFirstLayer:
        layer1_slider_on_changed(layer1_slider.val)
    else:
        layer1_slider_on_changed(layer2_slider.val)
LabelAs_0_1_ToggleButton.sigClick = _redraw1stLayer

def setLayer1SliderMaxFromTrainning (val: int):
    layer1_slider.valmax = val
    layer1_slider.ax.set_xlim(0, val)
    layer1_slider.ax.draw(fig.canvas.get_renderer())
    layer1_slider.set_val(val)
    # fig.canvas.draw_idle()

if SHOW_PROGRESS_IN_REAL_TIME:
    layer1_signal.connect(setLayer1SliderMaxFromTrainning)

if bTrainFirstLayer:
    layer1_slider.set_val(len(layer1_weights_history.weights) - 1)

# endregion layer 1

# --------------------------------------------------------------
# region layer 2
layer2_ax: Axes3D = fig.add_subplot(133, projection='3d')

layer2_loss_text = fig.text(0.89, 0.97, f'Loss: ', color='black')
layer2_accuracy_text = fig.text(0.89, 0.945, f'Accuracy: ', color='black')

# add a matplotlib.widgets.TextBox ("Rng Seed") mapping an integer and feed keras with this value on text update
layer2_rng_seed_ax_left = 0.6
layer2_rng_seed_ax_width = 0.04
layer2_rng_seed_ax = plt.axes([layer2_rng_seed_ax_left, 0.95, layer2_rng_seed_ax_width, 0.03])
layer2_rng_seed_txt = IntTextBox(layer2_rng_seed_ax, 'Rng Seed: ', initial=str(layer2_init_seed))

# add a button to randomly(0 to 9999) reset the seed when pressed. After the press event put the focus into the TextBox
layer2_btn_randomize_ax = plt.axes([layer2_rng_seed_ax_left + layer2_rng_seed_ax_width + 0.01, 0.95, 0.04, 0.03])
layer2_btn_randomize = Button(layer2_btn_randomize_ax, 'Rnd', color='lightgoldenrodyellow', hovercolor='0.975')
def layer2_randomize_seed(event):
    np.random.seed(int(time.time()))
    rnd_int = np.random.randint(0, 1000)
    layer2_rng_seed_txt.set_val(str(rnd_int))  # Set the value to the corresponding rng seed IntTextBox

layer2_btn_randomize.on_clicked(layer2_randomize_seed)

# add a "GO" button after the Rnd button to start the training of the second layer
layer2_btn_go_ax = plt.axes([layer2_btn_randomize_ax.get_position().x1 + 0.02, 0.95, 0.05, 0.03])
layer2_btn_go = Button(layer2_btn_go_ax, 'Train layer 2', color='lightgreen', hovercolor='0.975')
def on_click2(event):
    layer2_ax.cla()
    layer2_ax.draw(fig.canvas.get_renderer())
    fig.canvas.flush_events()
    train_second_layer_model(int(layer2_rng_seed_txt.get_val()))
    # layer2_slider.set_val(len(layer2_weights_history.weights) - 1)
    if not SHOW_PROGRESS_IN_REAL_TIME: #if REAL_TIME already updated
        layer2_slider.valmax = len(layer2_weights_history.weights) - 1
        layer2_slider.ax.set_xlim(0, layer2_slider.valmax)
        layer2_slider.ax.draw(fig.canvas.get_renderer())    
    layer2_slider.reset() # set the slider to the initial value
layer2_btn_go.on_clicked(on_click2)

# Create a slider for the epochs of the second layer model
layer2_slider_ax = plt.axes([layer2_rng_seed_ax_left, 0.03, 0.37, 0.02])
layer2_slider = Slider(layer2_slider_ax, 'Epoch: ', 0, len(layer2_weights_history.weights) - 1, valinit=0, valstep=1)

# Define a function to update the decision boundary
def layer2_slider_on_changed(val:int):
    print(f"Layer 2 current epoch = {val}")
    if not bTrainFirstLayer:
        if not layer2_signal.onTraining or (layer2_signal.onTraining and SHOW_PROGRESS_IN_REAL_TIME): # aka if not layer2_signal.onTraining Or REAL_TIME
            # using escape characters, clear the previous line in the console... have to try it 😁
            print("\033[F\033[K", end="")
            print(f"Layer 2 and Layer 1 current epoch = {val}")
            layer1_slider_on_changed(val) # update also the first layer model decision boundary when the second layer slider is changed
    # always update the texts
    layer2_loss_text.set_text(f'Loss: {layer2_weights_history.loss[val]:.5f}')
    layer2_accuracy_text.set_text(f'Accuracy: {layer2_weights_history.accuracy[val]:.2f}')
    if layer2_signal.onTraining:
        if SHOW_PROGRESS_IN_REAL_TIME:
            layer2_loss_text.draw(fig.canvas.get_renderer())
            layer2_accuracy_text.draw(fig.canvas.get_renderer())
            fig.canvas.flush_events()
    else:
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
            layer2_ax.text(x0_data[i], x1_data[i], z1_fixed_output_data[i], 
                    f'  ({x0_data[i]}, {x1_data[i]}, {z1_fixed_output_data.ravel()[i]:.2f})\n= {z2_output_pred[i][0]:.2f}',
                    va='center_baseline', ha='left', ma='center',
                    )

        layer2_ax.set_title('Second Layer (XOR) Model Decision Boundary')
        layer2_ax.set_xlabel('x0_data')
        layer2_ax.set_ylabel('x1_data')
        layer2_ax.set_zlabel('z1_fixed_output_data')

        # Input data
        # X = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])  # Features
        # construct X from x0_data, x1_data and z1_fixed_output_data
        X = np.column_stack((x0_data, x1_data, z1_fixed_output_data))

        Y = np.array([0, 1, 1, 0])  # Labels

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
        # ax.set_zticks(ticks) 

        fig.canvas.draw_idle()
        
    # end if second_layer_signal.onTraining
        
# Connect the function with the slider value changes event
layer2_slider.on_changed(layer2_slider_on_changed)

def setLayer2SliderMaxFromTrainning (val: int):
    layer2_slider.valmax = val
    layer2_slider.ax.set_xlim(0, val)
    layer2_slider.ax.draw(fig.canvas.get_renderer())
    layer2_slider.set_val(val)
    # fig.canvas.draw_idle()

if SHOW_PROGRESS_IN_REAL_TIME:
    layer2_signal.connect(setLayer2SliderMaxFromTrainning)

layer2_slider.set_val(0)

# endregion layer 2


# --------------------------------------------------------------
# Create the 4 instances of LabelAs_0_1_ToggleButton for zip(x0_data, x1_data)
lst_label_x00_x11: List[LabelAs_0_1_ToggleButton] = [
        LabelAs_0_1_ToggleButton(goals_ax, 0.1, 0.05 * (i + 1), 
                             str(x0_data[i]) + " ∘ " + str(x1_data[i]) + " = ", y_true_layer1[i]) 
        for i in range(4)
        ]

# some cosmetic improvements
def on_hover_HAND(event):
    hover_active = False
    if (layer1_btn_randomize_ax.contains_point((event.x, event.y)) 
        or layer1_btn_go_ax.contains_point((event.x, event.y)) 
        or layer2_btn_randomize_ax.contains_point((event.x, event.y)) 
        or layer2_btn_go_ax.contains_point((event.x, event.y))
        or checkTrain1st.contains(event)[0] 
        ):
        hover_active = True
    else:
        for instance in lst_label_x00_x11:
            if instance.text_result.contains(event)[0]:
                hover_active = True
                break
    if hover_active:
        fig.canvas.set_cursor(Cursors.HAND)  # Change cursor to hand
    else:
        fig.canvas.set_cursor(Cursors.POINTER)  # Revert cursor when not hovering over any text

fig.canvas.mpl_connect('motion_notify_event', on_hover_HAND)

xArtist = []
left_ax_width_inch = 1.5
def resize_figure(event):
    """ keep the width of the parameters axis constant when resizing """

    # Get the current figure size in inches
    fig_width_inch = fig.get_figwidth()
    
    # Recalculate the normalized width of the left axis
    left_ax_width_normalized = left_ax_width_inch / fig_width_inch
    
    # Update the positions of the axes
    width1 = left_ax_width_normalized
    goals_ax.set_position([0, 0.1, width1, 0.8])

    width2 = (1 - width1) / 2
    layer1_ax.set_position([width1, 0.1, width2, 0.8])

    layer2_ax.set_position([width1 + width2, 0.1, width2, 0.8])

    global xArtist
    # Remove existing artists
    if xArtist:
        for artist in xArtist:
            artist.remove()
        xArtist.clear()  # Clear the list after removing the artists

    if not bTrainFirstLayer:
        # draw a X shape formed by 2 lines on the figure fig = plt.figure()
        # the X shape should cover 30 % of top of goals_ax axes and 100% width, by fig.add_artist
        _eps = 0.007
        xArtist.append(fig.add_artist(
            plt.Line2D([0 + _eps, width1 * 0.6 ], [0.71, 0.9 - 2 * _eps], 
                       transform=fig.transFigure, figure=fig, color='red', linestyle='--', lw=1.5)))
        xArtist.append(fig.add_artist(
            plt.Line2D([0 + _eps, width1 * 0.6 ], [0.9 - 2 * _eps, 0.71], 
                       transform=fig.transFigure, figure=fig, color='red', linestyle='--', lw=1.5)))
        
        # xArtist.append(fig.add_artist(plt.Line2D([0.05, width1 - 0.2], [0.75, 0.7], transform=fig.transFigure, figure=fig, color='red', lw=1)))
        
    fig.canvas.draw_idle()

# connect the resize_figure function to the resize event
fig.canvas.mpl_connect('resize_event', resize_figure)

def set_UI_visibility():
    layer1_accuracy_txt.set_visible(bTrainFirstLayer)
    layer1_loss_txt.set_visible(bTrainFirstLayer)
    layer1_rng_seed_ax.set_visible(bTrainFirstLayer)
    layer1_btn_randomize_ax.set_visible(bTrainFirstLayer)
    layer1_btn_go_ax.set_visible(bTrainFirstLayer)
    layer1_slider_ax.set_visible(bTrainFirstLayer)
    # if not bTrainFirstLayer:
    #     xtext = goals_ax.text(0.3, 0.8, 'X', ha='center', va='center', fontsize=90, color='red',)
    #                                     # bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='square, pad=0.5', lw=0.5))
    #     xtext.set_backgroundcolor('none')

set_UI_visibility()

fig.set_size_inches(18, 9, forward=True)

# endregion UI

# Let's see the wonder :D
plt.show()


