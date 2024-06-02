import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError as mse
import get_cv
import get_targets
import get_audio

model = load_model('my_model.keras', custom_objects={'mse': mse}, compile=False)


def get_current_state(data):
    # ---------------------------------------------------------------------------- #
    # The function returns the actually, current-scale values of the room's state. #
    # input: data - the dataset containing the room's state values.                #
    # output: input_values - the current-scale values of the room's state.         #
    # ---------------------------------------------------------------------------- #

    # get the amount of people in the room thanks to the computer vision model
    n_people = get_cv.main()

    # get the other values from the dataset
    ext_temperature = data['ext_temperature'].tail(5).mean()
    temperature_now = data['temperature_now'].tail(5).mean()
    pressure_now = data['pressure_now'].tail(5).mean()
    brightness_now = data['brightness_now'].tail(5).mean()
    humidity_now = data['humidity_now'].tail(5).mean()
    co2_now = data['co2_now'].tail(5).mean()

    window1_now, window2_now, window3_now, window4_now = np.array(data.tail(1)['window1_now'])[0], np.array(data.tail(1)['window2_now'])[0], np.array(data.tail(1)['window3_now'])[0], np.array(data.tail(1)['window4_now'])[0]
    shutter1_now, shutter2_now, shutter3_now, shutter4_now = np.array(data.tail(1)['shutter1_now'])[0], np.array(data.tail(1)['shutter2_now'])[0], np.array(data.tail(1)['shutter3_now'])[0], np.array(data.tail(1)['shutter4_now'])[0]

    date = np.array(data.tail(1)['date'])[0]
    time = np.array(data.tail(1)['time'])[0]
    room_size = np.array(data.tail(1)['room_size'])[0]

    # creating a pd.DataFrame with the current values
    columns = ['n_people', 'room_size', 'date', 'time', 'ext_temperature', 'temperature_now', 'co2_now', 'pressure_now', 'brightness_now', 'humidity_now', 'temperature_opt', 'co2_opt', 'pressure_opt', 'brightness_opt', 'humidity_opt', 'window1_now', 'window2_now', 'window3_now', 'window4_now', 'shutter1_now', 'shutter2_now', 'shutter3_now', 'shutter4_now']
    values = [n_people, room_size, date, time, ext_temperature, temperature_now, co2_now, pressure_now, brightness_now, humidity_now, 0, 0, 0, 0, 0, window1_now, window2_now, window3_now, window4_now, shutter1_now, shutter2_now, shutter3_now, shutter4_now]
    input_values = pd.DataFrame([values], columns=columns)

    return input_values


def get_opt(input_values):
    # ---------------------------------------------------------------------------- #
    # The function returns the target values to achieve.                           #
    # input: input_values - the dataset containing the room's state values.        #
    # output: input_values - the current-scale values of the room's state and      #
    #                           targets.                                           #
    # ---------------------------------------------------------------------------- #

    # Call the function to get the target values
    targets = get_targets.main()

    # Add targets to the input_values DataFrame
    opt_columns = ['temperature_opt', 'co2_opt', 'pressure_opt', 'brightness_opt', 'humidity_opt']
    input_values[opt_columns] = targets

    return input_values


def preprocess_data(data, input_values):
    # ---------------------------------------------------------------------------- #
    # The function returns normalized and cleaned data                             #
    # input: data - the dataset containing the room's date values.                 #
    #        input_values - the current-scale values of the room's state.          #
    # output: input_values - the current-scale values of the room's state.         #
    # ---------------------------------------------------------------------------- #

    # targeting the numerical columns and normalizing them
    numerical_columns = ['n_people', 'room_size', 'ext_temperature', 'temperature_now', 'co2_now', 'pressure_now', 'brightness_now', 'humidity_now', 'temperature_opt', 'co2_opt', 'pressure_opt', 'brightness_opt', 'humidity_opt', 'window1_now', 'window2_now', 'window3_now', 'window4_now', 'shutter1_now', 'shutter2_now', 'shutter3_now', 'shutter4_now']
    input_values[numerical_columns] = (input_values[numerical_columns] - data[numerical_columns].mean()) / data[numerical_columns].std()

    # targeting the date and time columns and convert them into numbers
    input_values['date'] = pd.to_datetime(input_values['date']).dt.month
    input_values['time'] = np.array(pd.to_datetime(data['time'].tail(1), format='%H:%M:%S').dt.hour * 60 + pd.to_datetime(data['time'].tail(1), format='%H:%M:%S').dt.minute)[0]

    return input_values


def denormalize_data(data, prediction):
    # ---------------------------------------------------------------------------- #
    # The function returns denormalized data                                       #
    # input: data - the dataset containing the room's date values.                 #
    #        prediction - the prediction of the model.                             #
    # output: prediction_denormalized - the denormalized prediction.               #
    # ---------------------------------------------------------------------------- #

    # Denormalize the prediction
    prediction_denormalized = prediction.copy()
    prediction_denormalized = prediction_denormalized * data[['window1_tg', 'window2_tg', 'window3_tg', 'window4_tg', 'shutter1_tg', 'shutter2_tg', 'shutter3_tg', 'shutter4_tg']].std() + data[['window1_tg', 'window2_tg', 'window3_tg', 'window4_tg', 'shutter1_tg', 'shutter2_tg', 'shutter3_tg', 'shutter4_tg']].mean()

    return prediction_denormalized


def simulate_adjustments(input_values, current_state_values, prediction_denormalized):
    # ---------------------------------------------------------------------------- #
    # The function returns the adjustments to make to reach the target values.     #
    # input: input_values - the dataset containing the room's state values.        #
    #        current_state_values - the current-scale values of the room's state.  #
    #        prediction_denormalized - the denormalized prediction.                #
    # output: adjustments - the adjustments to make to reach the target values.    #
    # ---------------------------------------------------------------------------- #

    # Get the current and predicted values for windows and shutters
    current_windows = np.array(input_values['window1_now'])[0], np.array(input_values['window2_now'])[0], np.array(input_values['window3_now']), np.array(input_values['window4_now'])[0]
    current_shutters = np.array(current_state_values['shutter1_now'])[0], np.array(current_state_values['shutter2_now'])[0], np.array(current_state_values['shutter3_now']), np.array(current_state_values['shutter4_now'])[0]
    predicted_windows = prediction_denormalized[:4].copy()
    predicted_shutters = prediction_denormalized[4:].copy()

    # Create a dictionary to store the adjustments
    adjustments = {}

    for i, (predicted, current) in enumerate(zip(predicted_windows, current_windows)):
        if int(predicted) - current == 0:           # they do not change
            pass
        elif int(predicted) - current == 1:         # From open to close
            adjustments[f'window{i+1}'] = "Close window" + str(i+1)
        else:                                       # From close to open
            adjustments[f'window{i+1}'] = "Open window" + str(i+1)

    for i, (predicted, current) in enumerate(zip(predicted_shutters, current_shutters)):
        if abs(predicted-current) < 0.2:            # No need to change
            pass
        elif predicted > current:                   # Shutter to lower
            adjustments[f'shutter{i+1}'] = "Lower the shutter " + str(i+1) + " of about " + str(int((predicted-current)*100)) + "%"
        else:                                       # Shutter to upper
            adjustments[f'shutter{i+1}'] = "Upper the shutter " + str(i+1) + " of about " + str(int((current-predicted)*100)) + "%"

    return adjustments


def get_instructions(adjustments):
    # ---------------------------------------------------------------------------- #
    # The function returns the instructions to reach the target values.            #
    # input: adjustments - the adjustments to make to reach the target values.     #
    # output: instructions - the instructions to reach the target values.          #
    # ---------------------------------------------------------------------------- #

    instructions = []
    for key, value in adjustments.items():
        instructions.append(value)
    return instructions


def setup_data():
    # ---------------------------------------------------------------------------- #
    # The function returns the dataset to use for the actions.                     #
    # output: data - the dataset to use for the actions.                           #
    # ---------------------------------------------------------------------------- #

    # Caricare il dataset per preprocessing
    data = pd.read_csv('final_dataset.csv')

    # ordering columns
    new_order = ['n_people', 'room_size', 'date', 'time', 'ext_temperature', 'temperature_now', 'co2_now', 'pressure_now', 'brightness_now', 'humidity_now', 'temperature_opt', 'co2_opt', 'pressure_opt', 'brightness_opt', 'humidity_opt', 'window1_now', 'window2_now', 'window3_now', 'window4_now', 'shutter1_now', 'shutter2_now', 'shutter3_now', 'shutter4_now', 'window1_tg', 'window2_tg', 'window3_tg', 'window4_tg', 'shutter1_tg', 'shutter2_tg', 'shutter3_tg', 'shutter4_tg']
    data = data[new_order]

    return data


def get_current_data(data):
    # ---------------------------------------------------------------------------- #
    # The function returns the current state values and the input values.          #
    # input: data - the dataset containing the room's date values.                 #
    # output: input_values_preprocessed - the current-scale values of the room's   #
    #                                    state.                                    #
    #         current_state_values - the current-scale values of the room's state. #
    # ---------------------------------------------------------------------------- #

    input_values = get_current_state(data)
    current_state_values = input_values[['window1_now', 'window2_now', 'window3_now', 'window4_now', 'shutter1_now', 'shutter2_now', 'shutter3_now', 'shutter4_now']].copy()
    input_values = get_opt(input_values)
    input_values_preprocessed = preprocess_data(data, input_values)

    return input_values_preprocessed, current_state_values


def get_predictions(data, input_values, current_state_values):
    # ---------------------------------------------------------------------------- #
    # The function returns the predictions of the model.                           #
    # input: data - the dataset containing the room's date values.                 #
    #        input_values - the current-scale values of the room's state.          #
    #        current_state_values - the current-scale values of the room's state.  #
    # output: prediction_denormalized - the denormalized prediction.               #
    # ---------------------------------------------------------------------------- #

    # Predict the values
    prediction = model.predict(input_values_preprocessed)

    # Denormalize the prediction
    prediction_denormalized = np.array(denormalize_data(data, prediction[0]))

    # Assume prediction_denormalized is already defined and has 8 elements
    # Adjust the values
    prediction_denormalized[:4] = [int(float(1 if x >= 0.5 else 0)) for x in prediction_denormalized[:4]]
    prediction_denormalized[4:8] = [round(x, 1) for x in prediction_denormalized[4:8]]

    return prediction_denormalized


def get_adjustments(input_values_preprocessed, current_state_values, prediction_denormalized):
    # ---------------------------------------------------------------------------- #
    # The function returns the adjustments to make to reach the target values.     #
    # input: input_values_preprocessed - the current-scale values of the room's    #
    #                                    state.                                    #
    #        current_state_values - the current-scale values of the room's state.  #
    #        prediction_denormalized - the denormalized prediction.                #
    # output: adjustments - the adjustments to make to reach the target values.    #
    # ---------------------------------------------------------------------------- #

    # Get the adjustments
    adjustments = simulate_adjustments(input_values, current_state_values, prediction_denormalized)

    instructions = get_instructions(adjustments)
    for instruction in instructions:
        print(instruction)


def get_audio_file(instructions):
    # ---------------------------------------------------------------------------- #
    # The function returns the audio file to play.                                 #
    # input: instructions - the instructions to reach the target values.           #
    # output: audio - the audio file to play.                                      #
    # ---------------------------------------------------------------------------- #

    # Call external file
    audio = get_audio.get_audio(instructions)
    return audio


def get_microphone():
    # ---------------------------------------------------------------------------- #
    # The function returns the microphone to listen to the user.                   #
    # output: microphone - the microphone to listen to the user.                   #
    # ---------------------------------------------------------------------------- #

    optimal_values = get_opt()
    return optimal_values


def main():
    return
