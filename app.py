from flask import Flask, render_template, request
import numpy as np
import pickle
import os
import numpy as np
from scipy.signal import butter, filtfilt

app = Flask(__name__)
model_path = os.path.join('model', 'sub1.pickle')
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

def get_plot(x):


    # Function to create a bandpass filter
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    # Define frequency bands
    delta_band = (0.5, 4)    # Delta band (in Hz)
    theta_band = (4, 8)      # Theta band
    alpha_band = (8, 12)     # Alpha band
    beta_band = (12, 30)     # Beta band
    gamma_band = (30, 45)    # Gamma band

    # Function to split data into frequency bands
    def split_into_frequency_bands(data, fs):
        delta = butter_bandpass_filter(data, delta_band[0], delta_band[1], fs)
        theta = butter_bandpass_filter(data, theta_band[0], theta_band[1], fs)
        alpha = butter_bandpass_filter(data, alpha_band[0], alpha_band[1], fs)
        beta = butter_bandpass_filter(data, beta_band[0], beta_band[1], fs)
        gamma = butter_bandpass_filter(data, gamma_band[0], gamma_band[1], fs)
        return delta, theta, alpha, beta, gamma

    # Example usage
    # Assuming X is your EEG data for one trial
    trial_index = 0  # Choose a specific trial
    fs=256
    # Loop through channels and split into frequency bands
    data_bands = []
    delta, theta, alpha, beta, gamma = split_into_frequency_bands(x, fs)
    channel_bands = np.stack([delta, theta, alpha, beta, gamma], axis=-1)
    data_bands.append(channel_bands)

    data_bands = np.array(data_bands)

    # Display the shape of the resulting data
    print("Data shape after splitting into frequency bands: [channels x samples x bands]")
    print(data_bands.shape)
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    bands= ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    for i in range(5):
        row=data_bands[0][:,i]
        plt.figure(figsize=(12,4))
        plt.axhline(0, color='grey')
        plt.plot(row)
        plt.xlabel('Time')
        plt.ylabel('eV')
        plt.title(f'{bands[i]} Band')
        plt.savefig(os.path.join('static', 'plots', f'{bands[i]}.png')) 

@app.route('/showbands')
def display_bands():
    return render_template('bands.html') 

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        # Check if the file is selected
        if file.filename == '':
            return render_template('index.html', message='No file selected')

        # Check if the file has a valid extension
        if file and file.filename.endswith('.npz'):
            try:
                # Load data from the uploaded .npz file
                data = np.load(file)
                
                # Access the arrays in the file
                x = data['x']
                y = data['y']
                loaded_x=np.squeeze(x)
                pred = model.predict(x)
                # Process the data (replace this with your processing logic)
                # result = f"result={result}"
                pred=f'{int(pred)}'
                y=f'{int(y)}'
                classes=['Up','Down','Left','Right']
                pred_class=classes[int(pred)]
                y_class=classes[int(y)]
                get_plot(loaded_x)
                return render_template('index.html', predicted=pred,y=y,pred_class=pred_class,y_class=y_class)

            except Exception as e:
                return render_template('index.html', message=f"Error processing file: {e}")

        else:
            return render_template('index.html', message='Invalid file format. Please upload a .npz file.')

    return render_template('index.html', message='Upload a .npz file')
if __name__ == '__main__':
   app.run(debug=True)