    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit
    from statsmodels.api import OLS
    from statsmodels.tools.tools import add_constant
    import matplotlib.pyplot as plt
    import json
    from PIL import Image
    from io import BytesIO
    import tkinter as tk
    from tkinter import filedialog


    # Function to open a file dialog for Excel file selection
    def select_excel_file():
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        file_path = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        return file_path


    # Get the Excel file via a file dialog
    file_path = select_excel_file()

    # Load the selected Excel file
    measurements = pd.read_excel(file_path)

    # Sort the data by the 'Day' column, in ascending order
    measurements = measurements.sort_values(by='Day')

    # Convert the sorted DataFrame to JSON format for further processing
    measurements = measurements.to_json(orient='records')
    measurements = json.loads(measurements)

    t = []
    y = []

    for measurement in measurements:
        day = measurement["Day"]
        width = measurement["Width"]  # Convert width to the desired unit
        t.append(day)
        y.append(width)


    def logit(p):
        # Handle edge cases where p is 0 or 100 to avoid division by zero
        p = np.clip(p, 1e-8, 100 - 1e-8)  # Clip values to avoid 0 and 100
        return np.log(p / (100 - p))


    def richards_model(t, L, K, m, W0):
        return L / (1 + np.exp(-K * (t - m))) + W0


    def richards(t, y, name=None):
        t = np.array(t)  # Convert t to a NumPy array
        y = np.array(y)  # Ensure y is a NumPy array
        t0 = [y[idx] for idx in range(len(t)) if t[idx] == 1]
        W0 = sum(t0) / len(t0) if t0 else 0

        # Linear regression to find initial parameters
        df = pd.DataFrame({'t': t, 'y': y})
        df['logit_y'] = logit(df['y'])
        model = OLS(df['logit_y'], add_constant(df['t'])).fit()
        parameter = model.params

        # Non-linear regression using the Richards growth model
        try:
            popt, pcov = curve_fit(
                lambda t, L, K, m: richards_model(t, L, K, m, W0),
                t,
                y,
                p0=[max(y), 1, np.median(t)],
                maxfev=5000  # Increase the maximum number of function evaluations
            )
        except RuntimeError as e:
            print(f"Error in curve fitting: {e}")
            return None

        L, K, m = popt
        print(f"L: {L}, K: {K}, m: {m}")

        # Calculate the standard errors from the covariance matrix
        perr = np.sqrt(np.diag(pcov))
        L_err, K_err, m_err = perr
        W0_err = 0  # As W0 is derived, there is no direct standard error

        # Calculate 95% confidence intervals
        z_value = 1.96  # Z-value for 95% confidence interval
        L_CI = [L - z_value * L_err, L + z_value * L_err]
        K_CI = [K - z_value * K_err, K + z_value * K_err]
        m_CI = [m - z_value * m_err, m + z_value * m_err]

        # Save the parameters, their standard errors, and confidence intervals into a CSV file
        param_df = pd.DataFrame({
            'Parameter': ['L', 'K', 'm', 'W0'],
            'Value': [L, K, m, W0],
            'Standard Error': [L_err, K_err, m_err, W0_err],
            '95% CI Lower': [L_CI[0], K_CI[0], m_CI[0], None],
            '95% CI Upper': [L_CI[1], K_CI[1], m_CI[1], None]
        })
        param_df.to_csv(f'{file_path.split("/")[-1].replace(".xlsx", "")}_richards_parameters.csv', index=False)

        t2 = np.array(range(1, max(t) + 1))
        predict_BW = richards_model(t, L, K, m, W0)
        predict_BW_p = richards_model(t2, L, K, m, W0)
        AGR = np.gradient(predict_BW_p, t2)
        RGR = AGR / (predict_BW_p + 1e-8)  # Adding a small constant to avoid division by zero

        return t, t2, predict_BW, AGR, RGR


    t, t2, predict_BW, AGR, RGR = richards(t, y)

    # Plotting
    # model fitting
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('', fontsize=18) # Age (day)
    ax1.set_ylabel('', fontsize=18) #Predicted body width (cm)
    ax1.plot(t, y, 'o', label='Observed data', markersize=10)
    ax1.plot(t, predict_BW, '-', label='Model prediction', linewidth=3)

    # Ensure the x-axis is limited between 0 and 15 days with ticks every 1 day
    ax1.set_xlim(0, 15)
    ax1.set_xticks(range(0, 16, 3))  # Set ticks at every day from 0 to 15

    # Ensure the y-axis is limited between 0 and 1.0 cm
    ax1.set_ylim(0, 1.0)

    ax1.tick_params(axis='x', labelsize=24)
    ax1.tick_params(axis='y', labelsize=24)

    fig.tight_layout()

    png1 = BytesIO()
    fig.savefig(png1, format='png')
    png2 = Image.open(png1)
    png2.save(f'{file_path.split("/")[-1].replace(".xlsx", "")}_Richards_model_fitting.tiff')
    png1.close()

    # AGR RGR
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('', fontsize=18) # Age (day)
    ax1.set_ylabel('', color=color, fontsize=18) # Absolute growth rate
    ax1.set_ylim(0, 0.1)
    ax1.plot(t2, AGR, color=color, linewidth=3)

    # Ensure the x-axis is limited between 0 and 15 days with ticks every 1 day
    ax1.set_xlim(0, 15)
    ax1.set_xticks(range(0, 16, 3))  # Set ticks at every day from 0 to 15

    ax1.tick_params(axis='x', labelsize=24)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=24)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('', color=color, fontsize=18)  # RGR we already handled the x-label with ax1
    ax2.set_ylim(0, 0.5)
    ax2.plot(t2, RGR, color=color, linewidth=3)

    ax2.tick_params(axis='y', labelcolor=color, labelsize=24)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    png1 = BytesIO()
    fig.savefig(png1, format='png')
    png2 = Image.open(png1)
    png2.save(f'{file_path.split("/")[-1].replace(".xlsx", "")}_AGRvsRGR.tiff')
    png1.close()
