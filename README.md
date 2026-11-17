# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date:31/10/2026 



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data = pd.read_csv('NVIDIA_STOCK.csv', skiprows=2)

data.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

data['Date'] = pd.to_datetime(data['Date'])

data.set_index('Date', inplace=True)

N = 1000

plt.rcParams['figure.figsize'] = [12, 6]

X = data['Close'].astype(float)

plt.plot(X)
plt.title('Original NVIDIA Stock Data')
plt.show()

plt.subplot(2, 1, 1)
plot_acf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.show()

plot_pacf(ARMA_1)
plt.show()

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.show()

plot_pacf(ARMA_2)
plt.show()
```
OUTPUT:
SIMULATED ARMA(1,1) PROCESS:
<img width="932" height="501" alt="image" src="https://github.com/user-attachments/assets/bb78f84f-36ea-44e9-8436-e4738e79538d" />


Partial Autocorrelation
<img width="932" height="493" alt="image" src="https://github.com/user-attachments/assets/9e61a809-1d9f-412d-adde-514d7884b5fc" />

Autocorrelation
<img width="931" height="500" alt="image" src="https://github.com/user-attachments/assets/90e65596-220f-408f-8aeb-a1666dfde391" />


SIMULATED ARMA(2,2) PROCESS:
<img width="931" height="498" alt="image" src="https://github.com/user-attachments/assets/e18c0b94-bcac-42f6-9823-2a3d80b4fa62" />


Partial Autocorrelation
<img width="933" height="500" alt="image" src="https://github.com/user-attachments/assets/a4f547fb-069c-4fa6-bcd2-0087ecfcf75f" />



Autocorrelation
<img width="932" height="497" alt="image" src="https://github.com/user-attachments/assets/54975094-fb8c-468b-afb1-eb9b6376a9f2" />

### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
