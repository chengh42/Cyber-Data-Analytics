import pandas as pd
import numpy as np
import time, datetime
import math
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import acf, pacf, ARMA
from statsmodels.tools.eval_measures import rmse

'''
## functions for ARMA model
'''
class ARMA_detection:
    def __init__(self, train, test, signal):
        self.train = train[signal]
        self.test = test[signal]
        self.predictions = []
        self.max_residual = 0
        self.anomalies = []

    # plot autocorrelation plot and partial autocorrelation plot
    def acf_plots(self, lag = 30):
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize = (6, 4))

        fig = plot_acf(self.train, lags = lag, ax = ax1)
        fig = plot_pacf(self.train, lags = lag, ax = ax2)

        ax2.set_xlabel('Lag (hour)')

    # perform aic calculations given orders (p, q) and plot the aic grid
    def aic_heatmap(self, ps = range(1, 3), qs = range(1, 3), title = 'AIC Grid'):
        aic = [[p, q, ARMA(self.train, order=(p, q), freq='H').fit().aic]
               for p in ps for q in qs if ((p!=0)|(q!=0))]
        aic = pd.DataFrame(aic, columns = ['p', 'q', 'aic'])

        # plot heatmap showing aic values vs (p, q) parameters
        sns.heatmap(aic.pivot('p', 'q', 'aic'), 
                    annot = True, fmt='0.1f', linewidths=1)
        plt.title(title)

    # train model to predict the next value per time tick
    def arma_train_predict(self, p, q, learninghr = 100, tolfactor = 1, stepsize = 1, showlog = False):
        history = [x for x in self.train] # set history (=training data)
        model = ARMA(history, order = (p, q)).fit() # first fit of model with train data
        self.max_residual = np.abs(model.resid).max() # get max residual from train data
        residual = [self.max_residual] # initiate an empty list for residuals
        
        # train model and predict the next time tick
        for t in range(math.ceil(len(self.test)/stepsize)):
            model = ARMA(history, order = (p, q)).fit()
            vPred = model.forecast(steps = stepsize)[0]
            vReal = self.test.values[t*stepsize:min((t+1)*stepsize, len(self.test))]
            #print('t: %i\tvPred: %i\tvReal: %i' % (t, len(vPred), len(vReal)))
            if abs(vReal - vPred[:len(vReal)]).max() < self.max_residual*tolfactor: # if within tolerance, update max residual
                self.max_residual = np.abs(residual).max()
            elif (t+1)*stepsize < learninghr: # model still learning
                self.max_residual = np.abs(residual).max()
            else: # residual beyond tolerance = anomaly
                residual_in_step = abs(vReal - vPred[:len(vReal)])
                id_max = np.where(residual_in_step == np.amax(residual_in_step))[0]
                dateAnomaly = self.test.index[t*stepsize + id_max][0]
                self.anomalies += [dateAnomaly]
                print('Anomaly detected at %s : real = %0.2f, predict = %0.2f' % (dateAnomaly, vReal[id_max], vPred[id_max]))
            # book keeping
            self.predictions += vPred[:len(vReal)].tolist()
            history += vReal.tolist()
            residual += (vReal - vPred[:len(vReal)]).tolist()            
            if showlog:
                print('i = %i\treal:%0.2f\tpredict: %0.2f\tresidual: %0.2f' % (t, vReal, vPred, vReal-vPred))

        # output final statistics
        print('Prediction ends!')
        print('The root mean square error for the process is %0.2f.' % rmse(self.test.values, np.array(self.predictions)))
        print('The max tolerated residual (for normal data) is %0.2f.' % self.max_residual)

    # compute and plot residuals
    def compute_residuals(self):
        res = np.subtract(self.test, self.predictions)
        res = abs(res - np.mean(res))
        res.plot()

'''
## functions for Discrete model
'''
class Discrete_detection:
    def __init__(self, train, test, signal):
        self.train = train[signal]
        self.test = test[signal]

'''
## functions for PCA method
'''
class PCA_detection:
    def __init__(self, train, test, signal):
        self.train = train[signal]
        self.test = test[signal]

'''
## functions for making plots
'''

# to be updated