import pandas as pd
import os
import matplotlib.pyplot as plt 
import re
import numpy as np
import torch
import torch.nn as nn
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf 
from statsmodels.tsa.arima_model import ARIMA  
import torch.optim as optim
class F1():

    static_curr_drivers = {"Verstappen": 830, "Norris": 846, "Piastri": 857, 
                            "Leclerc": 844, "Sainz": 832, "Hamilton": 1, "Perez": 815, 
                             "Russell": 847, "Alonso": 4, "Stroll": 840, "Hulkenberg": 807, 
                              "Tsunoda": 852, "Ricciardo": 817, 'Gasly': 824, 'Magnussen': 825, 
                               'Albon': 848, 'Ocon': 839, 'Zhou': 855, 'Sargeant': 858, 'Bottas': 822}
   
    def __init__(self):
        self.df = None
        self.csv_paths = None
        self.currDrivers = {"Verstappen": 830, "Norris": 846, "Piastri": 857, 
                            "Leclerc": 844, "Sainz": 832, "Hamilton": 1, "Perez": 815, 
                             "Russell": 847, "Alonso": 4, "Stroll": 840, "Hulkenberg": 807, 
                              "Tsunoda": 852, "Ricciardo": 817, 'Gasly': 824, 'Magnussen': 825, 
                               'Albon': 848, 'Ocon': 839, 'Zhou': 855, 'Sargeant': 858, 'Bottas': 822}
        self.curr_drivers_id = list(self.currDrivers.values())
        self.all_drivers_id = None
        self.fastest_lap_times = None
        self.corresponding_fastest_lap_times = {}
        self.lstm = None
        self.selected_player = None
        self.lstm_sequences_x = None
        self.lstm_sequences_y = None
        self.selected_player_fastest_lap_times = None
        self.sequence_length_X = None
        self.sequence_length_Y = None
    def df_info(self):
        print(f"length: {len(self.df)}")
        #print(self.df.head())
    def extract_csv_data(self, input_csv):
        p = os.getcwd()
        p = os.path.join(p, "F1_Multi_CSV_DATA")
        p = os.path.join(p, input_csv)
        self.df = pd.read_csv(p)
       # print(self.df.head())
    
    def collect_csv(self):
        path = os.getcwd()
        path  = os.path.join(path, "F1_Multi_CSV_DATA")
        csv_paths = os.listdir(path)
        self.csv_paths = csv_paths
       # print(self.csv_paths)

# make it so that the in GUI you enter the CSV?? 
    def clean_organize_race_win_predictions(self):
        self.fastest_lap_times = self.df['fastestLapTime']
        self.all_drivers_id = self.df["driverId"]
       # print(self.fastest_lap_times)
        for id in self.curr_drivers_id:
            times= []
            for index, time in enumerate(self.fastest_lap_times):
                 if self.all_drivers_id[index] == id:
                     times.append(self.fastest_lap_times[index])
            
            self.corresponding_fastest_lap_times[id] = times
        #print(self.corresponding_fastest_lap_times)

        for key, value, in self.corresponding_fastest_lap_times.items():
            v = value
           # print(v)
           # print(len(v))
            index = 0
            for value in v:
               # print(i)
                if v[index] == "\\N":
                    #print("hello")
                    del v[index]
                    index = index - 1
                index = index + 1
            self.corresponding_fastest_lap_times[key] = v
            #print(self.corresponding_fastest_lap_times[key])
        #print(self.corresponding_fastest_lap_times)
        
    def display_player_times(self, player_name):
        fastest_lap_times = self.corresponding_fastest_lap_times[self.currDrivers[player_name]]
        # print(fastest_lap_times)
        x = [i + 1 for i in range(len(fastest_lap_times))]
        fastest_lap_times = [float(x) for x in fastest_lap_times]
        plt.ylabel("Fastest Lap Time in Seconds")
        plt.xlabel("Fastest lap times for Grand Prix(1950-2024)")
        plt.plot(x, np.array(fastest_lap_times))
        plt.show()
    def display_player_times_scatter(self, player_name):
        fastest_lap_times = self.corresponding_fastest_lap_times[self.currDrivers[player_name]]
        # print(fastest_lap_times)
        x = [i + 1 for i in range(len(fastest_lap_times))]
        fastest_lap_times = [float(x) for x in fastest_lap_times]
        plt.ylabel("Fastest Lap Time in Seconds")
        plt.xlabel("Fastest lap times for each Grand Prix(1950-2024)")
        plt.plot(x, np.array(fastest_lap_times), 'o')
        plt.show()
    def time_parsing(self):
        pass
        #1:30.454
         #self.
        for index, racer in enumerate(list(self.corresponding_fastest_lap_times.values())):
           #  print(racer)
             updated_times = []
             for time in racer:
                 
                 pattern = r"\d[:]\d\d[.]\d\d\d"
                 result = re.match(pattern, time)
                 if result:
                    t = time.replace(":", "")
                    t = t.replace(".", "")
                   # print(t)
                    minutes = int(t[0])
                    seconds = int(t[1:3])
                    milliseconds = int(t[3:])

                    time_in_seconds = (60 * minutes) + seconds + (milliseconds/1000)
                    #print(time_in_seconds)
                    updated_times.append(time_in_seconds)
             id = self.curr_drivers_id[index]
             self.corresponding_fastest_lap_times[id] = updated_times
    
    def check_stationarity(self, player_name):
        self.selected_player = self.currDrivers[player_name]
        selected_player_fastest_lap_times = self.corresponding_fastest_lap_times[self.currDrivers[player_name]]
        result = adfuller(selected_player_fastest_lap_times)
        print(result)
      
    def plot_acf(self, player_name):
        fastest_lap_times = self.corresponding_fastest_lap_times[self.currDrivers[player_name]]
        d = pd.Series(fastest_lap_times)
        #plot_pacf(d, lags=50, ax=plt.gca())
        plot_acf(d, lags=len(fastest_lap_times) / 2, ax=plt.gca())
        plt.show()
    def plot_pacf(self, player_name):
        fastest_lap_times = self.corresponding_fastest_lap_times[self.currDrivers[player_name]]
        d = pd.Series(fastest_lap_times)
        #plot_pacf(d, lags=50, ax=plt.gca())
        plot_pacf(d, lags=50, ax=plt.gca())
        plt.show()

    def min_max_normalization(self):
        # incorrect normalization make sure to normalize it between -1 to 1
        # self.normalized_times = []
        # for index, racer in enumerate(list(self.corresponding_fastest_lap_times.values())):
        #      normalized_times = torch.tensor(racer, dtype= torch.float32)
        #      max = normalized_times.max()
        #      min = normalized_times.min()
        #      normalized_times = 2 * (normalized_times - min) / (max - min)  - 1
        #      normalized_times = list(normalized_times)
        #      normalized_times = [float(time) for time in normalized_times]
        #      self.normalized_times.append(normalized_times)
        self.selected_player_fastest_lap_times = torch.tensor(self.selected_player_fastest_lap_times, dtype= torch.float32)
        self.max = self.selected_player_fastest_lap_times.max()
        self.min = self.selected_player_fastest_lap_times.min()
        self.selected_player_fastest_lap_times = 2 * (self.selected_player_fastest_lap_times - self.min) / (self.max - self.min)  - 1
        self.selected_player_fastest_lap_times = list(self.selected_player_fastest_lap_times)
        self.selected_player_fastest_lap_times = [float(time) for time in self.selected_player_fastest_lap_times]
        #print(self.normalized_times)
    def sliding_window(self, fixed_window_size, future_steps = 1):
        counter = 0
        self.sequence_length_X = fixed_window_size -1
        self.lstm_sequences_x = []
        self.lstm_sequences_y = []
        self.lstm_sequence_test_X = []
        self.lstm_seqwuence_test_y = []
        self.lstm_pred_sequence_X = []
        self.lstm_pred_sequence_y = []
        #self.lstm
        if fixed_window_size <= len(self.selected_player_fastest_lap_times): 
            # establish a training and testing limit 
            training_limit = int(.95 * len(self.selected_player_fastest_lap_times))
            sequence = None

            for i in range(training_limit):
                sequence = self.selected_player_fastest_lap_times[i:i + fixed_window_size] 
                sequence = [[feature] for feature in sequence]
                if counter % fixed_window_size==0:
                    #print(counter, fixed_window_size)
                    self.lstm_sequences_x.append(sequence[:-1])
                    self.lstm_sequences_y.append(sequence[-1])
                    #print(self.lstm_sequences_x.shape)
                    
                counter +=1
            counter+=1

            print(counter)
            # generate Test Sequence
            test_sequence = self.selected_player_fastest_lap_times[counter:]
            test_sequence = [[feature] for feature in test_sequence]
            self.lstm_sequence_test_X.append(test_sequence[:-1])
            self.lstm_seqwuence_test_y.append(test_sequence[-1])
            #print(self.lstm_seqwuence_test_y[0])
            # generate Entire series for prediction

            pred_sequence = self.selected_player_fastest_lap_times
            pred_sequence = [[feature] for feature in pred_sequence]
            self.lstm_pred_sequence_X.append(pred_sequence)
            
            
        else:
            print("Too large of a window size")
            return
    def arima(self):
        
        pass
    def train_LSTM_Model(self, hidden_size, feature_size, num_layers, output_size, num_epochs):
    
        X = torch.tensor(self.lstm_sequences_x, requires_grad=True)
        y = torch.tensor(self.lstm_sequences_y)
        print(X.shape)
        self.model = LSTM(hidden_size, feature_size, num_layers, output_size )
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=.001)
        for epoch in range(num_epochs):# train
            self.optimizer.zero_grad()
            out = self.model.forward(X)
            
            #if epoch == num_epochs - 10: print(out, y)
            #print(out.shape, y.shape)
            loss = self.loss_function(out, y)
           #   print()
            arc_pred = ((float(out[6]) + 1)/2) * (self.max - self.min) + self.min
            arc_actual = ((float(y[6]) + 1)/2) * (self.max - self.min) + self.min
            print(f"Epoch Num: {epoch + 1} Training Loss: {loss}, Model Prediction: {arc_pred}, Target: {arc_actual}")
            loss.backward()
            self.optimizer.step()
        
    def test_LSTM_model(self):
        test_X = torch.tensor(self.lstm_sequence_test_X)
        test_y = torch.tensor(self.lstm_seqwuence_test_y)
        #print(test_X.size(), test_y.size())
        print(test_X.shape)
        out = self.model.forward(test_X)
        #print(out.shape,test_y.shape )
        loss = self.loss_function(out, test_y)
        arc_pred = ((float(out[0]) + 1)/2) * (self.max - self.min) + self.min
        arc_actual = ((float(test_y[0]) + 1)/2) * (self.max - self.min) + self.min
        print(f"Testing Loss: {loss}, Model Prediction: {arc_pred}, Target: {arc_actual}")
         # batch_size, sequence, feature_size - X input
         # batch_size, hidden_size - y label
    def predict_LSTM_model(self):
        X = torch.tensor(self.lstm_pred_sequence_X)

        out = self.model.forward(X)

        arc_pred = ((float(out[0]) + 1)/2) * (self.max - self.min) + self.min

        print(f"Prediciton 1st Step in Future: {arc_pred}")

        pass

         
class LSTM(nn.Module):
# input_size (batch_size, sequence_length, feature_size)

#
    def __init__(self, hidden_size, feature_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size = feature_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.lc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(.1)

    def forward(self, X):
        lstm_out, (hn, cn)  = self.lstm(X) 
                # batch_size, sequence_length, hidden_size
        out = lstm_out[:, -1, :] # batch_size, hidden_size
    
        out = self.dropout(out)

        output = self.lc(out) # batch_size, output_size
        return output


# f1 = F1()
# f1.collect_csv()
# f1.extract_csv_data("Race_Results.csv")
# f1.df_info()
# f1.clean_organize_race_win_predictions()
#f1.display_player_times("verstappen")
# f1.time_parsing()
# f1.display_player_times("norris")
# f1.check_stationarity("norris")
# f1.check_seasonality("norris")
# f1.min_max_normalization()
# f1.sliding_window(12)
# f1.train_LSTM_Model(128, 1, 10, 1, 200)# hidden size, feature size, layer size, outputsize, num_epochs
# f1.test_LSTM_model()
# f1.predict_LSTM_model()
