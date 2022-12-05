import os
import pdb
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import pdb
import warnings
warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Zillow(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ml_dataset.csv', 
                 target='OT', scale=True, inverse=False, timeenc=1, freq='m', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_raw = df_raw.rename(columns={"Unnamed: 0": "date"})
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        self.cols = [
            'Fulton', 'Gwinnett', 'Cobb', 'Dekalb', 'Chatham', 'Clayton', 
            'Cherokee', 'Forsyth', 'Henry', 'Richmond', 'Hall', 'Muscogee', 'Paulding',
            'Houston', 'Bibb', 'Columbia', 'Douglas', 'Coweta', 'Clarke', 'Carroll', 
            'Lowndes', 'Fayette', 'Newton', 'Bartow', 'Whitfield', 'Floyd', 'Walton', 
            'Rockdale', 'Dougherty', 'Glynn', 'Barrow', 'Bulloch', 'Troup', 'Walker', 
            'Jackson', 'Catoosa', 'Spalding', 'Liberty', 'Effingham', 'Gordon', 'Camden', 
            'Laurens', 'Colquitt', 'Baldwin', 'Thomas', 'Habersham', 'Coffee', 'Polk', 
            'Tift', 'Murray', 'Oconee', 'Bryan', 'Ware', 'Harris', 'Lumpkin', 'Pickens', 
            'Gilmer', 'Sumter', 'Wayne', 'Lee', 'White', 'Haralson', 'Madison', 'Jones', 
            'Monroe', 'Peach', 'Toombs', 'Decatur', 'Upson', 'Hart', 'Stephens', 'Tattnall', 
            'Fannin', 'Grady', 'Chattooga', 'Dawson', 'Butts', 'Union', 'Franklin', 'Crisp', 
            'Emanuel', 'Burke', 'Mitchell', 'Putnam', 'McDuffie', 'Meriwether', 'Dodge', 'Worth', 
            'Washington', 'Pierce', 'Elbert', 'Berrien', 'Brantley', 'Banks', 'Long', 'Lamar', 'Morgan', 
            'Appling', 'Pike', 'Greene', 'Cook', 'Ben Hill', 'Rabun', 'Dade', 'Telfair', 'Jefferson', 'Brooks', 
            'Jeff Davis', 'Oglethorpe', 'McIntosh', 'Screven', 'Jasper', 'Dooly', 'Macon', 'Charlton', 'Bleckley', 
            'Crawford', 'Heard', 'Towns', 'Pulaski', 'Bacon', 'Candler', 'Evans', 'Chattahoochee', 'Lanier', 'Early', 
            'Wilkes', 'Johnson', 'Irwin', 'Montgomery', 'Wilkinson', 'Wilcox', 'Jenkins', 'Terrell', 'Hancock', 'Marion', 
            'Seminole', 'Atkinson', 'Twiggs', 'Taylor', 'Turner', 'Wheeler', 'Lincoln', 'Randolph', 'Treutlen', 'Clinch', 
            'Calhoun', 'Talbot', 'Stewart', 'Miller', 'Warren', 'Schley', 'Echols', 'Baker', 'Glascock', 'Clay', 'Webster', 
            'Quitman', 'Taliaferro'
        ]

        self.valid_cols = df_raw.drop(columns="date").keys().values
        self.total_len = 0
        self.index_dict = {}
        self.index_county_map = {}
        for key in self.cols:
            if key not in self.valid_cols:
                continue
            cur_raw_df = df_raw.loc[:, ["date", key]]
            cur_raw_df.loc[:, "date"] = pd.to_datetime(cur_raw_df.loc[:, "date"])
            cur_len = cur_raw_df.dropna().shape[0] - self.seq_len - self.pred_len
            if cur_len <= 0:
                continue
            cur_data_stamp = time_features(cur_raw_df.dropna().loc[:, ["date"]], timeenc=self.timeenc, freq=self.freq)
            self.index_dict[key] = [cur_raw_df.dropna().drop(columns="date"), cur_data_stamp]
            null_offset_idx = self.index_dict[key][0].index.values.flatten()[0]
            for i in range(self.total_len, self.total_len+cur_len):
                self.index_county_map[i] = (key, i - self.total_len + null_offset_idx)
            self.total_len += cur_len

        num_train = int(self.total_len*0.7)
        num_test = int(self.total_len*0.2)
        num_vali = self.total_len - num_train - num_test

        self.data_length = [num_train, num_vali, num_test][self.set_type]

        border1s = [0, num_train-self.seq_len, self.total_len-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, self.total_len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # if self.features=='M' or self.features=='MS':
        #     cols_data = df_raw.columns[1:]
        #     df_data = df_raw[cols_data]
        # elif self.features=='S':
        #     df_data = df_raw[[self.target]]

        if self.scale:
            scaler_data = None
            for key in self.cols:
                if key in self.index_dict:
                    if scaler_data is None:
                        scaler_data = np.array(self.index_dict[key][0].values)
                        continue
                    if key == self.index_county_map[border2s[0]][0]:
                        scaler_data = np.concatenate((scaler_data, self.index_dict[key][0].loc[:self.index_county_map[border2s[0]][1]].values))
                        break
                    scaler_data = np.concatenate((scaler_data, self.index_dict[key][0].values))
            self.scaler.fit(scaler_data)
            for key in self.cols:
                if key in self.index_dict:
                    self.index_dict[key][0] = self.scaler.transform(self.index_dict[key][0])

        # df_stamp = df_raw[['date']]
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        # self.data_x = data[border1:border2]
        # self.data_y = data[border1:border2]
        # self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = self.index_county_map[index][1]
        s_end = s_begin + self.seq_len - 1
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len - 1

        county_key = self.index_county_map[index][0]
        cur_df = self.index_dict[county_key][0]
        cur_data_stamp = self.index_dict[county_key][1]
        
        seq_x = cur_df.loc[s_begin:s_end].values
        seq_y = cur_df.loc[r_begin:r_end].values

        min_offset = cur_df.index.values.min()
        seq_x_mark = cur_data_stamp[s_begin-min_offset:s_end-min_offset+1]
        seq_y_mark = cur_data_stamp[r_begin-min_offset:r_end-min_offset+1]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return self.data_length - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Zillow_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ml_dataset.csv', 
                 target='OT', scale=True, inverse=True, timeenc=1, freq='m', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_raw = df_raw.rename(columns={"Unnamed: 0": "date"})
        self.df_raw = df_raw
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        self.cols = [
            'Fulton', 'Gwinnett', 'Cobb', 'Dekalb', 'Chatham', 'Clayton', 
            'Cherokee', 'Forsyth', 'Henry', 'Richmond', 'Hall', 'Muscogee', 'Paulding',
            'Houston', 'Bibb', 'Columbia', 'Douglas', 'Coweta', 'Clarke', 'Carroll', 
            'Lowndes', 'Fayette', 'Newton', 'Bartow', 'Whitfield', 'Floyd', 'Walton', 
            'Rockdale', 'Dougherty', 'Glynn', 'Barrow', 'Bulloch', 'Troup', 'Walker', 
            'Jackson', 'Catoosa', 'Spalding', 'Liberty', 'Effingham', 'Gordon', 'Camden', 
            'Laurens', 'Colquitt', 'Baldwin', 'Thomas', 'Habersham', 'Coffee', 'Polk', 
            'Tift', 'Murray', 'Oconee', 'Bryan', 'Ware', 'Harris', 'Lumpkin', 'Pickens', 
            'Gilmer', 'Sumter', 'Wayne', 'Lee', 'White', 'Haralson', 'Madison', 'Jones', 
            'Monroe', 'Peach', 'Toombs', 'Decatur', 'Upson', 'Hart', 'Stephens', 'Tattnall', 
            'Fannin', 'Grady', 'Chattooga', 'Dawson', 'Butts', 'Union', 'Franklin', 'Crisp', 
            'Emanuel', 'Burke', 'Mitchell', 'Putnam', 'McDuffie', 'Meriwether', 'Dodge', 'Worth', 
            'Washington', 'Pierce', 'Elbert', 'Berrien', 'Brantley', 'Banks', 'Long', 'Lamar', 'Morgan', 
            'Appling', 'Pike', 'Greene', 'Cook', 'Ben Hill', 'Rabun', 'Dade', 'Telfair', 'Jefferson', 'Brooks', 
            'Jeff Davis', 'Oglethorpe', 'McIntosh', 'Screven', 'Jasper', 'Dooly', 'Macon', 'Charlton', 'Bleckley', 
            'Crawford', 'Heard', 'Towns', 'Pulaski', 'Bacon', 'Candler', 'Evans', 'Chattahoochee', 'Lanier', 'Early', 
            'Wilkes', 'Johnson', 'Irwin', 'Montgomery', 'Wilkinson', 'Wilcox', 'Jenkins', 'Terrell', 'Hancock', 'Marion', 
            'Seminole', 'Atkinson', 'Twiggs', 'Taylor', 'Turner', 'Wheeler', 'Lincoln', 'Randolph', 'Treutlen', 'Clinch', 
            'Calhoun', 'Talbot', 'Stewart', 'Miller', 'Warren', 'Schley', 'Echols', 'Baker', 'Glascock', 'Clay', 'Webster', 
            'Quitman', 'Taliaferro'
        ]
        self.valid_cols = df_raw.drop(columns="date").keys().values
        self.total_len = 0
        self.index_dict = {}
        self.index_county_map = {}
        for key in self.cols:
            if key not in self.valid_cols:
                continue
            # if key == "Glynn":
            #     pdb.set_trace()
            cur_raw_df = df_raw.loc[:, ["date", key]]
            cur_raw_df.loc[:, "date"] = pd.to_datetime(cur_raw_df.loc[:, "date"])
            cur_len = cur_raw_df.dropna().shape[0] - self.seq_len
            if cur_len <= 0:
                continue
            cur_data_stamp = time_features(cur_raw_df.dropna().loc[:, ["date"]], timeenc=self.timeenc, freq=self.freq)
            self.index_dict[key] = [cur_raw_df.dropna().drop(columns="date"), cur_data_stamp, cur_raw_df.dropna().loc[:, ["date"]]]
            null_offset_idx = self.index_dict[key][0].index.values.flatten()[0]
            for i in range(self.total_len, self.total_len+cur_len+1):
                self.index_county_map[i] = (key, i - self.total_len + null_offset_idx)
            self.total_len += cur_len

        # df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = self.total_len-self.seq_len-1
        border2 = self.total_len-1
        
        if self.scale:
            scaler_data = None
            for key in self.cols:
                if key in self.index_dict:
                    if scaler_data is None:
                        scaler_data = np.array(self.index_dict[key][0].values)
                        continue
                    if key == self.index_county_map[border2][0]:
                        scaler_data = np.concatenate((scaler_data, self.index_dict[key][0].loc[:self.index_county_map[border2][1]].values))
                        break
                    scaler_data = np.concatenate((scaler_data, self.index_dict[key][0].values))
            self.scaler.fit(scaler_data)
            for key in self.cols:
                if key in self.index_dict:
                    self.index_dict[key][0] = self.scaler.transform(self.index_dict[key][0])

        # if self.scale:
        #     self.scaler.fit(df_data.values)
        #     data = self.scaler.transform(df_data.values)
        # else:
        #     data = df_data.values
            
        # tmp_stamp = df_raw[['date']][border1:border2]
        # tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        # pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)

        self.pred_dates = pd.date_range(pd.to_datetime(df_raw["date"]).iloc[-1], periods=self.pred_len+1, freq=self.freq)
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(df_raw["date"].values) + list(self.pred_dates[1:])
        self.data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        # self.data_x = data[border1:border2]
        # if self.inverse:
        #     self.data_y = df_data.values[border1:border2]
        # else:
        #     self.data_y = data[border1:border2]
        # self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        # s_end = s_begin + self.seq_len - 1
        # r_end = r_begin + self.label_len + self.pred_len - 1
        # index = 6468
        s_begin = self.index_county_map[index][1]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len
        
        county_key = self.index_county_map[index][0]
        cur_df = self.index_dict[county_key][0]
        cur_data_stamp = self.index_dict[county_key][1]

        min_offset = cur_df.index.values.min()
        seq_x = cur_df.values[s_begin-min_offset:s_end-min_offset]
        seq_y = cur_df.values[r_begin-min_offset:r_begin+self.label_len-min_offset]
        
        seq_x_mark = cur_data_stamp[s_begin-min_offset:s_end-min_offset]
        time_stamps_left = (r_end-min_offset) - (cur_df.shape[0]-1)
        if time_stamps_left > 0:
            seq_y_mark = cur_data_stamp[r_begin-min_offset:r_end-min_offset-time_stamps_left]
            seq_y_mark = np.concatenate((seq_y_mark, self.data_stamp[-time_stamps_left:]))
        else:
            seq_y_mark = cur_data_stamp[r_begin-min_offset:r_end-min_offset]

        date = self.index_dict[county_key][2].loc[s_end]
        date_idx_arr = np.array([s_end for _ in range(seq_x.shape[0])]).reshape(-1,1)
        
        county = county_key
        county_idx = self.cols.index(county_key)
        county_idx_arr = np.array([county_idx for _ in range(seq_x.shape[0])]).reshape(-1,1)

        # seq_x = self.data_x[s_begin:s_end]
        # if self.inverse:
        #     seq_y = self.data_x[r_begin:r_begin+self.label_len]
        # else:
        #     seq_y = self.data_y[r_begin:r_begin+self.label_len]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, date_idx_arr, county_idx_arr
    
    def __len__(self):
        return self.total_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)