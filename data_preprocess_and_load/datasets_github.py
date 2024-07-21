import numpy as np
import pandas as pd
import scipy.io
import random
import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.autograd import Variable
import nibabel as nib
import scipy

import torch.nn.functional as F
import nitime
from scipy.optimize import curve_fit

import pickle

import warnings
warnings.filterwarnings("ignore")

from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer

def lorentzian_function(x, s0, corner):
    return (s0*corner**2) / (x**2 + corner**2)

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
    
    def register_args(self,**kwargs):
        self.index_l = []
        self.target = kwargs.get('target')
        self.fine_tune_task = kwargs.get('fine_tune_task')
        self.dataset_name = kwargs.get('dataset_name')
        self.fmri_type = kwargs.get('fmri_type')
        self.seq_len = kwargs.get('sequence_length')
        self.lorentzian = kwargs.get('divide_by_lorentzian')
        self.fmri_multimodality_type = kwargs.get('fmri_multimodality_type')
    
class UKB_fMRI_timeseries(BaseDataset):
    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        self.fine_tune_task = kwargs.get('fine_tune_task')
        self.data_dir = kwargs.get('ukb_path')
        self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','UKB_phenotype_gps_fluidint.csv'))
        self.subject_names = os.listdir(self.data_dir)
        self.subject_folders = []
        self.intermediate_vec = kwargs.get('intermediate_vec')
        self.filtering_type = kwargs.get('filtering_type')
        self.sequence_length = kwargs.get('sequence_length')
        self.use_raw_knee = kwargs.get('use_raw_knee')
        self.seq_part = kwargs.get('seq_part')
        self.use_high_freq = kwargs.get('use_high_freq')
                
        valid_sub = os.listdir(kwargs.get('ukb_path'))
        valid_sub = list(map(int, valid_sub))
        
        if self.target != 'reconstruction':
            non_na = self.meta_data[['eid',self.target]].dropna(axis=0)
            subjects = set(non_na['eid']) & set(valid_sub)
        else:
            subjects = valid_sub
        
        print('UKB')
        
        if self.fine_tune_task == 'regression':
            cont_mean = non_na[self.target].mean()
            cont_std = non_na[self.target].std()
            self.mean = cont_mean
            self.std = cont_std

        for i, subject in enumerate(subjects):
            # Normalization
            if self.fine_tune_task == 'regression':
                target = torch.tensor((self.meta_data.loc[self.meta_data['eid']==subject,self.target].values[0] - cont_mean) / cont_std)
                target = target.float()
            elif self.fine_tune_task == 'binary_classification':
                target = torch.tensor(self.meta_data.loc[self.meta_data['eid']==subject,self.target].values[0]) 
            else:
                if self.target == 'reconstruction': # for transformer reconstruction
                    target = torch.tensor(0)
                    
            if self.intermediate_vec == 180:
                path_to_fMRIs = os.path.join(self.data_dir, str(subject), 'hcp_mmp1_'+str(subject)+'.npy')
            elif self.intermediate_vec == 400:
                path_to_fMRIs = os.path.join(self.data_dir, str(subject), 'schaefer_400Parcels_17Networks_'+str(subject)+'.npy')

            self.index_l.append((i, subject, path_to_fMRIs, target))           

            

    def __len__(self):
        N = len(self.index_l)
        return N
    
        
    def __getitem__(self, index):
        subj, subj_name, path_to_fMRIs, target = self.index_l[index]
        
        if self.seq_part=='tail':
            y = np.load(path_to_fMRIs)[-self.sequence_length:].T # [ROI, seq_len]
        elif self.seq_part=='head':
            y = np.load(path_to_fMRIs)[20:20+self.sequence_length].T # [ROI, seq_len]
        
        ts_length = y.shape[1]
        pad = self.sequence_length - ts_length # 0

        TR = 0.735
        if self.lorentzian:
        
            '''
            get knee frequency
            '''

            sample_whole = np.zeros((self.sequence_length,))
            for i in range(self.intermediate_vec):
                sample_whole+=y[i]

            sample_whole /= self.intermediate_vec   

            T = TimeSeries(sample_whole, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)

            # Lorentzian function fitting
            xdata = np.array(S_original.spectrum_fourier[0][1:])
            ydata = np.abs(S_original.spectrum_fourier[1][1:])
            
            def lorentzian_function(x, s0, corner):
                return (s0*corner**2) / (x**2 + corner**2)

            # initial parameters
            p0 = [0, 0.006]

            popt, pcov = curve_fit(lorentzian_function, xdata, ydata, p0=p0, maxfev = 5000)
            
            f1 = popt[1]
            
            knee = round(popt[1]/(1/(sample_whole.shape[0]*TR))) #int에
            
            if knee==0:
                knee=1
                
            if self.fmri_multimodality_type == 'three_channels':
                def modified_lorentzian_function(x, beta_low, beta_high, A, B, corner):
                    return np.where(x < corner, A * x**beta_low, B * x**beta_high)

                p1 = [2, 1, 23, 25, 0.16]
                
                popt_mo, pcov = curve_fit(modified_lorentzian_function, xdata[knee:], ydata[knee:], p0=p1, maxfev = 50000)
                pink = round(popt_mo[-1]/(1/(sample_whole.shape[0]*TR)))
                f2 = popt_mo[-1]


        # no lorentzian
        else:
            ## just timeseries
            if self.fmri_type == 'timeseries':
                pass
            else:
                ## randomly dividing frequency ranges
                frequency_range = list(range(xdata.shape[0]))
                import random
                if self.fmri_multimodality_type == 'three_channels':
                    a,b = random.sample(frequency_range, 2)
                    knee = min(a,b)
                    pink = max(a,b)
                else:
                    knee = random.sample(frequency_range, 1)
        
        
        if self.fmri_type == 'timeseries':
            y = scipy.stats.zscore(y, axis=1) # (ROI, sequence length)
            y = torch.from_numpy(y).T.float() #.type(torch.DoubleTensor) # (sequence length, ROI)
            ans_dict = {'fmri_sequence':y,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'frequency':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            y = scipy.stats.zscore(np.abs(S_original.spectrum_fourier[1]), axis=None) # (ROI, sequence length//2)
            y = torch.from_numpy(y).T.float() # (sequence length//2, ROI)
            ans_dict = {'fmri_sequence':y,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'time_domain_low':
            if self.fmri_multimodality_type == 'three_channels':
                # 01 high ~ (low+ultralow)
                T1 = TimeSeries(y, sampling_interval=TR)
                S_original1 = SpectralAnalyzer(T1)
                FA1 = FilterAnalyzer(T1, lb= S_original1.spectrum_fourier[0][pink])
                ultralow_low = FA1.data-FA1.filtered_boxcar.data

                # 02 low ~ ultralow
                T2 = TimeSeries(ultralow_low, sampling_interval=TR)
                S_original2 = SpectralAnalyzer(T2)
                if self.use_raw_knee:
                    FA2 = FilterAnalyzer(T2, lb=raw_knee)
                else:    
                    FA2 = FilterAnalyzer(T2, lb= S_original2.spectrum_fourier[0][knee])
                    
                if self.filtering_type == 'FIR':
                    low = scipy.stats.zscore(FA2.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    low = scipy.stats.zscore(FA2.filtered_boxcar.data, axis=1)
                
                low = torch.from_numpy(low).T.float() # (sequence length, ROI)
            
            else: # only two frequencies
                T = TimeSeries(y, sampling_interval=TR)
                S_original = SpectralAnalyzer(T)
                if self.use_raw_knee:
                    FA = FilterAnalyzer(T, lb=raw_knee)
                else:    
                    FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

                if self.filtering_type == 'FIR':
                    low = scipy.stats.zscore(FA.fir.data, axis=1) # (ROI, sequence length) 
                elif self.filtering_type == 'Boxcar':
                    low = scipy.stats.zscore(FA.filtered_boxcar.data, axis=1) # (ROI, sequence length)  

                low = torch.from_numpy(low).T.float() # (sequence length, ROI)
            ans_dict = {'fmri_sequence':low,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'time_domain_ultralow':
            if self.fmri_multimodality_type == 'three_channels':
                # 01 high ~ (low+ultralow)
                T1 = TimeSeries(y, sampling_interval=TR)
                S_original1 = SpectralAnalyzer(T1)
                FA1 = FilterAnalyzer(T1, lb= S_original1.spectrum_fourier[0][pink])
                ultralow_low = FA1.data-FA1.filtered_boxcar.data

                # 02 low ~ ultralow
                T2 = TimeSeries(ultralow_low, sampling_interval=TR)
                
                if self.use_raw_knee:
                    FA2 = FilterAnalyzer(T2, lb=raw_knee)
                else:
                    S_original2 = SpectralAnalyzer(T2)
                    FA2 = FilterAnalyzer(T2, lb= S_original2.spectrum_fourier[0][knee])
                    
                if self.filtering_type == 'FIR':
                    ultralow = scipy.stats.zscore(FA2.data-FA2.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    ultralow = scipy.stats.zscore(FA2.data-FA2.filtered_boxcar.data, axis=1)
                
                ultralow = torch.from_numpy(ultralow).T.float() # (sequence length, ROI)
            
            else:
                T = TimeSeries(y, sampling_interval=TR)
                
                if self.use_raw_knee:
                    FA = FilterAnalyzer(T, lb=raw_knee)
                else:
                    S_original = SpectralAnalyzer(T)
                    FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

                if self.filtering_type == 'FIR':
                    ultralow = scipy.stats.zscore(FA.data-FA.fir.data, axis=1) #1) # (ROI, sequence length) 
                elif self.filtering_type == 'Boxcar':
                    ultralow = scipy.stats.zscore(FA.data-FA.filtered_boxcar.data, axis=1) #1) # (ROI, sequence length) 

                ultralow = torch.from_numpy(ultralow).T.float() # (sequence length, ROI)
            ans_dict = {'fmri_sequence':ultralow,'subject':subj,'subject_name':subj_name, self.target:target}
        
        elif self.fmri_type == 'time_domain_high':
            T1 = TimeSeries(y, sampling_interval=TR)
            if self.use_raw_knee:
                FA1 = FilterAnalyzer(T1, lb=f2)
            else:
                S_original1 = SpectralAnalyzer(T1)
                FA1 = FilterAnalyzer(T1, lb= S_original1.spectrum_fourier[0][pink])
            high = scipy.stats.zscore(FA1.filtered_boxcar.data, axis=1)
            high = torch.from_numpy(high).T.float()
            ans_dict = {'fmri_sequence':high,'subject':subj,'subject_name':subj_name, self.target:target}
            
        elif self.fmri_type == 'divided_timeseries':
            if self.fmri_multimodality_type =='three_channels':
                # 01 high ~ (low+ultralow)
                T1 = TimeSeries(y, sampling_interval=TR)
                if self.use_raw_knee:
                    FA1 = FilterAnalyzer(T1, lb=f2)
                else:
                    S_original1 = SpectralAnalyzer(T1)
                    FA1 = FilterAnalyzer(T1, lb= S_original1.spectrum_fourier[0][pink])
                high = scipy.stats.zscore(FA1.filtered_boxcar.data, axis=1)
                ultralow_low = FA1.data-FA1.filtered_boxcar.data

                # 02 low ~ ultralow
                T2 = TimeSeries(ultralow_low, sampling_interval=TR)
                
                if self.use_raw_knee:
                    FA2 = FilterAnalyzer(T2, lb=f1)
                else:
                    S_original2 = SpectralAnalyzer(T2)
                    FA2 = FilterAnalyzer(T2, lb= S_original2.spectrum_fourier[0][knee])
                if self.filtering_type == 'FIR':
                    low = scipy.stats.zscore(FA2.fir.data, axis=1)
                    ultralow = scipy.stats.zscore(FA2.data-FA2.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    low = scipy.stats.zscore(FA2.filtered_boxcar.data, axis=1)
                    ultralow = scipy.stats.zscore(FA2.data-FA2.filtered_boxcar.data, axis=1)
                
                high = torch.from_numpy(high).T.float() # (sequence length, ROI)
                low = torch.from_numpy(low).T.float() # (sequence length, ROI)
                ultralow = torch.from_numpy(ultralow).T.float() # (sequence length, ROI)
                
                    
                if self.use_high_freq:
                    ans_dict= {'fmri_highfreq_sequence':high, 'fmri_lowfreq_sequence':low, 'fmri_ultralowfreq_sequence':ultralow, 'subject':subj, 'subject_name':subj_name, self.target:target}
                else:
                    ans_dict= {'fmri_lowfreq_sequence':low, 'fmri_ultralowfreq_sequence':ultralow, 'subject':subj, 'subject_name':subj_name, self.target:target}
            
            else:    
                T = TimeSeries(y, sampling_interval=TR)
                if self.use_raw_knee:
                    FA = FilterAnalyzer(T, lb=raw_knee)
                else:
                    S_original = SpectralAnalyzer(T)
                    FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

                if self.filtering_type == 'FIR':
                    low = scipy.stats.zscore(FA.fir.data, axis=1) #1) # (ROI, sequence length) 
                    ultralow = scipy.stats.zscore(FA.data-FA.fir.data, axis=1) #1) # (ROI, sequence length) 
                elif self.filtering_type == 'Boxcar':
                    low = scipy.stats.zscore(FA.filtered_boxcar.data, axis=1) #1) # (ROI, sequence length) 
                    ultralow = scipy.stats.zscore(FA.data-FA.filtered_boxcar.data, axis=1) #1) # (ROI, sequence length) 

                low = torch.from_numpy(low).T.float() # (sequence length, ROI)
                ultralow = torch.from_numpy(ultralow).T.float()

                ans_dict= {'fmri_lowfreq_sequence':low, 'fmri_ultralowfreq_sequence':ultralow, 'subject':subj, 'subject_name':subj_name, self.target:target}

                
        elif self.fmri_type == 'frequency_domain_low':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            if self.use_raw_knee:
                FA = FilterAnalyzer(T, lb=raw_knee)
            else:
                S_original = SpectralAnalyzer(T)
                FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])
            T1 = TimeSeries((FA.fir.data), sampling_interval=TR)
            S_original1 = SpectralAnalyzer(T1)
            
            # complex number -> real number (amplitude)
            low = np.abs(S_original1.spectrum_fourier[1].T[1:].T)          
            low = torch.from_numpy(low).T.float() #.type(torch.DoubleTensor)  
            ans_dict = {'fmri_sequence':low,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'frequency_domain_ultralow':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            if self.use_raw_knee:
                FA = FilterAnalyzer(T, lb=raw_knee)
            else:
                S_original = SpectralAnalyzer(T)
                FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

            T1 = TimeSeries((FA.data-FA.fir.data), sampling_interval=TR)
            S_original1 = SpectralAnalyzer(T1)

            ultralow = np.abs(S_original1.spectrum_fourier[1].T[1:].T) 
            ultralow = torch.from_numpy(ultralow).T.float() #.type(torch.DoubleTensor) # (sequence length//2, ROI)  

            ans_dict = {'fmri_sequence':ultralow,'subject':subj,'subject_name':subj_name, self.target:target}
        
        elif self.fmri_type == 'divided_frequency':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            if self.use_raw_knee:
                FA = FilterAnalyzer(T, lb=raw_knee)
            else:
                S_original = SpectralAnalyzer(T)
                FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

            # low
            T1 = TimeSeries((FA.fir.data), sampling_interval=0.72)
            S_original1 = SpectralAnalyzer(T1)
            low = np.abs(S_original1.spectrum_fourier[1].T[1:].T)
            low = torch.from_numpy(low).T.float() #.type(torch.DoubleTensor)   # (sequence length//2, ROI) 
            
            # ultralow
            T2 = TimeSeries((FA.data-FA.fir.data), sampling_interval=TR)
            S_original2 = SpectralAnalyzer(T2)
            ultralow = np.abs(S_original2.spectrum_fourier[1].T[1:].T) 
            ultralow = torch.from_numpy(ultralow).T.float() #.type(torch.DoubleTensor)  # (sequence length//2, ROI) 

            ans_dict = {'fmri_lowfreq_sequence':low, 'fmri_ultralowfreq_sequence':ultralow,'subject':subj,'subject_name':subj_name, self.target:target}
        
       
        
        elif self.fmri_type == 'timeseries_and_frequency':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            if self.use_raw_knee:
                FA = FilterAnalyzer(T, lb=raw_knee)
            else:
                S_original = SpectralAnalyzer(T)
                FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])
            
            # timeseries (low)
            low = scipy.stats.zscore(FA.fir.data, axis=1)
            low = torch.from_numpy(low).T.float()
            
            # frequency (ultralow) 
            T1 = TimeSeries((FA.data-FA.fir.data), sampling_interval=TR)
            S_original1 = SpectralAnalyzer(T1)


            ultralow = np.abs(S_original1.spectrum_fourier[1].T[1:].T)
            ultralow = torch.from_numpy(ultralow).T.float()

            ans_dict= {'fmri_lowfreq_sequence':low,'fmri_ultralowfreq_sequence':ultralow,'subject':subj,'subject_name':subj_name, self.target:target}
        

        return ans_dict    
    
class ABCD_fMRI_timeseries(BaseDataset):
    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        self.feature_map_gen = kwargs.get('feature_map_gen')
        self.fine_tune_task = kwargs.get('fine_tune_task')
        self.data_dir = kwargs.get('fmri_timeseries_path')
        self.feature_map_size = kwargs.get('feature_map_size')
        self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','ABCD_phenotype_total.csv'))
        self.subject_names = os.listdir(self.data_dir)
        self.subject_folders = []
        self.intermediate_vec = kwargs.get('intermediate_vec')
        self.filtering_type = kwargs.get('filtering_type')
        self.sequence_length = kwargs.get('sequence_length')
        self.use_raw_knee = kwargs.get('use_raw_knee')
        self.seq_part = kwargs.get('seq_part')
        self.use_high_freq = kwargs.get('use_high_freq')
        self.pretrained_model_weights_path = kwargs.get('pretrained_model_weights_path')
        self.finetune = kwargs.get('finetune')
        self.transfer_learning =  bool(self.pretrained_model_weights_path) or self.finetune
        self.finetune_test = kwargs.get('finetune_test') # test phase of finetuning task
        print('self.finetune_test', self.finetune_test)
        
        valid_sub = [i.split('-')[1] for i in os.listdir(kwargs.get('fmri_timeseries_path'))]
        
        if self.target != 'reconstruction':
            non_na = self.meta_data[['subjectkey',self.target]].dropna(axis=0)
            subjects = list(non_na['subjectkey'])
            subjects = list(set(subjects) & set(valid_sub))
        else:
            subjects = valid_sub
        
        print('ABCD')
                       
        if self.fine_tune_task == 'regression':
            cont_mean = non_na[self.target].mean()
            cont_std = non_na[self.target].std()
            self.mean = cont_mean
            self.std = cont_std

        for i, subject in enumerate(subjects):
            
            if self.intermediate_vec == 180:
                path_to_fMRIs = os.path.join(self.data_dir, 'sub-'+subject, 'hcp_mmp1_sub-'+subject+'.npy')
            elif self.intermediate_vec == 400:
                if self.shaefer_num_networks == 7:
                    path_to_fMRIs = os.path.join(self.data_dir, 'sub-'+subject, 'schaefer_400Parcels_7Networks_sub-'+subject+'.npy')
                elif self.shaefer_num_networks == 17:
                    path_to_fMRIs = os.path.join(self.data_dir, 'sub-'+subject, 'schaefer_400Parcels_17Networks_sub-'+subject+'.npy') 
            elif self.intermediate_vec == 246:
                path_to_fMRIs = os.path.join(self.data_dir, 'sub-'+subject, 'Brainnectome_246_sub-'+subject+'.npy') 
            elif self.intermediate_vec == 200:
                path_to_fMRIs = os.path.join(self.data_dir, 'sub-'+subject, 'craddock200_sub-'+subject+'.npy')
            elif self.intermediate_vec == 333:
                path_to_fMRIs = os.path.join(self.data_dir, 'sub-'+subject, 'gordon_sub-'+subject+'.npy')            
            
                
            # Normalization
            if self.fine_tune_task == 'regression':
                target = torch.tensor((self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0] - cont_mean) / cont_std)
                target = target.float()
            elif self.fine_tune_task == 'binary_classification':
                target = torch.tensor(self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0]) 
            else:
                if self.target == 'reconstruction': # for transformer reconstruction
                    target = torch.tensor(0)
                    
            self.index_l.append((i, subject, path_to_fMRIs, target))

            

    def __len__(self):
        N = len(self.index_l)
        return N
    
        
    def __getitem__(self, index):
        subj, subj_name, path_to_fMRIs, target = self.index_l[index]
        if self.seq_part=='tail':
            y = np.load(path_to_fMRIs)[-self.sequence_length:].T # [180, seq_len]
        elif self.seq_part=='head':
            y = np.load(path_to_fMRIs)[20:20+self.sequence_length].T # [180, seq_len]
        
        if self.transfer_learning or self.finetune_test:
            # 324 -> 464 padding for transfer learning or finetuning (test phase)
            pad = 464 - self.sequence_length

        TR = 0.8
        if self.lorentzian:
        
            '''
            get knee frequency
            '''

            sample_whole = np.zeros((self.sequence_length,))
            for i in range(self.intermediate_vec):
                sample_whole+=y[i]

            sample_whole /= self.intermediate_vec    

            T = TimeSeries(sample_whole, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)

            # Lorentzian function fitting
            xdata = np.array(S_original.spectrum_fourier[0][1:])
            ydata = np.abs(S_original.spectrum_fourier[1][1:])
            
            def lorentzian_function(x, s0, corner):
                return (s0*corner**2) / (x**2 + corner**2)

            # initial parameter setting
            p0 = [0, 0.006]

            popt, pcov = curve_fit(lorentzian_function, xdata, ydata, p0=p0, maxfev = 5000)
            
            f1 = popt[1]
            
            knee = round(popt[1]/(1/(sample_whole.shape[0]*TR)))
            
            if knee <= 0:
                knee = 1
            
            if self.fmri_multimodality_type in ['three_channels', 'cross_attention']:
                def modified_lorentzian_function(x, beta_low, beta_high, A, B, corner):
                    return np.where(x < corner, A * x**beta_low, B * x**beta_high)

                p1 = [2, 1, 23, 25, 0.16]
                
                popt_mo, pcov = curve_fit(modified_lorentzian_function, xdata[knee:], ydata[knee:], p0=p1, maxfev = 50000)
                pink = round(popt_mo[-1]/(1/(sample_whole.shape[0]*TR)))
                f2 = popt_mo[-1]
        
        # no lorentzian
        else:
            ## just timeseries
            if self.fmri_type == 'timeseries':
                pass
            else:
                ## randomly dividing frequency ranges
                frequency_range = list(range(xdata.shape[0]))
                import random
                if self.fmri_multimodality_type == 'three_channels':
                    a,b = random.sample(frequency_range, 2)
                    knee = min(a,b)
                    pink = max(a,b)
                else:
                    knee = random.sample(frequency_range, 1)

        if self.fmri_type == 'timeseries':
            y = scipy.stats.zscore(y, axis=1) # (ROI, sequence length)
            y = torch.from_numpy(y).T.float() #.type(torch.DoubleTensor) # (sequence length, ROI)
            ans_dict = {'fmri_sequence':y,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'frequency':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            y = scipy.stats.zscore(np.abs(S_original.spectrum_fourier[1]), axis=None) # (ROI, sequence length//2)
            y = torch.from_numpy(y).T.float() # (sequence length//2, ROI)
            ans_dict = {'fmri_sequence':y,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'time_domain_low':
            if self.fmri_multimodality_type == 'three_channels':
                # 01 high ~ (low+ultralow)
                T1 = TimeSeries(y, sampling_interval=TR)
                S_original1 = SpectralAnalyzer(T1)
                FA1 = FilterAnalyzer(T1, lb= S_original1.spectrum_fourier[0][pink])
                ultralow_low = FA1.data-FA1.filtered_boxcar.data

                # 02 low ~ ultralow
                T2 = TimeSeries(ultralow_low, sampling_interval=TR)
                S_original2 = SpectralAnalyzer(T2)
                if self.use_raw_knee:
                    FA2 = FilterAnalyzer(T2, lb=raw_knee)
                else:    
                    FA2 = FilterAnalyzer(T2, lb= S_original2.spectrum_fourier[0][knee])
                    
                if self.filtering_type == 'FIR':
                    low = scipy.stats.zscore(FA2.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    low = scipy.stats.zscore(FA2.filtered_boxcar.data, axis=1)
                
                low = torch.from_numpy(low).T.float() # (sequence length, ROI)
            
            else: # only two frequencies
                T = TimeSeries(y, sampling_interval=TR)
                S_original = SpectralAnalyzer(T)
                if self.use_raw_knee:
                    FA = FilterAnalyzer(T, lb=raw_knee)
                else:    
                    FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

                if self.filtering_type == 'FIR':
                    low = scipy.stats.zscore(FA.fir.data, axis=1) # (ROI, sequence length) 
                elif self.filtering_type == 'Boxcar':
                    low = scipy.stats.zscore(FA.filtered_boxcar.data, axis=1) # (ROI, sequence length)  

                low = torch.from_numpy(low).T.float() # (sequence length, ROI)
            ans_dict = {'fmri_sequence':low,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'time_domain_ultralow':
            if self.fmri_multimodality_type == 'three_channels':
                # 01 high ~ (low+ultralow)
                T1 = TimeSeries(y, sampling_interval=TR)
                S_original1 = SpectralAnalyzer(T1)
                FA1 = FilterAnalyzer(T1, lb= S_original1.spectrum_fourier[0][pink])
                ultralow_low = FA1.data-FA1.filtered_boxcar.data

                # 02 low ~ ultralow
                T2 = TimeSeries(ultralow_low, sampling_interval=TR)
                
                if self.use_raw_knee:
                    FA2 = FilterAnalyzer(T2, lb=raw_knee)
                else:
                    S_original2 = SpectralAnalyzer(T2)
                    FA2 = FilterAnalyzer(T2, lb= S_original2.spectrum_fourier[0][knee])
                    
                if self.filtering_type == 'FIR':
                    ultralow = scipy.stats.zscore(FA2.data-FA2.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    ultralow = scipy.stats.zscore(FA2.data-FA2.filtered_boxcar.data, axis=1)
                
                ultralow = torch.from_numpy(ultralow).T.float() # (sequence length, ROI)
            
            else:
                T = TimeSeries(y, sampling_interval=TR)
                
                if self.use_raw_knee:
                    FA = FilterAnalyzer(T, lb=raw_knee)
                else:
                    S_original = SpectralAnalyzer(T)
                    FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

                if self.filtering_type == 'FIR':
                    ultralow = scipy.stats.zscore(FA.data-FA.fir.data, axis=1) #1) # (ROI, sequence length) 
                elif self.filtering_type == 'Boxcar':
                    ultralow = scipy.stats.zscore(FA.data-FA.filtered_boxcar.data, axis=1) #1) # (ROI, sequence length) 

                ultralow = torch.from_numpy(ultralow).T.float() # (sequence length, ROI)
            ans_dict = {'fmri_sequence':ultralow,'subject':subj,'subject_name':subj_name, self.target:target}
        
        elif self.fmri_type == 'time_domain_high':
            T1 = TimeSeries(y, sampling_interval=TR)
            if self.use_raw_knee:
                FA1 = FilterAnalyzer(T1, lb=f2)
            else:
                S_original1 = SpectralAnalyzer(T1)
                FA1 = FilterAnalyzer(T1, lb= S_original1.spectrum_fourier[0][pink])
            high = scipy.stats.zscore(FA1.filtered_boxcar.data, axis=1)
            high = torch.from_numpy(high).T.float()
            ans_dict = {'fmri_sequence':high,'subject':subj,'subject_name':subj_name, self.target:target}
            
        elif self.fmri_type == 'divided_timeseries':
            if self.fmri_multimodality_type =='three_channels':
                # 01 high ~ (low+ultralow)
                T1 = TimeSeries(y, sampling_interval=TR)
                if self.use_raw_knee:
                    FA1 = FilterAnalyzer(T1, lb=f2)
                else:
                    S_original1 = SpectralAnalyzer(T1)
                    FA1 = FilterAnalyzer(T1, lb= S_original1.spectrum_fourier[0][pink])
                high = scipy.stats.zscore(FA1.filtered_boxcar.data, axis=1)
                ultralow_low = FA1.data-FA1.filtered_boxcar.data

                # 02 low ~ ultralow
                T2 = TimeSeries(ultralow_low, sampling_interval=TR)
                
                if self.use_raw_knee:
                    FA2 = FilterAnalyzer(T2, lb=f1)
                else:
                    S_original2 = SpectralAnalyzer(T2)
                    FA2 = FilterAnalyzer(T2, lb= S_original2.spectrum_fourier[0][knee])
                if self.filtering_type == 'FIR':
                    low = scipy.stats.zscore(FA2.fir.data, axis=1)
                    ultralow = scipy.stats.zscore(FA2.data-FA2.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    low = scipy.stats.zscore(FA2.filtered_boxcar.data, axis=1)
                    ultralow = scipy.stats.zscore(FA2.data-FA2.filtered_boxcar.data, axis=1)
                
                high = torch.from_numpy(high).T.float() # (sequence length, ROI)
                low = torch.from_numpy(low).T.float() # (sequence length, ROI)
                ultralow = torch.from_numpy(ultralow).T.float() # (sequence length, ROI)
                
                    
                if self.use_high_freq:
                    ans_dict= {'fmri_highfreq_sequence':high, 'fmri_lowfreq_sequence':low, 'fmri_ultralowfreq_sequence':ultralow, 'subject':subj, 'subject_name':subj_name, self.target:target}
                else:
                    ans_dict= {'fmri_lowfreq_sequence':low, 'fmri_ultralowfreq_sequence':ultralow, 'subject':subj, 'subject_name':subj_name, self.target:target}
            
            else:    
                T = TimeSeries(y, sampling_interval=TR)
                if self.use_raw_knee:
                    FA = FilterAnalyzer(T, lb=raw_knee)
                else:
                    S_original = SpectralAnalyzer(T)
                    FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

                if self.filtering_type == 'FIR':
                    low = scipy.stats.zscore(FA.fir.data, axis=1) #1) # (ROI, sequence length) 
                    ultralow = scipy.stats.zscore(FA.data-FA.fir.data, axis=1) #1) # (ROI, sequence length) 
                elif self.filtering_type == 'Boxcar':
                    low = scipy.stats.zscore(FA.filtered_boxcar.data, axis=1) #1) # (ROI, sequence length) 
                    ultralow = scipy.stats.zscore(FA.data-FA.filtered_boxcar.data, axis=1) #1) # (ROI, sequence length) 

                low = torch.from_numpy(low).T.float() # (sequence length, ROI)
                ultralow = torch.from_numpy(ultralow).T.float()

                ans_dict= {'fmri_lowfreq_sequence':low, 'fmri_ultralowfreq_sequence':ultralow, 'subject':subj, 'subject_name':subj_name, self.target:target}

                
        elif self.fmri_type == 'frequency_domain_low':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            if self.use_raw_knee:
                FA = FilterAnalyzer(T, lb=raw_knee)
            else:
                S_original = SpectralAnalyzer(T)
                FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])
            T1 = TimeSeries((FA.fir.data), sampling_interval=TR)
            S_original1 = SpectralAnalyzer(T1)
            
            # complex number -> real number (amplitude)
            low = np.abs(S_original1.spectrum_fourier[1].T[1:].T)          
            low = torch.from_numpy(low).T.float() #.type(torch.DoubleTensor)  
            ans_dict = {'fmri_sequence':low,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'frequency_domain_ultralow':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            if self.use_raw_knee:
                FA = FilterAnalyzer(T, lb=raw_knee)
            else:
                S_original = SpectralAnalyzer(T)
                FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

            T1 = TimeSeries((FA.data-FA.fir.data), sampling_interval=TR)
            S_original1 = SpectralAnalyzer(T1)

            ultralow = np.abs(S_original1.spectrum_fourier[1].T[1:].T) 
            ultralow = torch.from_numpy(ultralow).T.float() #.type(torch.DoubleTensor) # (sequence length//2, ROI)  

            ans_dict = {'fmri_sequence':ultralow,'subject':subj,'subject_name':subj_name, self.target:target}
        
        elif self.fmri_type == 'divided_frequency':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            if self.use_raw_knee:
                FA = FilterAnalyzer(T, lb=raw_knee)
            else:
                S_original = SpectralAnalyzer(T)
                FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

            # low
            T1 = TimeSeries((FA.fir.data), sampling_interval=0.72)
            S_original1 = SpectralAnalyzer(T1)
            low = np.abs(S_original1.spectrum_fourier[1].T[1:].T)
            low = torch.from_numpy(low).T.float() #.type(torch.DoubleTensor)   # (sequence length//2, ROI) 
            
            # ultralow
            T2 = TimeSeries((FA.data-FA.fir.data), sampling_interval=TR)
            S_original2 = SpectralAnalyzer(T2)
            ultralow = np.abs(S_original2.spectrum_fourier[1].T[1:].T) 
            ultralow = torch.from_numpy(ultralow).T.float() #.type(torch.DoubleTensor)  # (sequence length//2, ROI) 

            ans_dict = {'fmri_lowfreq_sequence':low, 'fmri_ultralowfreq_sequence':ultralow,'subject':subj,'subject_name':subj_name, self.target:target}
        
       
        
        elif self.fmri_type == 'timeseries_and_frequency':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            if self.use_raw_knee:
                FA = FilterAnalyzer(T, lb=raw_knee)
            else:
                S_original = SpectralAnalyzer(T)
                FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])
            
            # timeseries (low)
            low = scipy.stats.zscore(FA.fir.data, axis=1)
            low = torch.from_numpy(low).T.float()
            
            # frequency (ultralow) 
            T1 = TimeSeries((FA.data-FA.fir.data), sampling_interval=TR)
            S_original1 = SpectralAnalyzer(T1)


            ultralow = np.abs(S_original1.spectrum_fourier[1].T[1:].T)
            ultralow = torch.from_numpy(ultralow).T.float()

            ans_dict= {'fmri_lowfreq_sequence':low,'fmri_ultralowfreq_sequence':ultralow,'subject':subj,'subject_name':subj_name, self.target:target}
        

        return ans_dict
    
