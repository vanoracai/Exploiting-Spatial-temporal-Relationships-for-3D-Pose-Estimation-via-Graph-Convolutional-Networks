"""
A bunch of utility functions for dealing with human3.6m data.
"""

from __future__ import division
import numpy as np


import pickle


"""
  For

  For human 3.6m
0  0  -- Hip
1  1  -- RHip
2  2  -- RKnee
3  3  -- RFoot
4  6  -- LHip
5  7  -- Lknee
6  8  -- Lfoot
7  12 -- Spine
8  13 -- Thorax
9  14 -- Neck/Nose
10  15 -- Head
11  17 -- LShoulder
12  18 -- LElbow
13  19 -- LWrist
14  25 -- RShoulder
15  26 -- RElbow
16  27 -- RWrist


"""



def normalize_data( data, data_mean, data_std, dim_to_use, actions,dim=3):

  data_out = {}
  nactions = len(actions)
  for key in data.keys():
    data[ key ] = data[ key ][ :, dim_to_use ]
    mu = data_mean[dim_to_use]
    stddev = data_std[dim_to_use]
    data_out[ key ] = np.divide( (data[key] - mu), stddev + 1E-8)

  return data_out

def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_use ):
  """Borrowed from Ashesh. Unnormalizes a matrix.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12
  """

  T = normalizedData.shape[0] # Batch size
  D = data_mean.shape[0] # Dimensionality
  origData = np.zeros((T, D), dtype=np.float32)
  origData[:, dimensions_to_use] = normalizedData

  # TODO this might be very inefficient? idk
  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  origData = np.multiply(origData, stdMat) + meanMat
  return origData

def unNormalizeData_gpu(normalizedData, data_mean, data_std, dimensions_to_use ):
  """Borrowed from Ashesh. Unnormalizes a matrix.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12
  """

  T = normalizedData.size(0)# Batch size
  data_mean_use = data_mean[dimensions_to_use].view(1, -1).repeat(T,1) #T*D
  data_std_use = data_std[dimensions_to_use].view(1, -1).repeat(T,1)#T*D
  out_data = normalizedData *data_std_use + data_mean_use



  return out_data



def define_actions( action ):

  actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

  if action == "All" or action == "all" or action == '*':
    return actions

  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )

  return [action]


def read_mean_std_from_file(filename_3d,filename_2d):
  pkl_file = open('%s.pkl' %(filename_3d), 'rb')
  mean_std_dic = pickle.load(pkl_file)
  mean_std_list_3d = [mean_std_dic['data_mean_3d'],mean_std_dic['data_std_3d'],mean_std_dic['dim_to_ignore_3d'],mean_std_dic['dim_to_use_3d']]
  pkl_file.close()

  pkl_file = open('%s.pkl' % (filename_2d), 'rb')
  mean_std_dic = pickle.load(pkl_file)
  mean_std_list_2d = [mean_std_dic['data_mean_2d'],mean_std_dic['data_std_2d'],mean_std_dic['dim_to_ignore_2d'],mean_std_dic['dim_to_use_2d']]
  pkl_file.close()

  return mean_std_list_3d, mean_std_list_2d

