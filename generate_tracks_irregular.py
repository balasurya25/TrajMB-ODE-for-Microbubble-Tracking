#test on irregularly sampled data

import argparse
import numpy as np  
import os
import copy
import sys
import random
#from data_util.data_visual import paint as visual
#from data_util.operate_util import divide as divide

def scenario_1(physical_width,physical_height):
  x = int(np.random.randint(physical_width/8, physical_width - (physical_width/8), 1)) #this kinda ensures that the points we select don't end up on the edge of the image
  z = int(np.random.randint(physical_height/8, physical_height-(physical_height/8), 1))
  #computing the velocity
  v= np.random.uniform(5.0,10.0)
  #get the initial phase
  phi= np.random.uniform(0,1)*np.pi
  #compute the angular velocity
  w= np.random.uniform(-5,5)*np.pi
  #acceleration
  a= 0.1
  #label - if a point belongs to a track or otherwise - this would be adjusted later - for now - we'd be setting all of these to 1
  label= 1
  vx,vz=0,0

  state_space= np.array([x,z,vx,vz,v,phi,w,a,label]).reshape((1,9)) #reshapes the entries into an np array of size 1x7
  return state_space

def scenario_2(physical_width,physical_height):
  x = int(np.random.randint(physical_width/8, physical_width - (physical_width/8))) #this kinda ensures that the points we select don't end up on the edge of the image
  z = int(np.random.randint(physical_height/8, physical_height-(physical_height/8)))
  #computing the velocity
  v= np.random.uniform(10.0,15.0)
  #get the initial phase
  phi= np.random.uniform(0,1)*np.pi
  #compute the angular velocity
  w= np.random.uniform(-8,8)*np.pi
  #acceleration
  a= 0.2
  #label - if a point belongs to a track or otherwise - this would be adjusted later - for now - we'd be setting all of these to 1
  label= 1
  vx,vz=0,0

  state_space= np.array([x,z,vx,vz,v,phi,w,a,label]).reshape((1,9)) #reshapes the entries into an np array of size 1x7
  return state_space

def scenario_3(physical_width,physical_height):
  x = int(np.random.randint(physical_width/8, physical_width - (physical_width/8), 1)) #this kinda ensures that the points we select don't end up on the edge of the image
  z = int(np.random.randint(physical_height/8, physical_height-(physical_height/8), 1))
  #computing the velocity
  v= np.random.uniform(15.0,20.0)
  #get the initial phase
  phi= np.random.uniform(0,2)*np.pi
  #compute the angular velocity
  w= np.random.uniform(-10,10)*np.pi
  #acceleration
  a= 0.5
  #label - if a point belongs to a track or otherwise - this would be adjusted later - for now - we'd be setting all of these to 1
  label= 1
  vx,vz=0,0

  state_space= np.array([x,z,vx,vz,v,phi,w,a,label]).reshape((1,9)) #reshapes the entries into an np array of size 1x7
  return state_space

def iterate_nonlinear_mm(state_space,delta_t): #delta_t is the time step
  x= state_space[:,0]
  z= state_space[:,1]
  vx= state_space[:,2]
  vz= state_space[:,3]
  v= state_space[:,4]
  phi= state_space[:,5]
  w= state_space[:,6]
  a= state_space[:,7]
  phi_tmp = (phi + w * delta_t)
  r_tmp = v / w
  #compute the positions
  x = x + r_tmp * (np.sin(phi) - np.sin(phi_tmp)) + np.random.normal(0,0.007,1)
  z = z + r_tmp * (np.cos(phi_tmp) - np.cos(phi)) + np.random.normal(0,0.007,1)
  #compute the velocities
  vx= v*np.cos(phi_tmp)
  vz= v*np.sin(phi_tmp)

  phi = phi_tmp
  v = v + a * delta_t
  w = v / r_tmp
  label = 1

  state_space = np.array([x.item(), z.item(), vx.item(), vz.item(), v.item(), phi.item(), w.item(), a.item(), label]).reshape((1, 9))
  return state_space

def linear_iterate(state_space, delta_t):
  x = state_space[:, 0]
  z = state_space[:, 1]
  vx= state_space[:,2]
  vz= state_space[:,3]
  v = state_space[:, 4]
  phi = state_space[:, 5]
  w = state_space[:, 6]
  a = state_space[:, 7]
  #compute the positions
  x = x + v*delta_t
  z = z + v*delta_t
  #compute the velocities
  vx= v*np.cos(phi)
  vz= v*np.sin(phi)

  label = 1

  state_space = np.array([x.item(), z.item(), vx.item(), vz.item(), v.item(), phi.item(), w.item(), a.item(), label]).reshape((1, 9))
  return state_space


def generate_tracks(physical_width,physical_height,delta_t,nums,t_points,scenario,drop_rate):
  #generate the necessary tracks by assigning a probability of dropping
  tracks=np.zeros((nums,t_points,3))

  for num in range(nums):
    for t_num in range(t_points):
      if t_num==0:
        if scenario=='1':
          pts= scenario_1(physical_width,physical_height)
        elif scenario=='2':
          pts= scenario_2(physical_width,physical_height)
        elif scenario=='3':
          pts= scenario_3(physical_width,physical_height)
        tracks[num, t_num, :2] = pts[:, :2]
        tracks[num, t_num, 2] = 1 #the label - an observation is present at that time point

      else:
        #for the other points in that trajectory, apply a frame dropping logic
        # Always evolve the system
        pts = iterate_nonlinear_mm(pts, delta_t)

        # Decide whether to record the observation
        if random.random() < drop_rate:
          continue  # skip recording
        else:
          tracks[num, t_num, :2] = pts[:, :2]
          tracks[num, t_num, 2] = 1 #observation is present at that time point
  return tracks


def divide(tracks, train_ratio, val_ratio, test_ratio):
    """
    Divides the tracks into training, validation, and test sets based on ratios.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Ratios must sum to 1.0")

    num_tracks = tracks.shape[0]
    indices = np.arange(num_tracks)
    np.random.shuffle(indices)

    train_split_point = int(num_tracks * train_ratio)
    val_split_point = int(num_tracks * (train_ratio + val_ratio))

    train_indices = indices[:train_split_point]
    val_indices = indices[train_split_point:val_split_point]
    test_indices = indices[val_split_point:]

    train_set = tracks[train_indices]
    val_set = tracks[val_indices]
    test_set = tracks[test_indices]

    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")

    return train_set, val_set, test_set

def main(scenario, drop_rate):
    """
    Main function to generate and save track data.
    """
    seed = 2
    np.random.seed(seed)

    physical_width = 15
    physical_height = 15
    delta_t = 0.02

    track_nums = 2000

    if scenario == '1':
        train_path = 'data/nonlinear/scenario_1/train_irreg'
        val_path = 'data/nonlinear/scenario_1/val_irreg'
        test_path = 'data/nonlinear/scenario_1/test_irreg'
    elif scenario == '2':
        train_path = 'data/nonlinear/scenario_2/train_irreg'
        val_path = 'data/nonlinear/scenario_2/val_irreg'
        test_path = 'data/nonlinear/scenario_2/test_irreg'
    elif scenario == '3':
        train_path = './src/data/nonlinear/scenario_3/train_irreg'
        val_path = './src/data/nonlinear/scenario_3/val_irreg'
        test_path = './src/data/nonlinear/scenario_3/test_irreg'
    else:
        print("scenario error")
        exit(0)

    train_tracks = {}
    val_tracks = {}
    test_tracks = {}

    train_per = 0.7
    val_per = 0.2
    test_per = 0.1

    for t_points in range(5, 51):
        tracks = generate_tracks(physical_width, physical_height, delta_t, track_nums, t_points, scenario, drop_rate)
        train_tracks[str(t_points)], val_tracks[str(t_points)], test_tracks[str(t_points)] = divide(tracks, train_per, val_per, test_per)

    np.save(train_path, train_tracks)
    np.save(val_path, val_tracks)
    np.save(test_path, test_tracks)
