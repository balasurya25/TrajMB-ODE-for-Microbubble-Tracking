import numpy as np
import argparse
import os
import copy
import sys
import random

import numpy as np
import copy
import os
import argparse

def scenario_1(physical_width,physical_height):
  x = int(np.random.randint(physical_width/8, physical_width - (physical_width/8))) #this kinda ensures that the points we select don't end up on the edge of the image
  z = int(np.random.randint(physical_height/8, physical_height-(physical_height/8)))
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
  x = int(np.random.randint(physical_width/8, physical_width - (physical_width/8))) #this kinda ensures that the points we select don't end up on the edge of the image
  z = int(np.random.randint(physical_height/8, physical_height-(physical_height/8)))
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

def scenario_crossing_pair(physical_width, physical_height):
    # Crossing point near center with small random offset
    cx = physical_width * 0.5 + np.random.uniform(-1.0, 1.0)
    cz = physical_height * 0.5 + np.random.uniform(-1.0, 1.0)

    # Two initial angles that cross (~90 degrees apart)
    phi1 = np.random.uniform(0, np.pi)
    phi2 = phi1 + np.pi/2 + np.random.uniform(-0.2, 0.2)

    # Velocities similar to S2 range
    v1 = np.random.uniform(10.0, 15.0)
    v2 = np.random.uniform(10.0, 15.0)

    # Angular velocities
    w1 = np.random.uniform(-6, 6) * np.pi
    w2 = np.random.uniform(-6, 6) * np.pi
    
    # Avoid near-zero w (straight line edge case)
    w1 = w1 if abs(w1) > 0.1 else 0.1
    w2 = w2 if abs(w2) > 0.1 else 0.1

    a1 = np.random.uniform(0.05, 0.2)
    a2 = np.random.uniform(0.05, 0.2)

    # State: [x, z, vx, vz, v, phi, w, a, label]
    vx1 = v1 * np.cos(phi1)
    vz1 = v1 * np.sin(phi1)
    vx2 = v2 * np.cos(phi2)
    vz2 = v2 * np.sin(phi2)

    s1 = np.array([cx, cz, vx1, vz1, v1, phi1, w1, a1, 1]).reshape(1, 9)
    s2 = np.array([cx, cz, vx2, vz2, v2, phi2, w2, a2, 1]).reshape(1, 9)

    return s1, s2

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
    
    #compute the velocity
    vx = v * np.cos(phi)
    vz = v * np.sin(phi)
    
    x = x + vx * delta_t + np.random.normal(0, 0.005, 1)
    z = z + vz * delta_t + np.random.normal(0, 0.005, 1)
    
    # Velocity update
    v = v + a * delta_t
    label = 1
    
    state_space = np.array([x.item(), z.item(), vx.item(), vz.item(), v.item(), phi.item(), w.item(), a.item(), label]).reshape((1, 9))
    return state_space

def generate_crossing_tracks(physical_width, physical_height, 
                              delta_t, nums, t_points,scenario):
    """
     Generate crossing track pairs.
    
    Args:
        physical_width:  domain width in mm
        physical_height: domain height in mm
        delta_t:         time step in seconds
        nums:            total number of tracks (must be even)
        t_points:        number of time points per track
    
    Returns:
        tracks: (nums, t_points, 2) array of [x, z] positions
    """
    assert nums % 2 == 0, "nums must be even for crossing pairs"
    
    tracks = np.zeros((nums, t_points, 2))
    idx = 0

    while idx < nums:
        s1, s2 = scenario_crossing_pair(physical_width, physical_height)

        for t in range(t_points):
            if t > 0:
                s1 = iterate_nonlinear_mm(s1, delta_t)
                s2 = iterate_nonlinear_mm(s2, delta_t)

            tracks[idx,   t, :] = s1[:, :2]  # [x, z]
            tracks[idx+1, t, :] = s2[:, :2]  # [x, z]

        idx += 2

    return tracks

def generate_tracks(physical_width, physical_height, delta_t, nums, t_points, scenario):
  tracks= np.zeros((nums,t_points,2))
  for num in range(nums):
    #choose motion profile for this specific track
    is_linear = (random.random() < 0.5)
    for t_num in range(t_points):
        if t_num==0:
            if scenario=='1':
                pts= scenario_1(physical_width,physical_height)
            elif scenario=='2':
                pts= scenario_2(physical_width,physical_height)
            elif scenario=='3':
                pts= scenario_3(physical_width,physical_height)
        else:
            if is_linear:
                pts= linear_iterate(pts, delta_t)
            else:
                pts= iterate_nonlinear_mm(pts, delta_t)

        tracks[num,t_num,:]= pts[:,:2] #simply get the [x,z] values
  return tracks

def divide(tracks, train_ratio, val_ratio, test_ratio):
    # Ensure ratios sum to 1
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Ratios must sum to 1.0")

    # Get the total number of tracks
    num_tracks = tracks.shape[0]

    # Create a list of indices and shuffle them
    indices = np.arange(num_tracks)
    np.random.shuffle(indices)

    # Determine the split points
    train_split_point = int(num_tracks * train_ratio)
    val_split_point = int(num_tracks * (train_ratio + val_ratio))

    # Split the indices
    train_indices = indices[:train_split_point]
    val_indices = indices[train_split_point:val_split_point]
    test_indices = indices[val_split_point:]

    # Use the indices to create the final datasets
    train_set = tracks[train_indices]
    val_set = tracks[val_indices]
    test_set = tracks[test_indices]

    # Print sizes for verification
    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")

    return train_set, val_set, test_set

def main(scenario):
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
        train_path = 'data/nonlinear/scenario_1/train'
        val_path = 'data/nonlinear/scenario_1/val'
        test_path = 'data/nonlinear/scenario_1/test'
    elif scenario == '2':
        train_path = 'data/nonlinear/scenario_2/train'
        val_path = 'data/nonlinear/scenario_2/val'
        test_path = 'data/nonlinear/scenario_2/test'
    elif scenario == '3':
        train_path = 'data/nonlinear/scenario_3/train'
        val_path = 'data/nonlinear/scenario_3/val'
        test_path = 'data/nonlinear/scenario_3/test'
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
        tracks = generate_tracks(physical_width, physical_height, delta_t, track_nums, t_points, scenario)

        train_tracks[str(t_points)], val_tracks[str(t_points)], test_tracks[str(t_points)] = divide(tracks, train_per, val_per, test_per)

        #print(np.median(v))

    np.save(train_path, train_tracks)
    np.save(val_path, val_tracks)
    np.save(test_path, test_tracks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='input',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="Bala Surya")
                                        
    # Scenario 2 is used by default
    parser.add_argument('-s', type=str, help='Choose scenario.', default="3")
    args = parser.parse_args()
    main(args.s)