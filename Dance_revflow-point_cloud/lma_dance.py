# -*- coding: utf-8 -*-
import numpy as np 
import math
import json
import glob
import os
import statistics as s
from scipy.spatial.distance import euclidean as e_d
import csv
from pandas import read_csv
#import pyny3d.geoms as pyny
from shapely.geometry import Polygon
#from area import area
import matplotlib.pyplot as plt

#determinant of matrix a
def det(a):
    return a[0][0]*a[1][1]*a[2][2] + a[0][1]*a[1][2]*a[2][0] + a[0][2]*a[1][0]*a[2][1] - a[0][2]*a[1][1]*a[2][0] - a[0][1]*a[1][0]*a[2][2] - a[0][0]*a[1][2]*a[2][1]

#unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    x = det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
    y = det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
    z = det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)
#dot product of vectors a and b
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

#cross product of vectors a and b
def cross(a, b):
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return (x, y, z)

#area of polygon poly
def area_p(poly):
    if len(poly) < 3: # not a plane - no area
        return 0

    total = [0, 0, 0]
    for i in range(len(poly)):
        vi1 = poly[i]
        if i is len(poly)-1:
            vi2 = poly[0]
        else:
            vi2 = poly[i+1]
        prod = cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)

if __name__ == '__main__':
    
    # select the model
    #model_name = "REV" #{"HPGAN", "MVAE", "REV", "TF"}
    model_names = ["HPGAN", "MVAE", "Dance_Rev"]#, "TransFlower"]  
    n_frames = " 200 frames"
    n_fs = 200
    
    co = 0
    for model_name in model_names:
        pose_file = []
        
        #load the dance files for analysis
        if model_name == "Dance_Rev":
            pose_file.append("results/new_preprocess_ptcloud_rev/rev_dance2.npz") # generated 
            pose_file.append("results/new_preprocess_ptcloud_rev/rev_dance_in2.npz") # ground truth
        elif model_name == "MVAE":
            pose_file.append("results/MVAE/mvae_0.npz")
            pose_file.append("results/MVAE/mvae_in_0.npz")
        elif model_name == "HPGAN":
            pose_file.append("results/HPGAN/hpgan_0.npz")
            pose_file.append("results/HPGAN/hpgan_in_0.npz")
        elif model_name == "TransFlower":
            pose_file.append("results/m_TransFlower/gen_.npz")
            pose_file.append("results/m_TransFlower/ref_trans.npz")

        for f in range(len(pose_file)):
            #print(pose_file[f])
            # load the file             
            with open(pose_file[f], "rb") as file:
                poses_all = np.load(file, allow_pickle=True)                    

            total_window_length = n_fs-1
            mov_window = 10
            overlap_window = 0
            n_features = 28
            tot_features = 61
            file_name = [model_name + '_gen', model_name + '_gnd']

            print(file_name[f], poses_all.shape)
            p = poses_all[:total_window_length+1,:,:]
            print(p.shape)

            # initialize joints #joint_n - 1 if excluded 0 (pelvis/hips) else same 
            r_foot = 29
            l_foot = 4
            hips = 0 # root joint
            r_hand = 22
            l_hand = 12
            head = 18
            r_shoulder = 19
            l_shoulder = 9
            neck = 17

            j = 0
            i = 0
            u_11 = 0 #initial velocity
            u_12 = 0
            u_13 = 0
            a_11 = 0
            gy_min = 1000
            hip_v = []
            hand_v = []
            feet_v = []
            w_f = []

            header_w = ['ϕ'+str(k) for k in range(tot_features)]
            header_f = ['f'+str(k) for k in range(n_features)]

            csv_file_w = os.path.join('results', 'lma_feats', 'feature_mat_'+file_name[f]+str(n_fs)+'.csv')
            with open(csv_file_w, 'w+') as csvfilew:
                writer_w = csv.writer(csvfilew)
                writer_w.writerow(header_w)

                while j < total_window_length-overlap_window-1:
                    v = 0        
                    #initialize features to zero (distance covered)
                    feature_set = np.zeros(tot_features)
                    T_feat = np.zeros(n_features)

                    csv_file_f = os.path.join('results', 'lma_feats', 'feature_m_'+file_name[f]+str(n_fs)+str(j)+'.csv')
                    with open(csv_file_f, 'w+') as csvfilef:
                        writer = csv.writer(csvfilef)
                        writer.writerow(header_f)

                        while v < mov_window:
                            i = v + j 

                            # mean of right and left sides 
                            #body features
                            gx,y,gz = p[i,:,hips]
                            gy = np.minimum(p[i,1,r_foot],p[i,1,l_foot]) # lowest point in frames as floor cooradinates
                            if gy < gy_min:
                                gy_min = gy
                            #print(i,p[i,:,hips])
                            gnd = [gx,gy_min,gz]

                            T_feat[1] = (e_d(p[i,:,r_foot],p[i,:,hips])+e_d(p[i,:,l_foot],p[i,:,hips]))/2 
                            T_feat[2] = (e_d(p[i,:,r_shoulder],p[i,:,r_hand])+e_d(p[i,:,l_shoulder],p[i,:,l_hand]))/2 
                            T_feat[3] = e_d(p[i,:,r_hand],p[i,:,l_hand])                
                            T_feat[4] = (e_d(p[i,:,head],p[i,:,r_hand])+e_d(p[i,:,head],p[i,:,l_hand]))/2                 
                            T_feat[5] = (e_d(p[i,:,hips],p[i,:,r_hand])+e_d(p[i,:,hips],p[i,:,l_hand]))/2                 
                            T_feat[6] = e_d(p[i,:,hips],gnd)                
                            T_feat[7] = T_feat[6]-T_feat[1]              
                            T_feat[8] = e_d(p[i,:,r_foot],p[i,:,l_foot])

                            # Effort feature
                            #p1 = p[i,:,head] 
                            #p2 = p[i+1,:,hips]
                            #ang1 = np.arctan2(*p1[::-1])
                            #ang2 = np.arctan2(*p2[::-1])
                            #f9 = np.rad2deg((ang1 - ang2) % (2 * np.pi))
                            #head orientation
                            d_v1 = p[i+1,:,head] - p[i,:,head] # direction vector
                            d_v2 = p[i+1,:,hips] - p[i,:,hips]
                            if e_d(d_v1, [0,0,0]) != 0 :
                                n_v1 = d_v1/ e_d(d_v1, [0,0,0]) # norm of vector
                            else:
                                n_v1 = 0                           
                            if e_d(d_v2, [0,0,0]) != 0:
                                n_v2 = d_v2/ e_d(d_v2, [0,0,0])
                            else:
                                n_v2 = 0
                            T_feat[9] = np.sum(n_v2 - n_v1)
                            T_feat[11] = e_d(p[i,:,hips],p[i+1,:,hips]) #distance in each frame [total distance/n_frames]  
                            #avg velocity of right and left per frame
                            hip_v.append(T_feat[11])
                            T_feat[12] = (e_d(p[i,:,r_hand],p[i+1,:,r_hand]) + e_d(p[i,:,l_hand],p[i+1,:,l_hand]))/2        
                            T_feat[13] = (e_d(p[i,:,r_foot],p[i+1,:,r_foot]) + e_d(p[i,:,l_foot],p[i+1,:,l_foot]))/2
                            hand_v.append(T_feat[12])
                            feet_v.append(T_feat[13])
                            #acceleration v-u                    
                            T_feat[14] = T_feat[11] - u_11
                            T_feat[10] = 0-T_feat[14] # -ve acceleration # deceleration 
                            w_f.append(T_feat[10])
                            u_11 = T_feat[11]
                            T_feat[15] = T_feat[12] - u_12
                            u_12 = T_feat[12]
                            T_feat[16] = T_feat[13] - u_13
                            u_13 = T_feat[13]
                            T_feat[17] = T_feat[14] - a_11
                            a_11 = T_feat[14]

                            #shape features
                            T_feat[24] = e_d(p[i,:,head],p[i,:,hips])
                            d_nh = e_d(p[i,:,neck],p[i,:,hips])
                            if T_feat[5]-T_feat[24] >0 and T_feat[4]-T_feat[24]<0:
                                T_feat[25] = 1 #'above head'
                            elif T_feat[5]- d_nh >0:
                                T_feat[25] = 0.5 #'btw head & chest'
                            else:
                                T_feat[25] = 0 #'below head'

                            #space features
                            T_feat[26] += e_d(gnd ,p[i,:,hips])
                            if v == mov_window-1 :
                                #if p_hips.all() != p[i,:,hips].all(): 
                                #print(p_hips, p[i,:,hips], p_gnd, gnd)
                                coordinates = np.array([p_hips, p[i,:,hips], p_gnd, gnd])
                                #polygon = pyny.Polygon(coordinates)
                                #obj = {'type':'Polygon','coordinates':coordinates}
                                T_feat[27] = area_p(coordinates)
                                #Polygon(coordinates).area #area_p(coordinates) #area(obj) #polygon.get_area()
                                if math.isnan(T_feat[27]):
                                    T_feat[27] = 0
                                #print(T_feat[27])
                            if v == 0:
                                p_hips = p[i,:,hips]
                                p_gnd = gnd
                            writer.writerow(T_feat)
                            v+=1


                    if j == 0:
                        j += mov_window-overlap_window-1
                    else:
                        j += mov_window-overlap_window
                    d_frame = read_csv(csv_file_f)
                    q = 0
                    x_feat = [10,25,26,27]
                    x_feat1 = [11,12,13]
                    x_feat2 = [14,15,16,17] 
                    not_done = [18,19,20,21,22,23] # exclude volume features
                    for n in range(1,n_features): 
                        features = d_frame['f'+str(n)].tolist()
                        
                        if n in not_done:
                            continue
                        elif n in x_feat:
                            feature_set[q] = T_feat[n]
                            q+=1
                        elif n in x_feat1:
                            feature_set[q] = np.max(features)
                            q+=1
                            feature_set[q] = np.min(features)
                            q+=1
                            feature_set[q] = np.std(features)
                            q+=1
                        elif n in x_feat2:
                            feature_set[q] = np.max(features)
                            q+=1
                            feature_set[q] = np.std(features)
                            q+=1
                        else:

                            feature_set[q] = np.max(features)
                            q+=1
                            feature_set[q] = np.min(features)
                            q+=1
                            feature_set[q] = np.mean(features)
                            q+=1
                            feature_set[q] = np.std(features)
                            q+=1
                    #print(feature_set[60])
                    writer_w.writerow(feature_set)

            if f == 0 and overlap_window == 0:
                # plot movement velocity plots for evaluation
                #plot for deceleration
#                 a_hip = np.array(w_f)
#                 #colors = ['c','y','g','b']
#                 frames = np.arange(total_window_length+1)
#                 plt.plot(frames, a_hip, 'g')
#                 plt.title('Deceleration for ' + model_name+n_frames)
#                 plt.xlabel('no. of frames')
#                 plt.ylabel('deceleration')
#                 plt.savefig("results/lma_feats/figures/dec_"+file_name[f]+n_frames+".png")
#                 plt.clf() # clear plot
                v_hip = np.array(hip_v)
                v_hand = np.array(hand_v)
                v_feet = np.array(feet_v)
                #print(v_hip.shape,v_hand.shape,v_feet.shape)
                plt.plot(frames, v_hip, colors[co], label=model_name)
                #plt.plot(frames, v_hand, 'y', label='hand velocity')           
                #plt.plot(frames, v_feet, 'g', label='feet velocity')
                plt.title('Hip movement velocity of models' + n_frames, fontsize = 19)
                plt.xlabel('no. of frames', fontsize = 17)
                plt.ylabel('velocity', fontsize = 17)
                plt.legend(fontsize = 15)
                plt.tick_params(axis='y', labelsize=14)
                plt.tick_params(axis='x', labelsize=14)
                
                if co == len(model_names)-1:
                    #print("save")
                    plt.savefig("results/lma_feats/figures/hip_velocity_"+n_frames+".png")
                co+=1
