'''
This alg is based on object_tracker_tflite.py by Dror. It evaluates velocity of each track in history dicts of given pkl file
 by calculating velocity based on location(x,y) and timestamp of each frame of each track.
'''
#written by Eyal Cohen.
#imports
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os
import sys
import statistics as stats

#------PARAMETERS--------: file Name - may switch input pkl files
fileName = str(sys.argv[1])+ "/" + str(sys.argv[2]) 
# oldVersion Example: 'tracks_20230622-092714.pkl' ,Current Example : 2023072606_tracks/tracks_00M36S_1690351236.mp4.pkl
#fileName = "C:/Users/eyala/Documents/Eyal/work/fish_tracking/tracks_analayze_automation/2023072606_tracks/tracks_00M36S_1690351236.mp4.pkl"
img_h = 1296  # height in pixels
img_w = 2304  # width in pixels
pixel = 0.0002645833 # 1 pixel = 0.0002645833 meters
savePlotImage = False  # Change to True if want output image of plots as well as pkl.
velocity_dict = dict() #Feature 1
linear_regression_dict = dict() # Feature 2
acceleration_dict = dict() # Feature 3
curvature_dict = dict() # Feature 4
x0 ,y0 = 500,500 #Center of fish tank
dispersion_dict = dict() # Feature 5



#View data of input pkl.
with open(fileName,'rb') as file:
    history_dict, max_cosine_distance, nn_budget, max_age, max_iou_dist, n_init, roi, frame_height, frame_width, model_path = pickle.load(file)

'''
print('\nFile:', fileName, ' ---------------')
print('model_path =', model_path)
print('max_cosine_distance =', max_cosine_distance)
print('nn_budget =', nn_budget)
print('max_age =', max_age)
print('max_iou_dist =', max_iou_dist)
print('n_init =', n_init)
print('Number of valid trackers:', len(history_dict.keys()))
print('Available trackers are:', history_dict.keys())
'''

#velocity evaluation per track/trajectory
for key in history_dict.keys():
    velocity = []
    acceleration = []
    curvature = []
    time = []
    timestamp = np.array(history_dict[key]['timestamp']) 
    xy_data = np.array(history_dict[key]['center'])
    frame = np.array(history_dict[key]['frame'])
    xy_data = np.ndarray.tolist(xy_data) #converting from np array to list
    if len(xy_data) > 10: #if track exists more than 10 frames
        timestamp = np.ndarray.tolist(timestamp)
        x_arr,y_arr = [],[]
        for i in range(len(frame)-1): # frame-1 iterations cuz iteration includes i+1 
            # Feature 2 calcs
            x_arr.append(xy_data[i][0]) ; y_arr.append(xy_data[i][1])
            # Feature 1 calcs
            x_start = xy_data[i][0] ; x_end = xy_data[i+1][0]
            y_start = xy_data[i][1] ; y_end = xy_data[i+1][1] 
            powx = math.pow(np.abs(x_end - x_start),2)* img_w # (x2-x1)^2 , convert to pixel
            powy = math.pow(np.abs(y_end - y_start),2)* img_h # (y2-y1)^2 , convert to pixel 
            dist = math.sqrt(powx + powy) * pixel #square root of (powx + powy) = distance , convert to meters( * pixel to meters)
            start_timestamp = timestamp[i]/1000 # miliseconds / 1000 = seconds
            end_timestamp = timestamp[i+1]/1000 # miliseconds / 1000 = seconds
            if(end_timestamp - start_timestamp > 0.01 and 0.72 > dist > 0):
                velocity.append(dist / (end_timestamp - start_timestamp)) #[v1,v2,...,vn] -v = acceleration
                time.append(end_timestamp - start_timestamp)
        # Feature 1 calcs - velocity
        if len(velocity) > 0:
            velocity_dict[key] = round(stats.median(velocity),3) # velocity unit is meters/second #med(vn)
        # Feature 2 calcs - linear regression
        x_arr.append(xy_data[i+1][0]) ; y_arr.append(xy_data[i+1][1]) #full lists of [x1,x2,...,xn],[y1,y2,...,yn]
        coeff = np.polyfit(x_arr,y_arr,1)
        linear_poly = np.poly1d(coeff)
        x1_vec = linear_poly(x_arr[0]) ; xn_vec = linear_poly(x_arr[len(x_arr)-1])
        y1_vec = linear_poly(y_arr[0]) ; yn_vec = linear_poly(y_arr[len(y_arr)-1])
        alpha = np.arctan((yn_vec - y1_vec) / (xn_vec - x1_vec))
        linear_regression_dict[key] = alpha
        # Feature 3 calcs - acceleration
        if len(velocity) > 2:
            for j in range(1,len(velocity)-1):
                accel_val = (velocity[j] - velocity[j-1] / time[j] - time[j-1])
                acceleration.append(accel_val)
            acceleration_dict[key] = stats.median(acceleration)
        #Feature 4 calcs - curvature
        if len(acceleration) > 2:
            for k in range(1,len(acceleration)):
                curve_val = (acceleration[k] - acceleration[k-1] / time[k] - time[k-1])
                curvature.append(curve_val)
            curvature_dict[key] = stats.median(curvature)
        # Feature 5 calcs - dispersion
        sx = 0
        sy = 0
        for q in range(len(x_arr)):
            sx += math.pow(x_arr[q] - x0,2)
            sy += math.pow(y_arr[q] - y0,2)
        sx /= (len(x_arr)-1)
        sy /= (len(y_arr)-1)
        snPow2 = [sx,sy]
        dispersion_dict[key] = snPow2


    
#extracting trakcer number from dict to form a list of trackers indentifiers.
#trackers = []
trackers = history_dict.keys() #tryout
'''
i = 0
for key in history_dict.keys():
    for j in range(len(history_dict[key]['frame'])):
        if i in acceleration_dict.keys():
            trackers.append(i)
        i+=1        

'''
#extracting velocity from dict to form a list of trackers velocity.
trackervelocity = []
for track in velocity_dict:
    trackervelocity.append(velocity_dict[track])


median_velocity = stats.median(trackervelocity) # median velocity of all tracks in vid.

if savePlotImage:
    #generating random colors for bar chart.
    barColors = []
    for j in range(len(trackers)):
        rgbValue=""
        for i in range(6):
            rgbValue += random.choice("0123456789ABCDEF")
        rgbValue = "#"+rgbValue
        barColors.append(rgbValue)

    #plot view.
    plt.figure
    plt.xlabel('Track No.')
    plt.ylabel('Average velocity(m/sec)')
    plt.grid()
    plt.bar(trackers,trackervelocity,label = trackers,color = barColors)
    plt.title('Analyze velocity')
    #plt.show() #currently saving the figure as image (line 111)

#argv[1] = directory , argv[2] = file name.
try:
    newDirName = str(sys.argv[2]).split(".mp4")
    os.mkdir(os.path.join(str(sys.argv[1]) + "/" ,newDirName[0] + '_analyse'))
    #os.mkdir(os.path.join('tracks_analysis/','tracks_00M36S_1690351236.mp4.pkl'+'_analayse'))
    #print("Made DIR")

except OSError as error:
    print("") 

filename2save = os.path.join(str(sys.argv[1]) + "/" + newDirName[0] + '_analyse', 'velocity_' + newDirName[0] + '.pkl') #name used to be with timestr , now is with video name.
#filename2save = os.path.join('tracks_analysis/'+'tracks_00M36S_1690351236.mp4.pkl'+'_analayse', 'velocity_' + 'tracks_00M36S_1690351236.mp4.pkl') #name used to be with timestr , now is with video name.
# Save history_dict + plot
#print('Saving file:', filename2save)
if savePlotImage:
    plt.savefig(os.path.join(str(sys.argv[1]) + "/" + newDirName[0] + '_analyse', 'velocity_' + newDirName[0] + ".jpg"))
with open(filename2save, 'wb') as f:
    pickle.dump([velocity_dict,linear_regression_dict,acceleration_dict,curvature_dict,dispersion_dict], f) #add velocity_dict,linear_regression_dict,acceleration_dict,curvature_dict,dispersion_dict


#code to get dist without 2nd loop. has index problem...
'''
xy_data = np.array(history_dict[key]["center"])
frame = np.array(history_dict[key]["frame"])
dist = np.abs(xy_data[frame,:] - xy_data[frame+1,:]) #index problem here.(index 7 is out of bounds for axis 0 with size 7)
print('***********************2nd dist********************** \n')
print(dist)
'''
#dist = np.abs(xy[frame, :] - xy_data)