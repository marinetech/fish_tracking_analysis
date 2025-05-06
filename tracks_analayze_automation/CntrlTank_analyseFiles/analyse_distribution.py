#this file evaluates most populated grid over the entire input trackers generated from a video.
#This alg is based on object_tracker_tflite.py by Dror. 
#written by Eyal Cohen.
#imports
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

#------PARAMETERS--------: file Name - may switch input pkl files
fileName = str(sys.argv[1])+ "/" + str(sys.argv[2])#'tracks_20230622-092714.pkl' - was mannual.
#fileName = "C:/Users/eyala/Documents/Eyal/work/fish_tracking/tracks_analayze_automation/2023072606_tracks/tracks_00M36S_1690351236.mp4.pkl"
#img_h = 1296  # height in pixels
#img_w = 2304  # width in pixels
#pixel = 0.0002645833 # 1 pixel = 0.0002645833 meters
savePlotImage = False  # Change to True if want output image of plots as well as pkl.

#-----grid definition-----
# grid-i - [0] = min(x) , [1] - max(x) , [2] - min(y) , [3] - max(y).
grid1 ,grid2 ,grid3 ,grid4 = [0,250,0,250],[250,500,0,250],[500,750,0,250] ,[750,1000,0,250]
grid5 ,grid6 ,grid7 ,grid8 = [0,250,250,500],[250,500,250,500],[500,750,250,500] ,[750,1000,250,500]
grid9 ,grid10 ,grid11 ,grid12 = [0,250,500,750],[250,500,500,750],[500,750,500,750] ,[750,1000,500,750]
grid13 ,grid14 ,grid15 ,grid16 = [0,250,750,1000],[250,500,750,1000],[500,750,750,1000] ,[750,1000,750,1000]

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
print('\n')
'''

gridPop,frameTime,timeStampArr = np.zeros(16),np.zeros(16),np.zeros(16)
if savePlotImage:
    #plotting figure 1
    fig1 = plt.figure(figsize=(10,15))
    plt.subplot(311)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Analyze distribution (scattered tracks)')
    plt.grid()
maxframe,maxTimeStamp,minTimeStamp = 0 , 0 ,0# finding length of video in frames.
x,y = [],[] #list of (x,y) dot locations of a tracker over the vid.
for key in history_dict.keys():
    timestamp = np.array(history_dict[key]['timestamp']) 
    xy_data = np.array(history_dict[key]['center'])
    frame = np.array(history_dict[key]['frame'])
    xy_data = np.ndarray.tolist(xy_data) #converting from np array to list
    frame = np.ndarray.tolist(frame) #converting from np array to list
    timestamp = np.ndarray.tolist(timestamp) #converting from np array to list
    if len(xy_data) > 10:
        for i in range(len(frame)-1):
            maxTimeStamp = max(maxTimeStamp,timestamp[i],timestamp[i+1])
            maxframe = max(maxframe,frame[i],frame[i+1])
            minTimeStamp = min(minTimeStamp,timestamp[i],timestamp[i+1])
            if(grid1[0] < xy_data[i][0] < grid1[1] and grid1[2] < xy_data[i][1] < grid1[3]):
                gridPop[0]+= 1
                frameTime[0] += round(np.abs(frame[i+1]-frame[i]),6)
                timeStampArr[0] += np.abs(timestamp[i+1]-timestamp[i])
            elif(grid2[0] < xy_data[i][0] < grid2[1] and grid2[2] < xy_data[i][1] < grid2[3]):
                gridPop[1]+= 1
                frameTime[1] += round(np.abs(frame[i+1]-frame[i]),6)
                timeStampArr[1] += np.abs(timestamp[i+1]-timestamp[i])
            elif(grid3[0] < xy_data[i][0] < grid3[1] and grid3[2] < xy_data[i][1] < grid3[3]):
                gridPop[2]+= 1
                frameTime[2] += round(np.abs(frame[i+1]-frame[i]),6)
                timeStampArr[2] += np.abs(timestamp[i+1]-timestamp[i])
            elif(grid4[0] < xy_data[i][0] < grid4[1] and grid4[2] < xy_data[i][1] < grid4[3]):
                gridPop[3]+= 1
                frameTime[3] += round(np.abs(frame[i+1]-frame[i]),6)
                timeStampArr[3] += np.abs(timestamp[i+1]-timestamp[i])
            elif(grid5[0] < xy_data[i][0] < grid5[1] and grid5[2] < xy_data[i][1] < grid5[3]):
                gridPop[4]+= 1
                frameTime[4] += round(np.abs(frame[i+1]-frame[i]),6)
                timeStampArr[4] += np.abs(timestamp[i+1]-timestamp[i])
            elif(grid6[0] < xy_data[i][0] < grid6[1] and grid6[2] < xy_data[i][1] < grid6[3]):
                gridPop[5]+= 1
                frameTime[5] += round(np.abs(frame[i+1]-frame[i]),6) 
                timeStampArr[5] += np.abs(timestamp[i+1]-timestamp[i])
            elif(grid7[0] < xy_data[i][0] < grid7[1] and grid7[2] < xy_data[i][1] < grid7[3]):
                gridPop[6]+= 1
                frameTime[6] += round(np.abs(frame[i+1]-frame[i]),6)
                timeStampArr[6] += np.abs(timestamp[i+1]-timestamp[i])    
            elif(grid8[0] < xy_data[i][0] < grid8[1] and grid8[2] < xy_data[i][1] < grid8[3]):
                gridPop[7]+= 1
                frameTime[7] += round(np.abs(frame[i+1]-frame[i]),6)
                timeStampArr[7] += np.abs(timestamp[i+1]-timestamp[i])
            elif(grid9[0] < xy_data[i][0] < grid9[1] and grid9[2] < xy_data[i][1] < grid9[3]):
                gridPop[8]+= 1
                frameTime[8] += round(np.abs(frame[i+1]-frame[i]),6)
                timeStampArr[8] += np.abs(timestamp[i+1]-timestamp[i])
            elif(grid10[0] < xy_data[i][0] < grid10[1] and grid10[2] < xy_data[i][1] < grid10[3]):
                gridPop[9]+= 1
                frameTime[9] += round(np.abs(frame[i+1]-frame[i]),6)
                timeStampArr[9] += np.abs(timestamp[i+1]-timestamp[i])
            elif(grid11[0] < xy_data[i][0] < grid11[1] and grid11[2] < xy_data[i][1] < grid11[3]):
                gridPop[10]+= 1
                frameTime[10] += round(np.abs(frame[i+1]-frame[i]),6)
                timeStampArr[10] += np.abs(timestamp[i+1]-timestamp[i])
            elif(grid12[0] < xy_data[i][0] < grid12[1] and grid12[2] < xy_data[i][1] < grid12[3]):
                gridPop[11]+= 1
                frameTime[11] += round(np.abs(frame[i+1]-frame[i]),6)
                timeStampArr[11] += np.abs(timestamp[i+1]-timestamp[i])
            elif(grid13[0] < xy_data[i][0] < grid13[1] and grid13[2] < xy_data[i][1] < grid13[3]):
                gridPop[12]+= 1
                frameTime[12] += round(np.abs(frame[i+1]-frame[i]),6)
                timeStampArr[12] += np.abs(timestamp[i+1]-timestamp[i])
            elif(grid14[0] < xy_data[i][0] < grid14[1] and grid14[2] < xy_data[i][1] < grid14[3]):
                gridPop[13]+= 1
                frameTime[13] += round(np.abs(frame[i+1]-frame[i]),6)
                timeStampArr[13] += np.abs(timestamp[i+1]-timestamp[i])
            elif(grid15[0] < xy_data[i][0] < grid15[1] and grid15[2] < xy_data[i][1] < grid15[3]):
                gridPop[14]+= 1
                frameTime[14] += round(np.abs(frame[i+1]-frame[i]),6)
                timeStampArr[14] += np.abs(timestamp[i+1]-timestamp[i])
            elif(grid16[0] < xy_data[i][0] < grid16[1] and grid16[2] < xy_data[i][1] < grid16[3]):
                gridPop[15]+= 1
                frameTime[15] += round(np.abs(frame[i+1]-frame[i]),6)
                timeStampArr[15] += np.abs(timestamp[i+1]-timestamp[i])
            x.append(xy_data[i][0])
            y.append(xy_data[i][1])
        #print(len(x),len(y),'\n')
        if savePlotImage:
            plt.scatter(x,y)
        x,y = [],[]
x= []
if savePlotImage:
    #plotting figure 2
    plt.subplot(312)
    plt.xlabel('Frame')
    plt.ylabel('X')
    plt.title('X per frame (scattered X)')
    plt.grid()

    for key in history_dict.keys():
        xy_data = np.array(history_dict[key]['center'])
        frame = np.array(history_dict[key]['frame'])
        xy_data = np.ndarray.tolist(xy_data) #converting from np array to list
        frame = np.ndarray.tolist(frame) #converting from np array to list
        if len(xy_data) > 10:
            for i in range(len(frame)):
                x.append(xy_data[i][0])
                y.append(xy_data[i][1])
            #print(len(x),len(y),'\n')
            plt.scatter(frame,x)
            x = []
    y = []
    #plotting figure 3
    plt.subplot(313)
    plt.xlabel('Frame')
    plt.ylabel('Y')
    plt.title('Y per frame (scattered Y)')
    plt.grid()

    for key in history_dict.keys():
        xy_data = np.array(history_dict[key]['center'])
        frame = np.array(history_dict[key]['frame'])
        xy_data = np.ndarray.tolist(xy_data) #converting from np array to list
        frame = np.ndarray.tolist(frame) #converting from np array to list
        if len(xy_data) > 10:
            for i in range(len(frame)):
                x.append(xy_data[i][0])
                y.append(xy_data[i][1])
            #print(len(y),len(frame),'\n')
            plt.scatter(frame,y)
            y = []

'''
#*****CHANGE MIN/MAX NOT VALID ANSWER*********
#[max(population / frames) ]- NOT TRUE = most populated grid over the vid.
pOf = [] #population over frames
for i in range(len(gridPop)):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if gridPop[i]/frameTime[i] > 0: #make change here.
            pOf.append(gridPop[i]/frameTime[i])
        else:
            pOf.append(0)
#finding min/max pop over frame and its grid(index)
maxPoF,indxMax = max(pOf),pOf.index(max(pOf)) + 1
pOf_zeroClean = []
for pof in pOf:
    if pof > 0:
        pOf_zeroClean.append(pof)
minPoF,indxMin = min(pOf_zeroClean),pOf.index(min(pOf_zeroClean)) + 1  '''

vidLength = maxTimeStamp - minTimeStamp
mostPopulatedGrid_list = [] # calculations is: fish amount * (time/total time)
for i in range(16): #16 = len of gridPop & timeStampArr
    if gridPop[i] * (timeStampArr[i]/vidLength) > 0:
        mostPopulatedGrid_list.append(gridPop[i] * (timeStampArr[i]/vidLength))
    else :
        mostPopulatedGrid_list.append(0)
mostPopulatedGrid,mostPopulatedGrid_idx = round(max(mostPopulatedGrid_list),3),mostPopulatedGrid_list.index(max(mostPopulatedGrid_list))+1
mostPopulatedGrid_list_zeroClean = []
for i in range(16):
    if mostPopulatedGrid_list[i] > 0:
        mostPopulatedGrid_list_zeroClean.append(mostPopulatedGrid_list[i])
if len(mostPopulatedGrid_list_zeroClean) != 0:
    leastPopulatedGrid,leastPopulatedGrid_idx = round(min(mostPopulatedGrid_list_zeroClean),3),mostPopulatedGrid_list.index(min(mostPopulatedGrid_list_zeroClean)) + 1
else:
    leastPopulatedGrid_idx = -1

#plt.show() #instead of showing we currently save the figure as image (line 220)

#argv[1] = directory , argv[2] = file name.
newDirName = ''
try: 
    newDirName = str(sys.argv[2]).split(".mp4")
    os.mkdir(os.path.join(str(sys.argv[1]) + "/" , newDirName[0] + '_analyse'))
except OSError as error: 
        print("") 
filename2save = os.path.join(str(sys.argv[1]) + "/" + newDirName[0] + '_analyse', 'distribution_' + newDirName[0] + ".pkl") #name used to be with timestr , now is with video name.
# Save history_dict + plot
#print('Saving file:', filename2save)
if savePlotImage:
    plt.savefig(os.path.join(str(sys.argv[1]) + "/" + newDirName[0] + '_analyse', 'distribution_' + newDirName[0] + ".jpg")) #saving plot as image instead of showing it
with open(filename2save, 'wb') as f:
    pickle.dump([leastPopulatedGrid_idx,mostPopulatedGrid_idx , gridPop, timeStampArr,vidLength], f)
    #pickle.dump([gridPop ,frameTime], f)  #[minPoF,indxMin,maxPoF,indxMax] was part of pickle 
#[min(poplutaion/frameTime), min index , max(poplutaion/frameTime) ,max index ,population over all grids over entire vid, frames spent by fish in each grid by index(frameTime[0] = grid 1...)]

