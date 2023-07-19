
import numpy as np
import os


def readNNData():
    with open('.//sde//training_data.txt','r') as f:
        lines = f.readlines()
    f.close()
    return lines

def loadSIRInfo():

    lines = readNNData()
    x_data_list=[]
    y_data_list=[]
    days = 4
    day=1

    # Removing simulations with 0 infectious state in a day of 4 days
    for i in range(0,len(lines),days):
        write = True
        for j in range(days):
            line = lines[i+j]
            line = line.strip('\n').split('-')
            x_data_points = line[0].split(' ')
            y_data_points = line[1].split(' ')
            x_data_0 = round(float( x_data_points[0] ),3)
            x_data_1 = round(float( x_data_points[1] ),3)
            y_data_0 = round(float( y_data_points[0] ),3)
            y_data_1 = round(float( y_data_points[1] ),3)
            if x_data_0 == 0.0:
                write = False
        if write:
            for j in range(days):
                line = lines[i+j]
                line = line.strip('\n').split('-')
                x_data_points = line[0].split(' ')
                y_data_points = line[1].split(' ')
                x_data_0 = round(float( x_data_points[0] ),3)
                x_data_1 = round(float( x_data_points[1] ),3)
                y_data_0 = round(float( y_data_points[0] ),3)
                y_data_1 = round(float( y_data_points[1] ),3)
                x_data_list.append([x_data_0,x_data_1])
                y_data_list.append([y_data_0,y_data_1])



    x_data = np.array(x_data_list[:5000], dtype=np.float32)
    y_data = np.array(y_data_list[:5000], dtype=np.float32)
    x_data_test = np.array(x_data_list[5000:], dtype=np.float32)
    y_data_test = np.array(y_data_list[5000:], dtype=np.float32)
    
    return x_data,y_data,x_data_test,y_data_test

def loadSIRInfo1D():

    lines = readNNData()
    x_data_list=[]
    y_data_list=[]
    days = 4
    day=1

    # Removing simulations with 0 infectious state in a day of 4 days
    for i in range(0,len(lines),days):
        write = True
        for j in range(days):
            line = lines[i+j]
            line = line.strip('\n').split('-')
            x_data_points = line[0].split(' ')
            y_data_points = line[1].split(' ')
            x_data_0 = round(float( x_data_points[0] ),3)
            x_data_1 = round(float( x_data_points[1] ),3)
            y_data_0 = round(float( y_data_points[0] ),3)
            y_data_1 = round(float( y_data_points[1] ),3)
            if x_data_0 == 0.0:
                write = False
        if write:
            for j in range(days):
                line = lines[i+j]
                line = line.strip('\n').split('-')
                x_data_points = line[0].split(' ')
                y_data_points = line[1].split(' ')
                x_data_0 = round(float( x_data_points[0] ),3)
                x_data_1 = round(float( x_data_points[1] ),3)
                y_data_0 = round(float( y_data_points[0] ),3)
                y_data_1 = round(float( y_data_points[1] ),3)
                x_data_list.append([x_data_0])
                y_data_list.append([y_data_0])



    x_data = np.array(x_data_list[:5000], dtype=np.float32)
    y_data = np.array(y_data_list[:5000], dtype=np.float32)
    x_data_test = np.array(x_data_list[5000:], dtype=np.float32)
    y_data_test = np.array(y_data_list[5000:], dtype=np.float32)
    
    return x_data,y_data,x_data_test,y_data_test