import numpy as np
from scipy import signal
import pandas as pd 
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from madgwickahrs import MadgwickAHRS
import quaternion_toolbox

import graphPlots

# Select dataset
# Fs = 256
# filePath = 'Datasets/straightLine_CalInertialAndMag.csv'
# startTime = 6
# stopTime = 28
# tempo_parado = 2 #segundos parado
# mag_enabled = False

def Tracking(arq):
    
#Podem ser alterados para parâmetros de entrada.
    Fs = 256
    startTime = 6
    stopTime = 28
    tempo_parado = 2 #segundos parado
    mag_enabled = False

    #import Data

    samplePeriod = 1/Fs
    dataset = pd.read_csv(arq)
    time = dataset.iloc[:,0].values * samplePeriod
    gyrX = dataset.iloc[:,1].values
    gyrY = dataset.iloc[:,2].values
    gyrZ = dataset.iloc[:,3].values
    accX = dataset.iloc[:,4].values
    accY = dataset.iloc[:,5].values
    accZ = dataset.iloc[:,6].values
    magX = dataset.iloc[:,7].values
    magY = dataset.iloc[:,8].values
    magZ = dataset.iloc[:,9].values

    #Manually Frame Data
    # startTime = 0
    # stopTime = 10

    indexSel1 = time > startTime
    indexSel2 = time < stopTime
    indexSel = indexSel1 * indexSel2

    time = time[indexSel]
    gyrX = gyrX[indexSel]
    gyrY = gyrY[indexSel]
    gyrZ = gyrZ[indexSel]
    accX = accX[indexSel]
    accY = accY[indexSel]
    accZ = accZ[indexSel]
    magX = magX[indexSel]
    magY = magY[indexSel]
    magZ = magZ[indexSel]


    # -------------------------------------------------------------------------
    # Detect stationary periods

    # Compute accelerometer magnitude
    acc_mag = np.sqrt(accX**2 + accY**2 + accZ**2)

    # HP filter accelerometer data
    filtCutOff = 0.001
    [b, a] = signal.butter(1, (2*filtCutOff)/(1/samplePeriod), 'high')
    acc_magFilt = signal.filtfilt(b, a, acc_mag)

    # Compute absolute value
    acc_magFilt = abs(acc_magFilt)

    # LP filter accelerometer data
    filtCutOff = 5
    [b, a] = signal.butter(1, (2*filtCutOff)/(1/samplePeriod), 'low')
    acc_magFilt = signal.filtfilt(b, a, acc_magFilt)

    # Descomente para ver a relação de tempo de espera para calibracao
    # plt.plot(time, acc_magFilt)
    # plt.plot(time[:(tempo_parado)*Fs], acc_magFilt[:(tempo_parado)*Fs])

    # Threshold detection
    stationaty_start_time = acc_magFilt[:(tempo_parado)*Fs]
    statistical_stationary_threshold = np.mean(stationaty_start_time) + 2*np.std(stationaty_start_time)
    stationary_threshold = 0.048

    print('Limiar Calculado = %.4f + 2 * %.4f = %.4f' % (np.mean(stationaty_start_time),
                                                        np.std(stationaty_start_time),
                                                        statistical_stationary_threshold))
    print('Limiar fixo = %.4f' % (stationary_threshold*2))

    stationary = acc_magFilt < stationary_threshold
    # -------------------------------------------------------------------------
    # Plot data raw sensor data and stationary periods
    raw_sensor = graphPlots.build_rawSensor(time, gyrX, gyrY, gyrZ, accX, accY, accZ, magX, magY, magZ, acc_magFilt, stationary, mag_enabled)


    # -------------------------------------------------------------------------
    # Compute orientation

    quat = [None] * len(time)
    AHRSalgorithm = MadgwickAHRS(sampleperiod=1/Fs)

    # Initial convergence
    initPeriod = tempo_parado # usually 2 seconds
    indexSel = time < (tempo_parado+time[0])
    for i in range(2000):
        AHRSalgorithm.update_imu([0, 0, 0],
                            [accX[indexSel].mean(), accY[indexSel].mean(), accZ[indexSel].mean()])
    #                         [magX[indexSel].mean(), magY[indexSel].mean(), magZ[indexSel].mean()])

    # For all data
    for t in range(len(time)):
        if stationary[t]:
            AHRSalgorithm.beta = 0.5
        else:
            AHRSalgorithm.beta = 0
            
        AHRSalgorithm.update_imu(
                np.deg2rad([gyrX[t], gyrY[t], gyrZ[t]]),
                [accX[t], accY[t], accZ[t]])
        quat[t] = AHRSalgorithm.quaternion

    quats = []
    for quat_obj in quat:
        quats.append(quat_obj.q)
    quats =np.array(quats)
    quat = quats
    # -------------------------------------------------------------------------
    # Compute translational accelerations

    # Rotate body accelerations to Earth frame
    a = np.array([accX, accY, accZ]).T
    acc = quaternion_toolbox.rotate(a, quaternion_toolbox.conjugate(quat))

    # # Remove gravity from measurements
    # acc = acc - [zeros(length(time), 2) ones(length(time), 1)]     # unnecessary due to velocity integral drift compensation

    # Convert acceleration measurements to m/s/s
    acc = acc * 9.81

    # Plot translational accelerations
    trans_acc = graphPlots.build_translational_acc(time, acc)

    # -------------------------------------------------------------------------
    # Compute translational velocities

    acc[:,2] = acc[:,2] - 9.81

    # Integrate acceleration to yield velocity
    vel = np.zeros(np.shape(acc))
    for t in range(1,len(vel)):
        vel[t,:] = vel[t-1,:] + acc[t,:] * samplePeriod
        if stationary[t]:
            vel[t,:] = np.zeros((3))    # force zero velocity when foot stationary
    

    # Compute integral drift during non-stationary periods

    velDrift = np.zeros(np.shape(vel))

    d = np.append(arr = [0], values = np.diff(stationary.astype(np.int8)))
    stationaryStart = np.where( d == -1)
    stationaryEnd =  np.where( d == 1)
    stationaryStart = np.array(stationaryStart)[0]
    stationaryEnd = np.array(stationaryEnd)[0]

    for i in range(len(stationaryEnd)):
        driftRate = vel[stationaryEnd[i]-1, :] / (stationaryEnd[i] - stationaryStart[i])
        enum = np.arange(0, stationaryEnd[i] - stationaryStart[i])
        enum_t = enum.reshape((1,len(enum)))
        driftRate_t = driftRate.reshape((1,len(driftRate)))
        drift = enum_t.T * driftRate_t
        velDrift[stationaryStart[i]:stationaryEnd[i], :] = drift

    # Remove integral drift
    vel = vel - velDrift

    # Plot translational velocity
    trans_vel = graphPlots.build_translational_vel(time, vel)

    # -------------------------------------------------------------------------
    # Compute translational position

    # Integrate velocity to yield position
    pos = np.zeros(np.shape(vel))
    for t in range(1,len(pos)):
        pos[t,:] = pos[t-1,:] + vel[t,:] * samplePeriod    # integrate velocity to yield position


    # Plot translational position
    trans_pos = graphPlots.build_translational_pos(time, pos)

    print('Erro em Z: %.4f' % abs(pos[-1, 2]))
    # -------------------------------------------------------------------------
    #  Plot 3D foot trajectory

    # # Remove stationary periods from data to plot
    # posPlot = pos(find(~stationary), :)
    # quatPlot = quat(find(~stationary), :)
    posPlot = pos
    quatPlot = quat

    # Extend final sample to delay end of animation
    extraTime = 20
    onesVector = np.ones((extraTime*Fs, 1))
    #TODO: usar pading
    # np.pad()
    #posPlot = np.append(arr = posPlot, values = onesVector * posPlot[-1, :])
    #quatPlot = np.append(arr = quatPlot, values = onesVector * quatPlot[-1, :])

    # -------------------------------------------------------------------------
    # Create 6 DOF animation
    # TODO: improve it

    posPlot = posPlot.T

    #
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    data_x = posPlot[0,0:1500]
    data_y = posPlot[1,0:1500]
    data_z = posPlot[2,0:1500]
    # Creating fifty line objects.
    # NOTE: Can't pass empty arrays into 3d version of plot()
    line = ax.plot(data_x, data_y, data_z)
    line = line[0]

    # Setting the axes properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title('3D Animation')

    ax.set_xlim3d([-3.0, 3.0])
    ax.set_ylim3d([-3.0, 3.0])
    ax.set_zlim3d([-3.0, 3.0])

    #
    def update_lines(num):
        # NOTE: there is no .set_data() for 3 dim data...
        index = num*10
        line.set_data(posPlot[0:2, :index])
        line.set_3d_properties(posPlot[2,:index])
        return line

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig=fig, func=update_lines,
                                    frames = int(max(posPlot.shape)/10),
                                    fargs=None,
                                    interval=50, blit=False)

    plt.show()