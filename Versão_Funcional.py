import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import quaternion_toolbox
from madgwickahrs import MadgwickAHRS
import numpy as np
from scipy import signal
import pandas as pd

# Select dataset

# Fs = 100
# filePath = 'Datasets/coleta1_CalInertialAndMag.csv'
# startTime = 8
# stopTime = 37.5

# Fs = 100
#filePath = 'Datasets/coleta3_CalInertialAndMag.csv'
#startTime = 8
#stopTime = 55

# Fs = 256
# filePath = 'Datasets/straightLine_CalInertialAndMag.csv'
# startTime = 6
# stopTime = 26

# Fs = 256
# filePath = 'Datasets/stairsAndCorridor_CalInertialAndMag.csv'
# startTime = 5
# stopTime = 53

Fs = 256
filePath = 'Datasets/spiralStairs_CalInertialAndMag.csv'
startTime = 4
stopTime = 47

tempo_parado = 2  # segundos parado
mag_enabled = False

#import Data

samplePeriod = np.around(1/Fs, decimals=4)

dataset = pd.read_csv(filePath)
time = np.array(np.arange(0, len(dataset.iloc[:,0].values), samplePeriod))
gyrX = dataset.iloc[:, 1].values
gyrY = dataset.iloc[:, 2].values
gyrZ = dataset.iloc[:, 3].values
accX = dataset.iloc[:, 4].values
accY = dataset.iloc[:, 5].values
accZ = dataset.iloc[:, 6].values

# Manually Frame Data
# startTime = 0
# stopTime = 10

# indexSel = find(sign(time-startTime)+1, 1) : find(sign(time-stopTime)+1, 1);
# np.sign(time-startTime)+1


indexSel1 = np.nonzero((np.sign(time-startTime)+1) > 0)[0][0]
indexSel2 = np.nonzero((np.sign(time-stopTime)+1) > 0)
if (len(indexSel2) > 1):
        indexSel2 = indexSel2[0][len(indexSel2)-1]
else:
        indexSel2 = len(gyrX) - 1


time = time[indexSel1:indexSel2]
gyrX = gyrX[indexSel1:indexSel2]
gyrY = gyrY[indexSel1:indexSel2]
gyrZ = gyrZ[indexSel1:indexSel2]
accX = accX[indexSel1:indexSel2]
accY = accY[indexSel1:indexSel2]
accZ = accZ[indexSel1:indexSel2]

# -------------------------------------------------------------------------
# Detect stationary periods

# Compute accelerometer magnitude
acc_mag = np.around(np.sqrt(accX**2 + accY**2 + accZ**2), decimals=4)

# HP filter accelerometer data
filtCutOff = 0.001

# [b, a] = np.around(signal.butter(1, (2*filtCutOff)/(1/samplePeriod), 'high'), decimals=4) #Erro de Matriz singular
freq = np.double((filtCutOff)/((1/samplePeriod)/2))
[b, a] = signal.butter(1, (2*filtCutOff)/(1/samplePeriod), 'high', output='ba')

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
stationary_threshold = 0.05

stationary = acc_magFilt < stationary_threshold
# -------------------------------------------------------------------------
# Plot data raw sensor data and stationary periods
plt.figure(figsize=(20, 10))
plt.suptitle('Sensor Data', fontsize=14)
ax1 = plt.subplot(2, 1, 1)
plt.grid()
plt.plot(time, gyrX, 'r')
plt.plot(time, gyrY, 'g')
plt.plot(time, gyrZ, 'b')
plt.title('Gyroscope')
plt.ylabel('Angular velocity (º/s)')
plt.legend(labels=['X', 'Y', 'Z'])


plt.subplot(2, 1, 2)
plt.grid()
plt.plot(time, accX, 'r')
plt.plot(time, accY, 'g')
plt.plot(time, accZ, 'b')
plt.plot(time, acc_magFilt, ':k')
plt.plot(time, stationary.astype(np.uint8)*acc_magFilt.max(), 'k', linewidth=2)
plt.title('Accelerometer')
plt.ylabel('Acceleration (g)')
plt.legend(['X', 'Y', 'Z', 'Filtered', 'Stationary'])

plt.xlabel('Time (s)')


# -------------------------------------------------------------------------
# Compute orientation

quat = [[0]*4]*len(time)
AHRSalgorithm = MadgwickAHRS(sampleperiod=np.round(1/Fs, decimals=4))

# Initial convergence
initPeriod = tempo_parado  # usually 2 seconds

# indexSel = 1 : find(sign(time-(time(1)+initPeriod))+1, 1);
np.nonzero((np.sign(time-startTime)+1) > 0)[0][0]
indexSel = np.arange(0, np.nonzero(
    np.sign(time-(time[0]+initPeriod))+1)[0][0], 1)

for i in range(1, 2000):
    AHRSalgorithm.update_imu_new([0, 0, 0],
                                 [accX[indexSel].mean(), accY[indexSel].mean(), accZ[indexSel].mean()])

# For all data
for t in range(len(time)):
    if stationary[t]:
        AHRSalgorithm.beta = 0.5
    else:
        AHRSalgorithm.beta = 0

    AHRSalgorithm.update_imu_new(
        np.deg2rad([gyrX[t], gyrY[t], gyrZ[t]]),
        [accX[t], accY[t], accZ[t]])
    quat[t] = AHRSalgorithm.quaternion

quats = []
for quat_obj in quat:
    quats.append(quat_obj.q)
quats = np.array(quats)
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
plt.figure(figsize=(20, 10))
plt.suptitle('Accelerations', fontsize=14)
plt.grid()

plt.plot(time, acc[:, 0], 'r')
plt.plot(time, acc[:, 1], 'g')
plt.plot(time, acc[:, 2], 'b')
plt.title('Acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s/s)')
plt.legend(('X', 'Y', 'Z'))


# -------------------------------------------------------------------------
# Compute translational velocities

acc[:, 2] = acc[:, 2] - 9.81

# Integrate acceleration to yield velocity
vel = np.zeros(np.shape(acc))
for t in range(1, len(vel)):
    vel[t, :] = vel[t-1, :] + acc[t, :] * samplePeriod
    if stationary[t]:
        vel[t, :] = np.zeros((3))    # force zero velocity when foot stationary


# Compute integral drift during non-stationary periods

velDrift = np.zeros(np.shape(vel))

d = np.append(arr=[0], values=np.diff(stationary.astype(np.int8)))
stationaryStart = np.where(d == -1)
stationaryEnd = np.where(d == 1)
stationaryStart = np.array(stationaryStart)[0]
stationaryEnd = np.array(stationaryEnd)[0]

for i in range(len(stationaryEnd)):
    driftRate = vel[stationaryEnd[i]-1, :] / (stationaryEnd[i] - stationaryStart[i])
    enum = np.arange(0, stationaryEnd[i] - stationaryStart[i])
    enum_t = enum.reshape((1, len(enum)))
    driftRate_t = driftRate.reshape((1, len(driftRate)))
    drift = enum_t.T * driftRate_t
    velDrift[stationaryStart[i]:stationaryEnd[i], :] = drift

# Remove integral drift
vel = vel - velDrift

# Plot translational velocity
plt.figure(figsize=(20, 10))
plt.suptitle('Velocity', fontsize=14)
plt.grid()
plt.plot(time, vel[:, 0], 'r')
plt.plot(time, vel[:, 1], 'g')
plt.plot(time, vel[:, 2], 'b')
plt.title('Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend(('X', 'Y', 'Z'))


# -------------------------------------------------------------------------
# Compute translational position

# Integrate velocity to yield position
pos = np.zeros(np.shape(vel))
for t in range(1, len(pos)):
    # integrate velocity to yield position
    pos[t, :] = pos[t-1, :] + vel[t, :] * samplePeriod


# Plot translational position
plt.figure(figsize=(20, 10))
plt.suptitle('Position', fontsize=14)
plt.grid()
plt.plot(time, pos[:, 0], 'r')
plt.plot(time, pos[:, 1], 'g')
plt.plot(time, pos[:, 2], 'b')
plt.title('Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend(('X', 'Y', 'Z'))

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
# TODO: usar pading
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

data_x = posPlot[0, 0:]
data_y = posPlot[1, 0:]
data_z = posPlot[2, 0:]
# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
line = ax.plot(data_x, data_y, data_z)
line = line[0]

# Setting the axes properties
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_title('3D Animation')

ax.set_xlim3d([-5.0, 5.0])
ax.set_ylim3d([-5.0, 5.0])
ax.set_zlim3d([-5.0, 5.0])


def update_lines(num):
    # NOTE: there is no .set_data() for 3 dim data...
    index = num*10
    line.set_data(posPlot[0:2, :index])
    line.set_3d_properties(posPlot[2, :index])
    return line


# Creating the Animation object
line_ani = animation.FuncAnimation(fig=fig, func=update_lines,
                                   frames=int(max(posPlot.shape)/10),
                                   fargs=None,
                                   interval=50, blit=False)

plt.show()
