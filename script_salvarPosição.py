import sys
import csv
import quaternion_toolbox
from madgwickahrs import MadgwickAHRS
import numpy as np
from scipy import signal
from scipy import stats
import pandas as pd

filenames = [
    

'ensaio_01', 'ensaio_02', 'ensaio_03', 'ensaio_04', 'ensaio_05', 'ensaio_06', 'ensaio_07', 
'ensaio_08', 'ensaio_09', 'ensaio_10', 'ensaio_11', 'ensaio_12', 'ensaio_13', 'ensaio_14', 
'ensaio_15', 'ensaio_16', 'ensaio_17', 'ensaio_18', 'ensaio_19', 'ensaio_20', 'ensaio_21', 
'ensaio_22', 'ensaio_23', 'ensaio_24', 'ensaio_25', 'ensaio_26', 'ensaio_27', 'ensaio_28', 
'ensaio_29', 'ensaio_30',
 

'ensaio_01_50hz', 'ensaio_02_50hz', 'ensaio_03_50hz', 'ensaio_04_50hz', 'ensaio_05_50hz', 
'ensaio_06_50hz', 'ensaio_07_50hz', 'ensaio_08_50hz', 'ensaio_09_50hz', 'ensaio_10_50hz', 
'ensaio_11_50hz', 'ensaio_12_50hz', 'ensaio_13_50hz', 'ensaio_14_50hz', 'ensaio_15_50hz', 
'ensaio_16_50hz', 'ensaio_17_50hz', 'ensaio_18_50hz', 'ensaio_19_50hz', 'ensaio_20_50hz', 
'ensaio_21_50hz', 'ensaio_22_50hz', 'ensaio_23_50hz', 'ensaio_24_50hz', 'ensaio_25_50hz', 
'ensaio_26_50hz', 'ensaio_27_50hz', 'ensaio_28_50hz', 'ensaio_29_50hz', 'ensaio_30_50hz', 

'ensaio_01_25hz', 'ensaio_02_25hz', 'ensaio_03_25hz', 'ensaio_04_25hz', 'ensaio_05_25hz', 
'ensaio_06_25hz', 'ensaio_07_25hz', 'ensaio_08_25hz', 'ensaio_09_25hz', 'ensaio_10_25hz', 
'ensaio_11_25hz', 'ensaio_12_25hz', 'ensaio_13_25hz', 'ensaio_14_25hz', 'ensaio_15_25hz', 
'ensaio_16_25hz', 'ensaio_17_25hz', 'ensaio_18_25hz', 'ensaio_19_25hz', 'ensaio_20_25hz', 
'ensaio_21_25hz', 'ensaio_22_25hz', 'ensaio_23_25hz', 'ensaio_24_25hz', 'ensaio_25_25hz', 
'ensaio_26_25hz', 'ensaio_27_25hz', 'ensaio_28_25hz', 'ensaio_29_25hz', 'ensaio_30_25hz']

data100 = []
data50 = []
data25 = []

with open('Datasets/posicao_erro.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    for name in filenames:        
        try:
            filePath = 'Datasets/'+name+'.csv'
            if ('50hz' in name):
                Fs = 50
            elif ('25hz' in name):
                Fs = 25
            else:
                Fs = 100
            if(name == 'ensaio_10' or name == 'ensaio_10_25hz' or name == 'ensaio_10_50hz'):
                startTime = 5
            else:
                startTime = 1
            
            stopTime = 55

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
            if('ensaio_25' in name):
                stationary_threshold = 0.1
            else:    
                stationary_threshold = 0.1

            stationary = acc_magFilt < stationary_threshold
            
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
            acc = acc * 9.81 # Conversão de G para m/s² - Verificar qual a unidade de medida de retorno do Acelerômetro usado.

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

            # -------------------------------------------------------------------------
            # Compute translational position

            # Integrate velocity to yield position
            pos = np.zeros(np.shape(vel))
            for t in range(1, len(pos)):
                # integrate velocity to yield position
                pos[t, :] = pos[t-1, :] + vel[t, :] * samplePeriod

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

            print(name)
            spamwriter.writerow([name, ' - ',pos[len(pos) - 1][1]])
            if('50hz' in name):
                data50.append(pos[len(pos) - 1][1])
            elif('25hz' in name):
                data25.append(pos[len(pos) - 1][1])
            else:
                data100.append(pos[len(pos) - 1][1])
        except:
            spamwriter.writerow([name, ' - ','Erro ao calcular'])

T1,p1 = stats.wilcoxon(data100, data50, zero_method='wilcox', correction=False, alternative='lesser')
#T2,p2 = stats.wilcoxon(data100, data25, zero_method='wilcox', correction=False)
#T3,p3 = stats.wilcoxon(data50, data25, zero_method='wilcox', correction=False)

print('------------------------')
print(T1,p1)
#print(T2,p2)
#print(T3,p3)
