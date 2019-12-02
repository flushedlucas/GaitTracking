import sys
import pandas as pd
import csv

# filepath = sys.argv[0]
# Fs = sys.argv[1]
filenames = ['ensaio_01', 'ensaio_02', 'ensaio_03', 'ensaio_04', 'ensaio_05', 'ensaio_06', 
'ensaio_07', 'ensaio_08', 'ensaio_09', 'ensaio_10', 'ensaio_11', 'ensaio_12', 'ensaio_13', 
'ensaio_14', 'ensaio_15', 'ensaio_16', 'ensaio_17', 'ensaio_18', 'ensaio_19', 'ensaio_20', 
'ensaio_21', 'ensaio_22', 'ensaio_23', 'ensaio_24', 'ensaio_25', 'ensaio_26', 'ensaio_27', 
'ensaio_28', 'ensaio_29', 'ensaio_30']

for name in filenames:
    filepath = 'Datasets/'+name+'.csv'
    # Fs = sys.argv[1]

    dataset = pd.read_csv(filepath)
    packet = dataset.iloc[:, 0].values
    gyrX = dataset.iloc[:, 1].values
    gyrY = dataset.iloc[:, 2].values
    gyrZ = dataset.iloc[:, 3].values
    accX = dataset.iloc[:, 4].values
    accY = dataset.iloc[:, 5].values
    accZ = dataset.iloc[:, 6].values

    packet_copy = []
    gyrX_copy = []
    gyrY_copy = []
    gyrZ_copy = []
    accX_copy = []
    accY_copy = []
    accZ_copy = []

    for i, j in enumerate(packet):
        if i % 2 == 0:
            packet_copy.append(packet[j])
            gyrX_copy.append(gyrX[j])
            gyrY_copy.append(gyrY[j])
            gyrZ_copy.append(gyrZ[j])
            accX_copy.append(accX[j])
            accY_copy.append(accY[j])
            accZ_copy.append(accZ[j]) 

    with open('Datasets/'+name+'_50hz.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['Packet number','Gyroscope X (deg/s)','Gyroscope Y (deg/s)','Gyroscope Z (deg/s)','Accelerometer X (g)','Accelerometer Y (g)','Accelerometer Z (g)'])
        for j, i in enumerate(packet_copy):
            spamwriter.writerow([packet_copy[j], gyrX_copy[j], gyrY_copy[j], gyrZ_copy[j], accX_copy[j], accY_copy[j], accZ_copy[j]])