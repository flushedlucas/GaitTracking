import io
import base64
import numpy as np
import matplotlib.pyplot as plt

# Plot data raw sensor data and stationary periods
def build_rawSensor(time, gyrX, gyrY, gyrZ, accX, accY, accZ, magX, magY, magZ, acc_magFilt, stationary, mag_enabled:bool):
    img = io.BytesIO()
    plt.figure(figsize=(20,10))
    plt.suptitle('Sensor Data', fontsize=14)
    ax1 = plt.subplot(2+mag_enabled,1,1)
    plt.grid()
    plt.plot(time, gyrX, 'r')
    plt.plot(time, gyrY, 'g')
    plt.plot(time, gyrZ, 'b')
    plt.title('Gyroscope')
    plt.ylabel('Angular velocity (º/s)')
    plt.legend(labels=['X', 'Y', 'Z'])


    plt.subplot(2+mag_enabled,1,2,sharex=ax1)
    plt.grid()
    plt.plot(time, accX, 'r')
    plt.plot(time, accY, 'g')
    plt.plot(time, accZ, 'b')
    plt.plot(time, acc_magFilt, ':k')
    plt.plot(time, stationary.astype(np.uint8)*acc_magFilt.max(), 'k', linewidth= 2)
    plt.title('Accelerometer')
    plt.ylabel('Acceleration (g)')
    plt.legend(['X', 'Y', 'Z', 'Filtered', 'Stationary'])

    if mag_enabled:
        plt.subplot(3,1,3,sharex=ax1)
        plt.grid()
        plt.plot(time, magX, 'r')
        plt.plot(time, magY, 'g')
        plt.plot(time, magZ, 'b')
        plt.title('Magnetrometer')
        plt.ylabel('Magnetic Flux Density  (G)')
        plt.legend(['X', 'Y', 'Z'])

    plt.xlabel('Time (s)')

    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)

# Plot translational Acceleration
def build_translational_acc(time, acc):
    img = io.BytesIO()  

    plt.figure(figsize=(20,10))
    plt.suptitle('Accelerations', fontsize=14)
    plt.grid()

    plt.plot(time, acc[:,0], 'r')
    plt.plot(time, acc[:,1], 'g')
    plt.plot(time, acc[:,2], 'b')
    plt.title('Acceleration')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s/s)')
    plt.legend(('X', 'Y', 'Z'))

    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)

# Plot translational Velocity
def build_translational_vel (time, vel):
    img = io.BytesIO() 

    plt.figure(figsize=(20,10))
    plt.suptitle('Velocity', fontsize=14)
    plt.grid()
    plt.plot(time, vel[:,0], 'r')
    plt.plot(time, vel[:,1], 'g')
    plt.plot(time, vel[:,2], 'b')
    plt.title('Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend(('X', 'Y', 'Z'))

    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)

# Plot translational Position
def build_translational_pos (time, pos):
    img = io.BytesIO() 

    plt.figure(figsize=(20,10))
    plt.suptitle('Position', fontsize=14)
    plt.grid()
    plt.plot(time, pos[:,0], 'r')
    plt.plot(time, pos[:,1], 'g')
    plt.plot(time, pos[:,2], 'b')
    plt.title('Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend(('X', 'Y', 'Z'))

    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)



