import time
from Arm_Lib import Arm_Device
Arm = Arm_Device()
time.sleep(.2)
q0 = [80,80,80,80,80,0]
q1 = [70,70,70,70,70,0]
q2 = [60,60,60,60,60,0]
q3 = [50,50,50,50,50,0]
q4 = [40,40,40,40,40,0]
# Servo 1 (0-180) degrees
# Servo 2 (0-180) degrees
# Servo 3 (0-180) degrees
# Servo 4 (0-180) degrees
# Servo 5 (0-270) degrees
# Servo 6 (0-180) degrees
Arm.Arm_serial_servo_write6(q0[0],q0[1],q0[2],q0[3],q0[4],q0[5], 400)
time.sleep(0.5)
s = 0
for i in range(6):
    s += 1
    var = Arm.Arm_serial_servo_read(i+1)
    if var == None:
        i -= 1
        time.sleep(.05)
        continue
        
    else:
        print(i+1, var)
    if s >= 5000:
        break
    
Rot, Pot = fk_Dofbot(q)
Rot = Rot.as_matrix()
H = np.array([
    [Rot[0,0], Rot[0,1], Rot[0,2], Pot[0]],
    [Rot[1,0], Rot[1,1], Rot[1,2], Pot[1]],
    [Rot[2,0], Rot[2,1], Rot[2,2], Pot[2]],
    [0, 0, 0, 1]
])
print(H)