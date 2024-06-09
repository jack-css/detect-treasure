# import machine
from machine import Pin, PWM
import time
# import utime
servo = PWM(Pin(7))
servo.freq(50)

# 数值重映射
def my_map(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

# 控制舵机，value=[0, 180]
def servo_control(value):
    if value > 180 or value < 0:
        print('Please enter a limited speed value of 0-180 ')
        return
    duty = my_map(value, 0, 180, 500000, 2500000)
    servo.duty_ns(duty)

# PWM舵机在0度和180度之前来回摆动。
while True:
    servo_control(0)
    time.sleep(1)
    servo_control(180)
    time.sleep(1)
