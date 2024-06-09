import RPi.GPIO as GPIO
import time

def main():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    # pysical mode
    num_gpio = 12
    GPIO.setup(num_gpio,GPIO.OUT)
    frequency=5000
    pwm=GPIO.PWM(num_gpio,frequency)
    pwm.start(0)
    pwm.ChangeDutyCycle(0)
    GPIO.cleaup()
    # time.sleep(1)
    # for i in range(500):
    #     time.sleep(0.0082)
    #     pwm.ChangeDutyCycle(0)
    #     time.sleep(.0082)
    #     time.sleep(.5)
    #     # pwm.start(0)
    #     pwm.ChangeDutyCycle(i//5)
    # pwm.stop()
    # pwm.ChangeDutyCycle(0)
    # GPIO.cleanup()

if __name__ == "_main__":
    main()