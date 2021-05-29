import time    # import the time library for the sleep function
import gopigo3 # import the GoPiGo3 drivers
import easygopigo3 as easy
#from di_sensors.light_color_sensor import LightColorSensor
import gopigo3 # import the GoPiGo3 drivers

gpg = easy.EasyGoPiGo3()
GPG2 = gopigo3.GoPiGo3()
#cs = LightColorSensor(led_state = True)



def TurnDegrees(degrees, speed):
    # get the starting position of each motor
    StartPositionLeft      = GPG2.get_motor_encoder(GPG2.MOTOR_LEFT)
    StartPositionRight     = GPG2.get_motor_encoder(GPG2.MOTOR_RIGHT)
    
    # the distance in mm that each wheel needs to travel
    WheelTravelDistance    = ((GPG2.WHEEL_BASE_CIRCUMFERENCE * degrees) / 360)
    
    # the number of degrees each wheel needs to turn
    WheelTurnDegrees       = ((WheelTravelDistance / GPG2.WHEEL_CIRCUMFERENCE) * 360)
    
    # Limit the speed
    GPG2.set_motor_limits(GPG2.MOTOR_LEFT + GPG2.MOTOR_RIGHT, dps = speed)
    
    # Set each motor target
    GPG2.set_motor_position(GPG2.MOTOR_LEFT, (StartPositionLeft + WheelTurnDegrees))
    GPG2.set_motor_position(GPG2.MOTOR_RIGHT, (StartPositionRight - WheelTurnDegrees))
    

WHEEL_SPEED_CONSTANT=20
while True:
    gpg.set_motor_power(gpg.MOTOR_LEFT, WHEEL_SPEED_CONSTANT)
    gpg.set_motor_power(gpg.MOTOR_RIGHT, WHEEL_SPEED_CONSTANT)
    TurnDegrees(20, 20)
    
    