# Planned robot parts

## Drive

-   DC Motor & Driver

-   Differential

-   Servo + Steering mechanics

## Sensors

-   RPLidar C1

-   2x PiCam3 wide

-   Button

-   Gyro (MPU6050 or BNO055)

-   ADC for Voltage display

-   2x OLED Display

## Computation

-   RPi 5 8GB & Active Cooler

-   Pi AI Hat+ (26 TOPS) for AI block detection

## ROS nodes

### prebuilt:

-   rplidar_ros --> RPLidar C1

-   SLAM toolbox (Idk if it fully supports what we need)

-   Gyro sensor (testing needed if it needs to have an ESP32)

-   maybe for DS4 controller

### need own coding:

-   interface both PiCams and instantly merge their images

-   talk to the motor controller (testing needed if it needs to have an ESP32)

-   run AI model for block detection

-   main command node, that waits for the button press and starts / stops the robot

-   maybe for button
