#include <wiringPi.h>
#include <iostream>
#include <cstdlib>

// GPIO pin definitions
#define ENA_PIN 19 // PWM1 pin for speed control
#define IN1_PIN 16 // Direction control pin 1
#define IN2_PIN 13 // Direction control pin 2

class MotorController
{
private:
    bool initialized;

public:
    MotorController() : initialized(false) {}

    bool init()
    {
        if (wiringPiSetupGpio() == -1)
        {
            std::cerr << "Failed to initialize WiringPi" << std::endl;
            return false;
        }

        // Setup GPIO pins
        pinMode(IN1_PIN, OUTPUT);
        pinMode(IN2_PIN, OUTPUT);
        pinMode(ENA_PIN, PWM_OUTPUT);

        // Initialize pins to safe state
        digitalWrite(IN1_PIN, LOW);
        digitalWrite(IN2_PIN, LOW);

        // Setup hardware PWM (PWM1)
        pwmSetMode(PWM_MODE_MS);
        pwmSetRange(1024); // 0-1024 range
        pwmSetClock(375);  // ~1kHz frequency
        pwmWrite(ENA_PIN, 0);

        initialized = true;
        std::cout << "Motor controller initialized" << std::endl;
        return true;
    }

    void setSpeed(int speed)
    {
        if (!initialized)
            return;

        // Clamp speed to valid range
        if (speed < 0)
            speed = 0;
        if (speed > 1024)
            speed = 1024;

        pwmWrite(ENA_PIN, speed);
    }

    void forward(int speed = 512)
    {
        if (!initialized)
            return;

        digitalWrite(IN1_PIN, HIGH);
        digitalWrite(IN2_PIN, LOW);
        setSpeed(speed);
    }

    void backward(int speed = 512)
    {
        if (!initialized)
            return;

        digitalWrite(IN1_PIN, LOW);
        digitalWrite(IN2_PIN, HIGH);
        setSpeed(speed);
    }

    void stop()
    {
        if (!initialized)
            return;

        digitalWrite(IN1_PIN, LOW);
        digitalWrite(IN2_PIN, LOW);
        setSpeed(0);
    }

    void brake()
    {
        if (!initialized)
            return;

        digitalWrite(IN1_PIN, HIGH);
        digitalWrite(IN2_PIN, HIGH);
        setSpeed(1024);
    }

    void cleanup()
    {
        if (initialized)
        {
            stop();
            std::cout << "Motor controller cleanup complete" << std::endl;
            initialized = false;
        }
    }

    ~MotorController()
    {
        cleanup();
    }
};

// Global motor controller instance
MotorController motorController;

// C-style interface functions
extern "C"
{
    int motor_init()
    {
        return motorController.init() ? 0 : -1;
    }

    void motor_forward(int speed)
    {
        motorController.forward(speed);
    }

    void motor_backward(int speed)
    {
        motorController.backward(speed);
    }

    void motor_stop()
    {
        motorController.stop();
    }

    void motor_brake()
    {
        motorController.brake();
    }

    void motor_set_speed(int speed)
    {
        motorController.setSpeed(speed);
    }

    void motor_cleanup()
    {
        motorController.cleanup();
    }
}

// Example main function for testing
int main()
{

    if (motor_init() != 0)
    {
        std::cerr << "Failed to initialize motor controller" << std::endl;
        return -1;
    }
    try
    {

        std ::cout << "Testing motor controller..." << std::endl;

        // Test forward motion
        std::cout << "Moving forward..." << std::endl;
        motor_forward(50);
        delay(100000);

        // motor_stop();
        // delay(1000);

        // // Test backward motion
        // std::cout << "Moving backward..." << std::endl;
        // motor_backward(100);
        // delay(2000);

        // Test stop
        std::cout << "Stopping..." << std::endl;
        motor_stop();
        delay(1000);

        motor_cleanup();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        motor_stop();
        motor_cleanup();
        return -1;
    }
}
