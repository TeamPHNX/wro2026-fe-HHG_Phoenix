#include <wiringPi.h>
#include <iostream>
#include <unistd.h>

using namespace std;

const int PWM_pin = 1;

int main()
{
    cout << "Initializing WiringPi..." << endl;

    if (wiringPiSetup() == -1)
    {
        return 1;
    }

    cout << "Setting up PWM on pin " << PWM_pin << endl;

    pinMode(PWM_pin, PWM_OUTPUT);

    // Set PWM frequency to 50 Hz for servo control
    pwmSetMode(PWM_MODE_MS);
    pwmSetClock(384);  // 19.2MHz / 384 = 50kHz base
    pwmSetRange(1000); // 50kHz / 1000 = 50Hz

    cout << "Moving servo..." << endl;

    // 85 = middle, 70= left, 103 = right

    while (true)
    {
        // Smooth movement from 70 to 103
        for (int pos = 70; pos <= 103; pos++)
        {
            pwmWrite(PWM_pin, pos);
            usleep(25000); // 25ms delay for smooth movement
        }

        delay(1000); // Wait for 1 second at right position

        // Smooth movement from 103 back to 70
        for (int pos = 103; pos >= 70; pos--)
        {
            pwmWrite(PWM_pin, pos);
            usleep(25000); // 25ms delay for smooth movement
        }
    }

    return 0;
}