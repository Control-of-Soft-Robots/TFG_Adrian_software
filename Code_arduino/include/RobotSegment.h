#ifndef ROBOTSEGMENT_H
#define ROBOTSEGMENT_H

#include <Arduino.h>

namespace Robot
{
    class stretchSensor
    {
        private:
            uint8_t pin;

            uint16_t max_value, min_value;
            int prev_time;
            uint16_t  curr_value;
            bool first_time_calibration = true;
            
            const uint32_t calibration_time = 5000;

        public:

            // Constructor
            stretchSensor(uint8_t pin);

            // Para inicializar el sensor
            void init();

            // Se devuelve el valor leido (entre 0 y 4096)
            uint16_t readRaw();

            // Se devuelve el valor leido y procesado (entre 0 y 1000)
            uint16_t readCalibratedValue();

            // Se calibra la medicion maxima y minima durante un periodo de tiempo
            void calibrate();

            // Se calibra la medicion maxima y minima durante un periodo de tiempo
            void calibrateBloqueante();

            uint16_t getMaxCalibratedValue()
            {
                return max_value;
            };

            uint16_t getMinCalibratedValue()
            {
                return min_value;
            };
    };

};





#endif ROBOTSEGMENT_H