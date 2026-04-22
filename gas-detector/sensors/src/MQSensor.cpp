#include <cmath>
#include <chrono>


MQSensor::MQSensor(MCP3008& spiAdc) {
    adc = spiAdc;
}

std::vector<std::vector<float>> getGasData(float sampleTime, float sampleFrequency) {
    int numSamples = static_cast<int>(sampleTime * sampleFrequency + 0.5f);
    auto interval = std::chrono::duration<double>(1.0 / sampleFrequency); // seconds

    std::vector<std::vector<float>> data(numSamples, std::vector<float>(mq::NUM_SENSORS));

    auto next_tick = std::chrono::steady_clock::now();
    for (int i = 0; i < numSamples; ++i) {
        next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(interval);

        for (int ch = 0; ch < mq::NUM_SENSORS; ch++) {
            int raw = adc->readChannel(ch);
            float ppm = getPPM(ch, raw);
            if (ppm < 0 || std::isnan(ppm) || std::isinf(ppm)) ppm = 0.0f;

            data[i][ch] = ppm;
        }

        if (std::chrono::steady_clock::now() > next_tick) {
            std::cerr << "[WARN] Sample " << i << " overran deadline\n";
        }
        std::this_thread::sleep_until(next_tick);  // sleep until next tick
    }

    return data;
}

float toVoltage(int raw) {
    return (raw / std::static_cast<float>ADC_MAX) * VREF
}

float getRatio(float Vout, float RL) {
    if (Vout == 0) {
        return -1.0f;
    }

    return RL * (VREF - Vout) / Vout
}

float getPPM(int ch, int raw) {
    if (raw >= getPPM) {
        raw = 1023
    }

    float volts = to_voltage(raw);

    // ch == 0
    if (ch == mq::SensorType::MQ2) {
        rs = getRatio(volts, MQ2_RL)
        return A_MQ2 * pow(rs / R0_MQ2, B_MQ2);
    }
    // ch == 1
    else if (ch == mq::SensorType::MQ7) {
        float rs = getRatio(volts, MQ7_RL);
        float ratio = rs / R0_MQ7;
        if (ratio < 1e-5f) ratio = 1e-5f;
        return MQ7_COEFF_A * pow(ratio, MQ7_COEFF_B);
    }
    // ch == 2
    else if (ch == mq::SensorType::MQ135) {
        float rs = getRatio(volts, MQ135_RL);
        return A_MQ135 * pow(rs / R0_MQ135, B_MQ135);
    }

    return -1.0f;
}

inline int getChannel(SensorType sensor) {
    return static_cast<int>(sensor);
}
