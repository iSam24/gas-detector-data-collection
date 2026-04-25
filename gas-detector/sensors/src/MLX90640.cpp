#include <stdint.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

#include "MLX90640.h"
#include "MLX90640/MLX90640_I2C_Driver.h"

#define DEBUG 1

// Static member definitions
uint16_t MLX90640::eeMLX90640[832];
float MLX90640::mlx90640To[768];

// Valid frame rates are 1, 2, 4, 8, 16, 32 and 64
// The i2c baudrate is set to 400mhz to support these
#define FPS_DEFAULT 16  // dont change (breaks driver if changed)
#define FRAME_TIME_MICROS (1000000/FPS_DEFAULT)

MLX90640::MLX90640(int fps) : fps(fps), baudrate(400000), eTa(0.0f)
{
    if(setup() != 0) {
        throw std::runtime_error("MLX90640 setup failed");
    }
}

// Returns frames shaped (numSamples-1, 32, 24)
// First frame is dropped as it contains stale/zero data
std::vector<std::vector<std::vector<float>>> MLX90640::getIrData(int numSamples)
{
    if (numSamples < 2) {
        throw std::invalid_argument("numSamples must be >= 2 (first frame is dropped)");
    }

    // Capture flat buffer: numSamples * 768 floats
    std::vector<float> buffer;
    captureIntoBuffer(numSamples, buffer);

    // drop first bad frame
    int validFrames = numSamples - 1;
    std::vector<std::vector<std::vector<float>>> result(
        validFrames,
        std::vector<std::vector<float>>(32, std::vector<float>(24))
    );

    for (int f = 0; f < validFrames; ++f) {
        // skip first frame (first 768 samples)
        const float* frameStart = buffer.data() + (f + 1) * 768;

        for (int row = 0; row < 32; ++row) {
            for (int col = 0; col < 24; ++col) {
                result[f][row][col] = frameStart[row * 24 + col];
            }
        }
    }

    return result;
}


// taken from library: mlx90640-library/python/library/mlx90640-python.cpp
int MLX90640::setup()
{
	MLX90640_SetDeviceMode(MLX_I2C_ADDR, 0);
	MLX90640_SetSubPageRepeat(MLX_I2C_ADDR, 0);

	switch(fps){
		case 1:
			MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b001);
			baudrate = 400000;
			break;
		case 2:
			MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b010);
			baudrate = 400000;
			break;
		case 4:
			MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b011);
			baudrate = 400000;
			break;
		case 8:
			MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b100);
			baudrate = 400000;
			break;
		case 16:
			MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b101);
			baudrate = 1000000;
			break;
		case 32:
			MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b110);
			baudrate = 1000000;
			break;
		case 64:
			MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b111);
			baudrate = 1000000;
			break;
		default:
            std::cerr << "[MLX90640] Unsupported framerate: " << fps << "\n";
			return 1;
	}

	MLX90640_SetChessMode(MLX_I2C_ADDR);
	MLX90640_DumpEE(MLX_I2C_ADDR, eeMLX90640);
	MLX90640_ExtractParameters(eeMLX90640, &mlx90640);

	return 0;
}

void MLX90640::captureIntoBuffer(int numFrames, std::vector<float>& buffer)
{
    buffer.clear();
    buffer.reserve(numFrames * 768);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < numFrames; ++i) {
        // GetFrameData blocks until the sensor delivers the next subpage
        // At 4Hz chess mode this naturally takes ~250ms per call
        // No sleep needed, the sensor is the timer
        MLX90640_GetFrameData(MLX_I2C_ADDR, frame);
        MLX90640_InterpolateOutliers(frame, eeMLX90640);
        eTa = MLX90640_GetTa(frame, &mlx90640);
        MLX90640_CalculateTo(frame, &mlx90640, emissivity, eTa, mlx90640To);

        buffer.insert(buffer.end(), mlx90640To, mlx90640To + 768);
    }
#ifdef DEBUG
    auto total = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "[MLX90640] Captured " << numFrames << " frames in "
        << total << "s  (" << numFrames / total << " fps actual)\n";
#endif

}
