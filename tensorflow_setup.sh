cd ~/dev/MLX90640

# Install build dependencies
sudo apt install -y cmake git build-essential

# Download the exact commit of kleidiAI TFLite 2.19 expects
git clone https://gitlab.arm.com/kleidi/kleidiai.git kleidiai-source
cd kleidiai-source
git checkout 847ebd19d0192528659b0a0fa2c6057eed674c6a
cd ..

# Clone TFLite 
git clone --depth 1 --branch v2.19.0 \
    https://github.com/tensorflow/tensorflow.git

mkdir tflite_build && cd tflite_build

# Configure — builds only TFLite, not all of TensorFlow
cmake ../tensorflow/tensorflow/lite \
    -DCMAKE_BUILD_TYPE=Release \
    -DTFLITE_ENABLE_XNNPACK=ON \
    -DKLEIDIAI_SOURCE_DIR=/home/isam/dev/MLX90640/kleidiai-source \
    -DCMAKE_C_FLAGS="-Wno-incompatible-pointer-types" \
    -DCMAKE_CXX_FLAGS="-Wno-incompatible-pointer-types"

# Build using all 4 Pi 5 cores
cmake --build . -j4

# Install headers and library to /usr/local
sudo cmake --install .
