#ifndef MCP3008_H
#define MCP3008_H

class MCP3008 {
public:
    int readChannel(int ch);

private:
    static constexpr int maxSpiSpeedHz = 135000;
};

#endif // MCP3008_H
