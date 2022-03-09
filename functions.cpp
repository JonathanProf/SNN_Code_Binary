#include "functions.h"

void getWeights(float *weights, const std::string fileName, const uint32_t pixel, const uint32_t neuron )
{
    std::fstream raw( fileName, std::ios::in | std::ios::binary );

    assert( raw.is_open() == true );
    float n;
    for (uint32_t pos = 0; ; ++pos) {
        raw.read(reinterpret_cast<char*>(&n), sizeof(n));
        if( raw.eof() ){
            break;
        }
        assert( pos < pixel * neuron );
        weights[pos] = n;
    }

    raw.close();
}

void getTheta( float *theta, const std::string fileName, const uint32_t numNeurons )
{
    std::fstream raw( fileName, std::ios::in | std::ios::binary );

    assert( raw.is_open() == true );
    float n;
    for (uint32_t pos = 0; ; ++pos) {
        raw.read(reinterpret_cast<char*>(&n), sizeof(n));
        if( raw.eof() ){
            break;
        }
        assert( pos < numNeurons );
        theta[pos] = n;
    }

    raw.close();
}

void getInputSample( uint32_t *input, const std::string fileName, const int row, const int col )
{
    std::fstream raw( fileName, std::ios::in | std::ios::binary );

    assert( raw.is_open() == true );
    uint32_t n;
    for (uint16_t pos = 0; ; ++pos) {
        raw.read(reinterpret_cast<char*>(&n), sizeof(n));
        if( raw.eof() ){
            break;
        }
        assert( pos < row * col );
        input[pos] = n;

#if DEBUG_INPUT == 1
        std::cout << std::dec << pos;
        std::cout << " -> " << std::hex << input[pos] << std::endl;
#endif
    }

    raw.close();

}

void getAssignments(uint8_t *assignments , const std::string fileName, const uint32_t numNeurons)
{
    std::fstream raw( fileName, std::ios::in | std::ios::binary );

    assert( raw.is_open() == true );
    uint8_t n;
    for (uint32_t pos = 0; ; ++pos) {
        raw.read(reinterpret_cast<char*>(&n), sizeof(n));
        if( raw.eof() ){
            break;
        }
        assert( pos < numNeurons );
        assignments[pos] = n;
    }

    raw.close();
}
