#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include "functions.h"

/*
 *  TODO: Comprobar que los voltajes para t = 1 en la capa excitación son iguales luego de haber cargado la información de forma binaria
 */

using namespace std;

#define PRINT_ENABLE false

#define NUM_NEURONS 400

#define NUM_PIXELS 784

#define SINGLE_SAMPLE_TIME 64

#define PATH_SAMPLES_POISSON "../DATABASE_SNN/inputSamples_64ms/%05d_inputSpikesPoisson_64ms.dat"

#define PATH_PARAMETERS_NET "../DATABASE_SNN/window64ms/BD400_64ms/"

#define PATH_RESULTS_NET "../DATABASE_SNN/classification/"

#define TOTAL_SAMPLES static_cast<int>(10)

int main()
{
    /*
    float v_rest_e = -65.0;     // [mV]
    float v_reset_e = -60.0;    // [mV]
    float v_thresh_e = -52.0;   // [mV]
    int refrac_e = 5;           // [ms]

    float v_rest_i = -60.0;     // [mV]
    float v_reset_i = -45.0;    // [mV]
    float v_thresh_i = -40.0;   // [mV]
    int refrac_i = 2;           // [ms]
    int dt = 1;
    */
    assert( sizeof (uint32_t) == 4 );
    assert( sizeof (uint16_t) == 2 );
    assert( sizeof (float) == 4 );

    uint16_t tamVector = NUM_NEURONS / 16; //(sizeof (uint16_t)*8)
    uint16_t tamVectorPixels = NUM_PIXELS / 16; //(sizeof (uint16_t)*8)

    //! =====     =====     =====
    //! Variables Inicialization
    //! =====     =====     =====
    float vSyn = 0.0;

    float weights_Ae_Ai_constant = 22.5;
    float weights_Ai_Ae_constant = -120.0;

    float *theta = nullptr;
    float *weightsXeAe = nullptr; // connections Xe -> Ae
    float *weightsAeAi = nullptr; // connections Ae -> Ai
    float *weightsAiAe = nullptr; // connections Ae <- Ai
    uint32_t  *input_sample = nullptr;

    // Check for spiking neurons
    uint16_t *spikesXePre = nullptr; // Spike occurrences Input
    uint16_t *spikesXePos = nullptr; // Spike occurrences Input
    uint16_t *spikes_Ae_Ai_pre = nullptr;
    uint16_t *spikes_Ae_Ai_pos = nullptr;
    uint16_t *spikes_Ai_Ae_pre = nullptr;
    uint16_t *spikes_Ai_Ae_pos = nullptr;

    uint32_t *spike_count = nullptr;
    uint8_t *assignments = nullptr;

    unsigned short int digits[10] = {0};

    float *vE = nullptr;
    float *vI = nullptr;

    int *refrac_countE = nullptr;       // Refractory period counters
    int *refrac_countI = nullptr;       // Refractory period counters

    //! [Step 1] Data structure initialization

    spikesXePre = new(std::nothrow) uint16_t[tamVectorPixels]{0};
    assert(spikesXePre != nullptr);

    spikesXePos = new(std::nothrow) uint16_t[tamVectorPixels]{0};
    assert(spikesXePos != nullptr);

    spikes_Ae_Ai_pre = new(std::nothrow) uint16_t[tamVector]{0};
    assert(spikes_Ae_Ai_pre != nullptr);

    spikes_Ae_Ai_pos = new(std::nothrow) uint16_t[tamVector]{0};
    assert(spikes_Ae_Ai_pos != nullptr);

    spikes_Ai_Ae_pre = new(std::nothrow) uint16_t[tamVector]{0};
    assert(spikes_Ai_Ae_pre != nullptr);

    spikes_Ai_Ae_pos = new(std::nothrow) uint16_t[tamVector]{0};
    assert(spikes_Ai_Ae_pos != nullptr);

    vE = new(std::nothrow) float[NUM_NEURONS]{0.0};
    assert( vE != nullptr );

    vI = new(std::nothrow) float[NUM_NEURONS]{0.0};
    assert( vI != nullptr );

    for (int indx = 0; indx < NUM_NEURONS; ++indx) {
        vE[indx] = -65.0f;      // v_rest_e = -65.0[mV]
        vI[indx] = -60.0f;    // v_rest_i = -60.0[mV]
    }

    refrac_countE = new(std::nothrow) int[NUM_NEURONS]{0};
    assert( refrac_countE != nullptr );

    refrac_countI = new(std::nothrow) int[NUM_NEURONS]{0};
    assert( refrac_countI != nullptr );

    unsigned int tamArrSamples = SINGLE_SAMPLE_TIME/32;
    input_sample = new(std::nothrow) uint32_t[NUM_PIXELS * tamArrSamples]{0};
    assert( input_sample != nullptr );

    spike_count = new(std::nothrow) uint32_t[NUM_NEURONS]{0};
    assert( spike_count != nullptr );

    //! [Step 2] Loading data from files

    weightsXeAe = new(std::nothrow) float[NUM_PIXELS*NUM_NEURONS]{0};
    assert( weightsXeAe != nullptr );

    std::string filename = std::string(PATH_PARAMETERS_NET) + "XeAe_" + std::to_string(NUM_NEURONS) + "N_" + std::to_string(SINGLE_SAMPLE_TIME) + "ms.dat";
    getWeights(weightsXeAe, filename, NUM_PIXELS, NUM_NEURONS);

    // The weight vector for Ae -> Ai layer is initialized.
    weightsAeAi = new(std::nothrow) float[NUM_NEURONS*NUM_NEURONS]{0};
    assert( weightsAeAi != nullptr );

    for (int i = 0; i < NUM_NEURONS; ++i) {
        weightsAeAi[i*NUM_NEURONS+i] = weights_Ae_Ai_constant;
    }

    // The weight vector for Ai -> Ae layer is initialized.
    weightsAiAe = new(std::nothrow) float[NUM_NEURONS*NUM_NEURONS]{0};
    assert( weightsAiAe != nullptr );

    for (int i = 0; i < NUM_NEURONS; ++i) {
        for (int j = 0; j < NUM_NEURONS; ++j) {
            weightsAiAe[i*NUM_NEURONS+j] = (i != j) ? weights_Ai_Ae_constant : 0.0;
        }
    }

    //! theta values are loaded into the group of excitatory neurons
    theta = new(std::nothrow) float[NUM_NEURONS]{0};
    assert( theta != nullptr );

    filename = std::string(PATH_PARAMETERS_NET) + "theta_" + std::to_string(NUM_NEURONS) + "N_" + std::to_string(SINGLE_SAMPLE_TIME) + "ms.dat";
    getTheta( theta, filename, NUM_NEURONS );

    assignments = new(std::nothrow) uint8_t[NUM_NEURONS]{0};
    assert( assignments != nullptr );

    filename = std::string(PATH_PARAMETERS_NET) + "assignments_" + std::to_string(NUM_NEURONS) + "N_" + std::to_string(SINGLE_SAMPLE_TIME) + "ms.dat";
    getAssignments( assignments, filename, NUM_NEURONS );

    time_t start, end;
    time(&start);
    //! =====     =====     =====
    //! Run Simulation
    //! =====     =====     =====
    for (int numSample = 1; numSample <= TOTAL_SAMPLES; ++numSample) {

        char buffer[100];
        sprintf( buffer, PATH_SAMPLES_POISSON ,numSample);

        std::string filename(buffer);

        getInputSample( input_sample, filename, NUM_PIXELS, tamArrSamples);

        //! Simulate network activity for SINGLE_SAMPLE_TIME timesteps.
        for (int t = 0; t < SINGLE_SAMPLE_TIME; ++t)
        {
            uint32_t datoAnalisis = 0;
            uint32_t resultOP = 0;
            uint8_t desplazamiento = 0;
            uint16_t index = 0, group = 0;

            for (int j = 0; j < NUM_PIXELS; ++j)
            {
                // Los números pares serán el indice del pixel
                // La información del tiempo empieza desde el bit más significativo
                //  0 1
                //  2 3
                //  4 5
                //  6 7
                //  8 9
                // 10 11

                //!* Se desenvuelve el bucle ya que solo son dos posiciones del arreglo
                //input_sample[ j*2   ]
                //input_sample[ j*2+1 ]

                datoAnalisis = ( t < 32 ) ? input_sample[ j*2 ] : input_sample[ j*2+1 ];
                desplazamiento = ( t < 32 ) ? (31 - t) : (63 - t);
                resultOP = datoAnalisis & (1 << desplazamiento);
                resultOP >>= desplazamiento;

                // binary information is verified
                assert( resultOP == 0 or resultOP == 1);

                index = j % 16; //(sizeof (uint16_t) * 8)
                group = (j - index) / 16; //(sizeof (uint16_t) * 8)
                spikesXePre[group] |= (resultOP << index);
                /*
                if ( input_sample[j+t*NUM_PIXELS] == 1 ) {
                    spikesXePre[j] = 1;
                }
                else{
                    spikesXePre[j] = 0;
                }
                */
            }

            uint16_t indexExc = 0, groupExc = 0;
            // 1. Update E neurons

            for (unsigned int j = 0; j < NUM_NEURONS; ++j)
            {
                indexExc = j % 16; //(sizeof (uint16_t) * 8)
                groupExc = (j - indexExc) / 16; //(sizeof (uint16_t) * 8)

                assert( indexExc < 16);
                assert( groupExc < NUM_NEURONS/16);

                // Decay voltages and adaptive thresholds.
                vE[j] = 0.990049839019775390625 * (vE[j] - (-65.0f)) + (-65.0f);

                // Integrate inputs.
                vSyn = 0.0;

                int32_t res = 0;
                void *ptrF = nullptr;
                int32_t resultBits = 0;

                for (unsigned int i = 0; i < NUM_NEURONS; ++i) {

                    index = i % 16; //(sizeof (uint16_t) * 8)
                    group = (i - index) / 16; //(sizeof (uint16_t) * 8)

                    assert( index < 16);
                    assert( group < NUM_NEURONS/16);

                    ptrF = &weightsAiAe[j*NUM_NEURONS+i];

                    res = (spikes_Ai_Ae_pos[group] & (1 << index)) >> index;
                    res = ~res;
                    res++;
                    // Computo de la sinapsis entre flotante y entero usando casting de apuntadores
                    resultBits = res & *(int*)ptrF;

                    //vSyn += spikes_Ai_Ae_pos[i] ? weightsAiAe[j*NUM_NEURONS+i] : 0.0f ;
                    vSyn += *(float*)&resultBits;
                }

                for (int i = 0; i < NUM_PIXELS; ++i) {

                    index = i % 16; //(sizeof (uint16_t) * 8)
                    group = (i - index) / 16; //(sizeof (uint16_t) * 8)

                    assert( index < 16);
                    assert( group < NUM_PIXELS/16);

                    ptrF = &weightsXeAe[i*NUM_NEURONS+j];

                    res = (spikesXePos[group] & (1 << index)) >> index;
                    res = ~res;
                    res++;
                    // Computo de la sinapsis entre flotante y entero usando casting de apuntadores
                    resultBits = res & *(int*)ptrF;

                    //vSyn += spikesXePos[i] ? weightsXeAe[i*NUM_NEURONS+j] : 0.0f ;
                    vSyn += *(float*)&resultBits;
                }

                if( refrac_countE[j] <= 0 )
                    vE[j] += vSyn;

                if( PRINT_ENABLE and t >= 34 and (j < 5 or j >= 395)){
                    printf("vE[%d] = %.25f\n", j, vE[j]);
                    cout.flush();
                }

                // Decrement refractory counters.
                refrac_countE[j] -= 1; // dt = 1

                // Check for spiking neurons.
                spikes_Ae_Ai_pre[groupExc] |= ( (vE[j] > ( (-52.0) + theta[j])) ? 1 : 0 ) << indexExc;

                // Refractoriness, voltage reset, and adaptive thresholds.
                if( (spikes_Ae_Ai_pre[groupExc] & (1 << indexExc) )  != 0 ){
                    refrac_countE[j] = 5; // refrac_e = 5;      // [ms]
                    vE[j] = -60.0F; // v_reset_e = -60.0;    // [mV]
                }

            }



            int indexWin = -1;
            for (int indx = 0; indx < NUM_NEURONS; ++indx) {

                index = indx % 16; //(sizeof (uint16_t) * 8)
                group = (indx - index) / 16; //(sizeof (uint16_t) * 8)

                assert( index < 16);
                assert( group < NUM_NEURONS/16);

                if( ( spikes_Ae_Ai_pre[group] & (1 << index) ) != 0 ){
                    //cout << "[" << t << ", " << indx << "]" << endl;
                    indexWin = (indexWin == -1) ? indx : indexWin;
                    if( PRINT_ENABLE ){
                        printf("\nt=[%d]; indexWin[%d]",t,indx); cout << endl;
                    }
                }
            }

            if (indexWin >= 0) {

                for (int index = 0; index < NUM_NEURONS; ++index) {

                    indexExc = index % 16; //(sizeof (uint16_t) * 8)
                    groupExc = (index - indexExc) / 16; //(sizeof (uint16_t) * 8)

                    assert( indexExc < 16);
                    assert( groupExc < NUM_NEURONS/16);

                    if( ( spikes_Ae_Ai_pre[groupExc] & (1 << indexExc) ) != 0){
                        ++digits[ assignments[index] ];
                    }
                }
                for (int indx = 0; indx < tamVector; ++indx) {
                    spikes_Ae_Ai_pre[indx] = 0;
                }
                uint16_t indxWin = indexWin % 16; //(sizeof (uint16_t) * 8)
                uint16_t groupWin = (indexWin - indxWin) / 16; //(sizeof (uint16_t) * 8)

                assert( indxWin < 16);
                assert( groupWin < NUM_NEURONS/16);

                spikes_Ae_Ai_pre[groupWin] |= (1 << indxWin);
                ++spike_count[indexWin];
            }

            // 2. Update I neurons
            int32_t res = 0;
            void *ptrF = nullptr;
            int32_t resultBits = 0;

            uint16_t indexInh = 0, groupInh = 0;

            for (int i = 0; i < NUM_NEURONS; ++i) {
                // Decay voltages.
                vI[i] = 0.904837429523468017578125 * (vI[i] - (-60.0f)) + (-60.0f);

                // Integrate inputs.
                vSyn = 0.0;
                for (unsigned int j = 0; j < NUM_NEURONS; ++j) {

                    index = j % 16; //(sizeof (uint16_t) * 8)
                    group = (j - index) / 16; //(sizeof (uint16_t) * 8)

                    assert( index < 16);
                    assert( group < NUM_NEURONS/16);

                    ptrF = &weightsAeAi[i*NUM_NEURONS+j];

                    res = (spikes_Ae_Ai_pos[group] & (1 << index)) >> index;
                    res = ~res;
                    res++;
                    // Computo de la sinapsis entre flotante y entero usando casting de apuntadores
                    resultBits = res & *(int*)ptrF;

                    //vSyn += spikesXePos[i] ? weightsXeAe[i*NUM_NEURONS+j] : 0.0f ;
                    vSyn += *(float*)&resultBits;

                    //vSyn += spikes_Ae_Ai_pos[j] ? weightsAeAi[i*NUM_NEURONS+j] : 0.0f ;
                }

                if( refrac_countI[i] > 0 )
                    vSyn = 0;

                vI[i] += vSyn;

                // Decrement refractory counters.
                refrac_countI[i] -= 1;

                // Check for spiking neurons.
                if ( vI[i] > -40.0 ){
                    refrac_countI[i] =  2; // refrac_i = 2; // [ms]
                    vI[i] =  -45.0f; // v_reset_i = -45.0;    // [mV]
                    for (int j = 0; j < tamVector; ++j) {
                        spikes_Ai_Ae_pre[j] = 0;
                    }

                    indexInh = i % 16; //(sizeof (uint16_t) * 8)
                    groupInh = (i - index) / 16; //(sizeof (uint16_t) * 8)

                    assert( indexInh < 16);
                    assert( groupInh < NUM_NEURONS/16);

                    spikes_Ai_Ae_pre[groupInh] = (1 << indexInh);
                }

            }

            // The arrays are exchanged for the next iteration of time
            for (int indx = 0; indx < tamVectorPixels; ++indx) {
                spikesXePos[indx] = spikesXePre[indx];
                spikesXePre[indx] = 0;
            }

            for (int indx = 0; indx < tamVector; ++indx){
                spikes_Ae_Ai_pos[indx] = spikes_Ae_Ai_pre[indx];
                spikes_Ai_Ae_pos[indx] = spikes_Ai_Ae_pre[indx];
                spikes_Ae_Ai_pre[indx] = 0;
                spikes_Ai_Ae_pre[indx] = 0;
            }

        }

        // classification
        float rates[10] = {0.0f};

        //! Count the number of neurons with this label assignment.
        float n_assigns[10] = {0};

        for (int indx = 0; indx < NUM_NEURONS; ++indx) {
            ++n_assigns[assignments[indx]];

            rates[assignments[indx]] += spike_count[indx];
        }

        for (int indx = 0; indx < 10; ++indx) {
            rates[indx] /= n_assigns[indx];
        }

        int indWinner = 0;
        float ratesWin = 0;

        for (int indx = 0; indx < 10; ++indx) {
            if ( rates[indx] > ratesWin ) {
                indWinner = indx;
                ratesWin = rates[indx];
            }
        }

        std::cout << "Digit class: " << indWinner << std::endl;

        std::ofstream fileLabels;
        std::string filenameLabels = std::string(PATH_RESULTS_NET) + "labelsQt" + std::to_string(NUM_NEURONS) +"N_64ms_test_Exc_Inh_bin.csv";
        fileLabels.open(filenameLabels, std::ofstream::out | std::ofstream::app);
        if (!fileLabels.is_open())
        {
            std::cout << "Error opening file" << __LINE__ << std::endl;
            exit(1);
        }

        fileLabels << indWinner << std::endl;
        fileLabels.close();

        //! =====     =====     =====
        //! Reset Variables
        //! =====     =====     =====
        for (int i = 0; i < NUM_NEURONS; ++i) {

            vE[i] = -65.0f; // v_rest_e = -65.0 [mV]
            vI[i] = -60.0f; // v_rest_i = -60.0 [mV]

            refrac_countE[i] = 0;
            refrac_countI[i] = 0;

            spike_count[i] = 0;
        }

        for (uint16_t i = 0; i < tamVector; ++i) {
            spikes_Ae_Ai_pos[i] = 0;
            spikes_Ae_Ai_pre[i] = 0;
            spikes_Ai_Ae_pos[i] = 0;
            spikes_Ai_Ae_pre[i] = 0;
        }

        for (int i = 0; i < tamVectorPixels; ++i) {
            spikesXePre[i] = 0;
            spikesXePos[i] = 0;
        }
    }

    // Recording end time.
    time(&end);

    // Calculating total time taken by the program.
    double time_taken = double(end - start);
    cout << "Time taken by program is : " << time_taken << " sec" << endl;

    return 0;
}
