#ifndef _2016_latte_kernel_operations_

#define _2016_latte_kernel_operations_
#include "util.h"

/*Function to perform convolution.
*Parameters:
**layer: layer specific data.
**windowx: the x coordinate of the window in which convolution to be performed.
**windowy: the y coordiante of the window in which convolution to be performed.
**output_location: the location of the output in row major form.
**kernel_depth_connection: the depth of the kernel that has been used so far. This is used for compactness of the kernel weights.
**next_layer: the next layer specific data.

*Return:
**status flag
*/
int conv_kernel(Layer *layer, int windowx, int windowy, int output_location, int *kernel_depth_connection, Layer *next_layer);

/*Function to perform maxpooling.
*Parameters:
**layer: layer specific data.
**windowx: the x coordinate of the window in which convolution to be performed.
**windowy: the y coordiante of the window in which convolution to be performed.
**output_location: the location of the output in row major form.
**kernel_depth_connection: the depth of the kernel that has been used so far. This is used for compactness of the kernel weights.
**next_layer: the next layer specific data.

*Return:
**status flag
*/
int maxpool_kernel(Layer *layer, int windowx, int windowy, int output_location, Layer *next_layer);

/*Function to perform Average-Pooling.
*Parameters:
**layer: layer specific data.
**windowx: the x coordinate of the window in which convolution to be performed.
**windowy: the y coordiante of the window in which convolution to be performed.
**output_location: the location of the output in row major form.
**kernel_depth_connection: the depth of the kernel that has been used so far. This is used for compactness of the kernel weights.
**next_layer: the next layer specific data.

*Return:
**status flag
*/
int avgpool_kernel(Layer *layer, int windowx, int windowy, int output_location, Layer *next_layer);


void transpose(float *matrix, int rows, int columns);

#endif 
