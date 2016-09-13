#ifndef _2016_latte_io_buffer_
#define _2016_latte_io_buffer_

#include "util.h"

/*Function to set input and output buffer points to their appropriate location in the two buffers such 
once initialized, it does not change till the same network is used for forward propagation.
*Parameters:
** layers: pointer to the list of all the layers
**buffer: pointer to the two buffer

*Return:
**status flag
*/
int set_io_channels(Layer **layers, int number_of_layers, float **buffer);


/*Function to copy input to the buffers of the framework from the location where the user has specified.
*Parameters:
**h: handle
**param: data structure specifying the location and size of the input and output.

*Return:
status flag
*/
int copy_input(latte_handle *h, PARAM param);


/*Function to copy output from the buffers of the framework to the location where the user has specified.
*Parameters:
**h: handle
**param: data structure specifying the location and size of the input and output.

*Return:
status flag
*/
int copy_output(latte_handle *h, PARAM param);

/*Function to boundary pad the input to the layer
*Parameter:
**layer: layer specific data
*/
void pad_layer_input(Layer *layer);


void print_output(Layer *layer);


void print_input(Layer *layer);
#endif 
