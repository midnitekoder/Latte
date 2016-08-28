#ifndef _2016_latte_net_builder_

#define _2016_latte_net_builder_

#include "util.h"
#include "latte_config.h"

/*Function to initialize the network specific parameters based on the model_reader.
*Parameters:
**handle: handle to the network.
**config: The data read from the lattemodel using model_reader.
**param: parameters set by user to access input and output buffers.

*Return:
**status flag*/
int init_network(latte_handle* handle, LATTE_config *config,PARAM param);





/*Function to initialize the layer specific parameters based on the model_reader.
This includes creating nodes of the layer and setting up the weights and biases.
*Parameters:
**handle: handle to the network-specific fields.
**config: The data read from the lattemodel using model_reader.
**layer_index: index of the  layer
**connection_table_h: the number of nodes in the previous layer.

*Return:
**status flag*/
int init_layer(Network *net, LATTE_config *config, int layer_index,  int connection_table_h);

#endif 
