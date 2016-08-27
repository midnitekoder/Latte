#ifndef _2016_latte_dimension_analysis_

#define _2016_latte_dimension_analysis_

#include "util.h"

/*
Function to calculate and set the dimension of the output of each layer in the network.
*Parameters:
**net: handle to the network specific data.

*Return:
**status flag
*/
int get_output_dimension(Network *net);

/*
Function to calculate and set the dimension of the input to the layer specified.
*Parameters:
**net: handle to the network specific data.
**layer_index: index of the layer whose input dimension has to be calculated.

*Return:
**status flag
*/
int get_input_dimension(Network *net, int layer_index);

/*Function to calculate the depth of the weights of the node.
*Parameters:
**isize_d: the depth of the input of the current layer
**type_of_layer: type of layer
**connection_table: represents connection between current layer and previous layer. Refer util.h for details. 
**node_index: index of the current node.
**ksize_d: depth of the weights of the node.
**number_of_nodes: number of nodes in the layer.

*Return:
**status flag
*/
int get_ksize_d(int isize_d, layer_type type_of_layer, bool *connection_table, int node_index, int * ksize_d,int number_of_nodes);

/* Function to calculate maximum buffer size that is needed to the intermediate output of any layer.
*Parameters:
**layers: pointer to array of layers.
**number_of_layers: number of layers in the network.

*Return:
**maximum buffer size that is needed.
*/
int get_max_buffer_size(Layer **layers, int number_of_layers);

/*Function to validate the input and output paramters specified by the
user by comparing it with the configuration of the network.
*Parameters:
**h: handle to the network
**param: parameters specified by the user.

*Return:
**status flag
*/
int validate_param(latte_handle *h, PARAM param);
#endif 
