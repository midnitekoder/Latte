#ifndef _2016_latte_connection_handling_

#define _2016_latte_connection_handling_


#include <stdbool.h>
#include <stdio.h>
#include "util.h"

bool is_connected(int prev_layer_node_index, int current_layer_node_index, int number_of_nodes, layer_type type_of_layer, bool *connection_table);

int update_connection_table_height(Layer *layer, int *connection_table_h);


/* To generate connection table based on the weight configuration of the nodes of the layer.
to be implemented*/
int build_connection_table(Layer prev_layer, Layer current_layer);



#endif