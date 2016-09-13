#include <stdio.h>
#include <stdbool.h>
#include "util.h"


bool is_connected(int prev_layer_node_index, int current_layer_node_index , int number_of_nodes, layer_type type_of_layer, bool *connection_table)
{
	switch (type_of_layer)
	{
	case LT_CONV:
	case LT_PART_CONN:
		if (connection_table == NULL)
		{
			return true;
		}
		else
		{/*(r,c) is the connection between r input layer node and c current layer node*/
			return connection_table[prev_layer_node_index*number_of_nodes+current_layer_node_index];
			/*temporary change*/
			/*if (prev_layer_node_index < 48 && current_layer_node_index < 128)
				return true;
			else if (prev_layer_node_index >= 48 && current_layer_node_index >= 128)
				return true;
			else
				return false;
				*/
			//return connection_table[prev_layer_node_index*number_of_nodes+ current_layer_node_index];

		
		}
		break;
	case LT_MAXP:
	case LT_FULL_CONN:
	case LT_LRN_ACROSS:
	case LT_AVGP:
	case LT_DROPOUT:
	case LT_SOFTMAX:
		return true;
	default:
		return false;
	}
}

int update_connection_table_height(Layer *layer, int *connection_table_h)
{
	switch (layer->type_of_layer)
	{
	case LT_CONV:
	case LT_FULL_CONN:
	case LT_PART_CONN:
		*connection_table_h = layer->number_of_nodes;
		break;
	case LT_MAXP:
	case LT_AVGP:
	case LT_LRN_ACROSS:
	case LT_SOFTMAX:
	case LT_DROPOUT:
		break;
	default:
		return LATTE_ERR_INVALID_CONFIG;

	}
	return LATTE_SUCCESS;
}

int build_connection_table(Layer prev_layer, Layer current_layer)
{
	//to be implemented.
}

