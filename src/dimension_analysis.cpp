#include "util.h"
#include <stdio.h>


/*Function to calculate and set the dimension of the output of each layer in the network.
*Parameters:
**net: handle to the network specific data.

*Return:
**status flag
*/
int get_output_dimension(Network *net)
{
	int layer_index;

	/*Setting dimensions of input and output of the layers*/
	for (layer_index = 0; layer_index < net->number_of_layers;layer_index++)
	{

		switch (net->layers[layer_index]->type_of_layer)
		{
		case LT_AVGP:
		case LT_MAXP:
			net->layers[layer_index]->osize_w = ((net->layers[layer_index]->istride_w  - net->layers[layer_index]->ksize_w) / net->layers[layer_index]->k_stride_w) + 1;
			net->layers[layer_index]->osize_h = ((net->layers[layer_index]->isize_h - net->layers[layer_index]->ksize_h) / net->layers[layer_index]->k_stride_h) + 1;
			net->layers[layer_index]->osize_d = net->layers[layer_index]->isize_d;
			//net->layers[layer_index]->ostride_w = net->layers[layer_index]->osize_w + 2 * net->layers[layer_index + 1]->padding_size_w;

			break;
		case LT_DROPOUT:
			net->layers[layer_index]->osize_w = net->layers[layer_index]->isize_w;
			net->layers[layer_index]->osize_h = net->layers[layer_index]->isize_h;
			net->layers[layer_index]->osize_d = net->layers[layer_index]->isize_d;
		//	net->layers[layer_index]->ostride_w = net->layers[layer_index]->osize_w + 2 * net->layers[layer_index + 1]->padding_size_w;
	
			break;
		case LT_CONV:
			net->layers[layer_index]->osize_w = ((net->layers[layer_index]->istride_w - net->layers[layer_index]->ksize_w) / net->layers[layer_index]->k_stride_w) + 1;
			net->layers[layer_index]->osize_h = ((net->layers[layer_index]->isize_h - net->layers[layer_index]->ksize_h) / net->layers[layer_index]->k_stride_h) + 1;
			net->layers[layer_index]->osize_d = net->layers[layer_index]->number_of_nodes;
		//	net->layers[layer_index]->ostride_w = net->layers[layer_index]->osize_w + 2 * net->layers[layer_index + 1]->padding_size_w;
		
			break;
		case LT_FULL_CONN:
		case LT_PART_CONN:
			net->layers[layer_index]->osize_w = 1;
			net->layers[layer_index]->osize_h = 1;
			net->layers[layer_index]->osize_d = net->layers[layer_index]->number_of_nodes;
		//	net->layers[layer_index]->ostride_w = net->layers[layer_index]->osize_w + 2 * net->layers[layer_index + 1]->padding_size_w;
		
			break;
		case LT_LRN_ACROSS:
		case LT_SOFTMAX:
			net->layers[layer_index]->osize_w = net->layers[layer_index]->isize_w;
			net->layers[layer_index]->osize_h = net->layers[layer_index]->isize_h;
			net->layers[layer_index]->osize_d = net->layers[layer_index]->isize_d;
		//	net->layers[layer_index]->ostride_w = net->layers[layer_index]->osize_w + 2 * net->layers[layer_index + 1]->padding_size_w;
	break;
		default:
			return LATTE_ERR_INVALID_CONFIG;

		}
		if (net->layers[layer_index + 1] != NULL)
		{
			net->layers[layer_index]->ostride_w = net->layers[layer_index]->osize_w + 2 * net->layers[layer_index + 1]->padding_size_w;

		}
		else
		{
			net->layers[layer_index]->ostride_w = net->layers[layer_index]->osize_w;
		}
		if (net->layers[layer_index]->osize_w <= 0 || net->layers[layer_index]->osize_h <= 0 || net->layers[layer_index]->osize_d <= 0)
			return LATTE_ERR_DIMENSION_INCOMPATIBLE;
	}

	return LATTE_SUCCESS;
}

/*
Function to calculate and set the dimension of the input to the layer specified.
*Parameters:
**net: handle to the network specific data.
**layer_index: index of the layer whose input dimension has to be calculated.

*Return:
**status flag
*/
int get_input_dimension(Network *net,int layer_index)
{
	int prev_osize_w, prev_osize_h, prev_osize_d, prev_ostride_w;
	

	if (layer_index == 0)
	{
		net->layers[layer_index]->isize_h = net->isize_h + 2 * net->layers[layer_index]->padding_size_h;
		net->layers[layer_index]->istride_w = net->istride_w + 2 * net->layers[layer_index]->padding_size_w;
		net->layers[layer_index]->isize_d = net->isize_d;
		net->layers[layer_index]->isize_w = net->layers[layer_index]->istride_w;
	}
	/*Setting dimensions of input and output of the layers*/
	else
	{

			switch (net->layers[layer_index-1]->type_of_layer)
			{
			case LT_AVGP:
			case LT_MAXP:
				prev_osize_w = ((net->layers[layer_index-1]->isize_w  - net->layers[layer_index-1]->ksize_w) / net->layers[layer_index-1]->k_stride_w) + 1;
				prev_osize_h = ((net->layers[layer_index-1]->isize_h  - net->layers[layer_index-1]->ksize_h) / net->layers[layer_index-1]->k_stride_h) + 1;
				prev_osize_d = net->layers[layer_index-1]->isize_d;
				prev_ostride_w = prev_osize_w + 2 * net->layers[layer_index]->padding_size_w;
				net->layers[layer_index]->istride_w = prev_ostride_w;
				net->layers[layer_index]->isize_h = prev_osize_h + 2 * net->layers[layer_index]->padding_size_h;
				net->layers[layer_index]->isize_d = prev_osize_d;
				net->layers[layer_index]->isize_w = net->layers[layer_index]->istride_w;//in case of tiling this will change

				break;

			case LT_CONV:
				prev_osize_w = ((net->layers[layer_index-1]->isize_w  - net->layers[layer_index-1]->ksize_w) / net->layers[layer_index-1]->k_stride_w) + 1;
				prev_osize_h = ((net->layers[layer_index-1]->isize_h  - net->layers[layer_index-1]->ksize_h) / net->layers[layer_index-1]->k_stride_h) + 1;
				prev_osize_d = net->layers[layer_index-1]->number_of_nodes;
				prev_ostride_w = prev_osize_w + 2 * net->layers[layer_index]->padding_size_w;
				net->layers[layer_index]->istride_w = prev_ostride_w;
				net->layers[layer_index ]->isize_h = prev_osize_h + 2 * net->layers[layer_index]->padding_size_h;
				net->layers[layer_index]->isize_d = prev_osize_d;
				net->layers[layer_index]->isize_w = net->layers[layer_index]->istride_w;//in case of tiling this will change

				break;
			case LT_FULL_CONN:
			case LT_PART_CONN:
				prev_osize_w = 1;
				prev_osize_h = 1;
				prev_osize_d = net->layers[layer_index-1]->number_of_nodes;
				prev_ostride_w = prev_osize_w + 2 * net->layers[layer_index]->padding_size_w;
				net->layers[layer_index]->istride_w = prev_ostride_w;
				net->layers[layer_index]->isize_h = prev_osize_h + 2 * net->layers[layer_index]->padding_size_h;
				net->layers[layer_index ]->isize_d = prev_osize_d;
				net->layers[layer_index]->isize_w = net->layers[layer_index]->istride_w;//in case of tiling this will change

				break;
			case LT_LRN_ACROSS:
			case LT_SOFTMAX:
				prev_osize_w = net->layers[layer_index-1]->isize_w;
				prev_osize_h = net->layers[layer_index-1]->isize_h;
				prev_osize_d = net->layers[layer_index-1]->isize_d;
				prev_ostride_w = prev_osize_w + 2 * net->layers[layer_index]->padding_size_w;
				net->layers[layer_index]->istride_w = prev_ostride_w;
				net->layers[layer_index]->isize_h = prev_osize_h + 2 * net->layers[layer_index]->padding_size_h;
				net->layers[layer_index]->isize_d = prev_osize_d;
				net->layers[layer_index]->isize_w = net->layers[layer_index]->istride_w;//in case of tiling this will change
				break;
			default:
				return LATTE_ERR_INVALID_CONFIG;

		}

	}
	
	return LATTE_SUCCESS;
}

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
int get_ksize_d(int isize_d, layer_type type_of_layer, bool *connection_table, int node_index, int *ksize_d,int number_of_nodes)
{
	int i, count_number_of_connections;
	switch (type_of_layer)
	{
	case LT_AVGP:
	case LT_MAXP:
		*ksize_d = 1;
		break;
	case LT_CONV:
		if (connection_table == NULL)
		{
			*ksize_d = isize_d;
		}
		else
		{
			count_number_of_connections = 0;
			for (i = 0; i < isize_d; i++)
			{
				if (connection_table[i*number_of_nodes+node_index] == true)
				{
					count_number_of_connections++;
				}
			}
			*ksize_d = count_number_of_connections;

		}
		break;
	case LT_DROPOUT:
		break;
	case LT_FULL_CONN:
		*ksize_d = isize_d;
		break;
	case LT_PART_CONN:
		if (connection_table != NULL)
		{

			count_number_of_connections = 0;
			for (i = 0; i < isize_d; i++)
			{
				if (connection_table[i*number_of_nodes + node_index] == true)
				{
					count_number_of_connections++;
				}
			}
			*ksize_d = count_number_of_connections;
		}

		break;
	case LT_LRN_ACROSS:
		*ksize_d = 4;
		break;
	case LT_SOFTMAX:
		*ksize_d = 0;
		break;

	default:
		return LATTE_ERR_INVALID_CONFIG;
		break;
	}
	return LATTE_SUCCESS;
}

int max(int num1, int num2)
{
	if (num1 > num2)
		return num1;
	else
		return num2;
}

/* Function to calculate maximum buffer size that is needed to the intermediate output of any layer.
*Parameters:
**layers: pointer to array of layers.
**number_of_layers: number of layers in the network.

*Return:
**maximum buffer size that is needed.
*/
int get_max_buffer_size(Layer **layers, int number_of_layers)
{
	int layer_index, input_size, output_size,max_buffer_size;
	max_buffer_size = 0;
	for (layer_index = 0; layer_index < number_of_layers; layer_index++)
	{
		input_size=layers[layer_index]->isize_d*layers[layer_index]->isize_h*layers[layer_index]->istride_w;
		output_size = layers[layer_index]->osize_d*layers[layer_index]->osize_h*layers[layer_index]->ostride_w;
		max_buffer_size = max(input_size, max_buffer_size);
		max_buffer_size = max(output_size, max_buffer_size);
	}
	return max_buffer_size;
}

/*Function to validate the input and output paramters specified by the
user by comparing it with the configuration of the network.
*Parameters:
**h: handle to the network
**param: parameters specified by the user.

*Return:
**status flag
*/
int validate_param(latte_handle *h, PARAM param)
{
	if (h->net->isize_d == param.isize_d && h->net->isize_h == param.isize_h && h->net->isize_w == param.isize_w && h->net->istride_w==param.istride_w)
	{
		if (h->net->osize_d == param.osize_d && h->net->osize_h == param.osize_h && h->net->osize_w == param.osize_w)
		{
			return LATTE_SUCCESS;
		}
		else
			return LATTE_ERR_INVALID_IO_PARAM;
	}
	else
		return LATTE_ERR_INVALID_IO_PARAM;
}