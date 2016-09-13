#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include "util.h"
#include "latte_config.h"
#include "latte_api.h"
#include "dimension_analysis.h"
#include "io_buffer_handling.h"
#include "connection_handling.h"
#include "forward_prop.h"
#include "net_builder.h"
#include "model_reader.h"

float **buffer;

/*Function to create a handle of the network model
*Parameters:
void

*Return:
**handle of the network model*/
latte_handle* latte_create(void)
{
	latte_handle *handle;
	handle = NULL;
	handle = (latte_handle*)MALLOC(sizeof(latte_handle));
	return handle;
}

/*Function to initalize the network model by parsing the lattemodel file.
*Parameters:
**handle: handle of the network model
**param: parameters set by user to access input and output buffers.

Return:
status flag
*/
int latte_parse_and_init(latte_handle *handle, FILE* LATTE_FILE, PARAM param)
{
	int connection_table_h, max_buffer_size, status, node_index, layer_index,weight_index;
	LATTE_config *config;
	latte_handle *h;
	h = handle;
	


	config = (LATTE_config*)MALLOC(sizeof(LATTE_config));
	status=parse_network(LATTE_FILE,config);
	if (status != LATTE_SUCCESS)
	{
		return status;
	}
	if (h == NULL)
	{
		return LATTE_ERR_OUT_OF_MEM;
	}
	status=init_network(h,config,param);
	if (status != LATTE_SUCCESS)
	{
		return status;
	}
	connection_table_h = h->net->isize_d;
	for (layer_index = 0; layer_index < h->net->number_of_layers; layer_index++)
	{
		status = parse_layer(LATTE_FILE, config,layer_index);
		if (status != LATTE_SUCCESS)
		{
			return status;
		}
		status = init_layer(h->net, config, layer_index, connection_table_h);
		if (status != LATTE_SUCCESS)
		{
			return status;
		}
		

		status = update_connection_table_height(h->net->layers[layer_index], &connection_table_h);
		if (status != LATTE_SUCCESS)
		{
			return status;
		}
		for (node_index = 0; node_index <h->net->layers[layer_index]->number_of_nodes; node_index++)
		{
			free(config->nodes[node_index]->weight);
			free(config->nodes[node_index]);
		}
		free(config->nodes);
		free(config->layer.layer_name);
		

	}
	status = get_output_dimension(h->net);
	if (status != LATTE_SUCCESS)
	{
		return status;
	}
	max_buffer_size = get_max_buffer_size(h->net->layers, h->net->number_of_layers);
	buffer = (float**)MALLOC(sizeof(float*)* 2);
	if (buffer == NULL)
	{
		return LATTE_ERR_OUT_OF_MEM;
	}
	buffer[0] = (float*)MALLOC(sizeof(float)*max_buffer_size);
	buffer[1] = (float*)MALLOC(sizeof(float)*max_buffer_size);
	if (buffer[0] == NULL || buffer[1] == NULL)
	{
		return LATTE_ERR_OUT_OF_MEM;
	}

	status = set_io_channels(h->net->layers, h->net->number_of_layers, buffer);
	/*print statements*/
/*	for (layer_index = 0; layer_index < h->net->number_of_layers; layer_index++)
	{
		for (node_index = 0; node_index < h->net->layers[layer_index]->number_of_nodes; node_index++)
		{
			if (h->net->layers[layer_index]->nodes != NULL)
			{
				for (weight_index = 0; weight_index < h->net->layers[layer_index]->ksize_h*h->net->layers[layer_index]->ksize_w*h->net->layers[layer_index]->nodes[node_index]->ksize_d; weight_index++)
				if (h->net->layers[layer_index]->nodes[node_index]->weight != NULL)
				{

					printf("%f ", h->net->layers[layer_index]->nodes[node_index]->weight[weight_index]);
				}
				printf("\n\n");
			}
		}
		
	}
	*/
	int count;
	layer_index = 0;
	node_index = 0;
	count = 0;
	for (weight_index = 0; weight_index < h->net->layers[layer_index]->ksize_h*h->net->layers[layer_index]->ksize_w*h->net->layers[layer_index]->nodes[node_index]->ksize_d; weight_index++)
	if (h->net->layers[layer_index]->nodes[node_index]->weight != NULL)
	{

		printf("%.10f ", h->net->layers[layer_index]->nodes[node_index]->weight[weight_index]);
		count++;
		if (count == 11)
		{
			printf("\n");
			count = 0;
		}
	}
	printf("\n\n");
	/*free config*/

	free(config);

	/*testing code temporary*/
/*	h->net->layers[0]->ksize_h = 3;
	h->net->layers[0]->ksize_w = 3;
	h->net->layers[0]->connection_table = (bool*)malloc(sizeof(bool)*)
	*/

	return status;
}



/*Function to forward propagate the network based on the design of the network model.
*Parameters:
**handle: handle of the network model
**param: parameters set by user to access input and output buffers.

Return:
status flag
*/
int latte_forward_prop(latte_handle* handle, PARAM param)
{
	int status,layer_index;
	latte_handle *h;
	h = handle;
	if (h == NULL || h->net == NULL)
		return LATTE_ERR_OUT_OF_MEM;
	status= validate_param(h, param);
	if (status != LATTE_SUCCESS)
	{
		return status;
	}
	if (buffer != NULL)
	{

		if (buffer[0] != NULL && buffer[1] != NULL)
		{
			copy_input(h, param);
		}
		else
		{
			return LATTE_ERR_OUT_OF_MEM;
		}
	}
	for (layer_index = 0; layer_index < h->net->number_of_layers; layer_index++)
	{
		switch (h->net->layers[layer_index]->type_of_layer)
		{
		case LT_CONV:
			
			status = forward_propagate_CONV(h->net->layers[layer_index],h->net->layers[layer_index+1]);
			//print_output(h->net->layers[layer_index]);
			
			break;		
		case LT_MAXP:
			status = forward_propagate_MAXP(h->net->layers[layer_index], h->net->layers[layer_index + 1]);
			//print_output(h->net->layers[layer_index]);
			break;
		case LT_LRN_ACROSS:
			
			status = forward_propagate_LRN_ACROSS(h->net->layers[layer_index], h->net->layers[layer_index + 1]);
			print_output(h->net->layers[layer_index]);
			break;
		case LT_FULL_CONN:
			status = forward_propagate_FULL_CONN(h->net->layers[layer_index], h->net->layers[layer_index + 1]);
			break;
		case LT_SOFTMAX:
			status = forward_propagate_SOFTMAX(h->net->layers[layer_index], h->net->layers[layer_index + 1]);
			
			break;
		case LT_AVGP:
			status = forward_propagate_AVGP(h->net->layers[layer_index], h->net->layers[layer_index + 1]);
			break;
		case LT_DROPOUT:
			status = forward_propagate_DROPOUT(h->net->layers[layer_index], h->net->layers[layer_index + 1]);
			break;
		case LT_PART_CONN:
			status = forward_propagate_PART_CONN(h->net->layers[layer_index], h->net->layers[layer_index + 1]);
			break;
		default:
			return LATTE_ERR_INVALID_CONFIG;
		}
		if (status != LATTE_SUCCESS)
			return status;
		//print_output(h->net->layers[layer_index]);
		/*for (d = 0; d < h->net->layers[layer_index]->osize_d;d++)
		{
			for (h = 0;)
		}*/
		print_output(h->net->layers[layer_index]);
	}
	//print_output(h->net->layers[layer_index-1]);
	copy_output(h, param);
	return LATTE_SUCCESS;
}



/*Function to release all the resources allocated to the network model.
*Parameters:
**handle: handle of the network model

Return:
status flag
*/
int latte_close(latte_handle* handle)
{
	latte_handle *h;
	int layer_index, node_index;
	Layer *current_layer = NULL;
	h = handle;
	
	if (buffer != NULL)
	{
		if (buffer[0] != NULL)
		{
			free(buffer[0]);
		}
		if (buffer[1] != NULL)
		{
			free(buffer[1]);
		}
		free(buffer);
	}
	if (h == NULL || h->net == NULL || h->net->layers == NULL)
	{
		return LATTE_ERR_OUT_OF_MEM;
	}
	else
	{

		for (layer_index = 0; layer_index < h->net->number_of_layers; layer_index++)
		{
			current_layer = h->net->layers[layer_index];
			if (current_layer != NULL)
			{
				if (current_layer->nodes != NULL)
				{
					for (node_index = 0; node_index < current_layer->number_of_nodes; node_index++)
					{
						if (current_layer->nodes[node_index] != NULL)
						{

							if (current_layer->nodes[node_index]->weight != NULL)
							{
								free(current_layer->nodes[node_index]->weight);
							}
							free(current_layer->nodes[node_index]);
						}


					}
				}
				if (current_layer->connection_table != NULL)
				{
					free(current_layer->connection_table);
				}
				free(current_layer);
			}
		}
		free(h->net->layers);
		free(h->net);
		free(h);
	}
	return LATTE_SUCCESS;
}


