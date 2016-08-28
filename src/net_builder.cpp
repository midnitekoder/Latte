#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include "util.h"
#include "latte_config.h"
#include "latte_api.h"
#include "dimension_analysis.h"
#include "io_buffer_handling.h"
#include "forward_prop.h"





/*Function to initialize the network specific parameters based on the model_reader.
*Parameters:
**handle: handle to the network.
**config: The data read from the lattemodel using model_reader.
**param: parameters set by user to access input and output buffers.

*Return:
**status flag*/
int init_network(latte_handle *h, LATTE_config *config,PARAM param)
{

	int layer_index;

	Layer *current_layer = NULL;

	h->net = (Network*)MALLOC(sizeof(Network));
	if (h->net == NULL)
	{
		printf("Error in h->net");
		return LATTE_ERR_OUT_OF_MEM;
	}
	h->net->number_of_layers = config->network.number_of_layers;
	//h->net->ip_buffer = config->network.ip_buffer;
	h->net->isize_d = config->network.isize_d;
	h->net->isize_h = config->network.isize_h;
	h->net->isize_w = config->network.isize_w;
	h->net->istride_w = h->net->isize_w; // temporarily
	h->net->osize_d = param.osize_d;
	h->net->osize_h = param.osize_h;
	h->net->osize_w = param.osize_w;
	h->net->op_buffer = config->network.op_buffer;
	h->net->layers = (Layer**)MALLOC(sizeof(Layer*)*(h->net->number_of_layers+1));
	if (h->net->layers == NULL)
	{
		return LATTE_ERR_OUT_OF_MEM;
	}
	

	for (layer_index = 0; layer_index < h->net->number_of_layers; layer_index++)
	{
		h->net->layers[layer_index] = (Layer*)MALLOC(sizeof(Layer));
		if (h->net->layers[layer_index] == NULL)
		{
			return LATTE_ERR_OUT_OF_MEM;
		}
	}
	h->net->layers[layer_index] = NULL;/*To avoid segmentation faults during the inference phase.*/
	return LATTE_SUCCESS;
}



/*Function to initialize the layer specific parameters based on the model_reader.
This includes creating nodes of the layer and setting up the weights and biases.
*Parameters:
**handle: handle to the network-specific fields.
**config: The data read from the lattemodel using model_reader.
**layer_index: index of the  layer
**connection_table_h: the number of nodes in the previous layer.

*Return:
**status flag*/
int init_layer(Network *net, LATTE_config *config, int layer_index,int connection_table_h)
{
	int  node_index, status, name_length, index, weight_size;



	    Layer *current_layer = NULL;
	    current_layer = net->layers[layer_index];


		name_length = strlen(config->layer.layer_name);
		current_layer->layer_name = (char*)MALLOC(name_length + 1);

		strcpy(current_layer->layer_name, config->layer.layer_name);
		current_layer->activation = config->layer.activation;

		current_layer->type_of_layer = config->layer.type_of_layer;

		current_layer->padding_size_h = config->layer.padding_size_h;
		current_layer->padding_size_w = config->layer.padding_size_w;
		current_layer->padding_case = config->layer.padding_case;


		current_layer->number_of_nodes = config->layer.number_of_nodes;
		switch (current_layer->type_of_layer)
		{
		case LT_CONV:

			current_layer->ksize_h = config->layer.ksize_h;
			current_layer->ksize_w = config->layer.ksize_w;
			current_layer->k_stride_h = config->layer.k_stride_h;
			current_layer->k_stride_w = config->layer.k_stride_w;
			current_layer->nodes = (Node**)MALLOC(sizeof(Node*)*current_layer->number_of_nodes);
			if (current_layer->nodes == NULL)
				return LATTE_ERR_OUT_OF_MEM;
			if (config->layer.connection_table != NULL)
			{
				current_layer->connection_table = (bool*)malloc(sizeof(bool)*connection_table_h*current_layer->number_of_nodes);
				for (index = 0; index < connection_table_h*current_layer->number_of_nodes; index++)
					current_layer->connection_table[index] = config->layer.connection_table[index];
			}
			else
			{
				current_layer->connection_table = NULL;
			}

			current_layer->bias_applied = config->layer.bias_applied;

			status = get_input_dimension(net, layer_index);
			for (node_index = 0; node_index < current_layer->number_of_nodes; node_index++)
			{
				current_layer->nodes[node_index] = (Node*)MALLOC(sizeof(Node));
				{
					status = get_ksize_d(current_layer->isize_d, current_layer->type_of_layer, current_layer->connection_table, node_index, &(current_layer->nodes[node_index]->ksize_d), current_layer->number_of_nodes);

					weight_size = current_layer->ksize_h*current_layer->ksize_w*(current_layer->nodes[node_index]->ksize_d);
					if (weight_size != 0)
					{
						current_layer->nodes[node_index]->weight = (float*)MALLOC(weight_size*sizeof(float));
						memcpy(current_layer->nodes[node_index]->weight, config->nodes[node_index]->weight, weight_size*sizeof(float));

					}
					else
					{
						current_layer->nodes[node_index]->weight = NULL;
					}

					if (current_layer->bias_applied == true)
					{
						current_layer->nodes[node_index]->bias = config->nodes[node_index]->bias;
					}
					else
					{
						current_layer->nodes[node_index]->bias = 0.0;
					}
					
				}
			}


			break;
		case LT_MAXP:
		case LT_AVGP:
			current_layer->ksize_h = config->layer.ksize_h;
			current_layer->ksize_w = config->layer.ksize_w;
			current_layer->k_stride_h = config->layer.k_stride_h;
			current_layer->k_stride_w = config->layer.k_stride_w;

			current_layer->nodes = (Node**)MALLOC(sizeof(Node*)*(current_layer->number_of_nodes));
			if (current_layer->nodes == NULL)
				return LATTE_ERR_OUT_OF_MEM;
			current_layer->bias_applied = config->layer.bias_applied;
			current_layer->connection_table = NULL;
			status = get_input_dimension(net, layer_index);
			for (node_index = 0; node_index < current_layer->number_of_nodes; node_index++)
			{

				current_layer->nodes[node_index] = (Node*)MALLOC(sizeof(Node));




				status = get_ksize_d(current_layer->isize_d, current_layer->type_of_layer, current_layer->connection_table, node_index, &current_layer->nodes[node_index]->ksize_d, current_layer->number_of_nodes);
				current_layer->nodes[node_index]->weight = NULL;
				if (current_layer->bias_applied == true)
				{
					current_layer->nodes[node_index]->bias = config->nodes[node_index]->bias;
				}
				else
				{
					current_layer->nodes[node_index]->bias = 0.0;
				}
			}
			break;
		case LT_PART_CONN:
		case LT_FULL_CONN:

			//current_layer->padding_size_h = 0;
			//current_layer->padding_size_w = 0;
			//current_layer->padding_case = 0;
			//	current_layer->connection_table = config->layers[layer_index].connection_table;
//			current_layer->number_of_nodes = config->layers[layer_index].number_of_nodes;
			current_layer->nodes = (Node**)MALLOC(sizeof(Node*)*current_layer->number_of_nodes);
			if (current_layer->nodes == NULL)
				return LATTE_ERR_OUT_OF_MEM;
			current_layer->bias_applied = config->layer.bias_applied;
			if (config->layer.connection_table != NULL)
			{
				current_layer->connection_table = (bool*)malloc(sizeof(bool)*connection_table_h*current_layer->number_of_nodes);
				for (index = 0; index < connection_table_h*current_layer->number_of_nodes; index++)
				{
					current_layer->connection_table[index] = config->layer.connection_table[index];
				}
			}
			else
			{
				current_layer->connection_table = NULL;
			}
			status = get_input_dimension(net, layer_index);
			current_layer->ksize_h = current_layer->isize_h;
			current_layer->ksize_w = current_layer->isize_w;
			current_layer->k_stride_h = 0;
			current_layer->k_stride_w = 0;
			for (node_index = 0; node_index < current_layer->number_of_nodes; node_index++)
			{
				current_layer->nodes[node_index] = (Node*)MALLOC(sizeof(Node));
					status = get_ksize_d(current_layer->isize_d, current_layer->type_of_layer, current_layer->connection_table, node_index, &current_layer->nodes[node_index]->ksize_d, current_layer->number_of_nodes);
			//	current_layer->nodes[node_index]->ksize_d = config->nodes[node_index]->ksize_d;
				current_layer->nodes[node_index]->weight = (float*)MALLOC(current_layer->ksize_h*current_layer->ksize_w*current_layer->nodes[node_index]->ksize_d*sizeof(float));
					memcpy(current_layer->nodes[node_index]->weight, config->nodes[node_index]->weight, current_layer->ksize_h*current_layer->ksize_w*current_layer->nodes[node_index]->ksize_d*sizeof(float));
					if (current_layer->bias_applied == true)
					{
						current_layer->nodes[node_index]->bias = config->nodes[node_index]->bias;
					}
					else
					{
						current_layer->nodes[node_index]->bias = 0.0;
					}
			


			}

			break;
		case LT_DROPOUT:
			//current_layer->ksize_h = 0;
			//current_layer->ksize_w = 0;
			//current_layer->k_stride_h = 0;
			//current_layer->k_stride_w = 0;
			//current_layer->padding_size_h = 0;
			//current_layer->padding_size_w = 0;
			//current_layer->padding_case = 0;
			break;

		case LT_LRN_ACROSS:
			current_layer->ksize_h = 1;
			current_layer->ksize_w = 1;
			current_layer->k_stride_h = 1;
			current_layer->k_stride_w = 1;
			//current_layer->number_of_nodes = 1;
			current_layer->nodes = (Node**)MALLOC(sizeof(Node*)*current_layer->number_of_nodes);
			if (current_layer->nodes == NULL)
			{
				return LATTE_ERR_OUT_OF_MEM;
			}
			current_layer->connection_table = NULL;
			status = get_input_dimension(net, layer_index);
			//status = get_input_output_dimension(net, layer_index);
			for (node_index = 0; node_index < current_layer->number_of_nodes; node_index++)
			{
				current_layer->nodes[node_index] = (Node*)MALLOC(sizeof(Node));
					status = get_ksize_d(current_layer->isize_d, current_layer->type_of_layer, current_layer->connection_table, node_index, &current_layer->nodes[node_index]->ksize_d, current_layer->number_of_nodes);
				//current_layer->nodes[node_index]->ksize_d = config->nodes[node_index]->ksize_d;
				current_layer->nodes[node_index]->weight = (float*)MALLOC(current_layer->ksize_h*current_layer->ksize_w*current_layer->nodes[node_index]->ksize_d*sizeof(float));
					memcpy(current_layer->nodes[node_index]->weight, config->nodes[node_index]->weight, current_layer->ksize_h*current_layer->ksize_w*current_layer->nodes[node_index]->ksize_d*sizeof(float));
					/*weight[0]=n,weight[1]=alpha,weight[2]=beta,weight[3]=k*/


			}

			break;
		case LT_SOFTMAX:
//			current_layer->number_of_nodes = 0;

			current_layer->nodes = (Node**)MALLOC(sizeof(Node*)*current_layer->number_of_nodes);
			if (current_layer->nodes == NULL)
			{
				return LATTE_ERR_OUT_OF_MEM;
			}
			current_layer->connection_table = NULL;
			status = get_input_dimension(net, layer_index);
	
			current_layer->ksize_h = 1;
			current_layer->ksize_w = 1;

			break;

		default:
			return LATTE_ERR_INVALID_CONFIG;
			break;


		}

	
		return LATTE_SUCCESS;

	}




