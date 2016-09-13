#include <malloc.h>
#include <stdio.h>
#include "latte_config.h"
#include "util.h"

/*Function to set input and output buffer points to their appropriate location in the two buffers such 
once initialized, it does not change till the same network is used for forward propagation.
*Parameters:
** layers: pointer to the list of all the layers
**buffer: pointer to the two buffer

*Return:
**status flag
*/
int set_io_channels(Layer **layers, int number_of_layers, float **buffer)
{
	int layer_index, channel_index;
	Layer *current_layer;
	for (layer_index = 0; layer_index < number_of_layers; layer_index++)
	{
		current_layer = layers[layer_index];
		current_layer->op_buffer = (float**)MALLOC(sizeof(float*)*current_layer->osize_d);
		current_layer->ip_buffer = (float**)MALLOC(sizeof(float*)*current_layer->isize_d);
		
		for (channel_index = 0; channel_index < current_layer->osize_d; channel_index++)
		{
			if (layers[layer_index + 1] != NULL)
			{
				current_layer->op_buffer[channel_index] = buffer[(layer_index + 1) % 2] + channel_index*(current_layer->osize_h+2*layers[layer_index+1]->padding_size_h)*current_layer->ostride_w;
			}
			else
			{
				current_layer->op_buffer[channel_index] = buffer[(layer_index + 1) % 2] + channel_index*current_layer->osize_h*current_layer->ostride_w;

			}
		
			}
		for (channel_index = 0; channel_index < current_layer->isize_d; channel_index++)
		{
			current_layer->ip_buffer[channel_index] = buffer[(layer_index % 2)] + channel_index*current_layer->isize_h*current_layer->istride_w;
		}
	}
	return LATTE_SUCCESS;
}


void print_output(Layer *layer)
{
	int d, h, w;
	FILE *fpp;
	//fopen_s(&fpp,"output.txt", "w");
	fpp=fopen("output.txt", "w");
	for (d = 0; d < layer->osize_d; d++)
	{
		for (h = 0; h < layer->osize_h; h++)
		{
			for (w = 0; w < layer->ostride_w; w++)
			{
				fprintf(fpp,"%.10f ", *(layer->op_buffer[d] + h*layer->ostride_w + w));
			}
			fprintf(fpp,"\n");
		}
	}
	fclose(fpp);
}

void print_input(Layer *layer)
{
	int d, h, w;
	FILE *fpp;
	FILE *fp2;
	
	//fopen_s(&fpp, "input.txt", "w");
	fpp=fopen("input.txt", "w");
	for (d = 0; d < layer->isize_d; d++)
	{
		for (h = 0; h < layer->isize_h; h++)
		{
			for (w = 0; w < layer->istride_w; w++)
			{
				fprintf(fpp,"%.10f ", *(layer->ip_buffer[d] + h*layer->istride_w + w));
			}
			fprintf(fpp,"\n");
		}
	}
	fclose(fpp);
	if (layer->connection_table != NULL)
	{
		//fopen_s(&fp2, "connection.txt", "w");
		fp2=fopen( "connection.txt", "w");
		for (h = 0; h < layer->isize_d; h++)
		{
			for (w = 0; w < layer->number_of_nodes; w++)
			{
				fprintf(fp2, "%d ", layer->connection_table[h*layer->number_of_nodes + w]);
			}
			fprintf(fp2, "\n");
		}
		fclose(fp2);
	}
}



/*Function to copy input to the buffers of the framework from the location where the user has specified.
*Parameters:
**h: handle
**param: data structure specifying the location and size of the input and output.

*Return:
status flag
*/
int copy_input(latte_handle *h, PARAM param)
{
	int input_channel_index, planex,planey,padding_offset;

	Network *net;
	net = h->net;
	
	net->ip_buffer = (float**)MALLOC(sizeof(float*)*net->isize_d);
	for (input_channel_index = 0; input_channel_index < net->isize_d; input_channel_index++)
	{
		net->ip_buffer[input_channel_index] = buffer[0]+input_channel_index*net->layers[0]->isize_w*net->layers[0]->isize_h;
	}
	if (net->ip_buffer == NULL)
		return LATTE_ERR_INVALID_IO_PARAM;
	padding_offset = net->layers[0]->padding_size_h*net->layers[0]->istride_w + net->layers[0]->padding_size_w;
	for (input_channel_index = 0; input_channel_index < net->isize_d; input_channel_index++)
	{
		if (net->ip_buffer[input_channel_index] == NULL)
			return LATTE_ERR_INVALID_IO_PARAM;
		for (planey = 0; planey < net->isize_h; planey++)
		{
			for (planex = 0; planex < net->isize_w; planex++)
			{
				
				*(net->ip_buffer[input_channel_index] +padding_offset+ planey*net->layers[0]->istride_w+planex) = (*(param.ip_buffer[input_channel_index] + planey*net->istride_w+planex));

			}
		}
		
	}
	return LATTE_SUCCESS;
}


/*Function to copy output from the buffers of the framework to the location where the user has specified.
*Parameters:
**h: handle
**param: data structure specifying the location and size of the input and output.

*Return:
status flag
*/
int copy_output(latte_handle *h, PARAM param)
{
	int output_channel_index, data_copy_index, data_copy_size;
	data_copy_size = param.osize_h*param.osize_w;
	h->net->op_buffer = (float**)MALLOC(sizeof(float*)*h->net->osize_d);
	for (output_channel_index = 0; output_channel_index < h->net->osize_d; output_channel_index++)
	{
		h->net->op_buffer[output_channel_index] = h->net->layers[h->net->number_of_layers-1]->op_buffer[output_channel_index];
	}
	if (h->net->op_buffer == NULL)
		return LATTE_ERR_INVALID_IO_PARAM;
	for (output_channel_index = 0; output_channel_index < h->net->osize_d; output_channel_index++)
	{
		if (param.op_buffer[output_channel_index] == NULL)
			return LATTE_ERR_INVALID_IO_PARAM;
		for (data_copy_index = 0; data_copy_index < data_copy_size; data_copy_index++)
		{
			*(param.op_buffer[output_channel_index] + data_copy_index) = *(h->net->op_buffer[output_channel_index] + data_copy_index);

		}
	}
	return LATTE_SUCCESS;

}

/*Function to boundary pad the input to the layer
*Parameter:
**layer: layer specific data
*/
void pad_layer_input(Layer *layer)
{
	int idepth,planey,planex,index;
	float padding_element;
	switch (layer->padding_case)
	{
	case NO_PADDING:
		break;
	case ZERO_PADDING:
		padding_element = 0.0;
		for (idepth = 0; idepth < layer->isize_d; idepth++)
		{
			for (index = 0; index < (layer->padding_size_h*layer->istride_w + layer->padding_size_w);index++)
			{
				*(layer->ip_buffer[idepth] + index) = padding_element;
			}
			planex = layer->istride_w - layer->padding_size_w;
			for (planey = layer->padding_size_h; planey < (layer->isize_h - layer->padding_size_h); planey++)
			{
				for (index = 0; index < (2 * layer->padding_size_w); index++)
				{
					*(layer->ip_buffer[idepth] + planey*layer->istride_w + planex+index) = padding_element;
				}
			}
			for (index = 0; index < (layer->padding_size_h*layer->istride_w + layer->padding_size_w); index++)
			{
				*(layer->ip_buffer[idepth] + (layer->isize_h - layer->padding_size_h)*layer->istride_w - layer->padding_size_w + index) = padding_element;
			}
		}
		break;
	default:
		padding_element = 0;
	}
	

}
