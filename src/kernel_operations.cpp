#include<stdio.h>
#include<math.h>
#include<string.h>
#include<stdbool.h>
#include "util.h"
#include "connection_handling.h"
#include "activation_function.h"

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
int conv_kernel(Layer *layer, int windowx, int windowy, int output_location, int *kernel_depth_connection,Layer *next_layer)
{
	float *kbuffer,*temp, *temp2;
	int node_index, windowz, offset_index, row_index,padding_offset,z,h,w;
	
	if (next_layer != NULL)
	{
		padding_offset = layer->ostride_w*next_layer->padding_size_h + next_layer->padding_size_w;
	}
	else
	{
		padding_offset = 0.0;
	}

	for (node_index = 0; node_index < layer->number_of_nodes; node_index++)	
	{
		temp = layer->op_buffer[node_index] + padding_offset + (output_location / layer->osize_w)*layer->ostride_w + (output_location%layer->osize_w);
		*(temp) = 0.0;
	}
/*	for (z = 0; z < layer->isize_d; z++)
	{
		for (h = 0; h < layer->ksize_h; h++)
		{
			for (w = 0; w < layer->ksize_w; w++)
			{
				printf("%f ", *(layer->ip_buffer[z] + h*layer->istride_w + w));
				
			}
			printf("\n");

		}
	}
	*/


	for (windowz = 0; windowz < layer->isize_d; windowz++)
	{
		
		for (offset_index = 0; offset_index < layer->ksize_h; offset_index++)
		{
			kbuffer = (layer->ip_buffer[windowz] + windowy*layer->istride_w + windowx +offset_index* layer->istride_w);
			for (node_index = 0; node_index < layer->number_of_nodes; node_index++)
			{
				if (is_connected(windowz, node_index, layer->number_of_nodes, layer->type_of_layer, layer->connection_table) == true) 
				{
					for (row_index = 0; row_index < layer->ksize_w; row_index++)
					{
						temp = layer->op_buffer[node_index] + padding_offset + (output_location / layer->osize_w)*layer->ostride_w + (output_location%layer->osize_w);
						temp2 = layer->nodes[node_index]->weight + kernel_depth_connection[node_index] * layer->ksize_w*layer->ksize_h + offset_index*layer->ksize_w + row_index;
						*(temp) = *(temp)+  *(temp2) * kbuffer[row_index];
					}
				}
			}


		}
		for (node_index = 0; node_index < layer->number_of_nodes; node_index++)
		{
			if (is_connected(windowz, node_index, layer->number_of_nodes, layer->type_of_layer, layer->connection_table) == true)
			{
				kernel_depth_connection[node_index]=kernel_depth_connection[node_index]+1;
			}
		

		}
	
	
	}
	for (node_index = 0; node_index < layer->number_of_nodes; node_index++)
	{
		temp = layer->op_buffer[node_index] + padding_offset + (output_location / layer->osize_w)*layer->ostride_w + (output_location%layer->osize_w);
	
		*(temp) = activation(layer->activation, (*(temp)+(layer->bias_applied == true)*layer->nodes[node_index]->bias), layer->negative_slope);
	
	}

	return LATTE_SUCCESS;
}

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
int maxpool_kernel(Layer *layer, int windowx, int windowy, int output_location, Layer *next_layer)
{
	float *kbuffer, *temp;
	int node_index, windowz, offset_index, row_index, padding_offset, z, h, w;
	if (next_layer != NULL)
	{
		padding_offset = layer->ostride_w*next_layer->padding_size_h + next_layer->padding_size_w;
	}
	else
	{
		padding_offset = 0;
	}

	/*	for (z = 0; z < layer->isize_d; z++)
	{
	for (h = 0; h < layer->ksize_h; h++)
	{
	for (w = 0; w < layer->ksize_w; w++)
	{
	printf("%f ", *(layer->ip_buffer[z] + h*layer->istride_w + w));

	}
	printf("\n");

	}
	}
	*/
	for (windowz = 0; windowz < layer->isize_d; windowz++)
	{
		temp = layer->op_buffer[windowz] + padding_offset + (output_location / layer->osize_w)*layer->ostride_w + (output_location%layer->osize_w);
		*(temp) = *(layer->ip_buffer[windowz] + windowy*layer->istride_w + windowx);
		for (offset_index = 0; offset_index < layer->ksize_h; offset_index++)
		{
			kbuffer = (layer->ip_buffer[windowz] + windowy*layer->istride_w + windowx + offset_index* layer->istride_w);

			for (row_index = 0; row_index < layer->ksize_w; row_index++)
			{
				//temp = layer->op_buffer[windowz] + padding_offset + (output_location / layer->osize_w)*layer->ostride_w + (output_location%layer->osize_w);
				if (kbuffer[row_index]>*(temp))
					*(temp) = kbuffer[row_index];
			}
			
	


		}

	}
	for (node_index = 0; node_index < layer->number_of_nodes; node_index++)
	{
		temp = layer->op_buffer[node_index] + padding_offset + (output_location / layer->osize_w)*layer->ostride_w + (output_location%layer->osize_w);
		*(temp) = activation(layer->activation, *(temp), layer->negative_slope);
	}
	return LATTE_SUCCESS;
}

/*Function to perform average pooling.
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
int avgpool_kernel(Layer *layer,int  windowx,int  windowy,int output_location,Layer *next_layer)
{
	float *kbuffer, *temp;
	int node_index, windowz, offset_index, row_index, padding_offset, z, h, w;
	if (next_layer != NULL)
	{
		padding_offset = layer->ostride_w*next_layer->padding_size_h + next_layer->padding_size_w;
	}
	else
	{
		padding_offset = 0;
	}

	/*	for (z = 0; z < layer->isize_d; z++)
	{
	for (h = 0; h < layer->ksize_h; h++)
	{
	for (w = 0; w < layer->ksize_w; w++)
	{
	printf("%f ", *(layer->ip_buffer[z] + h*layer->istride_w + w));

	}
	printf("\n");

	}
	}
	*/
	for (windowz = 0; windowz < layer->isize_d; windowz++)
	{
		temp = layer->op_buffer[windowz] + padding_offset + (output_location / layer->osize_w)*layer->ostride_w + (output_location%layer->osize_w);
		*(temp) = 0.0;
		for (offset_index = 0; offset_index < layer->ksize_h; offset_index++)
		{
			kbuffer = (layer->ip_buffer[windowz] + windowy*layer->istride_w + windowx + offset_index* layer->istride_w);
			//temp = layer->op_buffer[windowz] + padding_offset + (output_location / layer->osize_w)*layer->ostride_w + (output_location%layer->osize_w);

			for (row_index = 0; row_index < layer->ksize_w; row_index++)
			{
				*(temp) = *(temp)+kbuffer[row_index];
			}

			*(temp) = *(temp) / (layer->ksize_h*layer->ksize_w);


		}

	}
	for (node_index = 0; node_index < layer->number_of_nodes; node_index++)
	{
		temp = layer->op_buffer[node_index] + padding_offset + (output_location / layer->osize_w)*layer->ostride_w + (output_location%layer->osize_w);
		*(temp) = activation(layer->activation, *(temp), layer->negative_slope);
	}
	return LATTE_SUCCESS;
}




void transpose(float *matrix, int rows, int columns)
{
	int i, j,size;
	float temp;
	if (rows > columns)
		size = rows;
	else
		size = columns;
	for (i = 0; i < rows/2; i++)
	{
		for (j = 0; j < columns/2; j++)
		{
			temp = matrix[i*columns + j];
			matrix[i*columns + j] = matrix[j*columns + i];
			matrix[j*columns + i] = temp;
		}
	}
}