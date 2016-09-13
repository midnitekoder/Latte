#include<stdio.h>
#include<math.h>
#include<string.h>
#include<stdbool.h>
#include <malloc.h>
#include "util.h"
#include "connection_handling.h"
#include "kernel_operations.h"
#include "activation_function.h"
#include "io_buffer_handling.h"

int maxint(int num1, int num2)
{
	if (num1 > num2)
		return num1;
	else
		return num2;
}


int forward_propagate_CONV(Layer *layer, Layer *next_layer)
{
	int windowx, windowy, output_location, status,i,j;
	int *kernel_depth_connection;

	
	pad_layer_input(layer);
	print_input(layer);
	if (layer->connection_table != NULL)
	{
		for (i = 0; i < 10; i++)
		{
			for (j = 0; j < layer->number_of_nodes; j++)
			{
				printf("%d ", layer->connection_table[i*layer->number_of_nodes + j]);
			}
			printf("\n");
		}
	}

		output_location = 0;
		for (windowy = 0; (windowy + layer->ksize_h - 1) < layer->isize_h; windowy = windowy + layer->k_stride_h)
		{
			for (windowx = 0; (windowx + layer->ksize_w - 1) < layer->isize_w; windowx = windowx + layer->k_stride_w)
			{
				//memset(kernel_depth_connection, 0, layer->number_of_nodes*sizeof(int));
				for (i = 0; i < layer->number_of_nodes; i++)
					kernel_depth_connection[i] = 0;
				status= conv_kernel(layer, windowx,  windowy, output_location,kernel_depth_connection,next_layer);
				output_location++;
			}
		}
		free(kernel_depth_connection);
	return LATTE_SUCCESS;

	//free(kbuffer);
}


/*The function forward propagates for maxpooling layer.
The implementation is as follows:
1. The evaluation happens plane by plane for the input.
2. In each plane, for each row, the max is calculated for each window of size ksize_w, horizontally and then appropriate stride k_stride_w and it is stored in the output buffer one after the another.
3. the output matrix is transposed so that evaluation vertically can still be done in row major order as in step 2.
4. The transposed matrix is evaluated as in step 2 for window of size ksize_h, horizontally and stride k_stride_h and the result is stored in the output matrix, itself one after the another.
5. The output matrix is transposed again.

Parameters:
layer: handle to layer specific data.
Return:
status: status flag
*/
/*int forward_propagate_MAXP(Layer *layer, Layer *next_layer)
{
	int prev_layer_node_index,  windowx, windowy, output_location,row_index,padding_offset,trans_padw,trans_padh,trans_ostride_w;
	float max, *temp;
	pad_layer_input(layer);
		for (prev_layer_node_index = 0; prev_layer_node_index < layer->isize_d; prev_layer_node_index++)
		{
			output_location = 0;
			for (windowy = 0; windowy < layer->isize_h; windowy++)
			{
				for (windowx = 0; (windowx + layer->ksize_w - 1) < layer->isize_w; windowx = windowx + layer->k_stride_w)
				{
					max = *(layer->ip_buffer[prev_layer_node_index] + (windowy*layer->istride_w + windowx)); //since some numbers are negative.
					for (row_index = 0; row_index < layer->ksize_w; row_index++)
					{
						temp = layer->ip_buffer[prev_layer_node_index] + (windowy*layer->istride_w + windowx) + row_index;
						if (*(temp)>max)
						{
							max = *(temp);
						}
					}
					*(layer->ip_buffer[prev_layer_node_index] + output_location) = max;
					output_location++;
				}
			}
			
			transpose(layer->ip_buffer[prev_layer_node_index], layer->isize_h, layer->osize_w);
			output_location = 0;
			if (next_layer != NULL)
			{
				trans_padw = next_layer->padding_size_h;//note that padding height and width are swapped for transpose case
				trans_padh = next_layer->padding_size_w;
			}
			else
			{
				trans_padw = 0;
				trans_padh = 0;
			}
			trans_ostride_w = layer->osize_h + 2 * trans_padw;
			padding_offset = trans_padh*(trans_ostride_w)+trans_padw;
			for (windowy = 0; windowy <layer->osize_w ; windowy++)
			{
				for (windowx = 0; (windowx + layer->ksize_h - 1) < layer->isize_h; windowx = windowx + layer->k_stride_h)
				{
					max = *(layer->ip_buffer[prev_layer_node_index] + (windowy*layer->isize_h + windowx));
					for (row_index = 0; row_index < layer->ksize_h; row_index++)
					{
						temp = layer->ip_buffer[prev_layer_node_index] + (windowy*layer->isize_h + windowx) + row_index;
						if (*(temp)>max)
						{
							max = *(temp);
						}
					}
						*(layer->op_buffer[prev_layer_node_index] +padding_offset+ (output_location/layer->osize_h)*trans_ostride_w+(output_location%layer->osize_h)) =activation( layer->activation,max,layer->negative_slope);
						output_location++;
					
				}
			}
			transpose(layer->op_buffer[prev_layer_node_index], layer->ostride_w,trans_ostride_w);

		}
	




	return CDNN_SUCCESS;

}
*/


int forward_propagate_MAXP(Layer *layer, Layer *next_layer)
{
	int windowx, windowy, output_location, status;
	/*float *kbuffer;
	max_k_length = max(layer->ksize_h, layer->ksize_w);
	kbuffer = (float*)malloc(sizeof(float)*max_k_length);
	*/
	/*In case of tiling just add tile offset to windowx and window y and their corresponding conditions in the loops below.*/
	pad_layer_input(layer);
	print_input(layer);
	output_location = 0;
	for (windowy = 0; (windowy + layer->ksize_h - 1) < layer->isize_h; windowy = windowy + layer->k_stride_h)
	{
		for (windowx = 0; (windowx + layer->ksize_w - 1) < layer->isize_w; windowx = windowx + layer->k_stride_w)
		{

			status = maxpool_kernel(layer, windowx, windowy, output_location, next_layer);
			output_location++;
		}
	}

	return LATTE_SUCCESS;

	//free(kbuffer);
}



/*the parameters n, k, alpha, beta are taken from the AlexNet paper.*/
int forward_propagate_LRN_ACROSS(Layer *layer, Layer *next_layer)
{
	int n,prev_layer_node_index,x,y,odepth,padding_offset;
	float k, alpha, beta,roll_sum,axy,alpha_by_n,*temp;/*weight[0]=n,weight[1]=alpha,weight[2]=beta,weight[3]=k*/
	n = (int)layer->nodes[0]->weight[0];
	alpha = layer->nodes[0]->weight[1];
	beta = layer->nodes[0]->weight[2];
	k = layer->nodes[0]->weight[3];
	/*special attention alpha here is in accordance with caffe, tinycnn and not alexnet paper. 
	Alexnet_alpha=alpha/n*/
	alpha_by_n = alpha / n;
	//alpha_by_n = alpha;
	if (next_layer != NULL)
	{
		padding_offset = layer->ostride_w*next_layer->padding_size_h + next_layer->padding_size_w;
	}
	else
	{
		padding_offset = 0;
	}
	pad_layer_input(layer);
	print_input(layer);
	for (y = 0; y < layer->isize_h; y++)
	{
		for (x = 0; x < layer->isize_w; x++)
		{
			roll_sum = 0.0;
			for (prev_layer_node_index = 0; prev_layer_node_index < (n / 2); prev_layer_node_index++)
			{
				axy = *(layer->ip_buffer[prev_layer_node_index] + y*layer->istride_w + x);
				roll_sum = roll_sum + axy*axy;

			}
			for (odepth = 0; odepth < layer->isize_d; odepth++)
			{
				if (odepth + (n / 2) < layer->isize_d)
				{
					axy = *(layer->ip_buffer[odepth + (n / 2)] + y*layer->istride_w + x);
					roll_sum = roll_sum + axy*axy;
				}
				if (odepth - (n/2)>0)
				{
					axy = *(layer->ip_buffer[odepth - (n / 2)-1] + y*layer->istride_w + x);
					roll_sum = roll_sum - axy*axy;
				}
				temp = layer->op_buffer[odepth] + padding_offset + y*layer->ostride_w + x;
				*(temp) = activation(layer->activation,(*(layer->ip_buffer[odepth] + y*layer->istride_w + x)*(float)pow((roll_sum*alpha_by_n+k), -beta)),layer->negative_slope);
			}
			
		}
	}




	return LATTE_SUCCESS;
}


int forward_propagate_FULL_CONN(Layer *layer, Layer *next_layer)
{
	int windowx, windowy, output_location, status;
	/*float *kbuffer;
	max_k_length = max(layer->ksize_h, layer->ksize_w);
	kbuffer = (float*)malloc(sizeof(float)*max_k_length);
	*/
	int *kernel_depth_connection;
	kernel_depth_connection = (int*)MALLOC(sizeof(int)*layer->number_of_nodes);
	pad_layer_input(layer);
	windowx = 0;
	windowy = 0;

		output_location = 0;
		memset(kernel_depth_connection, 0, layer->number_of_nodes*sizeof(int));
		status = conv_kernel(layer, windowx, windowy, output_location,kernel_depth_connection,next_layer);
		free(kernel_depth_connection);
		return status;
}

/*The function forward propagates for average pooling layer.
The implementation is as follows:
1. The evaluation happens plane by plane for the input.
2. In each plane, for each row, the sum is accumulated for each window of size ksize_w, horizontally and then appropriate stride k_stride_w and it is stored in the output buffer one after the another.
3. the output matrix is transposed so that evaluation vertically can still be done in row major order as in step 2.
4. The transposed matrix is evaluated as in step 2 for window of size ksize_h, horizontally and stride k_stride_h and the result is stored in the output matrix, itself one after the another.
5. Each output is divided by the size of the window.
6. The output matrix is transposed again.

Parameters:
layer: handle to layer specific data.
Return:
status: status flag
*/
/*
int forward_propagate_AVGP(Layer *layer, Layer *next_layer)
{
	int prev_layer_node_index, windowx, windowy, output_location, row_index, trans_padw, trans_padh, trans_ostride_w,padding_offset;
	float sum;
	pad_layer_input(layer);
		for (prev_layer_node_index = 0; prev_layer_node_index < layer->isize_d; prev_layer_node_index++)
		{
			output_location = 0;
			for (windowy = 0; windowy < layer->isize_h; windowy++)
			{
				for (windowx = 0; (windowx + layer->ksize_w - 1) < layer->isize_w; windowx = windowx + layer->k_stride_w)
				{
					sum = 0.0;
					for (row_index = 0; row_index < layer->ksize_w; row_index++)
					{

							sum+= *(layer->ip_buffer[prev_layer_node_index] + (windowy*layer->istride_w + windowx) + row_index);
	
					}
					*(layer->ip_buffer[prev_layer_node_index] + output_location) = sum;
					output_location++;
				}
			}

			transpose(layer->ip_buffer[prev_layer_node_index], layer->isize_h, layer->osize_w);
			output_location = 0;
			if (next_layer != NULL)
			{

				trans_padw = next_layer->padding_size_h;//note that padding height and width are swapped for transpose case
				trans_padh = next_layer->padding_size_w;
			}
			else
			{
				trans_padw = 0;
				trans_padh = 0;
			}
			trans_ostride_w = layer->osize_h + 2 * trans_padw;
			padding_offset = trans_padh*(trans_ostride_w)+trans_padw;
			for (windowy = 0; windowy <layer->osize_w; windowy++)
			{
				for (windowx = 0; (windowx + layer->ksize_h - 1) < layer->isize_h; windowx = windowx + layer->k_stride_h)
				{
					sum = 0.0;
					for (row_index = 0; row_index < layer->ksize_h; row_index++)
					{
							sum += *(layer->ip_buffer[prev_layer_node_index] + (windowy*layer->isize_h + windowx) + row_index);
					}
					*(layer->op_buffer[prev_layer_node_index] + padding_offset + (output_location / layer->osize_h)*trans_ostride_w + (output_location%layer->osize_h)) =activation(layer->activation,( sum / (layer->ksize_h*layer->ksize_w)),layer->negative_slope);
					output_location++;
				}
			}
			transpose(layer->op_buffer[prev_layer_node_index], layer->ostride_w, trans_ostride_w);

		}
	




	return CDNN_SUCCESS;

}
*/


int forward_propagate_AVGP(Layer *layer, Layer *next_layer)
{
	int windowx, windowy, output_location, status;

	/*In case of tiling just add tile offset to windowx and window y and their corresponding conditions in the loops below.*/
	pad_layer_input(layer);

	output_location = 0;
	for (windowy = 0; (windowy + layer->ksize_h - 1) < layer->isize_h; windowy = windowy + layer->k_stride_h)
	{
		for (windowx = 0; (windowx + layer->ksize_w - 1) < layer->isize_w; windowx = windowx + layer->k_stride_w)
		{

			status = avgpool_kernel(layer, windowx, windowy, output_location, next_layer);
			output_location++;
		}
	}

	return LATTE_SUCCESS;

	//free(kbuffer);
}


int forward_propagate_SOFTMAX(Layer *layer, Layer *next_layer)
{
	int prev_layer_idepth, planex, planey,node_index,padding_offset;
	float max, denominator;
	pad_layer_input(layer);
	max = 0;
	for (prev_layer_idepth = 0; prev_layer_idepth < layer->isize_d; prev_layer_idepth++)
	{
		for (planey = 0; planey < layer->isize_h; planey++)
		{
			for (planex = 0; planex < layer->isize_w; planex++)
			{
				if (max < *(layer->ip_buffer[prev_layer_idepth] + planey*layer->istride_w + planex))
					max = *(layer->ip_buffer[prev_layer_idepth] + planey*layer->istride_w + planex);

			}
		}
	}
	if (next_layer != NULL)
	{
		padding_offset = layer->ostride_w*next_layer->padding_size_h + next_layer->padding_size_w;
	}
	else
	{
		padding_offset = 0;
	}
	denominator = 0;
	for (node_index = 0; node_index < layer->isize_d; node_index++)
	{
		for (planey = 0; planey < layer->isize_h; planey++)	
		{
			for (planex = 0; planex < layer->isize_w; planex++)
			{
				*(layer->op_buffer[node_index] +padding_offset+ planey*layer->ostride_w + planex)=(float)exp((*(layer->ip_buffer[node_index] + planey*layer->istride_w + planex) - max));		
				denominator = denominator + *(layer->op_buffer[node_index] + padding_offset + planey*layer->ostride_w + planex);
			}
		}
	}
	for (node_index = 0; node_index < layer->isize_d; node_index++)
	{
		for (planey = 0; planey < layer->isize_h; planey++)
		{
			for (planex = 0; planex < layer->isize_w; planex++)
			{
				*(layer->op_buffer[node_index] + padding_offset + planey*layer->ostride_w + planex) = activation(layer->activation,(*(layer->op_buffer[node_index] + padding_offset + planey*layer->ostride_w + planex)/denominator),layer->negative_slope);
				
			}
		}
	}

	return LATTE_SUCCESS;
}



int forward_propagate_PART_CONN(Layer *layer, Layer *next_layer)
{
	int windowx, windowy, output_location, status;

	
	int *kernel_depth_connection;
	kernel_depth_connection = (int*)MALLOC(sizeof(int)*layer->number_of_nodes);
	pad_layer_input(layer);
	windowx = 0;
	windowy = 0;

		output_location = 0;
		memset(kernel_depth_connection, 0, layer->number_of_nodes*sizeof(int));
		status = conv_kernel(layer, windowx, windowy, output_location,kernel_depth_connection,next_layer);

		free(kernel_depth_connection);
		return status;
}



int forward_propagate_DROPOUT(Layer *layer, Layer *next_layer)
{
	int input_size;
	pad_layer_input(layer);
	input_size = layer->isize_d*layer->isize_h*layer->istride_w;
	memcpy(layer->op_buffer, layer->ip_buffer, input_size*sizeof(float));
	return LATTE_SUCCESS;
}
