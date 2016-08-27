#ifndef _2016_latte_util_

#define _2016_latte_util_

#include <stdbool.h>
#include "latte_config.h"


/*different types of layers*/
enum layer_type{
	LT_AVGP,      	/*layer type: average pooling*/
	LT_CONV,      	/*layer type: convolutional layer*/
	LT_DROPOUT,    	/*layer type: dropout layer*/
	LT_FULL_CONN,  	/*layer type: fully connected layer*/
	LT_LRN_ACROSS,  /*layer type: Local Response Normalization layer(across)*/
	LT_MAXP,      	/*layer type: maxpooling layer*/
	LT_PART_CONN,  	/*layer type: partially connected layer*/
	LT_SOFTMAX     	/*layer type: Softmax layer*/
};

/*different activation functions that are supported. 
Activation layer is combined with the previous layer.*/
enum activation_type{
	AF_NOACTIVATION,/*In case, no activation is applied on the layer*/
	AF_RELU,       	/*activation function: ReLU*/
	AF_SIGM,      	/*activation function: sigmoid*/
	AF_TANH,      	/*activation function: tanh*/
	AF_LEAKY_RELU,  /*activation function: Leaky ReLU*/
	AF_ELU         	/*activation function: ELU*/
};

/*boundary padding options.*/
enum padding_type{
	NO_PADDING,
	ZERO_PADDING
};

/*error codes*/
enum error_type{
	LATTE_SUCCESS=1,               		/*The block/method ran successfully.*/
	LATTE_ERR_INVALID_CONFIG=2,    		/*Error in specifying parameters of latte_config.h file.*/
	LATTE_ERR_OUT_OF_MEM=3,				/*Memory Error*/
	LATTE_ERR_INVALID_IO_PARAM=4,		/*Error in specifying the Input/Output buffers or dimensions. */
	LATTE_ERR_DIMENSION_INCOMPATIBLE=5	/*Input image dimensions are not in accordance with the trained network.*/
};

/*buffers used to store intermediate outputs of the layers.*/
extern float **buffer;

/*handle data specific to a node*/
struct Node{
	float *weight;		/*array of weights of the node*/
	float bias;         /*bias appied to the multiplication and accumulation function. In case, no bias applied, bias=0. */
	int ksize_d;        /*depth of the kernel of the node.*/
};


/*handle data specific to a layer*/
struct Layer{
	char *layer_name;			/*name of the layer*/                 
	layer_type type_of_layer;	/*type of layer*/        
	float **ip_buffer;          /*pointer to the array of channels of input to the layer. Prefer to keep all buffers contiguous.*/      
	int isize_w;                /*width of the input*/     
	int isize_h;                /*height of the input*/     
	int	isize_d;                /*depth of the input*/     
	int	istride_w;              /*stride applied along the width to shift by a row in 2-dimensional plane.*/     
	float **op_buffer;          /*pointer to the array of channels of output to the layer. Prefer to keep all buffers contiguous.*/     
	int osize_w;                /*width of the output*/    
	int osize_h;                /*height of the output*/   
	int osize_d;                /*depth of the output*/     
	int ostride_w;				/*stride applied along the width to shift by a row in 2-dimensional plane.*/ 
	int ksize_h;                /*height of the kernel*/    
	int ksize_w;                /*width of the kernel*/     
	int k_stride_w;             /*stride applied along the width while processing a plane.*/    
	int k_stride_h;             /*stride applied along the height while processing a plane.*/     
	padding_type padding_case;  /*type of boundary padding applied on the input of the layer*/     
	int padding_size_w;         /*width of padding along the boundary.*/     
	int padding_size_h;         /*height of padding along the boundary.*/     
	activation_type activation; /*type of activation layer after this layer or activation applied on this layer.*/    
	int number_of_nodes;        /*number of nodes in the layer.*/    
	Node **nodes;               /*pointer to array of nodes.*/     
	bool *connection_table;     /*2-d matrix such that (i,j) represents if ith node of previous layer is connected to the jth node of current layer.*/     
	bool bias_applied;          /*is bias applied to each node of the layer*/    
	float negative_slope;       /*for leakyRelu...code going to change, soon.*/     
	
};


/*handle data spefic to the Network*/
struct Network{

	float **ip_buffer;          /*pointer to the array of channels of input to the network. Prefer to keep all buffers contiguous.*/      
	int isize_w;                /*width of the input*/     
	int isize_h;                /*height of the input*/     
	int	isize_d;                /*depth of the input*/     
	int	istride_w;              /*stride applied along the width to shift by a row in 2-dimensional plane.*/     
	float **op_buffer;          /*pointer to the array of channels of output to the network. Prefer to keep all buffers contiguous.*/     
	int osize_w;                /*width of the output*/    
	int osize_h;                /*height of the output*/   
	int osize_d;                /*depth of the output*/     
                  

	int number_of_layers;       /*number of layers in the network*/  
	Layer **layers;             /*pointer to array of layers.*/   

};


/*Temporary*/
struct LATTE_config{

	Network network;
	Layer layer;
	Node **nodes;


};

struct latte_handle{
	Network *net;
};



/*Parameters specified by the user for input and output buffers.*/
struct PARAM{
	float **ip_buffer;          /*pointer to the array of channels of input to the network. Prefer to keep all buffers contiguous.*/      
	int isize_w;                /*width of the input*/     
	int isize_h;                /*height of the input*/     
	int	isize_d;                /*depth of the input*/     
	int	istride_w;              /*stride applied along the width to shift by a row in 2-dimensional plane.*/     
	float **op_buffer;          /*pointer to the array of channels of output to the network. Prefer to keep all buffers contiguous.*/     
	int osize_w;                /*width of the output*/    
	int osize_h;                /*height of the output*/   
	int osize_d;                /*depth of the output*/     
};



#endif
