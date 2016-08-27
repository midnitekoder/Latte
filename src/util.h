#ifndef _2016_latte_util_

#define _2016_latte_util_

#include <stdbool.h>
#include "latte_config.h"


/*different types of layers*/
enum layer_type{
	LT_AVGP,      /*layer type: average pooling*/
	LT_CONV,      /*layer type: convolutional layer*/
	LT_DROPOUT,    /*layer type: dropout layer*/
	LT_FULL_CONN,  /*layer type: fully connected layer*/
	LT_LRN_ACROSS,  /*layer type: Local Response Normalization layer(across)*/
	LT_MAXP,      /*layer type: maxpooling layer*/
	LT_PART_CONN,  /*layer type: partially connected layer*/
	LT_SOFTMAX     /*layer type: Softmax layer*/
};

enum activation_type{
	AF_NOACTIVATION, /*In case no activation is applied on the layer*/
	AF_RELU,       /*activation function: ReLU*/
	AF_SIGM,      /*activation function: sigmoid*/
	AF_TANH,      /*activation function: tanh*/
	AF_LEAKY_RELU,  /*activation function: Leaky ReLU*/
	AF_ELU         /*activation function: ELU*/
};

enum padding_type{
	NO_PADDING,
	ZERO_PADDING
};

enum error_type{
	LATTE_SUCCESS=1,
	LATTE_ERR_INVALID_CONFIG=2,
	LATTE_ERR_OUT_OF_MEM=3,
	LATTE_ERR_INVALID_IO_PARAM=4,
	LATTE_ERR_DIMENSION_INCOMPATIBLE=5
};


extern float **buffer;

/*handle data specific to a node*/
struct Node{
	float *weight;      
	float bias;         
	int ksize_d;        
};



struct Layer{
	char *layer_name;                 
	layer_type type_of_layer;        
	float **ip_buffer;                
	int isize_w;                     
	int isize_h;                     
	int	isize_d;                     
	int	istride_w;                   
	float **op_buffer;                
	int osize_w;                     
	int osize_h;                     
	int osize_d;                     
	int ostride_w;
	int ksize_h;                     
	int ksize_w;                     
	int k_stride_w;                  
	int k_stride_h;                  
	padding_type padding_case;       
	int padding_size_w;              
	int padding_size_h;              
	activation_type activation;      
	int number_of_nodes;             
	Node **nodes;                    
	bool *connection_table;          
	bool bias_applied;               
	float negative_slope;            
	
};



struct Network{

	float **ip_buffer;              
	int isize_w;                   
	int isize_h;                   
	int	isize_d;                   
	int	istride_w;                 
	float **op_buffer;              
	int osize_w;                   
	int osize_h;                   
	int osize_d;                   

	int number_of_layers;          
	Layer **layers;                

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


struct PARAM{
	float **ip_buffer;              
	int isize_w;                   
	int isize_h;                  
	int	isize_d;                   
	int	istride_w;                 
	float **op_buffer;              
	int osize_w;                  
	int osize_h;                   
	int osize_d;                   

};



#endif
