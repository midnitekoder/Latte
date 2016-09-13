#ifndef _2016_latte_forward_prop_

#define _2016_latte_forward_prop_

#include "util.h"


int forward_propagate_CONV(Layer *layer, Layer *next_layer);


int forward_propagate_MAXP(Layer *layer,Layer *next_layer);


int forward_propagate_LRN_ACROSS(Layer *layer, Layer *next_layer);


int forward_propagate_FULL_CONN(Layer *layer, Layer *next_layer);


int forward_propagate_AVGP(Layer *layer, Layer *next_layer);
	

int forward_propagate_SOFTMAX(Layer *layer, Layer *next_layer);
	

int forward_propagate_PART_CONN(Layer *layer, Layer *next_layer);


int forward_propagate_DROPOUT(Layer *layer, Layer *next_layer);
			

#endif 
