#ifndef _2016_latte_api_

#define _2016_latte_api_


#include <stdio.h>
#include "latte_config.h"
#include "util.h"



/*Function to create a handle of the network model
*Parameters:
void

*Return:
**handle of the network model*/
latte_handle* latte_create(void);


/*Function to initalize the network model by parsing the lattemodel file.
*Parameters:
**handle: handle of the network model
**param: parameters set by user to access input and output buffers.

Return:
status flag
*/
int latte_parse_and_init(latte_handle *handle,FILE* LATTE_FILE,PARAM param);  



/*Function to forward propagate the network based on the design of the network model.
*Parameters:
**handle: handle of the network model
**param: parameters set by user to access input and output buffers.

Return:
status flag
*/
int latte_forward_prop(latte_handle* handle, PARAM param);        


/*Function to release all the resources allocated to the network model.
*Parameters:
**handle: handle of the network model

Return:
status flag
*/
int latte_close(latte_handle* handle);                


#endif 
