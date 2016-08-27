#ifndef _2016_latte_api_

#define _2016_latte_api_


#include <stdio.h>
#include "latte_config.h"
#include "util.h"




latte_handle* latte_create(void);

/

int latte_parse_and_init(latte_handle *handle,FILE* LATTE_FILE,PARAM param);  




int latte_forward_prop(latte_handle* handle, PARAM param);        


int latte_close(latte_handle* h);                


#endif 
