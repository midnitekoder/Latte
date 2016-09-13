#include<stdio.h>
#include<malloc.h>
#include "latte_api.h"
#include "util.h"
#include "latte_config.h"
int main()
{

	latte_handle *handle;
	PARAM param;
	FILE *fp=NULL;
	FILE *imagePointer = NULL;
	int status,i;
	float* input_image;
	
	



	
	input_image = (float*)malloc(28 * 28 * 1 * sizeof(float));
	param.ip_buffer = (float**)malloc(sizeof(float*)* 3);
	param.ip_buffer[0] = input_image;
	//status = fopen_s(&imagePointer, "seven.latteinput", "rb");
	imagePointer = fopen("seven.latteinput", "rb");
	int read_elements = fread(param.ip_buffer[0], sizeof(float), 28 * 28 * 1, imagePointer);
	param.isize_d = 1;
	param.isize_h = 28;
	param.isize_w = 28;
	param.istride_w = 28;
	param.osize_d = 10;
	param.osize_h = 1;
	param.osize_w = 1;
	param.op_buffer = (float**)malloc(sizeof(float*)* 10);
	for (i = 0; i < 1000; i++)
		param.op_buffer[i] = (float*)malloc(sizeof(float));
	//status = fopen_s(&fp, "lenet5.lattemodel", "rb");
	fp = fopen("lenet5.lattemodel", "rb");
	

	//status= fopen_s(&fp,"model.lattemodel", "rb");
	/*
	input_image = (float*)malloc(227 * 227 * 3 * sizeof(float));
	param.ip_buffer = (float**)malloc(sizeof(float*)* 3);
	param.ip_buffer[0] = input_image;
	param.ip_buffer[1] = input_image + 227 * 227;
	param.ip_buffer[2] = input_image + 227 * 227*2;
	//status = fopen_s(&imagePointer, "catimage.latteinput", "rb"); //in windows
	imagePointer = fopen("catimage.latteinput", "rb");

	int read_elements = fread(param.ip_buffer[0], sizeof(float), 227 * 227 * 3, imagePointer);
	param.isize_d = 3;
	param.isize_h = 227;
	param.isize_w = 227;
	param.istride_w = 227;
	param.osize_d = 1000;
	param.osize_h = 1;
	param.osize_w = 1;
	param.op_buffer = (float**)malloc(sizeof(float*)* 1000);
	for (i = 0; i < 1000; i++)
		param.op_buffer[i] = (float*)malloc(sizeof(float));
	//status = fopen_s(&fp, "model.lattemodel", "rb"); //in windows
	fp = fopen("model.lattemodel", "rb");
	*/
	handle=latte_create();

	status = latte_parse_and_init(handle, fp,param);
	status = latte_forward_prop(handle, param);
	for (i = 0; i < 10; i++)                          /*change value here*/
	{
		printf("%.10f\n", *(param.op_buffer[i]));
	}
	status = latte_close(handle);
	printf("Everything ran successfully!\n");
	

	
	return 0;
}
