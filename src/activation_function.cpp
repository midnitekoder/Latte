#include<stdio.h>
#include<math.h>
#include "util.h"


float maxfloat(float num1, float num2)
{
	if (num1 > num2)
		return num1;
	else
		return num2;
}


float activation(activation_type type_of_activation, float val, float negative_slope)
{
	double positive_exp, negative_exp;
	switch (type_of_activation)
	{
	case AF_RELU:
		return maxfloat(0, val);
	case AF_SIGM:
		return (float)(1.0 / (1.0 + exp(-val)));
	case AF_TANH:
		positive_exp = exp(val);
		negative_exp = exp(-val);
		return (float)((positive_exp - negative_exp) / (positive_exp + negative_exp));
	case AF_LEAKY_RELU:
		return val > 0 ? val : val*negative_slope;
	case AF_ELU:
		return (float)(exp(val)*(val < 0) + val*(val >= 0));
	case AF_NOACTIVATION:
		return val;
	default:
		return 0.0;


	}
}