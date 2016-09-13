all:	latte


latte:	test_main.o	activation_function.o	model_reader.o	io_buffer_handling.o	connection_handling.o	dimension_analysis.o	net_builder.o	kernel_operations.o	forward_prop.o	latte_api.o
	g++ test_main.o	activation_function.o	model_reader.o	io_buffer_handling.o	connection_handling.o	dimension_analysis.o	net_builder.o	kernel_operations.o	forward_prop.o	latte_api.o	-o	latte -lm

test_main.o:	test_main.cpp
	g++ -c test_main.cpp

activation_function.o:	activation_function.cpp
	g++ -c activation_function.cpp -o activation_function.o -lm

model_reader.o:	model_reader.cpp
	g++ -c model_reader.cpp

io_buffer_handling.o:	io_buffer_handling.cpp
	g++ -c io_buffer_handling.cpp

connection_handling.o:	connection_handling.cpp
	g++ -c connection_handling.cpp

dimension_analysis.o:	dimension_analysis.cpp
	g++ -c dimension_analysis.cpp

net_builder.o:	net_builder.cpp
	g++ -c net_builder.cpp

kernel_operations.o:	kernel_operations.cpp
	g++ -c kernel_operations.cpp

forward_prop.o:	forward_prop.cpp
	g++ -c forward_prop.cpp

latte_api.o:	latte_api.cpp
	g++ -c latte_api.cpp

clean:
	rm -f *.o latte










