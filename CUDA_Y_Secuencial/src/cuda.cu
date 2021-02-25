//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~||
//                                _____          _____                       ||
//                          /\   / ____|   /\   |  __ \                      ||
//                         /  \ | |       /  \  | |__) |                     ||
//                        / /\ \| |      / /\ \ |  ___/                      ||
//                       / ____ \ |____ / ____ \| |                          ||
//                      /_/    \_\_____/_/    \_\_|                          ||
//                                                                           ||
// Práctica 4 - Operar con vectores v.CUDA                                   ||
// Antonio Priego Raya                                                       ||
//                                                                           ||
// COMPILACIÓN (nvcc)                                                        ||
//   nvcc src/cuda.cu -Iinclude -o bin/cuda                                  ||
//                                                                           ||
// DEPENDENCIAS                                                              ||
//   CUDA v.9.1                                                              ||
//                                                                           ||
// EJECUCIÓN                                                                 ||
//   ./bin/cuda data/0/input0.raw data/0/input1.raw                          ||
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~||

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <time.h>

void ShowDeviceProperties(char *nombre_prog, FILE *stream=stdout);
__global__ void operaVector(const float *h_A, const float *h_B, float *h_C, int num_elem);


int main(int argc, char **argv)
{
	if(argc<3) {
		printf("Introduzca los vectores de entrada\n");
		exit(1);
	}
	cudaError_t error = cudaSuccess;

	ShowDeviceProperties(argv[0]);

	// Lectura/Apertura de ficheros de entrada 
	FILE *datosA, *datosB;
	datosA = fopen(argv[1], "r");
	datosB = fopen(argv[2], "r");

	int num_elemA, num_elemB;
	fscanf(datosA, "%d", &num_elemA);
	fscanf(datosB, "%d", &num_elemB);

	if (num_elemA != num_elemB) {
	 fprintf(stderr, "Los vectores de entrada deben tener el mismo numero de elementos\n");
	 exit(-1);
	}

	// Reservar memoria vectores
	float *h_A = (float*)malloc(num_elemA * sizeof(float));
	float *h_B = (float*)malloc(num_elemA * sizeof(float));
	float *h_C = (float*)malloc(num_elemA * sizeof(float));

	if ( !h_A || !h_B || !h_C ) {
		fprintf(stderr, "Fallo al reservar memoria vectores\n");
		exit(-1);
	}

	// Asignar valores a los vectores
	float tmp;
	for (int i=0; i<num_elemA; ++i) {
		fscanf(datosA,"%f",&tmp); h_A[i] = tmp;
		fscanf(datosB,"%f",&tmp); h_B[i] = tmp;
	}

	// Reservar memoria en dispositivos
	float *device_A = NULL;
	float *device_C = NULL;
	float *device_B = NULL;
	
	if ( cudaMalloc((void **)&device_A, num_elemA*sizeof(float)) != cudaSuccess ) {
		fprintf(stderr, "Fallo al reservar memoria para vector en A!\n");
		exit(-1);
	}
	if ( cudaMalloc((void **)&device_B, num_elemA*sizeof(float)) != cudaSuccess ) {
		fprintf(stderr, "Fallo al reservar memoria para vector en B!\n");
		exit(-1);
	}
	if ( cudaMalloc((void **)&device_C, num_elemA*sizeof(float)) != cudaSuccess ) {
		fprintf(stderr, "Fallo al reservar memoria para vector en C!\n");
		exit(-1);
	}

	struct timespec t_0;
	clock_gettime(CLOCK_REALTIME, &t_0);

	// Llevar valores de entrada a memoria
	if ( cudaMemcpy(device_A, h_A, num_elemA*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ) {
		fprintf(stderr, "Fallo al copiar valores en memoria del dispositivo\n");
		exit(-1);
	}
	if ( cudaMemcpy(device_B, h_B, num_elemA*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ) {
		fprintf(stderr, "Fallo al copiar valores en memoria del dispositivo\n");
		exit(-1);
	}

	struct timespec t_1;
	clock_gettime(CLOCK_REALTIME, &t_1);

	// Cómputo (ejecución del kernel)
	int threadsXbloque = 512 ;
	int bloquesXGrid =(num_elemA+threadsXbloque - 1) / threadsXbloque;
	printf("Ejecucion del kernel con %d bloques de %d threads\n", bloquesXGrid, threadsXbloque);
	operaVector<<<bloquesXGrid, threadsXbloque>>>(device_A, device_B, device_C, num_elemA);

	error = cudaGetLastError();

	if (error != cudaSuccess) {
		fprintf(stderr, "Error en la ejecucion del kernel \n");
		exit(-1);
	}

	struct timespec t_2;
	clock_gettime(CLOCK_REALTIME, &t_2);

	// Copiar datos de salida
	if (cudaMemcpy(h_C, device_C, num_elemA*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
		fprintf(stderr, "Error al traer el vector c de memoria de GPU\n");
		exit(-1);
	}

	struct timespec t_f;
	clock_gettime(CLOCK_REALTIME, &t_f);
	
	// Cálculo de tiempos
	double tiempo_memoria = (t_1.tv_sec  - t_0.tv_sec )
	            + (double)  (t_1.tv_nsec - t_0.tv_nsec) / 1000000000;
	       tiempo_memoria +=(t_f.tv_sec  - t_2.tv_sec )
	            + (double)  (t_f.tv_nsec - t_2.tv_nsec) / 1000000000;

	double tiempo_ejec = (t_2.tv_sec  - t_1.tv_sec )
	          + (double) (t_2.tv_nsec - t_1.tv_nsec) / 1000000000;

	// Liberar memoria
	if (cudaFree(device_A) != cudaSuccess) {
		fprintf(stderr, "Fallo al liberar memoria del dispositivo A!\n");
		exit(-1);
	}
	if (cudaFree(device_B) != cudaSuccess) {
		fprintf(stderr, "Fallo al liberar memoria del dispositivo B!\n");
		exit(-1);
	}
	if (cudaFree(device_C) != cudaSuccess) {
		fprintf(stderr, "Fallo al liberar memoria del dispositivo C!\n");
		exit(-1);
	}

	// Guardar Info sobre GPUs
	FILE * infoGPU;
	if( !( infoGPU = fopen( "data/ejecucion/InfoGPUs", "w" ) ) )
		printf( "No se pudo acceder a data/ejecucion/InfoGPUs\n" );

	ShowDeviceProperties(argv[0], infoGPU);

	// Guardar resultados en fichero de salida
	FILE * datosC;
	if( !( datosC = fopen( "data/vectorC/CUDA.raw", "w" ) ) )
		printf( "No se pudo acceder a data/vectorC/CUDA.raw\n" );

	for (int i=0; i<num_elemA*10; i++)
		fprintf(datosC, "%f\n", h_C[i]);


	// Liberar memoria y cerrar ficheros
	free(h_A); fclose(datosA);
	free(h_B); fclose(datosB);
	free(h_C); fclose(datosC);
	
	
	// Valores de ejecución
	FILE * ejec;
	if( !( ejec = fopen("data/ejecucion/CUDA", "a") ) ) 
		printf( "No se ha podido abrir resultados de ejecucion.raw\n" );

	fprintf(ejec, "\nTamanio vector: %d\n\tTiempo ejecucion:\t%.8f\n", num_elemA, tiempo_ejec);
	fprintf(ejec, "\tTiempo R/W memoria:\t%.8f\n", tiempo_memoria);
	fprintf(ejec, "\tTiempo total:\t\t%.8f\n\n", tiempo_memoria+tiempo_ejec);	
	fclose (ejec);

	
	printf("\nTamanio vector: %d\n\tTiempo ejecucion:\t%.8f\n", num_elemA, tiempo_ejec);
	printf("\tTiempo R/W memoria:\t%.8f\n", tiempo_memoria);
	printf("\tTiempo total:\t\t%.8f\n\n", tiempo_memoria+tiempo_ejec);	

	return 0;
}

//=========================== Mostrar Info GPUs =============================\\

void ShowDeviceProperties(char *nombre_prog, FILE *stream)
{
	int num_devices;
	fprintf(stream,"\n%s se ejecutara en\n  ║\n", nombre_prog);
	cudaGetDeviceCount(&num_devices);
	for (int i=0; i<num_devices-1; i++) {
		cudaDeviceProp device_info;
		cudaGetDeviceProperties(&device_info, i);
		fprintf(stream,"  ╠Device Number: %d\n", i);
		fprintf(stream,"  ║ ╠Device name: %s\n", device_info.name);
		fprintf(stream,"  ║ ╠Memory Clock Rate (KHz): %d\n", device_info.memoryClockRate);
		fprintf(stream,"  ║ ╠Memory Bus Width (bits): %d\n", device_info.memoryBusWidth);
		fprintf(stream,"  ║ ╠Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*device_info.memoryClockRate*(device_info.memoryBusWidth/8)/1.0e6);
		fprintf(stream,"  ║ ╚Max threads per block: %d", device_info.maxThreadsPerBlock);	
	}
	cudaDeviceProp device_info;
	cudaGetDeviceProperties(&device_info, num_devices-1);
	fprintf(stream,"  ╚Device Number: %d\n", num_devices-1);
	fprintf(stream,"    ╠Device name: %s\n", device_info.name);
	fprintf(stream,"    ╠Memory Clock Rate (KHz): %d\n", device_info.memoryClockRate);
	fprintf(stream,"    ╠Memory Bus Width (bits): %d\n", device_info.memoryBusWidth);
	fprintf(stream,"    ╠Peak Memory Bandwidth (GB/s): %f\n", 2.0*device_info.memoryClockRate*(device_info.memoryBusWidth/8)/1.0e6);
	fprintf(stream,"    ╚Max threads per block: %d\n\n", device_info.maxThreadsPerBlock);	
	
}

//================================ KERNEL ===================================\\

__global__ void operaVector(const float *h_A, const float *h_B, float *h_C, int num_elem)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < num_elem)	
		h_C[i] = pow(pow(log(5*h_A[i]*100*h_B[i]+7*h_A[i]) / 0.33, 3), 7);
}