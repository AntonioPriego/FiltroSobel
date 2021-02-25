//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~||
//                                _____          _____                       ||
//                          /\   / ____|   /\   |  __ \                      ||
//                         /  \ | |       /  \  | |__) |                     ||
//                        / /\ \| |      / /\ \ |  ___/                      ||
//                       / ____ \ |____ / ____ \| |                          ||
//                      /_/    \_\_____/_/    \_\_|                          ||
//                                                                           ||
// Práctica 4 - Operar con vectores v.CPU                                    ||
// Antonio Priego Raya                                                       ||
//                                                                           ||
// COMPILACIÓN (gcc)                                                         ||
//   gcc src/secuencial.c -o bin/secuencial -lm -                            ||
//                                                                           ||
// EJECUCIÓN                                                                 ||
//   ./bin/secuencial data/0/input0.raw data/0/input1.raw                    ||
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~||

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(int argc, char **argv)
{
	if(argc<3) {
		printf("Introduzca los vectores de entrada\n");
		exit(1);
	}

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
	float *h_A = (float*)malloc( num_elemA * sizeof(float) );
	float *h_B = (float*)malloc( num_elemA * sizeof(float) );
	float *h_C = (float*)malloc( num_elemA * sizeof(float) );

	
	if ( !h_A || !h_B || !h_C) {
		fprintf(stderr, "Fallo al reservar memoria vectores\n");
		exit(-1);
	}


	struct timespec t_0;
	clock_gettime(CLOCK_REALTIME, &t_0);

	// Asignar valores a los vectores
	float tmp;
	for (int i=0; i<num_elemA; ++i) {
		fscanf(datosA, "%f", &tmp); h_A[i] = tmp;
		fscanf(datosB, "%f", &tmp); h_B[i] = tmp;
	}

	struct timespec t_1;
	clock_gettime(CLOCK_REALTIME, &t_1);
	// Computo
	for (int i=0; i<num_elemA; ++i)
		h_C[i] = pow(pow(log(5*h_A[i]*100*h_B[i]+7*h_A[i]) / 0.33, 3), 7);

	struct timespec t_2;
	clock_gettime(CLOCK_REALTIME, &t_2);

	// Escritura de datos
	FILE * datosC;
	if( !( datosC = fopen("data/vectorC/secuencial.raw", "w") ) ) 
		printf( "No se ha podido abrir mi_output.raw\n" );

	for (int i=0; i<num_elemA; i++) 
		fprintf(datosC, "%f\n", h_C[i]);

	struct timespec t_f;
	clock_gettime(CLOCK_REALTIME, &t_f);
	
	// Cálculo de tiempos
	double tiempo_memoria = (t_1.tv_sec  - t_0.tv_sec )
	            + (double)  (t_1.tv_nsec - t_0.tv_nsec) / 1000000000;
	       tiempo_memoria +=(t_f.tv_sec  - t_2.tv_sec )
	            + (double)  (t_f.tv_nsec - t_2.tv_nsec) / 1000000000;

	double tiempo_ejec = (t_2.tv_sec  - t_1.tv_sec )
	          + (double) (t_2.tv_nsec - t_1.tv_nsec) / 1000000000;


	// Liberar memoria y cerrar ficheros
	free(h_A); fclose(datosA);
	free(h_B); fclose(datosB);
	free(h_C); fclose(datosC);
	

	// Valores de ejecución
	FILE * ejec;
	if( !( ejec = fopen("data/ejecucion/secuencial", "a") ) ) 
		printf( "No se ha podido abrir resultados de ejecucion.raw\n" );

	fprintf(ejec, "\nTamanio vector: %d\n\tTiempo ejecucion:\t%.8f\n", num_elemA, tiempo_ejec);
	fprintf(ejec, "\tTiempo R/W memoria:\t%.8f\n", tiempo_memoria);
	fprintf(ejec, "\tTiempo total:\t\t%.8f\n\n", tiempo_memoria+tiempo_ejec);	
	fclose (ejec);


	printf("Tamanio vector: %d, Tiempo ejecucion:\t%.8f\n", num_elemA, tiempo_ejec);
	printf("Tiempo R/W memoria:\t%.8f\n\n", tiempo_memoria);

	return 0;
}
