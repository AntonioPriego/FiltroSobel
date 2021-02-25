//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~||
//                                   _____          _____                                ||
//                             /\   / ____|   /\   |  __ \                               ||
//                            /  \ | |       /  \  | |__) |                              ||
//                           / /\ \| |      / /\ \ |  ___/                               ||
//                          / ____ \ |____ / ____ \| |                                   ||
//                         /_/    \_\_____/_/    \_\_|                                   ||
//                                                                                       ||
// Práctica 3 - Filtro de sobel con MPI                                                  ||
// Antonio Priego Raya                                                                   ||
//                                                                                       ||
// COMPILACIÓN (MPI)                                                                     ||
//   mpic++ src/filtro_sobel_paral_v2.cpp -o bin/filtro_sobel_paral_v2                   ||
//                        `pkg-config --libs opencv` -fopenmp -lstdc++                   ||
//                                                                                       ||
// DEPENDENCIAS (openCV)                                                                 ||
//   sudo apt-get install libopencv-dev python3-opencv                                   ||
//                                                                                       ||
// EJECUCIÓN  (* = opcional)                                                             ||
//   <ruta_ejecutable> <*ruta_jpg> <*condicion_frontera>                                 ||
//       -ruta_jpg           : ubicación de jpg                                          ||
//       -condicion_frontera : nivel de exigencia para llevar un pixel procesado a       ||
//                             blanco o negro. +frontera = +definición y -detalle        ||
//                             Por defecto 0, no se modifica pixel procesado.            ||
//                             Máximo 127 (255/2).                                       ||
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~||

#include <iostream>
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <omp.h>
#include <mpi.h>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;

//
// Devuelve el gradiente de 'x' del punto(x,y) de la imagen
//    Proceso: Aplicar la máscara
//
//   Orden       Máscara (x)
// [z1 z2 z3]    [-1  0  1]
// [z4 z5 z6]    [-2  0  2]
// [z7 z8 z9]    [-1  0  1]
//
// Gx= (z3+2*z6+z9) - (z1+2*z4+z7)
int x_gradient(Mat image, int x, int y)
{
	int gradient;  
	if (x!=0 && y!=0) {  // Caso típico (pixel central)
		gradient = image.at<uchar>(y-1, x-1) + 2*image.at<uchar>(y, x-1) +
					  image.at<uchar>(y+1, x-1) - image.at<uchar>(y-1, x+1) -
					  2*image.at<uchar>(y, x+1) - image.at<uchar>(y+1, x+1);
	}
	else {               // Caso excepcional (pixel de marco imagen)
		if (x==0 && y!=0) {
			gradient = image.at<uchar>(y-1, x+1) -
						  2*image.at<uchar>(y, x+1) -
						  image.at<uchar>(y+1, x+1);
		}
		else if (x=!0 && y==0) {
			gradient = 2*image.at<uchar>(y, x-1) +
						  image.at<uchar>(y+1, x-1) -
						  2*image.at<uchar>(y, x+1) -
						  image.at<uchar>(y+1, x+1);
		}
		else if (x=!0 && y==image.cols) {
			gradient = image.at<uchar>(y-1, x-1) +
						  2*image.at<uchar>(y, x-1) +
						  image.at<uchar>(y-1, x+1) -
						  2*image.at<uchar>(y, x+1);
		}
	}

	return gradient;
}

//
// Devuelve el gradiente de 'y' del punto(x,y) de la imagen
//    Proceso: Aplicar la máscara 
//
//   Orden       Máscara (y)
// [z1 z2 z3]    [-1 -2 -1]
// [z4 z5 z6]    [ 0  0  0]
// [z7 z8 z9]    [ 1  2  1]
//
// Gy= (z7+2*z8+z9) - (z1+2*z2+z3)
int y_gradient(Mat image, int x, int y)
{
	int gradient;
	if (x!=0 && y!=0) {  // Caso típico (pixel central)
		gradient = image.at<uchar>(y+1, x-1) + 2*image.at<uchar>(y+1, x) +
					  image.at<uchar>(y+1, x+1) - image.at<uchar>(y-1, x-1) -
					  2*image.at<uchar>(y-1, x) - image.at<uchar>(y-1, x+1);
	}
	else {               // Caso excepcional (pixel de marco imagen)
		if (x==0 && y!=0)
			gradient = 2*image.at<uchar>(y+1, x) +
						  image.at<uchar>(y+1, x+1) -
						  2*image.at<uchar>(y-1, x) -
						  image.at<uchar>(y-1, x+1);
		else if (x=!0 && y==0)
			gradient = image.at<uchar>(y+1, x-1) +
						  2*image.at<uchar>(y+1, x) +
						  image.at<uchar>(y+1, x+1);
		else if (x=!0 && y==image.cols)
			gradient = image.at<uchar>(y-1, x-1) -
						  2*image.at<uchar>(y-1, x) -
						  image.at<uchar>(y-1, x+1);
	}

	return gradient;
}

int main(int argc, char** argv)
{
	double ini_creacion = omp_get_wtime(),
			 t_creacion,
			 ini_destruc,
			 t_destruc;
	Mat src;
	int sum;
	string directorio;

	int num_procesos, id_proceso;

	int condicion_frontera = 0; // Nivel de exigencia para llevar
										 // pixel procesado a blanco o negro

	// Carga archivo de imagen sin procesar
	if (argc == 2) {
		directorio = argv[1];
		src = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	}
	else if (argc == 3) {
		directorio = argv[1];
		src = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
		condicion_frontera= atoi(argv[2]);
		if (condicion_frontera<0  ||  condicion_frontera>127)
			exit(97);
	}
	else {
		directorio = "files/paisaje.jpg";
		src = imread(directorio.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	}
	int rows     = src.rows;
	int cols     = src.cols;
	int type     = src.type();
	int channels = src.channels();

	if ( !src.data ) exit(98);

	if (MPI_Init(&argc, &argv) != MPI_SUCCESS) 
		exit(99);
	t_creacion = omp_get_wtime() - ini_creacion;
	MPI_Comm_size(MPI_COMM_WORLD, &num_procesos);
	MPI_Comm_rank(MPI_COMM_WORLD, &id_proceso);
	MPI_Status status;

	int tam_trozos = rows/num_procesos; // +1 para transición de un procesador a
													// otro. Procesador 2 necesita 1 líneas de
													// proceso 1 para vecindad de su primera línea

	// Procesamiento imagen
	double ini_ejec = omp_get_wtime();
	if (id_proceso == 0) {  // Procesador de reparto/recepción

		for (int i=1; i<num_procesos; i++) {
			int ini = i*tam_trozos - 1,  
				 fin = (i+1)*tam_trozos + (i!=num_procesos-1 ? 1 : 0);
			 // ini: Iniciamos una fila atrás, para respetar vecindad
			 // fin: Si es último rango, no suma 1 -> fin imagen

			Mat trozo ( src, Range(ini, fin) );

			MPI_Send(trozo.data, (trozo.cols*trozo.rows), MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
		}
		
		Mat dst;
		dst.create(tam_trozos, src.cols, src.type());

		// Procesamiento del trozo 0
		int x,y; // Indices de columnas y filas que reconstruirán dst(archivo resultado)
		for (x=0; x<cols; x++) {
			for (y=0; y<tam_trozos; y++) {
				sum = abs( x_gradient(src, x, y) ) +
						abs( y_gradient(src, x, y) )   ;
				sum = sum > 255-condicion_frontera ? 255:sum;
				sum = sum < 0  +condicion_frontera ? 0  :sum;
				dst.at<uchar>(y,x) = sum;
			}
		}

		double t_ejec, ini_recepcion;
		for (int i=1; i<num_procesos; i++) {
			Mat rcib = Mat(tam_trozos, cols, type);
			MPI_Recv(rcib.data, tam_trozos*cols, MPI_UNSIGNED_CHAR, i, 1, MPI_COMM_WORLD, &status);

			if (i==1) {
				t_ejec = omp_get_wtime() - ini_ejec;
				ini_recepcion = omp_get_wtime();
			}

			for (int j=0; j<rcib.rows; j++)
				dst.push_back(rcib.row(j));
		}
		double t_recepcion = omp_get_wtime() - ini_recepcion;

		// Crea archivo de imagen final
		cout << "Tiempo de calculo: " << t_ejec << endl;
		cout << "Tiempo de recepción de resultados: " << t_recepcion << endl;
		for (int i=0; i<4; i++)  directorio.pop_back();
		directorio += "_procesado(" + (to_string)(condicion_frontera) + ").jpg";

		
		// Guarda archivo de imagen procesada
		if (imwrite(directorio.c_str(), dst))
			cout << "Imagen guardada con exito " << directorio << endl;
		else
			cout << "\nError al guardar la imagen." << endl;
	}
	else {
		int tam_rango = tam_trozos+(id_proceso!=num_procesos-1 ? 2 : 1);
		Mat rcib = Mat(tam_rango, cols, type);
		MPI_Recv(rcib.data, tam_rango*cols, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &status);
		Mat prcs = Mat(tam_trozos, cols, type);   

		// Procesamiento del trozo recibido
		for (int x=0; x<cols; x++) {
			for (int y=1; y<tam_trozos; y++) {
			//       y=1 porque y=0 es fila de vecindad, no de procesamiento
				sum = abs( x_gradient(rcib, x, y) ) +
						abs( y_gradient(rcib, x, y) )   ;
				sum = sum > 255-condicion_frontera ? 255:sum;
				sum = sum < 0  +condicion_frontera ? 0  :sum;
				prcs.at<uchar>(y-1,x) = sum;
				//             y-1 porque queremos almacenar desde 0 pero y empieza en 1
			}
		}

		MPI_Send(prcs.data, prcs.cols*prcs.rows, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD);      
	}
	
	ini_destruc = omp_get_wtime();

	MPI_Finalize();

	t_destruc = omp_get_wtime() - ini_destruc;
	cout << "Tiempo de creacion y destruccion procesos: " << t_creacion+t_destruc << endl;

	return 0;
}

/*
	ERRORES:
		99: no se pudo inicializar MPI
		98: imagen source vacía
		97: condicion_frontera no respetada [0 a 127]
*/
