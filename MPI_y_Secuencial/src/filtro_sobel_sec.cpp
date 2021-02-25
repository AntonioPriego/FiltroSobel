//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~||
//                                   _____          _____                             ||
//                             /\   / ____|   /\   |  __ \                            ||
//                            /  \ | |       /  \  | |__) |                           ||
//                           / /\ \| |      / /\ \ |  ___/                            ||
//                          / ____ \ |____ / ____ \| |                                ||
//                         /_/    \_\_____/_/    \_\_|                                ||
//                                                                                    ||
// Práctica 3 - Filtro de sobel con MPI                                               ||
// Antonio Priego Raya                                                                ||
//                                                                                    ||
// COMPILACIÓN (MPI)                                                                  ||
//   g++ src/filtro_sobel_sec.cpp -o bin/filtro_sobel_sec `pkg-config --libs opencv`  ||
//                                                                -fopenmp -lstdc++   ||
//                                                                                    ||
// DEPENDENCIAS (openCV)                                                              ||
//   sudo apt-get install libopencv-dev python3-opencv                                ||
//                                                                                    ||
// EJECUCIÓN  (* = opcional)                                                          ||
//   <ruta_ejecutable> <*ruta_jpg> <*condicion_frontera>                              ||
//       -ruta_jpg           : ubicación de jpg                                       ||
//       -condicion_frontera : nivel de exigencia para llevar un pixel procesado a    ||
//                             blanco o negro. +frontera = +definición y -detalle     ||
//                             Por defecto 0, no se modifica pixel procesado.         ||
//                             Máximo 127 (255/2).                                    ||
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~||

#include <iostream>
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <omp.h>
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
   Mat src, dst;
   int sum;
   string directorio;

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
         exit(1);
   }
   else {
      directorio = "files/paisaje.jpg";
      src = imread(directorio.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
   }

   if ( !src.data ) return -1;
   dst.create(src.rows, src.cols, src.type());

   double start_time = omp_get_wtime();
   for (int y=0; y<src.rows; y++){
      for (int x=0; x<src.cols; x++) {
         sum = abs(x_gradient(src, x, y)) + abs(y_gradient(src, x, y));
         sum = sum > 255-condicion_frontera ? 255:sum;
         sum = sum < 0  +condicion_frontera ? 0  :sum;
         dst.at<uchar>(y,x) = sum;
      }
   }
   double time = omp_get_wtime() - start_time;

   // Procesa archivo de imagen final
   cout << "Tiempo de procesado: " << time << endl;
   for (int i=0; i<4; i++)  directorio.pop_back();
   directorio += "_procesado(" + (to_string)(condicion_frontera) + ").jpg";

   // Guarda archivo de imagen procesada
   if (imwrite(directorio.c_str(), dst))
      cout << "Imagen guardada con exito " << directorio << endl;
   else
      cout << "\nError al guardar la imagen." << endl;


   return 0;
}
