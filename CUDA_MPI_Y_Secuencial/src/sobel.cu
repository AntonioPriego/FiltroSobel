//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~||
//                                _____          _____                       ||
//                          /\   / ____|   /\   |  __ \                      ||
//                         /  \ | |       /  \  | |__) |                     ||
//                        / /\ \| |      / /\ \ |  ___/                      ||
//                       / ____ \ |____ / ____ \| |                          ||
//                      /_/    \_\_____/_/    \_\_|                          ||
//                                                                           ||
// Práctica 5 - Filtro de Sobel con CUDA                                     ||
// Antonio Priego Raya                                                       ||
//                                                                           ||
// COMPILACIÓN (nvcc)                                                        ||
//   nvcc src/sobel.cu `pkg-config --libs opencv` -lstdc++ -Iinclude         ||
//                    -I/usr/include/opencv/ -I/usr/include/opencv2/         ||
//                    -o bin/sobel -Xcompiler -fopenmp -w -O2                ||
//                                                                           ||
// DEPENDENCIAS                                                              ||
//   CUDA v.9.1                                                              ||
//   OpenMP (medición de tiempos fiel a P3)                                  ||
//                                                                           ||
// EJECUCIÓN                                                                 ||
//   ./bin/sobel data/imagenes/i/planetas.jpg 127                            ||
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~||

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <time.h>
#include <omp.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#define TIPO_UNSIGNED_8b CV_8U //unsigned 8bit/pixel
typedef uchar* _Mat;


//================================= FUNCS ===================================\\

void ShowDeviceProperties(char *nombre_prog, FILE *stream=stdout);
__global__ void sobel(_Mat original, _Mat procesada, int x_max, int y_max, int frontera);



//================================== MAIN ===================================\\

int main(int argc, char **argv)
{
   cudaError_t error = cudaSuccess;


   // Lectura/Apertura de ficheros de entrada 
   cv::Mat orig;
   char* directorio;

   int tam_threads = 32;       //-Tamaño de threads
   int condicion_frontera = 0; //-Nivel de exigencia para llevar
                               // pixel procesado a blanco o negro

   // Carga archivo de imagen sin procesar
   if (argc == 2) {
      directorio = argv[1];
      orig = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
   }
   else if (argc == 3) {
      directorio = argv[1];
      orig = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
      condicion_frontera = atoi(argv[2]);
      if (condicion_frontera<0  ||  condicion_frontera>127)
         exit(97);
   }
   else if (argc == 4) {
      directorio = argv[1];
      orig = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
      condicion_frontera = atoi(argv[2]);
      if (condicion_frontera<0  ||  condicion_frontera>127)
         exit(97);

      tam_threads = atoi(argv[3]);
      if (tam_threads<1  ||  tam_threads>32) //32x32=1024 es mi máximo número de threads por bloque
         exit(96);

   }
   else {
      directorio = "data/imagenes/i/paisaje.jpg";
      orig = cv::imread(directorio, CV_LOAD_IMAGE_GRAYSCALE);
   }

   ShowDeviceProperties(argv[0]);
   
   // Reservar memoria en dispositivos
   _Mat device_A = NULL;
   _Mat device_B = NULL;
   
   if ( cudaMalloc((void **)&device_A, orig.rows*orig.cols*sizeof(uchar)) != cudaSuccess ) {
      fprintf(stderr, "Fallo al reservar memoria alojar imagen origen\n");
      exit(-1);
   }
   if ( cudaMalloc((void **)&device_B, orig.rows*orig.cols*sizeof(uchar)) != cudaSuccess ) {
      fprintf(stderr, "Fallo al reservar memoria alojar imagen procesada\n");
      exit(-1);
   }

   //TIEMPO 0
   double t_0 = omp_get_wtime();

   // Llevar valores de entrada a memoria
   if ( cudaMemcpy(device_A, orig.data, orig.rows*orig.cols*sizeof(uchar), cudaMemcpyHostToDevice) != cudaSuccess ) {
      fprintf(stderr, "Fallo al copiar valores en memoria del dispositivo\n");
      exit(-1);
   }

   //TIEMPO 1
   double t_1 = omp_get_wtime();

   dim3 threadsXbloque(tam_threads, tam_threads, 1);
   dim3 numBloques((int)(orig.cols/tam_threads), (int)(orig.rows/tam_threads), 1);

   sobel<<<numBloques, threadsXbloque>>>(device_A, device_B, orig.cols, orig.rows, condicion_frontera);

   if (error != cudaSuccess) {
      fprintf(stderr, "Error en la ejecucion del kernel \n");
      exit(-1);
   }

   //TIEMPO 2
   double t_2 = omp_get_wtime();

   _Mat data_proc = (uchar*)malloc( orig.rows*orig.cols );
   if ( !data_proc ) {
      fprintf(stderr, "Fallo al reservar memoria alojar imagen procesada\n");
      exit(-1);
   }

   cv::Mat proc(orig.rows, orig.cols, TIPO_UNSIGNED_8b);

   // Copiar datos de salida
   if (cudaMemcpy(data_proc, device_B, orig.rows*orig.cols*sizeof(uchar), cudaMemcpyDeviceToHost) != cudaSuccess) {
      fprintf(stderr, "Error al traer imagen de GPU\n");
      exit(-1);
   }

   // Trasladar datos de salida a imagen procesada
   for(int i=0; i<orig.rows; i++)
      for(int j=0; j<orig.cols; j++)
         proc.at<uchar>(i,j) = (uchar)data_proc[i*orig.cols+j];
  
   //TIEMPO FINAL
   double t_f = omp_get_wtime();

   // Cálculo de tiempos
   double tiempo_memoria  = t_1 - t_0; // Memoria CPU->GPU
   double tiempo_ejec     = t_2 - t_1; // Procesamiento en GPU
          tiempo_memoria += t_f - t_2; // Memoria CPU<-GPU

   // Liberar memoria
   if (cudaFree(device_A) != cudaSuccess) {
      fprintf(stderr, "Fallo al liberar memoria del dispositivo A!\n");
      exit(-1);
   }
   if (cudaFree(device_B) != cudaSuccess) {
      fprintf(stderr, "Fallo al liberar memoria del dispositivo B!\n");
      exit(-1);
   }
   free(data_proc);

   // Guarda archivo de imagen procesada
   directorio[14] = 'o';                           //Pasa de ficheros entrada(i) a salida(o)
   directorio[strlen(directorio)-4] = '\0';        //Elimina ".jpg" final
   sprintf(directorio, "%s(%d).jpg", directorio, condicion_frontera); //Aniade "(frontera).jpg"

   if ( imwrite(directorio, proc) )
      printf("Imagen guardada con exito en %s\n", directorio);
   else
      printf("\nError al guardar la imagen.");


   // Valores de ejecución
   FILE * ejec;
   if( !( ejec = fopen("data/ejecucion/sobel", "a") ) ) 
      printf( "No se ha podido abrir resultados de ejecucion\n" );

   fprintf(ejec, "\nTamanio imagen: %dx%d\n\tTiempo ejecucion:\t%.8f\n", orig.cols, orig.rows, tiempo_ejec);
   fprintf(ejec, "\tTiempo R/W memoria:\t%.8f\n", tiempo_memoria);
   fprintf(ejec, "\tTiempo total:\t\t%.8f\n\n", tiempo_memoria+tiempo_ejec);  
   fclose (ejec);

   //Impresión con cadenas
   printf("\nTamanio imagen: %dx%d\n\tTiempo ejecucion:\t%.8f\n", orig.rows, orig.cols, tiempo_ejec);
   printf("\tTiempo R/W memoria:\t%.8f\n", tiempo_memoria);
   printf("\tTiempo total:\t\t%.8f\n\n"  , tiempo_memoria+tiempo_ejec);


   //Libera imágenes
   orig.release();
   proc.release();


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

__global__ void sobel(_Mat original, _Mat procesada, int x_max, int y_max, int frontera=0)
{
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;

   int xgradient, ygradient, sum;  
   

   if ( (x>0 && y>0) && (x<x_max && y<y_max) ) { 
      //GRADIENTE X
      xgradient =   original[x-1 + (y-1)*x_max] + 2*original[x-1 +  y   *x_max] +
                    original[x-1 + (y+1)*x_max] -   original[x+1 + (y-1)*x_max] -
                  2*original[x+1 +  y   *x_max] -   original[x+1 + (y+1)*x_max];
      //GRADIENTE Y
      ygradient =   original[x-1 + (y+1)*x_max] + 2*original[x   + (y+1)*x_max] +
                    original[x+1 + (y+1)*x_max] -   original[x-1 + (y-1)*x_max] -
                  2*original[x   + (y-1)*x_max] -   original[x+1 + (y-1)*x_max];

      //SUMA
      sum = abs(xgradient) + abs(ygradient);
      sum = sum > 255-frontera ? 255:sum;
      sum = sum < 0  +frontera ? 0  :sum;
      procesada[x + y*x_max] = sum;
   }

}