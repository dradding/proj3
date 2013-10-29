#include <emmintrin.h>
#define KERNX 3//this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
//Authors: Ayush Mudgal (cs61c-cj) and Daniel Radding (cs61c-vh)
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{


   /*float newker[12]; // kerner conversion (padding to the right)*/

    float newker[KERNX * KERNY]; // kerner conversion (padding to the right)

   // int count2 = -1; 
    for (int count = 0; count < (KERNX * KERNY); count++) {
	newker[count] = kernel[count];
      }
    kernel = newker;
    
    int size = (data_size_X) * (data_size_Y);
    float newin[size]; // kerner conversion (padding to the right)

    //int count2 = -1; 
    for (int count = 0; count < size; count++) {
	newin[count] = in[count];
      }
    in = newin;

    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;
    //printf("center of the kernel %d %d", kern_cent_X, kern_cent_Y); //uncomment to print the center of the kernel
    // main convolution loop
	for(int y = 1; y < data_size_Y -1; y++){ // the y coordinate of theoutput location we're focusing on
		for(int x = 1; x < data_size_X -1; x++){ // the x coordinate of the output location we're focusing on
				for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
					for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
							out[x+y*data_size_X] += 
								kernel[(kern_cent_X-i)+(kern_cent_Y-j)*(KERNX)] * in[(x+i) + (y+j)*data_size_X];							
				}
			}
		}
	}

	//printf("top calculations \n");
	for(int i = 1; i <= data_size_X -2 ; i++){
	    for (int k = i; k < data_size_X + i; k = k + data_size_X) {
		out[i] = (in[k+1] * kernel[3]) + (in[k] * kernel[4]) + (in[k-1] * kernel[5]) + (in[k+data_size_X+1] * kernel[0]) + (in[k+data_size_X] * kernel[1]) + (in[k+data_size_X-1] * kernel[2]); 
	      }
	}
	//printf(" \n \n \nleft hand side calculations \n");
	for(int i = data_size_X; i < size - data_size_X ; i = i + data_size_X){
	    for (int k = i - data_size_X; k < i; k = k + data_size_X) {
		out[i] = (in[k] * kernel[7]) + (in[k+1] * kernel[6]) + (in[k+ data_size_X] * kernel[4]) + (in[k+data_size_X+1] * kernel[3]) + (in[data_size_X + (data_size_X + k)] * kernel[1]) + (in[data_size_X + (data_size_X + k)+1] * kernel[0]); 
	    }
	}
	//printf("\n \n \n bottom calculations \n");
	for(int i = size - data_size_X + 1 ; i <= size - 2 ; i++){
	    for (int k = i - data_size_X; k < i; k = k + data_size_X) {
		out[i] = (in[k-1] * kernel[8]) + (in[k] * kernel[7]) + (in[k+1] * kernel[6]) + (in[k+data_size_X-1] * kernel[5]) + (in[k+data_size_X] * kernel[4]) + (in[k+data_size_X+1] * kernel[3]); 
	    }
	}
	//printf("\n \n \n right hand side calculations \n");
	for(int i = (2 * data_size_X) - 1; i < size - data_size_X ; i = i + data_size_X){
	    for (int k = i - data_size_X; k < i ; k = k + data_size_X) {
		out[i] = (in[k-1] * kernel[8]) + (in[k] * kernel[7]) + (in[k + data_size_X -1] * kernel[5]) + (in[k + data_size_X] * kernel[4]) + (in[(k + data_size_X + data_size_X)-1] * kernel[2]) + (in[k + data_size_X + data_size_X] * kernel[1]); 
	    }
	}

	
	//Handles upper left corner
	for(int j = -kern_cent_Y; j <= kern_cent_Y-1; j++){ // kernel unflipped y coordinate
		for(int i = -1; i <= kern_cent_X-1; i++){ // kernel unflipped x coordinate
			out[0] += kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)] * in[((i) + (j)*data_size_X) * -1];
		}
	}
	
	//Handles upper right corner
	for(int j = -kern_cent_Y; j <= kern_cent_Y-1; j++){ // kernel unflipped y coordinate
		for(int i = 0; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
			out[data_size_X-1] += kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)] * in[(-(2*data_size_X-1) + (data_size_X+i) + (j)*data_size_X) * -1];
		}
	}
	
	
	
	//Handles lower right corner
	for(int j = 0; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
		for(int i = 0; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
			out[(data_size_X*data_size_Y)-1] += kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)] * in[(-2*(size-1) + data_size_X-1+i + (data_size_Y-1+j)*data_size_X) * -1];
		}
	}
	
	
	//Handles lower left corner
	for(int j = 0; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
		for(int i = -1; i <= kern_cent_X-1; i++){ // kernel unflipped x coordinate
			out[(data_size_X*data_size_Y)-data_size_X] += kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)] * in[((-i)*-1 + (2*data_size_X*j)- (data_size_Y-1+j)*data_size_X)*-1];
		}
	}
	
	
	return 1;
}
