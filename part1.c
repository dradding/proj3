#include <emmintrin.h>
#define KERNX 3//this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
  /*printf("old \n"); //print loop
  for (int i = 0; i < 9; i++) {
	  if (i % 3 == 0) {
	    printf("\n");
	  }
	  printf(" %f ", kernel[i]);
	}*/
	
    float newker[12]; // kerner conversion (padding to the right)
    int count2 = -1; 
    for (int count = 0; count < 12; count++) {
      if ((count) % 4 == 3) {
	    newker[count] = 0;
      }
      else {
	count2++;
	newker[count] = kernel[count2];
      }
    }
    kernel = newker;

    printf("\n padded kernel"); //print loop; new kernel
    for (int i = 0; i < 12; i++) {
	  if (i % 4 == 0) {
	    printf("\n");
	  }
	  printf(" %f ", newker[i]);
	}
	printf("\n");
    int size = (data_size_X) * (data_size_Y);
    /*data_size_X += 3; //padding the input matrix all around
    data_size_Y += 2;
    int size = (data_size_X) * (data_size_Y);
    float newin[size];
    count2 = -1; 
    for (int count = 0; count < size; count++) {
 0     if (count <  data_size_X|| count % data_size_X == 1 ||count % data_size_X == data_size_X - 1 || count % data_size_X == 0 || count > size - data_size_X) {
	    newin[count] = 0;
      }
      else {
	count2++;
	newin[count] = in[count2];
      }
    }
    in = newin;*/
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
					// only do the operation if not out of bounds
					//if(x+i>-1 && x+i<data_size_X && y+j>-1 && y+j<data_size_Y){
						//Note that the kernel is flipped
							out[x+y*data_size_X] += 
								kernel[(kern_cent_X-i)+(kern_cent_Y-j)*(KERNX+1)] * in[(x+i) + (y+j)*data_size_X];							
					//}
				}
			}
		}
	}
	
	printf("\n");
	//Handles upper left corner
	for(int j = 0; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
		for(int i = 0; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
			out[0] += kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX+1)] * in[(0+i) + (0+j)*data_size_X];
			printf("kern Index: %d, kern Value: %f, x value: %f\n", (kern_cent_X+i)+(kern_cent_Y+j)*(KERNX+1), kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX+1)], in[(0+i) + (0+j)*data_size_X]);
		}
	}
	
	/*for(int j = -kern_cent_Y; j <= 0; j++){ // kernel unflipped y coordinate
		for(int i = -kern_cent_X; i <= 0; i++){ // kernel unflipped x coordinate
			out[0] += kernel[(kern_cent_X-i)+(kern_cent_Y-j)*(KERNX+1)] * in[(0+i) + (0+j)*data_size_X];
			printf("kern Index: %d, kern Value: %f, x value: %f\n", (kern_cent_X-i)+(kern_cent_Y-j)*(KERNX+1), kernel[(kern_cent_X-i)+(kern_cent_Y-j)*(KERNX+1)], in[(0+i) + (0+j)*data_size_X]);
		}
	}*/
	printf("\n");
	//Handles upper right corner
	
	for(int j = 0; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
		for(int i = -1; i <= kern_cent_X-1; i++){ // kernel unflipped x coordinate
			out[data_size_X-1] += kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX+1)] * in[(data_size_X-1+i) + (0+j)*data_size_X];
			printf("kern Index: %d, kern Value: %f, x value: %f\n", (kern_cent_X+i)+(kern_cent_Y+j)*(KERNX+1), kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX+1)], in[(data_size_X-1+i) + (0+j)*data_size_X]);
		}
	}
	
	/*for(int j = -kern_cent_Y; j <= 0; j++){ // kernel unflipped y coordinate
		for(int i = 0; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
			out[data_size_X-1] += kernel[(kern_cent_X-i)+(kern_cent_Y-j)*(KERNX+1)] * in[(data_size_X-1-i) + (0-j)*data_size_X];
			printf("kern Index: %d, kern Value: %f, x value: %f\n", (kern_cent_X-i)+(kern_cent_Y-j)*(KERNX+1), kernel[(kern_cent_X-i)+(kern_cent_Y-j)*(KERNX+1)], in[(data_size_X-1-i) + (0-j)*data_size_X]);
		}
	}*/
	
	printf("\n");
	//Handles lower right corner
	for(int j = -1; j <= kern_cent_Y-1; j++){ // kernel unflipped y coordinate
		for(int i = -1; i <= kern_cent_X-1; i++){ // kernel unflipped x coordinate
			out[(data_size_X*data_size_Y)-1] += kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX+1)] * in[data_size_X-1+i + (data_size_Y-1+j)*data_size_X];
			printf("kern Index: %d, kern Value: %f, x value: %f\n", (kern_cent_X+i)+(kern_cent_Y+j)*(KERNX+1), kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX+1)], in[data_size_X-1+i + (data_size_Y-1+j)*data_size_X]);
		}
	}
	/*for(int j = 0; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
		for(int i = 0; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
			out[(data_size_X*data_size_Y-1)] += kernel[(kern_cent_X-i)+(kern_cent_Y-j)*(KERNX+1)] * in[data_size_X-1+i + (data_size_Y-1+j)*data_size_X];
			printf("kern Index: %d, kern Value: %f, x value: %f\n", (kern_cent_X-i)+(kern_cent_Y-j)*(KERNX+1), kernel[(kern_cent_X-i)+(kern_cent_Y-j)*(KERNX+1)], in[data_size_X-1+i + (data_size_Y-1+j)*data_size_X]);
		}
	}*/
	
	printf("\n");
	//Handles lower left corner
	for(int j = -1; j <= kern_cent_Y-1; j++){ // kernel unflipped y coordinate
		for(int i = 0; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
			out[(data_size_X*data_size_Y)-data_size_X] += kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX+1)] * in[0+i + (data_size_Y-1+j)*data_size_X];
			printf("kern Index: %d, kern Value: %f, x value: %f\n", (kern_cent_X+i)+(kern_cent_Y+j)*(KERNX+1), kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX+1)], in[0+i + (data_size_Y-1+j)*data_size_X]);
		}
	}
	
	/*for(int j = 0; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
		for(int i = -kern_cent_X; i <= 0; i++){ // kernel unflipped x coordinate
			out[(data_size_X*data_size_Y)-data_size_X] += kernel[(kern_cent_X-i)+(kern_cent_Y-j)*(KERNX+1)] * in[0-i + (data_size_Y-1+j)*data_size_X];
			printf("kern Index: %d, kern Value: %f, x value: %f\n", (kern_cent_X-i)+(kern_cent_Y-j)*(KERNX+1), kernel[(kern_cent_X-i)+(kern_cent_Y-j)*(KERNX+1)], in[0-i + (data_size_Y-1+j)*data_size_X]);
		}
	}*/
	
	
	
	printf("\n in");
	for (int i = 0; i < size; i++) {
	  if (i % (data_size_X) == 0) {
	    printf("\n");
	  }
	  printf(" %f ", in[i]);
	}
	
	printf("\n out") ;
	for (int i = 0; i < size; i++) {
	  if (i % (data_size_X) == 0) {
	    printf("\n");
	  }
	  printf(" %f ", out[i]);
	}
	return 1;
}
