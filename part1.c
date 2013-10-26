#include <emmintrin.h>
#define KERNX 3//this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{

  printf("Kernal \n"); //print loop

  printf("old \n"); //print loop

  for (int i = 0; i < 9; i++) {
	  if (i % 3 == 0) {
	    printf("\n");
	  }
	  printf(" %f ", kernel[i]);
	}
	

    /*float newker[12]; // kerner conversion (padding to the right)

  /*  float newker[12]; // kerner conversion (padding to the right)

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
    kernel = newker;*/

   // printf("\n padded kernel"); //print loop; new kernel
   // for (int i = 0; i < 12; i++) {
//	  if (i % 4 == 0) {
	//    printf("\n");
	//  }
//	  printf(" %f", newker[i]);
//	}
	//printf("\n");
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
								kernel[(kern_cent_X-i)+(kern_cent_Y-j)*(KERNX)] * in[(x+i) + (y+j)*data_size_X];							
					//}
				}
			}
		}
	}

	//printf("top calculations \n");
	for(int i = 1; i <= data_size_X -2 ; i++){
	 // printf("input matrix edge: %d \n", i);
	    for (int k = i; k < data_size_X + i; k = k + data_size_X) {
	     // printf("input matrix edges being multiplied: %d %d %d \n", k-1, k, k+1);
	     // for (int g = 3; g<=8; g = g + 3) {
		//printf("\n \n kernel index multiplied by \n %f * %f  + \n %f * %f  + \n %f * %f  + \n %f * %f + \n %f * %f + \n %f * %f \n", in[k+1], kernel[6], in[k], kernel[5], in[k-1], kernel[4], in[k+1], kernel[10], in[k], kernel[9], in[k-1], kernel[7]);
		out[i] = (in[k+1] * kernel[3]) + (in[k] * kernel[4]) + (in[k-1] * kernel[5]) + (in[k+data_size_X+1] * kernel[0]) + (in[k+data_size_X] * kernel[1]) + (in[k+data_size_X-1] * kernel[2]); 
	      }
	  //  }
	}
	//printf(" \n \n \nleft hand side calculations \n");
	for(int i = data_size_X; i < size - data_size_X ; i = i + data_size_X){
	//printf("input matrix edge: %d \n", i);
	    for (int k = i - data_size_X; k < i; k = k + data_size_X) {
	    //  printf("input matrix edges being multiplied: %d %d \n", k, k+1);
	     // for (int g = 1; g<=8; g = g + 3) {
		//printf("kernel index multiplied by %d %d\n", g, g+1);
		//printf("\n \n new print kernel index multiplied by \n 0%f * %f  + \n %f * %f  + \n %f * %f  + \n %f * %f + \n %f * %f + \n %f * %f \n printted value at %f", in[k], kernel[1], in[k+1], kernel[2], in[k+ data_size_X], kernel[5], in[k+data_size_X+1], kernel[6], in[data_size_X + (data_size_X + k)], kernel[9], in[data_size_X + (data_size_X + k)+1], kernel[10], in[i]);
		out[i] = (in[k] * kernel[7]) + (in[k+1] * kernel[6]) + (in[k+ data_size_X] * kernel[4]) + (in[k+data_size_X+1] * kernel[3]) + (in[data_size_X + (data_size_X + k)] * kernel[1]) + (in[data_size_X + (data_size_X + k)+1] * kernel[0]); 
	    //  }
	    }
	}
	//printf("\n \n \n bottom calculations \n");
	for(int i = size - data_size_X + 1 ; i <= size - 2 ; i++){
	//  printf("input matrix edge: %d \n", i);
	    for (int k = i - data_size_X; k < i; k = k + data_size_X) {
	   //   printf("input matrix edges being multiplied: %d %d %d \n", k-1, k, k+1);
	      //for (int g = 0; g<=5; g = g + 3) {
	//	printf("kernel index multiplied by %d %d %d \n", g, g+1, g+2);
		out[i] = (in[k-1] * kernel[8]) + (in[k] * kernel[7]) + (in[k+1] * kernel[6]) + (in[k+data_size_X-1] * kernel[5]) + (in[k+data_size_X] * kernel[4]) + (in[k+data_size_X+1] * kernel[3]); 
	      //}
	    }
	}
	//printf("\n \n \n right hand side calculations \n");
	for(int i = (2 * data_size_X) - 1; i < size - data_size_X ; i = i + data_size_X){
	//  printf("input matrix edge: %d \n", i);
	    for (int k = i - data_size_X; k < i ; k = k + data_size_X) {
	//      printf("input matrix edges being multiplied: %d %d\n", k-1, k);
	  //    for (int g = 0; g<=8; g = g + 3) {
		//printf("multiplied by index location:%d %d\n", i,  g, g+1);
		//printf("\n \n kernel index multiplied by \n %f * %f  + \n %f * %f  + \n %f * %f  + \n %f * %f + \n %f * %f \n %f %f \n",in[k-1], kernel[0], in[k] , kernel[1], in[k + data_size_X -1], kernel[4], in[k + data_size_X], kernel[5], in[(k + data_size_X + data_size_X)-1], kernel[8], in[k + data_size_X + data_size_X], kernel[9]);
		out[i] = (in[k-1] * kernel[8]) + (in[k] * kernel[7]) + (in[k + data_size_X -1] * kernel[5]) + (in[k + data_size_X] * kernel[4]) + (in[(k + data_size_X + data_size_X)-1] * kernel[2]) + (in[k + data_size_X + data_size_X] * kernel[1]); 
	      //}
	    }
	}

	
	printf("\n");
	//Handles upper left corner
	for(int j = -kern_cent_Y; j <= kern_cent_Y-1; j++){ // kernel unflipped y coordinate
		for(int i = -1; i <= kern_cent_X-1; i++){ // kernel unflipped x coordinate
			out[0] += kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)] * in[((0+i) + (0+j)*data_size_X) * -1];
			printf("kern Index: %d, kern Value: %f, in index: %d, in value: %f\n", (kern_cent_X+i)+(kern_cent_Y+j)*(KERNX), kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)], ((0+i) + (0+j)*data_size_X) * -1, in[((0+i) + (0+j)*data_size_X) * -1]);
		}
	}
	
	printf("\n");
	//Handles upper right corner
	
	for(int j = -kern_cent_Y; j <= kern_cent_Y-1; j++){ // kernel unflipped y coordinate
		for(int i = 0; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
			out[data_size_X-1] += kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)] * in[(-(2*data_size_X-1) + (data_size_X+i) + (0+j)*data_size_X) * -1];
			printf("kern Index: %d, kern Value: %f, in index: %d, in value: %f\n", (kern_cent_X+i)+(kern_cent_Y+j)*(KERNX), kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)], (-(2*data_size_X-1) + (data_size_X+i) + (0+j)*data_size_X) * -1, in[(-(2*data_size_X-1) + (data_size_X+i) + (0+j)*data_size_X) * -1]);
		}
	}
	
	
	printf("\n");
	//Handles lower right corner
	for(int j = 0; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
		for(int i = 0; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
			out[(data_size_X*data_size_Y)-1] += kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)] * in[(-2*(size-1) + data_size_X-1+i + (data_size_Y-1+j)*data_size_X) * -1];
			printf("kern Index: %d, kern Value: %f, in index: %d, in value: %f\n", (kern_cent_X+i)+(kern_cent_Y+j)*(KERNX), kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)], (-2*(size-1) + data_size_X-1+i + (data_size_Y-1+j)*data_size_X) * -1, in[(-2*(size-1) + data_size_X-1+i + (data_size_Y-1+j)*data_size_X) * -1]);
		}
	}
	
	printf("\n");
	//Handles lower left corner
	for(int j = 0; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
		for(int i = -1; i <= kern_cent_X-1; i++){ // kernel unflipped x coordinate
			out[(data_size_X*data_size_Y)-data_size_X] += kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)] * in[(-(data_size_X*data_size_Y+data_size_X)+ 0+i + (data_size_Y-1+j)*data_size_X) * -1];
			printf("kern Index: %d, kern Value: %f, in index: %d, in value: %f\n", (kern_cent_X+i)+(kern_cent_Y+j)*(KERNX), kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)], (-(data_size_X*data_size_Y+data_size_X)+ 0+i + (data_size_Y-1+j)*data_size_X) * -1, in[(-(data_size_X*data_size_Y+data_size_X)+ 0+i + (data_size_Y-1+j)*data_size_X) * -1]);
		}
	}
	
	

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
