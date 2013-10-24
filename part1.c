#include <emmintrin.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
    
    float newker[16];
    int count2 = -1; 
    for (int count = 0; count < 16; count++) {
      if (count <= 3 || count % 4 == 0) {
	    newker[count] = 0;
      }
      else {
	count2++;
	newker[count] = kernel[count2];
      }
    }
    kernel = newker;
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;
    
    // main convolution loop
	for(int y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on
		for(int x = 0; x < data_size_X; x++){ // the x coordinate of the output location we're focusing on
				for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
					for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
					// only do the operation if not out of bounds
						if(x+i>-1 && x+i<data_size_X && y+j>-1 && y+j<data_size_Y){
						//Note that the kernel is flipped
							out[x+y*data_size_X] += 
								kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X];
					}
				}
			}
		}
	}
	return 1;
}
