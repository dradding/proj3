#include <emmintrin.h>
#define KERNX 3//this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
#include <omp.h>
#define unroll 4
//Authors: Ayush Mudgal (cs61c-cj) and Daniel Radding (cs61c-vh)
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
    {

  //printf("Kernal \n"); //print loop

  /*printf("old \n"); //print loop

for (int i = 0; i < 9; i++) {
         if (i % 3 == 0) {
         printf("\n");
         }
         printf(" %f ", kernel[i]);
        }*/
        

    /*float newker[12]; // kerner conversion (padding to the right)*/

    float newker[KERNX * KERNY]; // kerner conversion (padding to the right)

   // int count2 = -1;
    //__m128 kern_intrin[KERNX * KERNY];
    for (int count = 0; count < (KERNX * KERNY); count++) {
        newker[count] = kernel[count];
        //kern_intrin[count] = _mm_set_ps1 (kernel[count]);
      }
    kernel = newker;
    int size = (data_size_X) * (data_size_Y);
    float newin[size]; // kerner conversion (padding to the right)

    //int count2 = -1;
 
    for (int count = 0; count < size; count++) {
        newin[count] = in[count];
      }
    in = newin;

   // printf("\n padded kernel"); //print loop; new kernel
   // for (int i = 0; i < 12; i++) {
//         if (i % 4 == 0) {
        // printf("\n");
        // }
//         printf(" %f", newker[i]);
//        }
        //printf("\n");
    /*data_size_X += 3; //padding the input matrix all around
data_size_Y += 2;
int size = (data_size_X) * (data_size_Y);
float newin[size];
count2 = -1;
for (int count = 0; count < size; count++) {
0 if (count < data_size_X|| count % data_size_X == 1 ||count % data_size_X == data_size_X - 1 || count % data_size_X == 0 || count > size - data_size_X) {
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
    __m128 outload;
    __m128 inload;
    __m128 kernelload;
    //printf("center of the kernel %d %d", kern_cent_X, kern_cent_Y); //uncomment to print the center of the kernel
    // main convolution loop
    /*for(int r = 0; r < 10; r++){
	  lsakjfdl;sajf
    }
    sum[x] += 2;
    sum[x+1] += 2;
    sum[x+2] += 2;
    sum[x+3] += 2;
    for(int r = 0; r < UNROLL; r++){
      sum[x+r] += in[x + r];
    }*/
    
    
	      for(int y = 1; y<data_size_Y-1; y++){   
	          for(int x = 1; x < (data_size_X-1)/4*4; x+=4){
		    for(int i = -kern_cent_X; i<=kern_cent_X; i++){
		      for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){
				  //printf("out index: %d\n",x+y*data_size_X); 
				  inload = _mm_loadu_ps (in+((x+i)+(y+j)*data_size_X));
				  outload = _mm_loadu_ps (out+(x+y*data_size_X));
				  kernelload = _mm_load_ps1(kernel + ((kern_cent_X-i)+(kern_cent_Y-j)*(KERNX)));
				  inload = _mm_mul_ps (kernelload , inload);
				  outload = _mm_add_ps (outload, inload);
				  _mm_storeu_ps ((out+(x+y*data_size_X)), outload);
				  
			   	
		    //for(int u = (data_size_X-1)/4*4 + (4 - data_size_X%4); u < (data_size_X-1); u++){
// 				  for(int u = (data_size_X- data_size_X%4); u < (data_size_X-1); u++){
// 				  //printf("out value %d \n", out[u+y*data_size_X]);
// 				   // printf("in index:%f, i:%d, j:%d \n", in[(u+i) + (y+j)*data_size_X],i,j);
// 				  //printf("in index %d \n", (u+i) + (y+j)*data_size_X);
// 				  //printf("%d \n", u);
// 				    out[u+y*data_size_X+1] +=
// 				    kernel[(kern_cent_X-i)+(kern_cent_Y-j)*(KERNX)] * in[(u+i) + (y+j)*data_size_X+1];                              }
		      }
			  }
		      }
		      
		 }
	      for(int y = 1; y<data_size_Y-1; y++){
		for(int u = ((data_size_X-1) - ((data_size_X-1)%4)); u < (data_size_X-2); u++){	    
		  for(int i = -kern_cent_X; i<=kern_cent_X; i++){
		      for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){
						printf("out index %d \n", u+y*data_size_X+1);
						// printf("in index:%f, i:%d, j:%d \n", in[(u+i) + (y+j)*data_size_X],i,j);
						//printf("in index %d \n", (u+i) + (y+j)*data_size_X);
						//printf("%d \n", u);
						  
						  out[u+y*data_size_X+1] +=
						  kernel[(kern_cent_X-i)+(kern_cent_Y-j)*(KERNX)] * in[(u+i) + (y+j)*data_size_X+1];
					    }
			      
				      }
			      }
		      }

    
       /*for(int i = -kern_cent_X; i<=kern_cent_X; i++){
            for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){
	      for(int y = 1; y<data_size_Y-1; y++){   
	          for(int x = 1; x < (data_size_X-1)/(unroll*unroll)*(unroll*unroll); x+=unroll){
		           for(int r = 0; r<unroll*unroll; r+=unroll){
				  inload = _mm_loadu_ps (in+((x+r*4+i)+(y+j)*data_size_X));
				  outload = _mm_loadu_ps (out+(x+r*4+y*data_size_X));
				  kernelload = _mm_load_ps1(kernel + ((kern_cent_X-i)+(kern_cent_Y-j)*(KERNX)));
				  inload = _mm_mul_ps (kernelload , inload);
				  outload = _mm_add_ps (outload, inload);
				  _mm_storeu_ps ((out+(x+r*4+y*data_size_X)), outload);
			   }
		      }
		 }
	    }
	 
       }*/
       
       /*for(int i = -kern_cent_X; i<=kern_cent_X; i++){
            for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){
	      for(int y = 1; y<data_size_Y-1; y++){   
	          for(int x = 1; x < ((data_size_X-1)/16*16); x+=16){
				  inload = _mm_loadu_ps (in+((x+i)+(y+j)*data_size_X));
				  outload = _mm_loadu_ps (out+(x+y*data_size_X));
				  kernelload = _mm_load_ps1(kernel + ((kern_cent_X-i)+(kern_cent_Y-j)*(KERNX)));
				  inload = _mm_mul_ps (kernelload , inload);
				  outload = _mm_add_ps (outload, inload);
				  _mm_storeu_ps ((out+(x+y*data_size_X)), outload);
		    
				  inload = _mm_loadu_ps (in+((x+4+i)+(y+j)*data_size_X));
				  outload = _mm_loadu_ps (out+(x+4+y*data_size_X));
				  kernelload = _mm_load_ps1(kernel + ((kern_cent_X-i)+(kern_cent_Y-j)*(KERNX)));
				  inload = _mm_mul_ps (kernelload , inload);
				  outload = _mm_add_ps (outload, inload);
				  _mm_storeu_ps ((out+(x+4+y*data_size_X)), outload);
				  
				  inload = _mm_loadu_ps (in+((x+8+i)+(y+j)*data_size_X));
				  outload = _mm_loadu_ps (out+(x+8+y*data_size_X));
				  kernelload = _mm_load_ps1(kernel + ((kern_cent_X-i)+(kern_cent_Y-j)*(KERNX)));
				  inload = _mm_mul_ps (kernelload , inload);
				  outload = _mm_add_ps (outload, inload);
				  _mm_storeu_ps ((out+(x+8+y*data_size_X)), outload);
				  
				  inload = _mm_loadu_ps (in+((x+12+i)+(y+j)*data_size_X));
				  outload = _mm_loadu_ps (out+(x+12+y*data_size_X));
				  kernelload = _mm_load_ps1(kernel + ((kern_cent_X-i)+(kern_cent_Y-j)*(KERNX)));
				  inload = _mm_mul_ps (kernelload , inload);
				  outload = _mm_add_ps (outload, inload);
				  _mm_storeu_ps ((out+(x+12+y*data_size_X)), outload);
				  
			   }
		      }
		 }
	    }*/
	 
       
       /*for(int i = -kern_cent_X; i<=kern_cent_X; i++){
            for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){
	      for(int y = 1; y<data_size_Y-1; y++){   
	          for(int x = 1; x < (data_size_X-1)/(unroll*unroll)*(unroll*unroll); x+=4){
			inload = _mm_loadu_ps (in+((x+i)+(y+j)*data_size_X));
			outload = _mm_loadu_ps (out+(x+y*data_size_X));
			kernelload = _mm_load_ps1(kernel + ((kern_cent_X-i)+(kern_cent_Y-j)*(KERNX)));
			inload = _mm_mul_ps (kernelload , inload);
			outload = _mm_add_ps (outload, inload);
			_mm_storeu_ps ((out+(x+y*data_size_X)), outload);
		      }
		 }
	    }
	 
       }*/
						  
        //printf("top calculations \n");
        for(int i = 1; i <= data_size_X -2 ; i++){
         // printf("input matrix edge: %d \n", i);
         for (int k = i; k < data_size_X + i; k = k + data_size_X) {
         // printf("input matrix edges being multiplied: %d %d %d \n", k-1, k, k+1);
         // for (int g = 3; g<=8; g = g + 3) {
                //printf("\n \n kernel index multiplied by \n %f * %f + \n %f * %f + \n %f * %f + \n %f * %f + \n %f * %f + \n %f * %f \n", in[k+1], kernel[6], in[k], kernel[5], in[k-1], kernel[4], in[k+1], kernel[10], in[k], kernel[9], in[k-1], kernel[7]);
                out[i] = (in[k+1] * kernel[3]) + (in[k] * kernel[4]) + (in[k-1] * kernel[5]) + (in[k+data_size_X+1] * kernel[0]) + (in[k+data_size_X] * kernel[1]) + (in[k+data_size_X-1] * kernel[2]);
         }
         // }
        }
        //printf(" \n \n \nleft hand side calculations \n");
        for(int i = data_size_X; i < size - data_size_X ; i = i + data_size_X){
        //printf("input matrix edge: %d \n", i);
         for (int k = i - data_size_X; k < i; k = k + data_size_X) {
         // printf("input matrix edges being multiplied: %d %d \n", k, k+1);
         // for (int g = 1; g<=8; g = g + 3) {
                //printf("kernel index multiplied by %d %d\n", g, g+1);
                //printf("\n \n new print kernel index multiplied by \n 0%f * %f + \n %f * %f + \n %f * %f + \n %f * %f + \n %f * %f + \n %f * %f \n printted value at %f", in[k], kernel[1], in[k+1], kernel[2], in[k+ data_size_X], kernel[5], in[k+data_size_X+1], kernel[6], in[data_size_X + (data_size_X + k)], kernel[9], in[data_size_X + (data_size_X + k)+1], kernel[10], in[i]);
                out[i] = (in[k] * kernel[7]) + (in[k+1] * kernel[6]) + (in[k+ data_size_X] * kernel[4]) + (in[k+data_size_X+1] * kernel[3]) + (in[data_size_X + (data_size_X + k)] * kernel[1]) + (in[data_size_X + (data_size_X + k)+1] * kernel[0]);
         // } # pragma omp parallel
         }
        }
        //printf("\n \n \n bottom calculations \n");
        for(int i = size - data_size_X + 1 ; i <= size - 2 ; i++){
        // printf("input matrix edge: %d \n", i);
         for (int k = i - data_size_X; k < i; k = k + data_size_X) {
         // printf("input matrix edges being multiplied: %d %d %d \n", k-1, k, k+1);
         //for (int g = 0; g<=5; g = g + 3) {
        //        printf("kernel index multiplied by %d %d %d \n", g, g+1, g+2);
                out[i] = (in[k-1] * kernel[8]) + (in[k] * kernel[7]) + (in[k+1] * kernel[6]) + (in[k+data_size_X-1] * kernel[5]) + (in[k+data_size_X] * kernel[4]) + (in[k+data_size_X+1] * kernel[3]);
         //}
         }
        }
        //printf("\n \n \n right hand side calculations \n");
        for(int i = (2 * data_size_X) - 1; i < size - data_size_X ; i = i + data_size_X){
        // printf("input matrix edge: %d \n", i);
         for (int k = i - data_size_X; k < i ; k = k + data_size_X) {
        // printf("input matrix edges being multiplied: %d %d\n", k-1, k);
         // for (int g = 0; g<=8; g = g + 3) {
                //printf("multiplied by index location:%d %d\n", i, g, g+1);
                //printf("\n \n kernel index multiplied by \n %f * %f + \n %f * %f + \n %f * %f + \n %f * %f + \n %f * %f \n %f %f \n",in[k-1], kernel[0], in[k] , kernel[1], in[k + data_size_X -1], kernel[4], in[k + data_size_X], kernel[5], in[(k + data_size_X + data_size_X)-1], kernel[8], in[k + data_size_X + data_size_X], kernel[9]);
                out[i] = (in[k-1] * kernel[8]) + (in[k] * kernel[7]) + (in[k + data_size_X -1] * kernel[5]) + (in[k + data_size_X] * kernel[4]) + (in[(k + data_size_X + data_size_X)-1] * kernel[2]) + (in[k + data_size_X + data_size_X] * kernel[1]);
         //}
         }
        }

        
        //printf("\n");
        //Handles upper left corner
        for(int j = -kern_cent_Y; j <= kern_cent_Y-1; j++){ // kernel unflipped y coordinate
                for(int i = -1; i <= kern_cent_X-1; i++){ // kernel unflipped x coordinate
                        out[0] += kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)] * in[((i) + (j)*data_size_X) * -1];
                        //printf("kern Index: %d, kern Value: %f, in index: %d, in value: %f\n", (kern_cent_X+i)+(kern_cent_Y+j)*(KERNX), kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)], ((0+i) + (0+j)*data_size_X) * -1, in[((0+i) + (0+j)*data_size_X) * -1]);
                }
        }
        
        //printf("\n");
        //Handles upper right corner
        for(int j = -kern_cent_Y; j <= kern_cent_Y-1; j++){ // kernel unflipped y coordinate
                for(int i = 0; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
                        out[data_size_X-1] += kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)] * in[(-(2*data_size_X-1) + (data_size_X+i) + (j)*data_size_X) * -1];
                        //printf("kern Index: %d, kern Value: %f, in index: %d, in value: %f\n", (kern_cent_X+i)+(kern_cent_Y+j)*(KERNX), kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)], (-(2*data_size_X-1) + (data_size_X+i) + (0+j)*data_size_X) * -1, in[(-(2*data_size_X-1) + (data_size_X+i) + (0+j)*data_size_X) * -1]);
                }
        }
        
        
        //printf("\n");
        //Handles lower right corner
        for(int j = 0; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
                for(int i = 0; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
                       out[(data_size_X*data_size_Y)-1] += kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)] * in[(-2*(size-1) + data_size_X-1+i + (data_size_Y-1+j)*data_size_X) * -1];
                        //printf("kern Index: %d, kern Value: %f, in index: %d, in value: %f\n", (kern_cent_X+i)+(kern_cent_Y+j)*(KERNX), kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)], (-2*(size-1) + data_size_X-1+i + (data_size_Y-1+j)*data_size_X) * -1, in[(-2*(size-1) + data_size_X-1+i + (data_size_Y-1+j)*data_size_X) * -1]);
                }
        }
        
        //printf("\n");
        //Handles lower left corner
        for(int j = 0; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
                for(int i = -1; i <= kern_cent_X-1; i++){ // kernel unflipped x coordinate
                        out[(data_size_X*data_size_Y)-data_size_X] += kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)] * in[((-i)*-1 + (2*data_size_X*j)- (data_size_Y-1+j)*data_size_X)*-1];
                        //printf("kern Index: %d, kern Value: %f, in index: %d, in value: %f\n", (kern_cent_X+i)+(kern_cent_Y+j)*(KERNX), kernel[(kern_cent_X+i)+(kern_cent_Y+j)*(KERNX)],((-i)*-1 + (2*data_size_X*j)- (data_size_Y-1+j)*data_size_X)*-1, in[((-i)*-1 + (2*data_size_X*j)- (data_size_Y-1+j)*data_size_X)*-1]);
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
        
   
  }
  return 1;
}

