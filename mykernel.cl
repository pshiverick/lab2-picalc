__kernel void calculatePi(
    __global int *bufferDen,
		__local float *sumBuf,
		__global float *resultBuf,
		__global float *debug
		)
{
	/* initialize memory items */
		sumBuf[0] = 0;
		sumBuf[1] = 0;

		resultBuf[0] = 0;
		resultBuf[1] = 0;

		barrier(CLK_GLOBAL_MEM_FENCE);
		
	  const int num_vars = 2;
		float values[num_vars];
		int i = 0;
		int index = 0;

    int gid = get_global_id(0); /* work group 	*/
		int lid = get_local_id(0);  /* work item 		*/
		int wi_size = get_local_size(0); /* # work items */

		for(i = 0; i < num_vars; i++)
		{
			index = gid*num_vars*wi_size + wi_size*lid + i;
			values[i] = (float) bufferDen[index];
		}

		sumBuf[lid] = 1.0/values[0];
		for(i = 1; i < num_vars; i++)
		{
			sumBuf[lid] -= 1.0/values[i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if(lid == 0)
		{
			for(i = 0; i < wi_size; i++) 
			{
				resultBuf[gid] += sumBuf[i];
			}
		}
}
