__kernel void calculatePi(
    __global int *bufferDen,
		__local float *sumBuf,
		__global float *resultBuf
		)
{
	/* Execute 16 calculations per work item */
		const int num_vars_per_wi = 16;
		float values[num_vars_per_wi];
		int i = 0;
		int index = 0;
		float wi_sum = 0;
		float local_sum = 0;

		/* Need the following values to find the index of the
			 * denominator that we need to find */
    int gid = get_group_id(0); /* work group 	*/
		int lid = get_local_id(0);  /* work item 		*/
		int wi_size = get_local_size(0); /* # work items */
		
		/* Extract array of denominators that we will work on */
		for(i = 0; i < num_vars_per_wi; i++)
		{
			index = gid*num_vars_per_wi*wi_size + wi_size*lid + i;
			values[i] = (float) bufferDen[index];
		}

		/* Add up 1/X - 1/Y + 1/Z - 1/A. etc... */
		float small_sum = 0;
		index = 0;
		for(i = 0; i < num_vars_per_wi/2; i++){
			small_sum = (float)1/values[index] - (float)1/values[index+1];
			wi_sum += small_sum;
			index += 2;
		}

		/* Add final sum of work item to local memory */
		sumBuf[lid] = wi_sum;

		/* Wait for all work items in work group to finish processing */
		barrier(CLK_LOCAL_MEM_FENCE);

		/* Work item 0 will add up all of the sums of the work items in each
			 * work group */
		if(lid == 0)
		{
			for(i = 0; i < wi_size; i++) 
			{
				local_sum += sumBuf[i];
			}

			/* Add final result to output buffer. This value is the result
				 * of all the work items operations. */
			resultBuf[gid] = local_sum;
		}
}
