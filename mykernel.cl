__kernel void calculatePi(
    __global float *wgSum,
		)
{
	  /* get work item id */
    int wi = get_global_id(0);
}
