#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define TRANSFER_GRAPH_NODE 1

#define REPS 10

#define BOOL_FMT(bool_expr) "%s\n", #bool_expr, (bool_expr) ? "true" : "false"

int no_of_nodes;
int edge_list_size;
FILE *fp;

struct timeval tTime1, tTime2;

//Structure to hold a node information
typedef struct Node {
	int starting;
	int no_of_edges;
} Node;

typedef enum bool{false=0, true=1} bool;

void BFSGraph(int argc, char** argv);

void Usage(int argc, char**argv) {
	fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
}



/*
 * This function ensures the device is awake.
 * It is more portable than acc_init().
 */
void wul(){

  int data = 8192;
  double *arr_a = (double *)malloc(sizeof(double) * data);
  double *arr_b = (double *)malloc(sizeof(double) * data);
  int i = 0;

  if (arr_a==NULL||arr_b==NULL) {
      printf("Unable to allocate memory in wul.\n");
  }

  for (i=0;i<data;i++){
    arr_a[i] = (double) (rand()/(1.0+RAND_MAX));
  }

#pragma acc data copy(arr_b[0:data]), copyin(arr_a[0:data])
  {
#pragma acc parallel loop
    for (i=0;i<data;i++){
      arr_b[i] = arr_a[i] * 2;
    }
  }

  if (arr_a[0] < 0){
    printf("Error in WUL\n");
    /*
     * This should never be called as rands should be in the range (0,1].
     * This stops clever optimizers.
     */
  }

  free(arr_a);
  free(arr_b);

}





////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	no_of_nodes = 0;
	edge_list_size = 0;
        wul();
int rep = 0;

//for (rep = 0; rep < REPS; rep++) {
	BFSGraph(argc, argv);
//}
//
return(0);
}

////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(int argc, char** argv) {
	char *input_f;

//	int* h_cost;
	int* h_graph_edges;

	if( (argc != 2) && (argc !=4)){
		Usage(argc, argv);
		exit(0);
	}

	input_f = argv[1];

	//printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f, "r");
	if (!fp) {
		printf("Error Reading graph file\n");
		return;
	}

	int source = 0;

	fscanf(fp, "%d", &no_of_nodes);
	// allocate host memory
	int *h_graph_nodes_starting = (int*)  malloc(sizeof(int) * no_of_nodes);
	int *h_graph_nodes_no_of_edges = (int*)  malloc(sizeof(int) * no_of_nodes);
	bool *h_graph_mask = (bool*) malloc(sizeof(bool) * no_of_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool) * no_of_nodes);
	bool *h_graph_visited = (bool*) malloc(sizeof(bool) * no_of_nodes);

	int start, edgeno;
	unsigned int i1 = 0;
	int i2 = 0;
	// initalize the memory
	for (i1 = 0; i1 < no_of_nodes; i1++) {
		fscanf(fp, "%d %d", &start, &edgeno);
		h_graph_nodes_starting[i1] = start;
		h_graph_nodes_no_of_edges[i1] = edgeno;

	}

	//read the source node from the file
	fscanf(fp, "%d", &source);
	//printf("source = %d\n",source);
	source = 0;



	fscanf(fp, "%d", &edge_list_size);
	int id, cost;
	h_graph_edges = (int*) malloc(sizeof(int) * edge_list_size);
	for (i2 = 0; i2 < edge_list_size; i2++) {
		fscanf(fp, "%d", &id);
		fscanf(fp, "%d", &cost);
		h_graph_edges[i2] = id;
	}

	if (fp)
		fclose(fp);

	//printf("finish read file\n");
        // allocate mem for the result on host side
        int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
		int i3=0;
//		#pragma acc kernels 	
		{
//		#pragma acc loop independent
		for (i3 = 0; i3 < no_of_nodes; i3++) {
			h_updating_graph_mask[i3] = false;
			h_graph_mask[i3] = false;
			h_graph_visited[i3] = false;

			//set the source node as true in the mask
			if (i3 == source) {
				h_graph_mask[source] = true;
				h_graph_visited[source] = true;
			}
		}

		 int i4=0;	
//		#pragma acc loop independent
		for (i4 = 0; i4 < no_of_nodes; i4++) {
			h_cost[i4] = -1.0;
			if (i4 == source)
				h_cost[source] = 0.0;
		}

		}
	
      gettimeofday(&tTime1, NULL);
#pragma acc data copyin(h_updating_graph_mask[0:no_of_nodes]), \
		copyin(h_graph_mask[0:no_of_nodes],h_graph_visited[0:no_of_nodes]), \
		copyin(h_graph_nodes_starting[0:no_of_nodes], h_graph_nodes_no_of_edges[0:no_of_nodes], \
			h_graph_edges[0:edge_list_size]), \
		copy(h_cost[0:no_of_nodes])
	{	
	// finish transfer node and edge to target
	//printf("Start traversing the tree\n");

		int k = 0;
		bool stop;
		int temp;
		do {
			//if no thread changes this value then the loop stops
			#pragma acc kernels 
			{
			stop = false;
			int tid1;
			#ifdef INDEPENDENT
			#pragma acc loop independent	 
			#else
			#pragma acc loop 
			#endif
			for (tid1 = 0; tid1 < no_of_nodes; tid1++) {
				if (h_graph_mask[tid1] == true) {
					h_graph_mask[tid1] = false;
					int i5=0;
					#ifdef ILP
					#pragma hmppcg(OPENCL) unroll(8), jam
					#endif
					for (i5 = h_graph_nodes_starting[tid1];i5 < (h_graph_nodes_no_of_edges[tid1] + h_graph_nodes_starting[tid1]);i5++) {
						int id = h_graph_edges[i5];
						if (h_graph_visited[id]==false){
							h_cost[id]= h_cost[tid1]+1;
							h_updating_graph_mask[id] = true;
						}
						}
				}
			  }
			

			int tid2;
			/*default version came with gang(192) and reduction directive*/
			#pragma acc kernels loop gang(192) reduction(||:stop)
			for (tid2  = 0; tid2 < no_of_nodes; tid2++) {
				if (h_updating_graph_mask[tid2] == true) {
					h_graph_mask[tid2] = true;
					h_graph_visited[tid2] = true;
					stop = true;
					h_updating_graph_mask[tid2] = false;
				}
			}
			}
	
		} while (stop);
	}//end pragma data
        gettimeofday(&tTime2, NULL);
	 /* end acc data */


	double copy_time = 11.0;
	 copy_time = (tTime2.tv_sec - tTime1.tv_sec)*1000000 + (tTime2.tv_usec - tTime1.tv_usec);
	


	int i6 = 0;
	//Store the result into a file
	FILE *fpo = fopen("out.txt", "w");
	for (i6 = 0; i6 < no_of_nodes; i6++)
		fprintf(fpo, "%d) cost:%d\n", i6, h_cost[i6]);
	fclose(fpo);
	//printf("Result stored in file out\n");

	printf("Time cost: %.6lf millisecond \n", copy_time/1000);


	// cleanup memory
	free(h_graph_nodes_starting);
	free(h_graph_nodes_no_of_edges);
	free(h_graph_edges);
	free(h_graph_mask);
	free(h_updating_graph_mask);
	free(h_graph_visited);
	free(h_cost);

}
