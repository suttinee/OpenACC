
default:
	cd BFS && make TARGET=OPENCL && make TARGET=CUDA;
	cd GE && make TARGET=OPENCL && make TARGET=CUDA;
	cd LUD  && make TARGET=OPENCL && make TARGET=CUDA;
	cd BP && make TARGET=OPENCL && make TARGET=CUDA;


clean:
	cd BFS && make clean
	cd LUD && make clean
	cd GE && make clean
	cd BP && make clean

run:
	echo "BFS APPLICATION+++++++++++++++++++++++++ "
	cd bin/bfs/ && ./run.sh
	echo "GE APPLICATION+++++++++++++++++++++++++ "
	cd bin/ge/ && ./run.sh
	echo "LUD APPLICATION+++++++++++++++++++++++++ "
	cd bin/lud/ && ./run.sh
	echo "BP APPLICATION+++++++++++++++++++++++++ "
	cd bin/bp/ && ./run.sh

