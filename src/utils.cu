#include "config.h"

void print_tensor_size(const std::string& info,const at::Tensor data,int level){
	for(int i=0;i< level;i++)
	{
		std::cout<<"\t";
	}
	std::cout<<info<<" ";
	for(int i=0;i< data.ndimension();i++)
	{
		std::cout<<data.size(i)<<" ";
	}
	std::cout<<std::endl;
}

void function_info(const std::string& info,int level){

	for(int i=0;i< level;i++)
	{
		std::cout<<"\t";
	}
	std::cout<<info<<" ";
	std::cout<<std::endl;
}





