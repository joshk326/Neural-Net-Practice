#include <iostream>
#include <vector>
#include "Network.h"

int main()
{
	std::vector<unsigned> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);
	Net n(topology);
    
	std::vector<double> inputVals;
	n.feedForward(inputVals);
	
	std::vector<double> targetVals;
	n.backProp(targetVals);
	
	std::vector<double> resultVals;
	n.getResults(resultVals);
}
