#include <vector>
#include "Neuron.h"

class Net {
public:
	Net(const std::vector<unsigned>& topology);
	void feedForward(const std::vector<double>& inputVals);
	void backProp(const std::vector<double>& targetVals);
	void getResults(std::vector<double>& resultVals) const;
private:
	std::vector<Layer> layers; //layers[layernum][neuronNum]
	double error;
	double recentAverageError;
	double recentAverageSmoothingFactor;

};