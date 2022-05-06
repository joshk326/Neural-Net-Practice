#include <vector>

struct Connection {
	double weight;
	double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron {
public:
	Neuron(unsigned numOutput, unsigned myIndex);
	void setOutputVal(double val) { outputVal = val;  };
	double getOutputVal(void) const { return outputVal; };
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);

private:
	static double eta;
	static double alpha;
	static double transferFunction(double x);
	static double transferFunctionDeriv(double x);
	double sumDOW(const Layer &nextLayer) const;
	double outputVal;
	std::vector<Connection> outputWeights;
	static double randomWeight(void) { return (rand() / double(RAND_MAX)); }
	unsigned myIndex;
	double gradient;
};
