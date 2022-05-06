#include "Network.h"

Neuron::Neuron(unsigned numOutput, unsigned myIndexInput) {
	for (unsigned c = 0; c < numOutput; ++c) {
		outputWeights.push_back(Connection());
		outputWeights.back().weight = randomWeight();
	}

	myIndexInput = myIndex;
}

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

void Neuron::updateInputWeights(Layer& prevLayer) {
	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		Neuron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron.outputWeights[myIndex].deltaWeight;

		double newDeltaWeight = eta * neuron.getOutputVal() * gradient + alpha * oldDeltaWeight;
	
		neuron.outputWeights[myIndex].deltaWeight = newDeltaWeight;
		neuron.outputWeights[myIndex].weight += newDeltaWeight;
	}
}

double Neuron::sumDOW(const Layer& nextLayer) const {
	double sum = 0.0;
	
	for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
		sum += outputWeights[n].weight * nextLayer[n].gradient;
	}

	return sum;
}

void Neuron::calcOutputGradients(double targetVal) {
	double delta = targetVal - outputVal;
	gradient = delta * Neuron::transferFunctionDeriv(outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
	double dow = sumDOW(nextLayer);
	gradient = dow * Neuron::transferFunctionDeriv(outputVal);
}

double Neuron::transferFunction(double x) {
	return tanh(x);
}

double Neuron::transferFunctionDeriv(double x) {
	return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer) {
	double sum = 0.0;

	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		sum += prevLayer[n].getOutputVal() * prevLayer[n].outputWeights[myIndex].weight;
	}

	outputVal = Neuron::transferFunction(sum);
}