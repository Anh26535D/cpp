#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>

class ClusteringAlgo {
public:
    virtual void fit(const std::vector<double>& data, int k, int maxIterations, double tolerance) = 0;
    virtual void printResults() const = 0;
    virtual ~ClusteringAlgo() = default;
};

struct Gaussian {
    double mean;
    double variance;
    double weight;
};

double gaussianProbability(double x, double mean, double variance) {
    double exponent = std::exp(-0.5 * std::pow((x - mean) / std::sqrt(variance), 2));
    return (1.0 / std::sqrt(2 * M_PI * variance)) * exponent;
}

class GMMClustering : public ClusteringAlgo {
private:
    std::vector<Gaussian> components;
    std::vector<std::vector<double>> responsibilities;
    double prevLogLikelihood;

public:
    void fit(const std::vector<double>& data, int k, int maxIterations, double tolerance) override {
        int n = data.size();
        components.resize(k);
        responsibilities.assign(n, std::vector<double>(k, 0.0));
        prevLogLikelihood = -std::numeric_limits<double>::infinity();

        // Initialize GMM components
        std::default_random_engine generator;
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double minVal = *std::min_element(data.begin(), data.end());
        double maxVal = *std::max_element(data.begin(), data.end());

        for (int i = 0; i < k; ++i) {
            components[i].mean = minVal + (maxVal - minVal) * dist(generator);
            components[i].variance = 1.0;
            components[i].weight = 1.0 / k;
        }

        for (int iteration = 0; iteration < maxIterations; ++iteration) {
            // E-step
            for (int i = 0; i < n; ++i) {
                double totalProb = 0.0;
                for (int j = 0; j < k; ++j) {
                    responsibilities[i][j] = components[j].weight * gaussianProbability(data[i], components[j].mean, components[j].variance);
                    totalProb += responsibilities[i][j];
                }
                for (int j = 0; j < k; ++j) {
                    responsibilities[i][j] /= totalProb;
                }
            }

            // M-step
            for (int j = 0; j < k; ++j) {
                double weightSum = 0.0;
                double meanSum = 0.0;
                double varianceSum = 0.0;

                for (int i = 0; i < n; ++i) {
                    weightSum += responsibilities[i][j];
                    meanSum += responsibilities[i][j] * data[i];
                }

                components[j].mean = meanSum / weightSum;

                for (int i = 0; i < n; ++i) {
                    varianceSum += responsibilities[i][j] * std::pow(data[i] - components[j].mean, 2);
                }

                components[j].variance = varianceSum / weightSum;
                components[j].weight = weightSum / n;
            }

            // Log-likelihood
            double logLikelihood = 0.0;
            for (const auto& x : data) {
                double prob = 0.0;
                for (const auto& component : components) {
                    prob += component.weight * gaussianProbability(x, component.mean, component.variance);
                }
                logLikelihood += std::log(prob);
            }

            if (std::abs(logLikelihood - prevLogLikelihood) < tolerance) {
                break;
            }

            prevLogLikelihood = logLikelihood;
        }
    }

    void printResults() const override {
        for (size_t j = 0; j < components.size(); ++j) {
            std::cout << "Component " << j + 1 << ":\n";
            std::cout << "  Mean: " << components[j].mean << "\n";
            std::cout << "  Variance: " << components[j].variance << "\n";
            std::cout << "  Weight: " << components[j].weight << "\n";
        }
    }
};

class KMeanClustering : public ClusteringAlgo {
private:
    std::vector<double> centroids;
    std::vector<int> assignments;

public:
    void fit(const std::vector<double>& data, int k, int maxIterations, double tolerance) override {
        int n = data.size();
        centroids.resize(k);
        assignments.resize(n, -1);

        // Initialize centroids randomly
        std::default_random_engine generator;
        std::uniform_int_distribution<int> dist(0, n - 1);
        for (int i = 0; i < k; ++i) {
            centroids[i] = data[dist(generator)];
        }

        for (int iteration = 0; iteration < maxIterations; ++iteration) {
            // Assign data points to nearest centroid
            bool changed = false;
            for (int i = 0; i < n; ++i) {
                int bestCluster = -1;
                double bestDistance = std::numeric_limits<double>::infinity();
                for (int j = 0; j < k; ++j) {
                    double distance = std::abs(data[i] - centroids[j]);
                    if (distance < bestDistance) {
                        bestDistance = distance;
                        bestCluster = j;
                    }
                }
                if (assignments[i] != bestCluster) {
                    assignments[i] = bestCluster;
                    changed = true;
                }
            }

            if (!changed) break;

            // Update centroids
            std::vector<double> newCentroids(k, 0.0);
            std::vector<int> counts(k, 0);

            for (int i = 0; i < n; ++i) {
                newCentroids[assignments[i]] += data[i];
                counts[assignments[i]]++;
            }

            for (int j = 0; j < k; ++j) {
                if (counts[j] > 0) {
                    centroids[j] = newCentroids[j] / counts[j];
                }
            }
        }
    }

    void printResults() const override {
        for (size_t i = 0; i < centroids.size(); ++i) {
            std::cout << "Centroid " << i + 1 << ": " << centroids[i] << "\n";
        }
    }
};

int main() {
    std::vector<double> data = {1.0, 1.2, 1.1, 2.5, 2.4, 2.3, 5.0, 5.2, 5.1};
    int k = 3;
    int maxIterations = 100;
    double tolerance = 1e-6;

    std::cout << "Gaussian Mixture Model Clustering:\n";
    GMMClustering gmm;
    gmm.fit(data, k, maxIterations, tolerance);
    gmm.printResults();

    std::cout << "\nK-Means Clustering:\n";
    KMeanClustering kmeans;
    kmeans.fit(data, k, maxIterations, tolerance);
    kmeans.printResults();

    return 0;
}
