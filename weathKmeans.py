from mrjob.job import MRJob
from mrjob.step import MRStep
import json

class KMeansMR(MRJob):
    def steps(self):
        return [
            MRStep(mapper_init=self.mapper_init, mapper=self.mapper, combiner=self.combiner, reducer=self.reducer),
            MRStep(reducer=self.final_reducer)
        ]

    def configure_args(self):
        super(KMeansMR, self).configure_args()
        self.add_passthru_arg('--k', type=int, default=3, help='Number of clusters (k)')
        self.add_passthru_arg('--iterations', type=int, default=10, help='Number of iterations')

    def mapper_init(self):
        # Load centroids from the initial centroids file or generate random centroids
        centroids_path = 'initial_centroids.txt'
        try:
            with open(centroids_path, 'r') as f:
                self.centroids = json.load(f)
        except FileNotFoundError:
            self.centroids = None

    def mapper(self, _, line):
        # Parse the CSV line
        # Assuming the last 5 elements are numeric and need to be considered for clustering
        # Ignoring the first element which is not numeric
        data = list(map(float, line.strip().split(',')[9:]))
    
        # Assign each data point to the nearest centroid
        closest_centroid = min(self.centroids, key=lambda c: sum((x - y) ** 2 for x, y in zip(data, self.centroids[c])))
        yield closest_centroid, (data, 1)

    def combiner(self, key, values):
        # Combine the features for each cluster (key)
        combined_features = [0.0] * len(next(values)[0])
        count = 0

        for features, count_in_cluster in values:
            combined_features = [x + y for x, y in zip(combined_features, features)]
            count += count_in_cluster

        yield key, (combined_features, count)

    def reducer(self, key, values):
        # Aggregate the combined features and counts
        combined_features = [0.0] * len(next(values)[0])
        count = 0

        for features, count_in_cluster in values:
            combined_features = [x + y for x, y in zip(combined_features, features)]
            count += count_in_cluster

        # Calculate the new centroid
        new_centroid = [x / count for x in combined_features]
        yield None, (key, new_centroid)

    def final_reducer(self, _, values):
        # Output the final centroids
        k = self.options.k
        centroids = dict(sorted(values, key=lambda x: x[0])[:k])
        yield None, centroids

if __name__ == '__main__':
    KMeansMR.run()
