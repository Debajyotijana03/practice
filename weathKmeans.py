from mrjob.job import MRJob
from mrjob.step import MRStep
import json
import random

class KMeansMR(MRJob):

    def configure_args(self):
        super(KMeansMR, self).configure_args()
        self.add_passthru_arg('--k', type=int, default=3, help='Number of clusters (k)')
        self.add_passthru_arg('--iterations', type=int, default=10, help='Number of iterations')

    def steps(self):
        return [
            MRStep(mapper_init=self.mapper_init, mapper=self.mapper, combiner=self.combiner, reducer=self.reducer),
            MRStep(reducer=self.final_reducer)
        ]

    def mapper_init(self):
        # Load centroids from the initial centroids file or generate random centroids
        centroids_path = 'initial_centroids.txt'
        try:
            with open(centroids_path, 'r') as f:
                self.centroids = json.load(f)
        except FileNotFoundError:
            self.centroids = self.generate_random_centroids()

    def generate_random_centroids(self):
        # Generate random centroids if the file is not found
        centroids = {}
        for _ in range(self.options.k):
            centroid = [random.uniform(0, 100) for _ in range(14)]  # Assuming 14 features in the data
            centroids[str(_)] = centroid
        return centroids

    def mapper(self, _, line):
        data = list(map(float, line.strip().split(',')))
        closest_centroid = min(self.centroids, key=lambda c: sum((x - y) ** 2 for x, y in zip(data, self.centroids[c])))
        yield closest_centroid, (data, 1)

    def combiner(self, key, values):
        combined_features = [0.0] * len(next(values)[0])
        count = 0

        for features, count_in_cluster in values:
            combined_features = [x + y for x, y in zip(combined_features, features)]
            count += count_in_cluster

        yield key, (combined_features, count)

    def reducer(self, key, values):
        combined_features = [0.0] * len(next(values)[0])
        count = 0

        for features, count_in_cluster in values:
            combined_features = [x + y for x, y in zip(combined_features, features)]
            count += count_in_cluster

        new_centroid = [x / count for x in combined_features]
        yield None, (key, new_centroid)

    def final_reducer(self, _, values):
        k = self.options.k
        centroids = dict(sorted(values, key=lambda x: x[0])[:k])
        yield None, centroids

if __name__ == '__main__':
    KMeansMR.run()
