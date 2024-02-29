from mrjob.job import MRJob
from mrjob.step import MRStep
import random
import math

class KMeans(MRJob):

    def configure_args(self):
        super(KMeans, self).configure_args()
        self.add_passthru_arg('--k', type=int, help='Number of clusters (k)')
        self.add_passthru_arg('--iterations', type=int, default=10, help='Number of iterations')

    def steps(self):
        return [MRStep(mapper_init=self.mapper_init,
                       mapper=self.mapper,
                       combiner=self.combiner,
                       reducer=self.reducer)] * self.options.iterations

    def mapper_init(self):
        # Initialize k random centroids in the mapper
        self.centroids = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(self.options.k)]

    def mapper(self, _, line):
        userID, movieID, rating, timestamp = map(int, line.split('\t')[:3])
        point = (userID, movieID)

        # Calculate the nearest centroid index for each point
        centroid_index = min(range(self.options.k),
                             key=lambda i: math.sqrt((point[0] - self.centroids[i][0])**2 +
                                                    (point[1] - self.centroids[i][1])**2))

        # Emit the centroid index as key and the point as value
        yield centroid_index, (point, 1)

    def combiner(self, key, values):
        # Combine points locally before sending to reducers
        combined_point = (sum(v[0][0] for v in values), sum(v[0][1] for v in values))
        count = sum(v[1] for v in values)
        yield key, (combined_point, count)

    def reducer(self, key, values):
        # Combine points globally and update centroids
        combined_point = (sum(v[0][0] for v in values), sum(v[0][1] for v in values))
        count = sum(v[1] for v in values)
        new_centroid = (combined_point[0] / count, combined_point[1] / count)

        yield None, new_centroid

if __name__ == '__main__':
    KMeans.run()
