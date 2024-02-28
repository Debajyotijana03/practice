from mrjob.job import MRJob
from mrjob.step import MRStep
import numpy as np

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
        self.centroids = np.random.rand(self.options.k, 2)

    def mapper(self, _, line):
        userID, movieID, rating, timestamp = map(int, line.split('\t')[:3])
        point = np.array([userID, movieID])

        # Calculate the nearest centroid index for each point
        centroid_index = np.argmin(np.linalg.norm(point - self.centroids, axis=1))

        # Emit the centroid index as key and the point as value
        yield centroid_index, (point, 1)

    def combiner(self, key, values):
        # Combine points locally before sending to reducers
        combined_point = np.sum([v[0] for v in values], axis=0)
        count = sum([v[1] for v in values])
        yield key, (combined_point, count)

    def reducer(self, key, values):
        # Combine points globally and update centroids
        combined_point = np.sum([v[0] for v in values], axis=0)
        count = sum([v[1] for v in values])
        new_centroid = combined_point / count

        yield None, new_centroid.tolist()

if __name__ == '__main__':
    KMeans.run()
