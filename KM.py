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
        # Set initial centroids manually
        self.centroids = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]

    def mapper(self, _, line):
        # Split the line into fields using tabs
        fields = line.split('\t')

        # Check if there are at least two values
        if len(fields) >= 2:
            # Unpack the two values from the line
            x, y = map(float, fields[:2])

            # Rest of your mapper logic...
            point = (x, y)
            centroid_index = min(range(self.options.k),
                                 key=lambda i: math.sqrt((point[0] - self.centroids[i][0])**2 +
                                                        (point[1] - self.centroids[i][1])**2))

            # Add debug print statements
            print("Point: {}, Assigned to Centroid: {}".format(point, centroid_index))

            yield centroid_index, (point, 1)
        else:
            # Handle lines with fewer than two values as needed
            pass

    def combiner(self, key, values):
        # Combine points locally before sending to reducers
        combined_point = (sum(v[0][0] for v in values), sum(v[0][1] for v in values))
        count = sum(v[1] for v in values)
        yield key, (combined_point, count)

    def reducer(self, key, values):
        # Combine points globally and update centroids
        combined_point = (sum(v[0][0] for v in values), sum(v[0][1] for v in values))
        count = sum(v[1] for v in values)

        # Check if count is not zero before performing division
        if count != 0:
            new_centroid = (combined_point[0] / count, combined_point[1] / count)
            yield None, new_centroid
        else:
            # Handle the case when count is zero (optional)
            # You can choose to emit a special value or skip this centroid
            pass

if __name__ == '__main__':
    KMeans.run()
