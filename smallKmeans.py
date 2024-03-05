from mrjob.job import MRJob
from math import sqrt

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

class MRKMeans(MRJob):

    def configure_options(self):
        super(MRKMeans, self).configure_options()
        self.add_passthrough_option('--k', default=2, type='int', help='Number of clusters (default: 2)')
        self.add_file_option('--centroids', help='Path to the centroids file')

    def load_options(self, args):
        super(MRKMeans, self).load_options(args)
        self.k = self.options.k
        self.centroids = {}

    def mapper_init(self):
        # Load initial centroids from the file specified in the command line
        with open(self.options.centroids, 'r') as centroids_file:
            for line in centroids_file:
                centroid_id, centroid_values = line.strip().split(',')
                self.centroids[centroid_id] = list(map(float, centroid_values.split()))

    def mapper(self, _, line):
        data = line.strip().split(',')
        point = list(map(float, data[1:]))
        
        # Calculate distances to centroids
        nearest_centroid = min(self.centroids.keys(),
                               key=lambda x: euclidean_distance(point, self.centroids[x]))

        yield nearest_centroid, (point, 1)

    def reducer(self, centroid_id, points):
        count = 0
        centroid_sum = [0.0] * len(self.centroids[centroid_id])

        for point, c in points:
            count += c
            for i in range(len(centroid_sum)):
                centroid_sum[i] += point[i]

        new_centroid = [x / count for x in centroid_sum]
        self.centroids[centroid_id] = new_centroid

        yield centroid_id, new_centroid

if __name__ == '__main__':
    MRKMeans.run()
