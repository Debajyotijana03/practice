from mrjob.job import MRJob
from math import sqrt

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

# Function to parse initial centroids
def parse_centroids(centroid_string):
    centroid_data = centroid_string.split(',')
    return centroid_data[0], [float(x) for x in centroid_data[1:]]

class MRKMeans(MRJob):

    def __init__(self, *args, **kwargs):
        super(MRKMeans, self).__init__(*args, **kwargs)
        self.centroids = {
            'A1': [2, 10],
            'B1': [5, 8],
            'C1': [1, 2]
        }

    def mapper(self, _, line):
        data = line.strip().split(',')
        point_id, point = data[0], list(map(float, data[1:]))

        # Calculate distances to centroids
        nearest_centroid = min(self.centroids.keys(),
                               key=lambda x: euclidean_distance(point, self.centroids[x]))

        yield nearest_centroid, (point, 1)

    def reducer_init(self):
        self.converged = False

    def reducer(self, centroid_id, points):
        count = 0
        centroid_sum = [0] * len(self.centroids[centroid_id])

        for point, c in points:
            count += c
            for i in range(len(centroid_sum)):
                centroid_sum[i] += point[i]

        new_centroid = [x / count for x in centroid_sum]

        # Check for convergence
        if new_centroid != self.centroids[centroid_id]:
            self.converged = False
            self.centroids[centroid_id] = new_centroid

        yield centroid_id, new_centroid

    def steps(self):
        while not self.converged:
            yield self.mr(mapper=self.mapper, reducer_init=self.reducer_init, reducer=self.reducer)

if __name__ == '__main__':
    MRKMeans().run()
