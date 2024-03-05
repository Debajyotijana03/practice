from mrjob.job import MRJob
from mrjob.step import MRStep
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
        self.converged = False

    def configure_options(self):
        super(MRKMeans, self).configure_options()
        self.add_file_option('--centroids', help='Path to the centroids file')

    def mapper(self, _, line):
        data = line.strip().split(',')
        point_id, point = data[0], list(map(float, data[1:]))

        # Calculate distances to centroids
        nearest_centroid = min(self.centroids.keys(),
                               key=lambda x: euclidean_distance(point, self.centroids[x]))

        yield nearest_centroid, (point, 1)

    def reducer_init(self):
        self.converged = True

    def reducer(self, centroid_id, points):
        count = 0
        centroid_sum = [0] * len(self.centroids[centroid_id])
        point_dimensions = len(centroid_sum)

        for point, c in points:
            count += c
            # Ensure the point has the expected number of dimensions
            point = point + [0] * (point_dimensions - len(point))

            for i in range(point_dimensions):
                centroid_sum[i] += point[i]

        new_centroid = [x / count for x in centroid_sum]

        # Check for convergence
        if new_centroid != self.centroids[centroid_id]:
            self.converged = False

        yield centroid_id, new_centroid

    def steps(self):
        return [
            MRStep(mapper=self.mapper, reducer_init=self.reducer_init, reducer=self.reducer)
        ]

if __name__ == '__main__':
    mr_job = MRKMeans(args=['input_data.txt', '--centroids', 'your_centroids_file.txt'])

    # Run the job until convergence
    while not mr_job.converged:
        with mr_job.make_runner() as runner:
            runner.run()

        # Update centroids for the next iteration
        mr_job.centroids = {}

        with open('output_file.txt', 'w') as output_file:
            with mr_job.make_runner() as runner:
                runner.run()
                for line in runner.stream_output():
                    key, value = mr_job.parse_output_line(line)
                    mr_job.centroids[key] = value

        # Check if the job converged
        mr_job.converged = True

        # Print the final centroids
        for centroid_id, final_centroid in mr_job.centroids.items():
            print('{}\t{}'.format(centroid_id, final_centroid))
