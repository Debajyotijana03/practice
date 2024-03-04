from mrjob.job import MRJob
from mrjob.step import MRStep

class KMeansMR(MRJob):

    def configure_args(self):
        super(KMeansMR, self).configure_args()
        self.add_passthru_arg('--k', type=int, help='Number of clusters')

    def steps(self):
        return [
            MRStep(mapper=self.mapper,
                   reducer=self.reducer),
            MRStep(mapper=self.centroid_mapper,
                   reducer=self.centroid_reducer)
        ]

    def mapper(self, _, line):
        # Skip the header line
        if line.startswith("Data.Temperature.Avg"):
            return

        # Split the input data into cluster ID and attributes
        cluster_id, *attributes = map(float, line.split(','))
        yield int(cluster_id), (attributes, 1)

    def reducer(self, cluster_id, points):
        # Aggregate points for each cluster
        total_points = [0.0] * len(next(points)[0])
        total_count = 0

        for point, count in points:
            total_points = [sum(x) for x in zip(total_points, point)]
            total_count += count

        # Check if total_count is zero before calculating the new centroid
        if total_count > 0:
            # Calculate the new centroid
            centroid = [x / total_count for x in total_points]
            yield None, (cluster_id, centroid)
        else:
            # Handle the case where there are no points in the cluster
            # You can emit a special value, skip this cluster, or handle it in another way
            pass

    def centroid_mapper(self, _, cluster_data):
        # Emit each cluster's centroid as the new key
        cluster_id, centroid = cluster_data
        yield cluster_id, centroid

    def centroid_reducer(self, _, centroids):
        # Choose the centroid with the lowest cluster ID as the final centroid
        min_cluster_id = None
        min_centroid = None

        for cluster_data in centroids:
            c_id, centroid = cluster_data
            if min_cluster_id is None or c_id < min_cluster_id:
                min_cluster_id = c_id
                min_centroid = centroid

        yield min_cluster_id, min_centroid

if __name__ == '__main__':
    KMeansMR.run()
