from mrjob.job import MRJob
from mrjob.step import MRStep

class KMeansMR(MRJob):

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

        # Split the input data into attributes
        attributes = list(map(float, line.split(',')))
        yield None, (attributes, 1)

    def reducer(self, _, points):
        total_points = [0.0] * len(next(points)[0])
        total_count = 0

        for point, count in points:
            for i in range(len(total_points)):
                total_points[i] += point[i]
            total_count += count

        if total_count > 0:
            centroid = [x / total_count for x in total_points]
            yield None, centroid
        else:
            # Emit a special value or handle the case in another way
            yield None, None

    def centroid_mapper(self, _, centroid):
        yield "final_centroid", centroid

    def centroid_reducer(self, _, centroids):
        min_centroid = None

        for data in centroids:
            if min_centroid is None or data[0] < min_centroid:
                min_centroid = data[0]

        yield "final_centroid", min_centroid

if __name__ == '__main__':
    KMeansMR.run()
