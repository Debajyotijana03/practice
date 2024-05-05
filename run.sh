#!/bin/bash

i=1

while :

do

	hadoop jar hadoop-3.2.4/share/hadoop/tools/lib/hadoop-streaming-3.2.4.jar -file /home/debajyoti/Downloads/Map/centroids.txt -file /home/debajyoti/Downloads/Map/mapper.py -mapper /home/debajyoti/Downloads/Map/mapper.py -file /home/debajyoti/Downloads/Map/reducer.py -reducer /home/debajyoti/Downloads/Map/reducer.py -input /testMapReduce/dataset.txt -output /testMapReduce/mapreduce-output$i

	rm -f centroids1.txt

	hadoop fs -copyToLocal /testMapReduce/mapreduce-output$i/part-00000 centroids1.txt

	seeiftrue=`python3 reader.py`

	if [ $seeiftrue = 1 ]

	then

		rm centroids.txt

		hadoop fs -copyToLocal /testMapReduce/mapreduce-output$i/part-00000 centroids.txt

		break

	else

		rm centroids.txt

		hadoop fs -copyToLocal /testMapReduce/mapreduce-output$i/part-00000 centroids.txt

	fi

	i=$((i+1))

done
