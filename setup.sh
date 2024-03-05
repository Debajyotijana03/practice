#!/bin/bash

# Upgrade Spark Python to work with Python 3
sudo yum upgrade -y spark-python hive

# Remove old Derby JAR so only one version is on Spark's classpath
sudo rm /usr/lib/flume-ng/lib/derby-10.8.2.2.jar

# Decompress datasets
gzip -d *.csv.gz

# Download and install Anaconda for Pandas, Jupyter
rm -f Anaconda3-latest-Linux-x86_64.sh
wget https://repo.anaconda.com/archive/Anaconda3-latest-Linux-x86_64.sh
bash Anaconda3-latest-Linux-x86_64.sh

# Add Spark CSV JARs to the classpath
echo export SPARK_CLASSPATH="$(pwd)/lib/spark-csv_2.10-1.5.0.jar:$(pwd)/lib/commons-csv-1.1.jar" >> ~/.bashrc

# Set environment variables to load Spark libs in Jupyter
echo "export PYSPARK_DRIVER_PYTHON_OPTS=\"notebook\"" >> ~/.bashrc
echo "export PYSPARK_DRIVER_PYTHON=jupyter"  >> ~/.bashrc

echo "Run 'source ~/.bashrc' to complete the setup."
