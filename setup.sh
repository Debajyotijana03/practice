#!/bin/bash

# Upgrade Spark Python to work with Python 3
sudo yum upgrade -y spark-python hive

# Remove old Derby JAR if it exists
sudo [ -e /usr/lib/flume-ng/lib/derby-10.8.2.2.jar ] && sudo rm /usr/lib/flume-ng/lib/derby-10.8.2.2.jar

# Decompress datasets if they exist
gzip -d *.csv.gz 2>/dev/null

# Download Anaconda manually
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

# Run the Anaconda installation script
bash Anaconda3-2022.05-Linux-x86_64.sh

# Add Spark CSV JARs to the classpath
echo export SPARK_CLASSPATH="$(pwd)/lib/spark-csv_2.10-1.5.0.jar:$(pwd)/lib/commons-csv-1.1.jar" >> ~/.bashrc

# Set environment variables to load Spark libs in Jupyter
echo "export PYSPARK_DRIVER_PYTHON_OPTS=\"notebook\"" >> ~/.bashrc
echo "export PYSPARK_DRIVER_PYTHON=jupyter"  >> ~/.bashrc

echo "Run 'source ~/.bashrc' to complete the setup."
