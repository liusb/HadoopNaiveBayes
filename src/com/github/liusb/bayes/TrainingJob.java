package com.github.liusb.bayes;


import org.apache.hadoop.conf.Configuration;

public class TrainingJob {

	public static void main(String[] args) throws Exception {
		
		Configuration conf = new Configuration();
		conf.set("mapred.job.tracker", "192.168.56.120:9001");
		conf.set("fs.default.name", "hdfs://192.168.56.120:9000");
		conf.set("mapred.jar", "F://WorkSpace//javaWorkspace//NaiveBayes.jar");
		//FileCount.run(conf);
		//WordCount.run(conf);
		Classifier.run(conf);
	}

}
