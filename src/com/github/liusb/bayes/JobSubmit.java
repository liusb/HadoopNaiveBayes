package com.github.liusb.bayes;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public class JobSubmit {

  public static void main(String[] args) throws Exception {

    Configuration conf = new Configuration();
    conf.set("mapred.job.tracker", "192.168.56.120:9001");
    conf.set("fs.default.name", "hdfs://192.168.56.120:9000");
    conf.set("mapred.jar", "F://WorkSpace//javaWorkspace//NaiveBayes.jar");

    String base_path = "hdfs://192.168.56.120:9000/user/hadoop/Bayes/Country";

    Path training_input = new Path(base_path + "_training");
    Path prior_output = new Path(base_path + "_PriorCount");
    FileSystem fs = prior_output.getFileSystem(conf);
    fs.delete(prior_output, true);
    PriorCount.run(conf, training_input, prior_output);

    Path feature_output = new Path(base_path + "_FeatureCount");
    fs.delete(feature_output, true);
    FeatureCount.run(conf, training_input, feature_output);

    Path test_input = new Path(base_path + "_test");
    Path test_output = new Path(base_path + "_testout");
    fs.delete(test_output, true);
    Classifier.run(conf, test_input, test_output, prior_output, feature_output);
  }
}
