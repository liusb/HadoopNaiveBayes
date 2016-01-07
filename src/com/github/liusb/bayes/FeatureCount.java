package com.github.liusb.bayes;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.util.LineReader;

public class FeatureCount {

  public static class FeatureRecordReader extends BaseRecordReader {

    private LineReader in = null;
    private long curlength;
    private long alllength;

    @Override
    public void initialize(InputSplit genericSplit, TaskAttemptContext context)
        throws IOException, InterruptedException {
      MultiPathSplit split = (MultiPathSplit) genericSplit;
      Path[] dirs = split.getPaths();
      alllength = 0;
      conf = context.getConfiguration();
      for (Path dir : dirs) {
        FileSystem fs = dir.getFileSystem(conf);
        for (FileStatus file : fs.listStatus(dir)) {
          files.add(file);
          alllength += file.getLen();
        }
      }
      index = 0;
      curlength = 0;
    }

    @Override
    public boolean nextKeyValue() throws IOException, InterruptedException {
      if (curlength == alllength) {
        return false;
      }
      int newSize = 0;
      while (index < files.size()) {
        if (in == null) {
          Path path = files.get(index).getPath();
          FileSystem fs = path.getFileSystem(conf);
          in = new LineReader(fs.open(path), conf);
          key.set(path.getParent().getName());
        }
        newSize = in.readLine(value);
        if (newSize == 0) {
          in.close();
          in = null;
          index++;
          continue;
        }
        curlength += newSize;
        String str = this.filter(value.toString());
        if (str.length() == 0) {
          continue;
        }
        value.set(str);
        return true;
      }
      return false;
    }
  }

  public static class CategoryInputFormat extends BaseInputFormat {

    @Override
    public RecordReader<Text, Text> createRecordReader(InputSplit split,
        TaskAttemptContext context) throws IOException, InterruptedException {
      return new FeatureRecordReader();
    }
  }

  public static class FeatureMapper extends Mapper<Text, Text, Text, IntWritable> {

    private final static IntWritable one = new IntWritable(1);
    private final static IntWritable zero = new IntWritable(0);
    private Text word = new Text();

    public void map(Text key, Text value, Context context) throws IOException,
        InterruptedException {
      word.set(key.toString() + "\t" + value.toString());
      context.write(word, one);
      context.write(key, one);
      context.write(value, zero);
    }
  }

  public static class FeatureCombiner extends
      Reducer<Text, IntWritable, Text, IntWritable> {

    private IntWritable result = new IntWritable();

    public void reduce(Text key, Iterable<IntWritable> values, Context context)
        throws IOException, InterruptedException {
      int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();
      }
      result.set(sum);
      context.write(key, result);
    }
  }

  public static class FeatureReducer extends
      Reducer<Text, IntWritable, Text, IntWritable> {
    private IntWritable result = new IntWritable();
    private MultipleOutputs<Text, IntWritable> mos;

    protected void setup(Context context) throws IOException, InterruptedException {
      mos = new MultipleOutputs<Text, IntWritable>(context);
    }

    public void reduce(Text key, Iterable<IntWritable> values, Context context)
        throws IOException, InterruptedException {
      int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();
      }
      if (sum == 0) {
        context.getCounter("FEATURE", "ALL").increment(1);
      } else {
        result.set(sum);
        String str[] = key.toString().split("\t");
        if (str.length == 2) {
          mos.write(new Text(str[1]), result, "WORD/" + str[0]);
        } else if (str.length == 1) {
          mos.write(key, result, "CATEGORY/result");
        }
      }
    }

    protected void cleanup(Context context) throws IOException,
        InterruptedException {
      mos.close();
    }
  }

  public static boolean run(Configuration conf, Path input, Path output)
      throws Exception {
    Job job = new Job(conf, "Feature Count");
    job.setJarByClass(FeatureCount.class);
    job.setInputFormatClass(CategoryInputFormat.class);
    job.setMapperClass(FeatureMapper.class);
    job.setCombinerClass(FeatureCombiner.class);
    job.setReducerClass(FeatureReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);

    CategoryInputFormat.setInputPath(job, input);
    FileOutputFormat.setOutputPath(job, output);

    FileSystem fs = output.getFileSystem(job.getConfiguration());

    if (job.waitForCompletion(true) == false) {
      return false;
    }

    Long all_count = job.getCounters().findCounter("FEATURE", "ALL").getValue();
    Path feature = new Path(output.toUri().toString() + "/FeatureNum/result.txt");
    FSDataOutputStream out = fs.create(feature);
    out.write(all_count.toString().getBytes("UTF-8"));
    out.close();
    return true;
  }
}
