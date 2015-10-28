package com.github.liusb.bayes.training;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.BlockLocation;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.LineReader;
import org.apache.hadoop.util.StringUtils;


public class WordCount {

	public static class FeatureRecordReader extends RecordReader<Text, Text> {

		private FileStatus[] files = null;
		private Text key = null;
		private Text value = null;
		private LineReader in = null;
		private int index;
		private long curlength;
		private long alllength;
		private FileSystem fs;
		private Configuration job;

		@Override
		public void initialize(InputSplit genericSplit,
				TaskAttemptContext context) throws IOException,
				InterruptedException {
			FileSplit split = (FileSplit) genericSplit;
			Path dir = split.getPath();
			job = context.getConfiguration();
			fs = dir.getFileSystem(job);
			key = new Text();
			value = new Text();
			key.set(dir.getName());
			files = fs.listStatus(dir);
			index = 0;
			curlength = 0;
			alllength = 0;
			for (FileStatus st : files) {
				alllength += st.getLen();
			}
		}

		@Override
		public boolean nextKeyValue() throws IOException, InterruptedException {
			if (files == null || curlength == alllength) {
				return false;
			}
			int newSize = 0;
			while (index < files.length) {
				if (in == null) {
					in = new LineReader(fs.open(files[index].getPath()), job);
				}
				newSize = in.readLine(value);
				if (newSize == 0) {
					in.close();
					in = null;
					index++;
					continue;
				}
				curlength += newSize;
				return true;
			}
			return false;
		}

		@Override
		public Text getCurrentKey() throws IOException, InterruptedException {
			return key;
		}

		@Override
		public Text getCurrentValue() throws IOException, InterruptedException {
			return value;
		}

		@Override
		public float getProgress() throws IOException, InterruptedException {
			if (alllength == 0) {
				return 0.0f;
			} else {
				return Math.min(1.0f, curlength / (float) alllength);
			}
		}

		@Override
		public void close() throws IOException {
			// nothing to do
		}

	}

	public static class CategoryInputFormat extends InputFormat<Text, Text> {

		public static void setInputPath(JobContext context, Path path)
				throws IOException {
			Configuration conf = context.getConfiguration();
			path = path.getFileSystem(conf).makeQualified(path);
			String dir = StringUtils.escapeString(path.toString());
			conf.set("mapred.input.dir", dir);
		}

		public static Path getInputPath(JobContext context) throws IOException {
			String dir = context.getConfiguration().get("mapred.input.dir", "");
			if (dir.length() == 0) {
				throw new IOException("No input paths specified in job");
			}
			return new Path(StringUtils.unEscapeString(dir));
		}

		@Override
		public List<InputSplit> getSplits(JobContext context)
				throws IOException {

			List<InputSplit> splits = new ArrayList<InputSplit>();
			List<FileStatus> categoryDirs = new ArrayList<FileStatus>();
			Path inputDir = getInputPath(context);
			FileSystem fs = inputDir.getFileSystem(context.getConfiguration());

			for (FileStatus stat : fs.listStatus(inputDir)) {
				categoryDirs.add(stat);
			}

			for (FileStatus dir : categoryDirs) {
				Path path = dir.getPath();
				long length = dir.getLen();
				if ((length != 0)) {
					BlockLocation[] blkLocations = fs.getFileBlockLocations(
							dir, 0, length);
					splits.add(new FileSplit(path, 0, length, blkLocations[0]
							.getHosts()));
				} else {
					splits.add(new FileSplit(path, 0, length, new String[0]));
				}
			}

			return splits;
		}

		@Override
		public RecordReader<Text, Text> createRecordReader(InputSplit split,
				TaskAttemptContext context) throws IOException,
				InterruptedException {
			return new FeatureRecordReader();
		}

	}

	public static class FeatureMapper extends
			Mapper<Text, Text, Text, IntWritable> {

		private final static IntWritable one = new IntWritable(1);
		private Text word = new Text();

		public void map(Text key, Text value, Context context)
				throws IOException, InterruptedException {
			word.set(key.toString() + '\t' + value.toString());
			context.write(word, one);
		}
	}

	public static class FeatureReducer extends
			Reducer<Text, IntWritable, Text, IntWritable> {
		private IntWritable result = new IntWritable();

		public void reduce(Text key, Iterable<IntWritable> values,
				Context context) throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			result.set(sum);
			context.write(key, result);
		}
	}

	public static boolean run(Configuration conf) throws Exception {
		Job job = new Job(conf, "word count");
		job.setJarByClass(WordCount.class);
		job.setInputFormatClass(CategoryInputFormat.class);
		job.setMapperClass(FeatureMapper.class);
		job.setCombinerClass(FeatureReducer.class);
		job.setReducerClass(FeatureReducer.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
		
		Path input = new Path(
				"hdfs://192.168.56.120:9000/user/hadoop/Bayes/Country");
		Path output = new Path(
				"hdfs://192.168.56.120:9000/user/hadoop/Bayes/WordCount");
		CategoryInputFormat.setInputPath(job, input);
		FileOutputFormat.setOutputPath(job, output);

		FileSystem fs = output.getFileSystem(job.getConfiguration());
		fs.delete(output, true);

		return job.waitForCompletion(true);
	}
}
