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
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.JobID;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.StringUtils;

public class FileCount {

	public static class FileRecordReader extends RecordReader<Text, Text> {

		private FileStatus[] files = null;
		private Text key = null;
		private Text value = null;
		private int pos;

		@Override
		public void initialize(InputSplit genericSplit,
				TaskAttemptContext context) throws IOException,
				InterruptedException {
			FileSplit split = (FileSplit) genericSplit;
			Configuration job = context.getConfiguration();
			Path dir = split.getPath();
			FileSystem fs = dir.getFileSystem(job);
			files = fs.listStatus(dir);
			pos = 0;
		}

		@Override
		public boolean nextKeyValue() throws IOException, InterruptedException {
			if (key == null) {
				key = new Text();
			}
			if (value == null) {
				value = new Text();
			}
			if (files == null || pos == files.length) {
				return false;
			} else {
				key.set(files[pos].getPath().getParent().getName());
				value.set(files[pos].getPath().getName());
				pos++;
				return true;
			}
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
			if (files == null || files.length == 0) {
				return 0.0f;
			} else {
				return Math.min(1.0f, pos / (float) files.length);
			}
		}

		@Override
		public void close() throws IOException {
			// nothing to do
		}

	}

	public static class DirInputFormat extends InputFormat<Text, Text> {

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
				throws IOException, InterruptedException {

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
			return new FileRecordReader();
		}

	}

	public static class FileCountMapper extends
			Mapper<Text, Text, Text, IntWritable> {

		private final static IntWritable one = new IntWritable(1);
		private Text category = new Text();

		public void map(Text key, Text value, Context context)
				throws IOException, InterruptedException {
			category.set(key);
			context.write(category, one);
		}
	}

	public static class FileCountReducer extends
			Reducer<Text, IntWritable, Text, DoubleWritable> {
		private DoubleWritable result = new DoubleWritable();
		private double all_count = 0;

		protected void setup(Context context) throws IOException,
				InterruptedException {
			JobConf conf = (JobConf) context.getConfiguration();
			JobClient client = new JobClient(conf);
			RunningJob job = client.getJob(JobID.forName(context.getJobID()
					.toString()));
			all_count = (double) job
					.getCounters()
					.findCounter("org.apache.hadoop.mapred.Task$Counter",
							"MAP_OUTPUT_RECORDS").getValue();
		}

		public void reduce(Text key, Iterable<IntWritable> values,
				Context context) throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			result.set(sum / all_count);
			context.write(key, result);
		}
	}

	public static boolean run(Configuration conf) throws Exception {
		Job job = new Job(conf, "file count");
		job.setJarByClass(FileCount.class);
		job.setInputFormatClass(DirInputFormat.class);
		job.setMapperClass(FileCountMapper.class);
		job.setReducerClass(FileCountReducer.class);
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(IntWritable.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(DoubleWritable.class);

		Path input = new Path(
				"hdfs://192.168.56.120:9000/user/hadoop/Bayes/Country");
		Path output = new Path(
				"hdfs://192.168.56.120:9000/user/hadoop/Bayes/FileCount");
		DirInputFormat.setInputPath(job, input);
		FileOutputFormat.setOutputPath(job, output);

		FileSystem fs = output.getFileSystem(job.getConfiguration());
		fs.delete(output, true);

		return job.waitForCompletion(true);
	}
}
