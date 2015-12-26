package com.github.liusb.bayes;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
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
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class PriorCount {

	public static class FileNameReader extends RecordReader<Text, Text> {

		private List<FileStatus> files = new ArrayList<FileStatus>();
		private Text key = null;
		private Text value = null;
		private int pos;

		@Override
		public void initialize(InputSplit genericSplit,
				TaskAttemptContext context) throws IOException,
				InterruptedException {
			MultiPathSplit split = (MultiPathSplit) genericSplit;
			Path[] dirs = split.getPaths();
			for (Path dir : dirs) {
				FileSystem fs = dir.getFileSystem(context.getConfiguration());
				for (FileStatus file : fs.listStatus(dir)) {
					files.add(file);
				}
			}
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
			if (pos == files.size()) {
				return false;
			} else {
				Path path = files.get(pos).getPath();
				key.set(path.getParent().getName());
				value.set(path.getName());
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
			if (files.size() == 0) {
				return 0.0f;
			} else {
				return Math.min(1.0f, pos / (float) files.size());
			}
		}

		@Override
		public void close() throws IOException {
			// nothing to do
		}

	}

	public static class DirInputFormat extends BaseInputFormat {

		@Override
		public RecordReader<Text, Text> createRecordReader(InputSplit split,
				TaskAttemptContext context) throws IOException,
				InterruptedException {
			return new FileNameReader();
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
			result.set(Math.log(sum / all_count));
			context.write(key, result);
		}
	}

	public static boolean run(Configuration conf, Path input, Path output)
			throws Exception {
		Job job = new Job(conf, "file count");
		job.setJarByClass(PriorCount.class);
		job.setInputFormatClass(DirInputFormat.class);
		job.setMapperClass(FileCountMapper.class);
		job.setReducerClass(FileCountReducer.class);
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(IntWritable.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(DoubleWritable.class);

		DirInputFormat.setInputPath(job, input);
		FileOutputFormat.setOutputPath(job, output);

		return job.waitForCompletion(true);
	}
}
