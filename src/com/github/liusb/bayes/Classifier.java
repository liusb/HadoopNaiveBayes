package com.github.liusb.bayes;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.BlockLocation;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
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

import com.github.liusb.bayes.WordCount.CategoryInputFormat;


public class Classifier {
	
	public static class SingleFileRecordReader extends RecordReader<Text, Text> {

		private Configuration conf = null;
		private FileSystem fs = null;
		private FileStatus[] files = null;
		private Text key = null;
		private Text value = null;
		private int index;
		private LineReader in = null;

		@Override
		public void initialize(InputSplit genericSplit,
				TaskAttemptContext context) throws IOException,
				InterruptedException {
			FileSplit split = (FileSplit) genericSplit;
			Path dir = split.getPath();
			conf = context.getConfiguration();
			fs = dir.getFileSystem(conf);
			files = fs.listStatus(dir);
			key = new Text();
			value = new Text();
			index = 0;
		}

		@Override
		public boolean nextKeyValue() throws IOException, InterruptedException {
			if (files == null || index == files.length) {
				return false;
			}
			Path filePath = files[index].getPath();
			key.set(filePath.getParent().getName() + "|" + filePath.getName());
			int newSize = 0;
			StringBuffer totalValue = new StringBuffer();
			in = new LineReader(fs.open(filePath), conf);
			while (true) {
				newSize = in.readLine(value);
				if(newSize != 0) {
					break;
				}
				totalValue.append(value.toString()+"\n");
			}
			value.set(totalValue.toString());
			in.close();
			in = null;
			return true;
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
			if (files.length == 0) {
				return 0.0f;
			} else {
				return Math.min(1.0f, index / (float) files.length);
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
			return new SingleFileRecordReader();
		}

	}


	public static class ClassifierMapper extends
			Mapper<Text, Text, Text, DoubleWritable> {
		
		private final static DoubleWritable weight = new DoubleWritable();
		private Text text = new Text();
		private HashMap<String, HashMap<String, Double>> wordWeight = new HashMap<String, HashMap<String, Double>>();
		private HashMap<String, Double> fileWeight = new HashMap<String, Double>();
		private HashMap<String, Double> lossWeight = new HashMap<String, Double>();

		protected void setup(Context context) throws IOException, InterruptedException {

			Path path = new Path("hdfs://192.168.56.120:9000/user/hadoop/Bayes/WordCount/FEATURE_NUM_RESULT.txt");
			FileSystem fs = path.getFileSystem(context.getConfiguration());
			LineReader in = new LineReader(fs.open(path), context.getConfiguration());
			int newSize = 0;
			while (true) {
				// TODO
				if(newSize == 0){
					break;
				}
			}
			in.close();
		}

		public void map(Text key, Text value, Context context)
				throws IOException, InterruptedException {
			StringTokenizer itr = new StringTokenizer(value.toString());
			while (itr.hasMoreTokens()) {
				text.set(itr.nextToken());
				context.write(text, weight);
			}
		}
	}

	public static class ClassifierReducer extends
			Reducer<Text, DoubleWritable, Text, Text> {
		private IntWritable result = new IntWritable();

		public void reduce(Text key, Iterable<IntWritable> values,
				Context context) throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			result.set(sum);
			//context.write(key, result);
		}
	}

	public static boolean run(Configuration conf) throws Exception {
		
		Job job = new Job(conf, "Classifier");
		job.setJarByClass(Classifier.class);
		job.setInputFormatClass(DirInputFormat.class);
		job.setMapperClass(ClassifierMapper.class);
		job.setCombinerClass(ClassifierReducer.class);
		job.setReducerClass(ClassifierReducer.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
		DirInputFormat.setInputPath(job, new Path(""));
		FileOutputFormat.setOutputPath(job, new Path(""));

		Path input = new Path(
				"hdfs://192.168.56.120:9000/user/hadoop/Bayes/Industry");
		Path output = new Path(
				"hdfs://192.168.56.120:9000/user/hadoop/Bayes/ClassifyResult");
		CategoryInputFormat.setInputPath(job, input);
		FileOutputFormat.setOutputPath(job, output);

		FileSystem fs = output.getFileSystem(job.getConfiguration());
		fs.delete(output, true);

		return job.waitForCompletion(true);

	}
}
