package com.github.liusb.bayes;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.BlockLocation;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.LineReader;
import org.apache.hadoop.util.StringUtils;


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
			key.set(filePath.getParent().getName() + "/" + filePath.getName());
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
			index++;
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
			Mapper<Text, Text, Text, Text> {
		
		private Text text = new Text();
		private HashMap<String, HashMap<String, Double>> wordWeight = new HashMap<String, HashMap<String, Double>>();
		private HashMap<String, Double> fileWeight = new HashMap<String, Double>();
		private HashMap<String, Double> lossWeight = new HashMap<String, Double>();

		protected void setup(Context context) throws IOException, InterruptedException {

			Path path = new Path("hdfs://192.168.56.120:9000/user/hadoop/Bayes/WordCount/FeatureNum/result.txt");
			FileSystem fs = path.getFileSystem(context.getConfiguration());
			LineReader in = new LineReader(fs.open(path), context.getConfiguration());
			int newSize = in.readLine(text);
			long feature_num_result = Long.parseLong(text.toString());
			in.close();
			
			path = new Path("hdfs://192.168.56.120:9000/user/hadoop/Bayes/FileCount/[^_]*");
			FileStatus[] files = fs.globStatus(path);
			in = new LineReader(fs.open(files[0].getPath()), context.getConfiguration());
			newSize = 0;
			String[] splitResult;
			while (true) {
				newSize = in.readLine(text);
				if(newSize == 0){
					break;
				}
				splitResult = text.toString().split("\t");
				fileWeight.put(splitResult[0], Double.parseDouble(splitResult[1]));
			}
			in.close();
			
			path = new Path("hdfs://192.168.56.120:9000/user/hadoop/Bayes/WordCount/CATEGORY/");
			files = fs.listStatus(path);
			in = new LineReader(fs.open(files[0].getPath()), context.getConfiguration());
			newSize = 0;
			while (true) {
				newSize = in.readLine(text);
				if(newSize == 0){
					break;
				}
				splitResult = text.toString().split("\t");
				lossWeight.put(splitResult[0], Double.parseDouble(splitResult[1]));
			}
			in.close();
						
			path = new Path("hdfs://192.168.56.120:9000/user/hadoop/Bayes/WordCount/WORD/");
			files = fs.listStatus(path);
			for(FileStatus f: files) {
				if(f.getLen() == 0) {
					continue;
				}
				String category = f.getPath().getName().split("-")[0];
				HashMap<String, Double> hashWeight = new HashMap<String, Double>();
				in = new LineReader(fs.open(f.getPath()), context.getConfiguration());
				double count = 0;				
				double all = lossWeight.get(category) + feature_num_result;
				newSize = 0;
				while (true) {
					newSize = in.readLine(text);
					if(newSize == 0){
						break;
					}
					splitResult = text.toString().split("\t");
					count = Double.parseDouble(splitResult[1]) + 1;
					hashWeight.put(splitResult[0], Math.log(count/all));
				}
				wordWeight.put(category, hashWeight);
				lossWeight.put(category, Math.log(1/all));
				in.close();
			}
		}

		public void map(Text key, Text value, Context context)
				throws IOException, InterruptedException {
			String token;
			double tokenLoss;
			double categoryWeight;
			double maxWeight = Double.NEGATIVE_INFINITY;
			String maxCategory = "";
			for (Map.Entry<String, HashMap<String, Double>> category : wordWeight.entrySet()) {
				tokenLoss = lossWeight.get(category.getKey());				
				categoryWeight = fileWeight.get(category.getKey());
				StringTokenizer itr = new StringTokenizer(value.toString());
				while (itr.hasMoreTokens()) {
					token = itr.nextToken();
					if(category.getValue().containsKey(token)){
						categoryWeight += category.getValue().get(token);
					}
					else {
						categoryWeight += tokenLoss;
					}
				}
				if(categoryWeight > maxWeight) {
					maxWeight = categoryWeight;
					maxCategory = category.getKey();
				}
			}
			context.write(key, new Text(maxCategory));
		}
		
	}

	public static boolean run(Configuration conf) throws Exception {
		
		Job job = new Job(conf, "Classifier");
		job.setJarByClass(Classifier.class);
		job.setInputFormatClass(DirInputFormat.class);
		job.setMapperClass(ClassifierMapper.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		Path input = new Path(
				"hdfs://192.168.56.120:9000/user/hadoop/Bayes/Industry");
		Path output = new Path(
				"hdfs://192.168.56.120:9000/user/hadoop/Bayes/ClassifyResult");
		DirInputFormat.setInputPath(job, input);
		FileOutputFormat.setOutputPath(job, output);

		FileSystem fs = output.getFileSystem(job.getConfiguration());
		fs.delete(output, true);

		return job.waitForCompletion(true);

	}
}
