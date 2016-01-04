package com.github.liusb.bayes;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
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
import org.apache.hadoop.util.StringUtils;

public class Classifier {

	public static class SingleFileRecordReader extends BaseRecordReader {

		private LineReader in = null;

		@Override
		public boolean nextKeyValue() throws IOException, InterruptedException {
			if (index == files.size()) {
				return false;
			}
			Path path = files.get(index).getPath();
			key.set(path.getParent().getName() + "/" + path.getName());
			int newSize = 0;
			StringBuffer totalValue = new StringBuffer();
			in = new LineReader(path.getFileSystem(conf).open(path), conf);
			while (true) {
				newSize = in.readLine(value);
				if (newSize == 0) {
					break;
				}
				String str = this.filter(value.toString());
				if (str.length() != 0) {
					totalValue.append(str + "\n");
				}
			}
			value.set(totalValue.toString());
			in.close();
			in = null;
			index++;
			return true;
		}
	}

	public static class DirInputFormat extends BaseInputFormat {

		@Override
		public RecordReader<Text, Text> createRecordReader(InputSplit split,
				TaskAttemptContext context) throws IOException,
				InterruptedException {
			return new SingleFileRecordReader();
		}
	}

	public static class ClassifierMapper extends Mapper<Text, Text, Text, Text> {

		private Text text = new Text();
		private HashMap<String, HashMap<String, Double>> featureWeight = new HashMap<String, HashMap<String, Double>>();
		private HashMap<String, Double> priorWeight = new HashMap<String, Double>();
		private HashMap<String, Double> lossWeight = new HashMap<String, Double>();
		private Text valueA = new Text("A");
		private Text valueR = new Text("R");
		private Text valueP = new Text("P");

		protected void setup(Context context) throws IOException,
				InterruptedException {
			Configuration conf = context.getConfiguration();
			String prior_path = StringUtils.unEscapeString(conf.get(
					"bayes.training.prior.dir", ""));
			String feature_path = StringUtils.unEscapeString(conf.get(
					"bayes.training.feature.dir", ""));
			if (prior_path.length() == 0 || feature_path.length() == 0) {
				throw new IOException(
						"No training result paths specified in job");
			}

			Path path = new Path(feature_path + "/FeatureNum/result.txt");
			FileSystem fs = path.getFileSystem(conf);
			LineReader in = new LineReader(fs.open(path), conf);
			int newSize = in.readLine(text);
			long feature_num_result = Long.parseLong(text.toString());
			in.close();

			path = new Path(prior_path + "/[^_]*");
			FileStatus[] files = fs.globStatus(path);
			in = new LineReader(fs.open(files[0].getPath()), conf);
			newSize = 0;
			String[] splitResult;
			while (true) {
				newSize = in.readLine(text);
				if (newSize == 0) {
					break;
				}
				splitResult = text.toString().split("\t");
				priorWeight.put(splitResult[0],
						Double.parseDouble(splitResult[1]));
			}
			in.close();

			path = new Path(feature_path + "/CATEGORY/");
			files = fs.listStatus(path);
			in = new LineReader(fs.open(files[0].getPath()), conf);
			newSize = 0;
			while (true) {
				newSize = in.readLine(text);
				if (newSize == 0) {
					break;
				}
				splitResult = text.toString().split("\t");
				lossWeight.put(splitResult[0],
						Double.parseDouble(splitResult[1]));
			}
			in.close();

			path = new Path(feature_path + "/WORD/");
			files = fs.listStatus(path);
			for (FileStatus f : files) {
				if (f.getLen() == 0) {
					continue;
				}
				String category = f.getPath().getName().split("-")[0];
				HashMap<String, Double> hashWeight = new HashMap<String, Double>();
				in = new LineReader(fs.open(f.getPath()), conf);
				double count = 0;
				double all = lossWeight.get(category) + feature_num_result;
				newSize = 0;
				while (true) {
					newSize = in.readLine(text);
					if (newSize == 0) {
						break;
					}
					splitResult = text.toString().split("\t");
					count = Double.parseDouble(splitResult[1]) + 1;
					hashWeight.put(splitResult[0], Math.log(count / all));
				}
				featureWeight.put(category, hashWeight);
				lossWeight.put(category, Math.log(1 / all));
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
			for (Map.Entry<String, HashMap<String, Double>> category : featureWeight
					.entrySet()) {
				tokenLoss = lossWeight.get(category.getKey());
				categoryWeight = priorWeight.get(category.getKey());
				StringTokenizer itr = new StringTokenizer(value.toString());
				while (itr.hasMoreTokens()) {
					token = itr.nextToken();
					if (category.getValue().containsKey(token)) {
						categoryWeight += category.getValue().get(token);
					} else {
						categoryWeight += tokenLoss;
					}
				}
				if (categoryWeight > maxWeight) {
					maxWeight = categoryWeight;
					maxCategory = category.getKey();
				}
			}
			String trueCategory = key.toString().split("/")[0];
			Text calCategory = new Text(maxCategory);
			context.write(key, calCategory);
			if (maxCategory.equals(trueCategory)) {
				context.write(calCategory, valueA);
			}
			context.write(new Text(trueCategory), valueR);
			context.write(calCategory, valueP);
		}
	}

	public static class ClassifierReducer extends
			Reducer<Text, Text, Text, Text> {
		private MultipleOutputs<Text, Text> mos;

		protected void setup(Context context) throws IOException,
				InterruptedException {
			mos = new MultipleOutputs<Text, Text>(context);
		}

		public void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			if (key.find("/") == -1) {
				double countA = 0, countR = 0, countP = 0;
				for (Text val : values) {
					if (val.toString().equals("A")) {
						countA++;
					} else if (val.toString().equals("R")) {
						countR++;
					} else {
						countP++;
					}
				}
				double recall = countA / countR;
				double precision = countA / countP;
				double F1 = 2 * recall * precision / (recall + precision);
				Text result = new Text("Recall:" + Double.toString(recall)
						+ "\tPrecision:" + Double.toString(precision) + "\tF1:"
						+ Double.toString(F1));
				mos.write(key, result, "Fmeasure");
			} else {
				mos.write(key, values.iterator().next(), "result");
			}
		}

		protected void cleanup(Context context) throws IOException,
				InterruptedException {
			mos.close();
		}
	}

	public static boolean run(Configuration conf, Path input, Path output,
			Path prior_path, Path feature_path) throws Exception {
		conf.set("bayes.training.prior.dir",
				StringUtils.escapeString(prior_path.toString()));
		conf.set("bayes.training.feature.dir",
				StringUtils.escapeString(feature_path.toString()));
		Job job = new Job(conf, "Classifier");
		job.setJarByClass(Classifier.class);
		job.setInputFormatClass(DirInputFormat.class);
		job.setMapperClass(ClassifierMapper.class);
		job.setReducerClass(ClassifierReducer.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		DirInputFormat.setInputPath(job, input);
		FileOutputFormat.setOutputPath(job, output);

		return job.waitForCompletion(true);
	}
}
