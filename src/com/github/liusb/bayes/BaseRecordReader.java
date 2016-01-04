package com.github.liusb.bayes;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;

public abstract class BaseRecordReader extends RecordReader<Text, Text> {
	protected Configuration conf = null;
	protected List<FileStatus> files = new ArrayList<FileStatus>();
	protected Text key = new Text();
	protected Text value = new Text();
	protected int index;

	@Override
	public void initialize(InputSplit genericSplit, TaskAttemptContext context)
			throws IOException, InterruptedException {
		conf = context.getConfiguration();
		MultiPathSplit split = (MultiPathSplit) genericSplit;
		Path[] dirs = split.getPaths();
		for (Path dir : dirs) {
			FileSystem fs = dir.getFileSystem(conf);
			for (FileStatus file : fs.listStatus(dir)) {
				files.add(file);
			}
		}
		index = 0;
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
		return files.size() == 0 ? 0.0f : Math.min(1.0f,
				index / (float) files.size());
	}

	@Override
	public void close() throws IOException {
	}
	
	public String filter(String text) {
//		if (text.equals("said"))
//			return "";
		return text.replaceAll("[0-9,-./]", "");
	}
}
