package com.github.liusb.bayes;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.BlockLocation;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.util.StringUtils;


public abstract class BaseInputFormat extends InputFormat<Text, Text> {

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

		// 按分类进行分片，每个分类作为一个InputSplit
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

}
