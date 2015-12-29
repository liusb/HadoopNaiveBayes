package com.github.liusb.bayes;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
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
	public List<InputSplit> getSplits(JobContext context) throws IOException,
			InterruptedException {

		List<InputSplit> splits = new ArrayList<InputSplit>();
		List<FileStatus> categoryDirs = new ArrayList<FileStatus>();
		Path inputDir = getInputPath(context);
		FileSystem fs = inputDir.getFileSystem(context.getConfiguration());

		for (FileStatus stat : fs.listStatus(inputDir)) {
			categoryDirs.add(stat);
		}

		int splitSize = 4;
		int splitLength = categoryDirs.size() / splitSize;
		for (int i = 1; i <= splitSize; i++) {
			int start = (i - 1) * splitLength;
			if (i == splitSize) {
				splitLength = categoryDirs.size() - start;
			}
			Path[] paths = new Path[splitLength];
			Set<String> hosts = new HashSet<String>();
			for (int j = 0; j < splitLength; j++) {
				FileStatus stat = categoryDirs.get(start + j);
				paths[j] = stat.getPath();
				FileStatus file = fs.listStatus(paths[j])[0];
				for (String host : fs.getFileBlockLocations(file, 0,
						file.getLen())[0].getHosts()) {
					hosts.add(host);
				}
			}
			if ((splitLength != 0)) {
				splits.add(new MultiPathSplit(paths, splitLength, hosts
						.toArray(new String[0])));
			}
		}
		return splits;
	}

}
