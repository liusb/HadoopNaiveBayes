package com.github.liusb.bayes;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.InputSplit;

public class MultiPathSplit extends InputSplit implements Writable {

	private Path[] paths;
	private long length;
	private String[] hosts;

	public MultiPathSplit() { }

	public MultiPathSplit(Path[] paths, long length, String[] hosts) {
		this.paths = paths;
		this.length = length;
		this.hosts = hosts;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeLong(length);
		out.writeInt(paths.length);
		for (Path p : paths) {
			Text.writeString(out, p.toString());
		}
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		length = in.readLong();
		int filesLength = in.readInt();
		paths = new Path[filesLength];
		for (int i = 0; i < filesLength; i++) {
			paths[i] = new Path(Text.readString(in));
		}
	}

	public Path[] getPaths() {
		return paths;
	}

	@Override
	public long getLength() throws IOException, InterruptedException {
		return length;
	}

	@Override
	public String[] getLocations() throws IOException {
		return this.hosts;
	}

	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < paths.length; i++) {
			if (i == 0) {
				sb.append("Paths:");
			}
			sb.append(paths[i].toUri().getPath());
			if (i < paths.length - 1) {
				sb.append(",");
			}
		}
		if (hosts != null) {
			StringBuffer hostsb = new StringBuffer();
			for (int i = 0; i < hosts.length; i++) {
				hostsb.append(hosts[i] + ":");
			}
			sb.append(" hosts:" + hostsb.toString() + "; ");
		}
		return sb.toString();
	}
}