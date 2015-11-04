package com.github.liusb.bayes;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;

public class PairWritable<A extends WritableComparable<A>, B extends WritableComparable<B>>
		implements WritableComparable<PairWritable<A, B>> {

	public A left;
	public B right;

	@Override
	public void write(DataOutput out) throws IOException {
		left.write(out);
		right.write(out);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		left.readFields(in);
		right.readFields(in);
	}

	@Override
	public int compareTo(PairWritable<A, B> o) {
		int result = left.compareTo(o.left);
		if (result == 0) {
			result = right.compareTo(o.right);
		}
		return result;
	}
	
	@Override
	public String toString() {
		return left.toString() + "\t" + right.toString();
	}

}
