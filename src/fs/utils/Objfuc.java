/** 
 * Copyright (c) 2023, An-Da Li. All rights reserved. 
 * Please read LICENCE for license terms.
 * Coded by An-Da Li
 * Email: andali1989@163.com
 *
 * Li, A.-D., Xue, B., & Zhang, M. (2021). A Forward Search Inspired Particle Swarm Optimization Algorithm 
 * for Feature Selection in Classification. IEEE Congress on Evolutionary Computation, CEC 2021, Kraków, 
 * Poland, June 28 - July 1, 2021, 786–793. https://doi.org/10.1109/CEC45853.2021.9504949
 *
 */

package fs.utils;

public class Objfuc {
	private int type = 0;
	public double[] W = {0.9, 0.1};
	public Objfuc() {
		super();
		setOptions(type);
	}
	public Objfuc(int type) {
		setOptions(type);
	}
	/**
	 * 
	 * @param type 
	 * type = 0 acc and feature number		
	 *  type = 1 gmean and feature number
	 */
	public void setOptions(int type) {
		
		this.type = type;
	}
	public void setWeight(double accW, double fnumW) {
		W[0] = accW;
		W[1] = fnumW;
	}
	public double fit(double[] acc, int fsNum, int fNum) {
		
		double fitness = 0;
		double fs = (double) fsNum;
		double full = (double) fNum;
		if(type == 0) {
			fitness = W[0] * acc[acc.length - 1] + W[1] * (1 - fs / full); 
		}else if(type == 1) {
			double gm = acc[0];
			int sub = 0;
			for(int i = 1; i < acc.length -1; i++) {
				if(Double.isNaN(acc[i])) {
					sub++;
				}else {
					gm = gm * acc[i];
				}
				
			}
			fitness = W[0] * Math.pow(gm, 1.0/(acc.length - sub - 1)) + W[1] * (1 - fs / full); 
			//System.out.println(String.valueOf(fitness));
		}
		return fitness;
		
	}
	public double fit(double[] acc, int fsNum, int fNum, double[] W) {
		

		double fitness = 0;
		double fs = (double) fsNum;
		double full = (double) fNum;
		if(type == 0) {
			fitness = W[0] * acc[acc.length - 1] + W[1] * (1 - fs / full); 
		}else if(type == 1) {
			double gm = 1;
			int sub = 0;
			for(int i = 0; i < acc.length -1; i++) {
				if(Double.isNaN(acc[i])) {
					sub++;
				}else {
					gm = gm * acc[i];
				}
				
			}
			fitness = W[0] * Math.pow(gm, 1.0/(acc.length - 1 -sub)) + W[1] * (1 - fs / full); 
		}
		return fitness;

	}
	public double fit(double acc, int fsNum, int fNum) {
		double fitness = 0;
		double fs = (double) fsNum;
		double full = (double) fNum;
		
		fitness = W[0] * acc + W[1] * (1 - fs / full); 
		
		return fitness;
	}

}
