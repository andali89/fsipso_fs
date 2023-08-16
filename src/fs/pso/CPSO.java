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

package fs.pso;

import java.util.BitSet;
import java.util.Random;

import fs.utils.Matcd;

import fs.utils.Wkc;
import weka.attributeSelection.SubsetEvaluator;
import weka.core.Instances;
import weka.core.Utils;

public class CPSO  extends Absbpso {
	
	protected double W = 0.7298;
	protected double C1 = 1.49618;
	protected double C2 = 1.49618;
	public double vThred = 4;
	public double bitThred = 0.6;

	public CPSO(Instances data, Wkc classi) {
		super(data, classi);
	}
	public CPSO(Instances data, Wkc classi, int popNum, int iterTime) {
		super(data, classi);
		double [] a = {1.0, 0};
		this.objWeight = a;
		this.setPara(popNum, iterTime);
	}


	public void setPara(int popNum, int iterTime) {
		// set the parameters of the sbpso
		this.popNum = popNum;
		this.iterTime = iterTime;
	}
	
	protected void resetPara() {
		this.iterInfo = new double[this.iterTime];
		this.iterNum = new double[iterTime + 1];
		this.iterMeanFit = new double[iterTime + 1];
		this.iterMeanNum = new double[iterTime +1];
		this.iterMeanPbestFit = new double[iterTime + 1];
		this.iterMeanPbestNum = new double[iterTime +1];
		//this.addedFnum = (int) Math.ceil((double)this.fNum * step / this.iterTime);
		
	}

	@Override
	public void run()  {
		// run the sbpso to find the best solution
		this.resetPara();
		System.out.printf("The VThred is %f", this.vThred);
		long startTime = System.currentTimeMillis(); // start time
		double[][] pos = new double[popNum][fNum];
		double[][] vel = new double[popNum][fNum];
		newInitialization(pos, vel, popNum / 3, bitThred, rnd);
		double[] fitness = null;
		 
		try {
			fitness = this.getFitness(pos);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		double[] index = Matcd.maxVector(fitness);
		double[] gbest = pos[(int)index[0]].clone();
		double gbestFit = fitness[(int) index[0]];
		
       
        
		//copy the pos to best0 and pbest1
		double[][] pbest = new double[this.popNum][this.fNum];

		for(int i = 0; i < this.popNum; i++) {
			pbest[i] = pos[i].clone();

		}

		double[] pbestFit = fitness.clone();
		
		
		// getInfo
		
		iterNum[0] = Matcd.sumVector(pos[(int)index[0]]);
		this.fitZero = gbestFit;		
		iterMeanFit[0] = Matcd.meanVector(fitness);
		double[] posNums = getNum(pos);
		this.iterMeanNum[0] = Matcd.meanVector(posNums);
		double[] pbestNums = posNums.clone();
		this.iterMeanPbestFit = iterMeanFit.clone();
		this.iterMeanPbestNum = iterMeanNum.clone();
		double gbestNum = posNums[(int)index[0]];

		
		// start the iteration
		for(int i = 0; i < this.iterTime; i++) {
			//iteration stop until the number of iterations
			for(int j = 0; j < this.popNum; j++) {
				// for each individual in the population
				for(int k = 0; k < this.fNum; k++) {
					// for each element in the individual
					vel[j][k] =  W * vel[j][k] + C1 * rnd.nextDouble()*(pbest[j][k] - pos[j][k])
							+ C2 * rnd.nextDouble() * (gbest[k] - pos[j][k]);
					
					if(vel[j][k] > vThred) {
						vel[j][k] = vThred;
					}else if(vel[j][k] < -vThred){
						vel[j][k] = - vThred;
					}
					
					
					pos[j][k] = pos[j][k] + vel[j][k];
					
					// normarlize the position to [0 , 1]
					if(pos[j][k] > 1) {
						pos[j][k] = 1;
					}else if(pos[j][k] < 0) {
						pos[j][k] = 0;
					}
					
					
				}
				// get the fitness of individual j
				fitness[j] = this.getFitness(pos[j]);
				//System.out.printf("i th acc is %f", fitness[j]);
				posNums[j] = getNum(pos[j]);
				//update the gbest and pbest
				
				if(fitness[j] > pbestFit[j] ||
						(fitness[j] == pbestFit[j] && posNums[j] < pbestNums[j]) ) {
					pbest[j] = pos[j].clone();
					pbestFit[j] = fitness[j];
					pbestNums[j] = posNums[j];
				}

			}

			int cpbestIn = Utils.maxIndex(fitness);
			if(fitness[cpbestIn] > gbestFit || 
					(fitness[cpbestIn] == gbestFit && posNums[cpbestIn] < gbestNum)  ) {
				gbest = pos[cpbestIn].clone();
				gbestFit = fitness[cpbestIn];
				gbestNum = posNums[cpbestIn];

			}


			this.iterInfo[i] = gbestFit;
			this.iterNum[i + 1] = gbestNum;			
			
			this.iterMeanFit[i + 1] = Matcd.meanVector(fitness);
			this.iterMeanNum[i + 1] = Matcd.meanVector(posNums);
			this.iterMeanPbestFit[i + 1] = Matcd.meanVector(pbestFit);
			this.iterMeanPbestNum[i + 1] = Matcd.meanVector(pbestNums);
			
			
			System.out.printf("iteration time: %d, fitness value: %f\n", i + 1, iterInfo[i]);
		}
		this.gbest = gbest;
		this.gbestFit = gbestFit;
		//measure the runtime of the pso
		this.runtime = (System.currentTimeMillis() - startTime) / 1000.0;
	}
    
	@Override
	protected double[] getNum(double[][] pos) {
		// TODO Auto-generated method stub
		double[] nums = new double[pos.length];
		for(int i = 0; i < pos.length; i++) {
			nums[i] = getNum(pos[i]);
		}
		return nums;
	}
	
	@Override
	public double getNum(double[] pos) {
		// TODO Auto-generated method stub
		double num = 0;
		for(int i = 0; i < pos.length; i++) {
			num += (pos[i] > bitThred ? 1 : 0);
		}
		return num;
	}
	
	@Override
	public double getFitness(double[] pos) {
		double acc = 0;
		BitSet bitset = new BitSet(fNum + 1);
		for(int i = 0; i < fNum; i++) {
			if(pos[i] > bitThred) {
				bitset.set(i);
			}else {
				bitset.clear(i);
			}
		}
		//bitset.clear(fNum);

		try {
			acc = ((SubsetEvaluator) ASEval).evaluateSubset(bitset);

		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

//		try {
//			acc = getAcc(pos);
//		} catch (Exception e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//		return getFitness(acc, this.obj);
	
		return acc;
	}
	
	/**
	 * Generate the swarm position and velocity, the small number of position is generated 
	 * as over 50 percent of features are selected, the large number of postion is generated 
	 * as 10 percent of features are selected. the velocity is generated in [-4, 4] randomly.
	 * 
	 * @param pos
	 * @param velocity
	 * @param smallnum
	 * @param thred
	 * @param rnd
	 */
	public void newInitialization(double[][] pos, double[][] velocity, 
			int smallnum, double thred, Random rnd) {
		double scaleV = vThred;
		int fnum = pos[0].length;
		int popNum = pos.length;
		int largenum = popNum - smallnum;
		
		// for large part of swarm
		double[][] largeswarm = new double[largenum][fnum];
		
		for(int s = 0; s < largenum; s++) {
			for(int i = 0; i <fnum; i++) {
				if(rnd.nextDouble() < 0.1) {
					// select
					largeswarm[s][i] = thred +  rnd.nextDouble() * (1 - thred);
				}else {
					largeswarm[s][i] = 1 - (thred +  rnd.nextDouble() * (1 - thred));
				}
			}		
			
		}
		
		
		
		// for that in half to whole
		double[][] smallswarm = new double[smallnum][fnum];
		
		for(int s = 0; s < smallnum; s++) {
			
			int toselec = Matcd.randomInt(fnum / 2, fnum, rnd);
			// generate a set of weights, then sort to find first tosec number of features.
			double[] weights = new double[fnum];
			for(int i = 0; i < fnum; i++) {
				weights[i] = rnd.nextDouble();			
			}
			int[] index = Utils.sort(weights);
			for(int i = 0; i < fnum; i++) {
				if(i < toselec) {
					// to select the feature
					smallswarm[s][index[i]] = thred +  rnd.nextDouble() * (1 - thred);
				}else {
					smallswarm[s][index[i]] = 1 - (thred +  rnd.nextDouble() * (1 - thred));
				}				
			}			
			
		}
		
		// copy the generated large and small swarm to the pos
		for(int i = 0; i < popNum; i++) {
			if(i < largenum) {
				System.arraycopy(largeswarm[i], 0, pos[i], 0, fnum);
			}else {
				System.arraycopy(smallswarm[i - largenum], 0, pos[i], 0, fnum);
			}		
			
		}
		
		
		//initialize velocity
		for(int i = 0; i < popNum; i++) {
			for(int j = 0; j < fnum; j++) {
				velocity[i][j] = - scaleV + rnd.nextDouble() * 2 * scaleV;				
			}			
		}
		System.out.printf("The VThred is %f", scaleV);
		
	}
	
	
	

}
