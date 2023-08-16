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

import java.util.Random;
import java.util.ArrayList;
import fs.utils.GetRank;
import fs.utils.Matcd;
import fs.utils.Wkc;
import weka.core.Instances;
import weka.core.Utils;
// Compared with SubCPSO, this version initialize velocity as 0.
public class SubCPSO2 extends CPSO {
	
	protected int step;
	protected int[] ranklist; // get the rank list according to the ranking method in descending order
	protected ArrayList<int[]> arrayIndex;
	public String iniMethod = "SU";
    public boolean expandAll = true;
    
	public SubCPSO2(Instances datain, Wkc classi) {
		super(datain, classi);
		// TODO Auto-generated constructor stub
	}
	
	
	public SubCPSO2(Instances datain, Wkc classi, int popNum, int iterTime, int step) {
		super(datain, classi);
		// TODO Auto-generated constructor stub
		this.fNum = datain.numAttributes() - 1;
		this.setPara(popNum, iterTime, step);
		
	}
	
	/**
	 * 
	 * @param popNum
	 * @param iterTime
	 * @param step  the step to increase the search space
	 *
	 */
	public void setPara(int popNum, int iterTime, int step) {
		// set the parameters of the sbpso
		this.popNum = popNum;
		this.iterTime = iterTime;
		this.step = step;
		
	}
	
	/**
	 * reset the parameters at each run
	 */
	protected void resetPara() {
		this.iterInfo = new double[this.iterTime];
		this.iterNum = new double[iterTime + 1];
		this.iterMeanFit = new double[iterTime + 1];
		this.iterMeanNum = new double[iterTime +1];
		this.iterMeanPbestFit = new double[iterTime + 1];
		this.iterMeanPbestNum = new double[iterTime +1];
		//this.addedFnum = (int) Math.ceil((double)this.fNum * step / this.iterTime);
		
	}
	
	/**
	 * 
	 * @param iter  the iteration counter "t"
	 * @param index the index of the global best in the pos
	 * @param posNums the number of features of each position
	 * @param fits    the fitness of each position
	 * @param pbestFits the fitness of each pbest position 
	 * @param pbestNums the number of features of each pbest position
	 */
	protected void setIterInfo(int iter, double[] posNums, double[] fits,
			double[] pbestFits, double[] pbestNums) {
		
		iterNum[iter] = this.getNum(gbest);
		
		if(iter==0) {
			fitZero = gbestFit;	
			System.out.printf("iteration time: %d, fitness value: %f\n", 0, fitZero);
		}else {
			iterInfo[iter - 1] = this.gbestFit;
			System.out.printf("iteration time: %d, fitness value: %f\n", iter, iterInfo[iter -1]);
		}					
		iterMeanFit[iter] = Matcd.meanVector(fits);		
		iterMeanNum[iter] = Matcd.meanVector(posNums);		
		iterMeanPbestFit[iter] = Matcd.meanVector(pbestFits);
		iterMeanPbestNum[iter] = Matcd.meanVector(pbestNums);		
	}
	

	@Override
	public void run() {
		// TODO Auto-generated method stub
		System.out.println("SubCPSO2 version 2020.11.01");
		double[][] pos = Matcd.sameNums(popNum, fNum, 0.0);
		double[][] vel = Matcd.sameNums(popNum, fNum, 0.0);
		
		// Initialization and setup
		long startTime = System.currentTimeMillis(); // start time
		this.resetPara();	
		// Get the rank list in descending order
		if(iniMethod.toLowerCase().equals("acc")) {
			this.ranklist = GetRank.getRankListDe(data, ASEval);
		}else {
			this.ranklist = GetRank.getRankListDe(iniMethod, data);
		}
		
		
		// Get a set of index for the initialization at different initialization phases
		getAddIndex();
		int addPhase = 0;
		
		// Initialization in the first phase, the first "numInFeature" features are initialized
		int[] curIndex = this.arrayIndex.get(addPhase);
		this.iniPop(pos, vel, curIndex, rnd);
		addPhase++;
		
		// get the fitness of the initialized population
		double[] fits = null;	
		try {
			fits = this.getFitness(pos);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		int index = Utils.maxIndex(fits);
		gbestFit = fits[index];
		//double[] index = Matcd.maxVector(fitness);
		gbest = pos[index].clone();
		double[] posNums = getNum(pos);		
		double[][] pbest = new double[this.popNum][this.fNum];
		for(int i = 0; i < this.popNum; i++) {
			pbest[i] = pos[i].clone();

		}
		double[] pbestFits = fits.clone();		
		double[] pbestNums = posNums.clone();
		setIterInfo(0, posNums, fits, pbestFits, pbestNums);		
		
		int[] ableIndex = curIndex.clone();
		// iterations
		for(int i = 0; i < this.iterTime; i++) {
			//iteration stop until the number of iterations
			if(Math.floorMod(i, this.step) == 0 && i > 0) {
				
				
				ableIndex = reInitilization(vel, pos, fits, pbest, pbestFits, addPhase, posNums, pbestNums, ableIndex);			
				addPhase++;
				setIterInfo(i + 1, posNums, fits, pbestFits, pbestNums);	
				i++; // add iterator by 1 
			}		

			for(int j = 0; j < this.popNum; j++) {
				// for each individual in the population
				for(int k : ableIndex) {
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
				fits[j] = this.getFitness(pos[j]);				
				posNums[j] = getNum(pos[j]);
				
				// update pbest
				if(fits[j] > pbestFits[j]) {
					pbest[j] = pos[j].clone();
					pbestFits[j] = fits[j];
					pbestNums[j] = posNums[j];
				}
				
			}
			
			// update gbest
			index = Utils.maxIndex(fits);
			if(fits[index] > gbestFit) {
				gbest = pos[index].clone();
				gbestFit = fits[index];
			}
			setIterInfo(i + 1, posNums, fits, pbestFits, pbestNums);	
		}
		
		this.runtime = (System.currentTimeMillis() - startTime) / 1000.0;
		

	}
	
	/**
	 * 
	 * @param vel
	 * @param pos
	 * @param fits
	 * @param pbest
	 * @param pbestFits
	 * @param curIndex
	 * @param posNums
	 * @param pbestNums
	 * 
	 * reInitilization and update gbest and pbests
	 */
	protected int[] reInitilization(double[][] vel, double[][] pos, double[] fits, 
			double[][] pbest, double[] pbestFits, int addPhase, double[] posNums, double[] pbestNums, int[] ableIndexIn) {
		int[] curIndex = this.arrayIndex.get(addPhase);
		int[] keepIndex = null;
		if(expandAll) {
			// all the feature space in current phase is kept
		    keepIndex = ableIndexIn.clone();
		}else {
			keepIndex = getKeepIndex(pos);
			// only the feature space selected by pos are kept
		}
		
		int[] ableIndex= new int[keepIndex.length + curIndex.length]; // the Index that are able to update
		System.arraycopy(keepIndex, 0, ableIndex, 0, keepIndex.length);
		System.arraycopy(curIndex, 0, ableIndex, keepIndex.length, curIndex.length);
		this.iniPop(pos, vel, curIndex, rnd);
		for (int j = 0; j < pos.length; j++) {
			try {
				fits[j] = this.getFitness(pos[j]);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			posNums[j] = this.getNum(pos[j]);
			if(fits[j] > pbestFits[j]) {
				pbest[j] = pos[j].clone();
				pbestFits[j] = fits[j];
				pbestNums[j] = posNums[j];
			}			
		}
		int index = Utils.maxIndex(pbestFits);
		if(pbestFits[index] > gbestFit) {
			gbestFit = pbestFits[index];
			gbest = pbest[index];			
		}
		return ableIndex;
		
	}

    /**
     * 
     * @param pos
     * @return
     * If any position select a feature the index of this feature is kept.
     * 
     */
	protected int[] getKeepIndex(double[][] pos) {
		// TODO Auto-generated method stub
		ArrayList<Integer> arrayKeep = new ArrayList<Integer>();
		for(int j = 0; j < pos[0].length; j++) {
			for(int i = 0; i < pos.length; i++) {
				if(pos[i][j] > bitThred) {
					arrayKeep.add(j);
					break;
				}
			}
		}
		int [] keepIndex = new int[arrayKeep.size()];
		for(int i = 0; i < keepIndex.length; i++) {
			keepIndex[i] = arrayKeep.get(i);
		}
		return keepIndex;
	}


	/**
	 * 
	 * @return
	 * get the index to initialize and add to arrayIndex;
	 * return the number of features to initialize for each initialization
	 * 
	 */
	protected int  getAddIndex() {
		int numIn = Math.floorDiv(iterTime, step);
		int addedFnum = (int) Math.ceil((double)this.fNum * step / this.iterTime);
		int rankJ = 0; // Counter on the ranklist
		arrayIndex = new ArrayList<int[]>();
		int[] tempIndex = new int[addedFnum];
		for(int i = 0; i < numIn - 1; i++) {
			for(int j = 0; j < addedFnum; j++) {
				tempIndex[j] = this.ranklist[rankJ];
				rankJ++;
			}
			arrayIndex.add(tempIndex.clone());
		}
		
		tempIndex = new int[fNum - rankJ];
		for(int j = 0; j < tempIndex.length; j++) {
			tempIndex[j] = this.ranklist[rankJ];
			rankJ++;
		}
		arrayIndex.add(tempIndex.clone());
		
		return addedFnum;
	}
	
	/**
	 * 
	 * @param pos
	 * @param velocity
	 * @param addIndex
	 * @param rnd
	 * 
	 * Initialization
	 * 
	 */
	protected void iniPop(double[][] pos, double[][] velocity, int[] addIndex, Random rnd) {		
		//double scaleV = vThred;
		
		int popNum = pos.length;
		
		//initialize population and velocity
		for(int i = 0; i < popNum; i++) {
			for(int j : addIndex) {
				pos[i][j] = rnd.nextDouble();
				// this version do not initialize velocity			
			}			
		}
		
	}

}
