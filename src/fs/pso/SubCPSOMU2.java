package fs.pso;

import fs.utils.GetRank;
import fs.utils.Matcd;
import fs.utils.Wkc;
import weka.core.Instances;
import weka.core.Utils;

//Compared with SubCPSO, this version initialize velocity as 0.
public class SubCPSOMU2 extends SubCPSO2 {
    // combined with mutation
	public SubCPSOMU2(Instances datain, Wkc classi, int popNum, int iterTime, int step) {
		// TODO Auto-generated constructor stub
		super(datain, classi, popNum, iterTime, step);
	}
	
	
	@Override
	public void run() {
		// TODO Auto-generated method stub
		System.out.println("SubCPSOMU2 version 2020.11.01");
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
            double mrate = 1.0 / ableIndex.length;
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
					
					//Mutation
					if(rnd.nextDouble() < mrate) {
						pos[j][k] = 1 - pos[j][k];
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
	
}
