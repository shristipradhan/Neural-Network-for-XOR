/*
 * A Neural network implementation of XOR training set.
 * Created by: Shristi Pradhan (shristi.pr@gmail.com)
 */

package NNXOR;

import java.io.*;

/*
 * NeuralNetMain class is the controller class to the NeuralNetAlgo class  
 */
public class NeuralNetMain {
	
	// Initialization of parameters: 
	// Number of patters, learning rate, momentum, number of hidden layers, error limit, binary/bipolar flag, epochs limit
	public static int numPattern=4;
	public static double rateLearn=0.2;
	public static double momentum=0.9;
	public static int numHidLayNeuron=4;
	public static double errLimit=0.05;
	public static int flagBinary=0;
	public static int maxEpoch=99999;
	
	/*
	 *  Main Program
	 */
	
	public static void main(String[] args){
		int countEpoch=0;
		double deltaEpoch;
		double outPattern;
		double avgAbsErrPattern;
		
		NeuralNetAlgo neuralNetXOR=new NeuralNetAlgo(numPattern,numHidLayNeuron,flagBinary,rateLearn,momentum,errLimit);
		
		neuralNetXOR.readData();
		neuralNetXOR.initializeWeight();
		
		try{
			FileWriter fstreamNumEpoch=new FileWriter("NumEpoch.txt");
			BufferedWriter outNumEpoch=new BufferedWriter(fstreamNumEpoch);
			
			FileWriter fstreamErrEpoch=new FileWriter("ErrEpoch.txt");
			BufferedWriter outErrEpoch=new BufferedWriter(fstreamErrEpoch);
		
		// while loop terminates when the total error from the network
		// is less than the error limit or the total epochs crosses the maximum epochs limit
		while(true){
			
			deltaEpoch=0.0;
			
			for(int p=0;p<numPattern;p++){
				outPattern=neuralNetXOR.outputFor(p);
				
				neuralNetXOR.train(p,outPattern);
				
				deltaEpoch=deltaEpoch+neuralNetXOR.computeTotErr(p,outPattern);
			}
				countEpoch=countEpoch+1;
				
				System.out.println("Epoch# "+countEpoch);
				outNumEpoch.write(countEpoch+"");
				outNumEpoch.newLine();
				
				outErrEpoch.write(deltaEpoch+"");
				outErrEpoch.newLine();
				
				if(deltaEpoch<errLimit || countEpoch>maxEpoch){
					outNumEpoch.close();
					outErrEpoch.close();
					
					break;
				}					
		}	
	}
		catch(Exception e){
			System.err.println("ERROR: "+e.getMessage());
		}
		
		avgAbsErrPattern=neuralNetXOR.computeAvgErr();
		
		neuralNetXOR.saveWeight();
		
		System.out.println("Number of Epochs="+countEpoch);
		
		System.out.println("Average absolute error="+avgAbsErrPattern);
		
	}
	
}
