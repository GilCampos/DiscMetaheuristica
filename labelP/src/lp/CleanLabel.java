/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package lp;

/**
 *
 * @author Gilma
 */
import java.io.IOException;
import java.util.concurrent.ExecutionException;


public class CleanLabel {

	
	public static void main(String[] args) throws IOException, InterruptedException, ExecutionException {
		LabelPropagation lp = new LabelPropagation();
		int numNodes = 916836;
		
		//input is "edgelist" format "id id" sorted by first id
		lp.readEdges(numNodes, "edges.list");
		
		lp.readMemberships("iterXXmemberships.txt");
		lp.writeMembershipsSmart("iterXXmemberships_smart.txt");

	}
}